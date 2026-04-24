"""V6 Batch 3 — mambanet_2ch scaling experiment.

Pretrain / finetune / eval entry points callable from sweep_v6b3.py or CLI.

CLI usage:
  python src/train/train_v6b3.py pretrain --size 500K --seed 0
  python src/train/train_v6b3.py finetune --size 500K --seed 0
  python src/train/train_v6b3.py finetune --size 500K --seed 2 --from-seed 1
  python src/train/train_v6b3.py eval     --size 500K --seed 0
"""

import sys, csv, time, math, argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.competitors import build_model
from src.models.v5_model import v5_loss, _SNR_MIN, _SNR_RANGE
from src.data_zhu import zhu_train_dataset, zhu_test_dataset, TEST_CONDITIONS
from src.train.train_v5 import (
    _calibrate_snr_estimator, estimate_snr, _preload_zhu, _norm_snr,
)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR   = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results"
DATA_B3    = ROOT / "data" / "synth_v6b3"

CKPT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

SIZE_MAP = {"500K": 500_000, "1M": 1_000_000, "2M": 2_000_000, "5M": 5_000_000}

# Batch 3 fixed hyperparameters
PRE_BATCH      = 512
PRE_LR         = 1e-3
PRE_LR_MIN     = 1e-5
PRE_MAX_EPOCHS = 40
PRE_PATIENCE   = 5
PRE_WARMUP     = 500   # steps

FT_BATCH       = 128
FT_LR          = 3e-4
FT_LR_MIN      = 1e-6
FT_EPOCHS      = 30
FT_PATIENCE    = 10

TEST_N = 700   # frames per condition


# ── Utilities ──────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % (2**32))


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_condition(cond: str) -> tuple:
    """Return (bt, m_str) for CSV."""
    bt = 0.3 if "Tb0d3" in cond else 0.5
    if "Awgn" in cond:
        m = "AWGN"
    else:
        m = 1.2 if "m1d2" in cond else 1.4
    return bt, m


# ── Dataset ────────────────────────────────────────────────────────────────

class SynthNpyDataset(Dataset):
    """Loads pre-generated .npy memmap files from data/synth_v6b3/<size>/."""

    def __init__(self, data_dir: Path, mmap: bool = False):
        mode = "r" if mmap else None
        self.xs   = np.load(str(data_dir / "xs.npy"),   mmap_mode=mode)
        self.ys   = np.load(str(data_dir / "ys.npy"),   mmap_mode=mode)
        self.snrs = np.load(str(data_dir / "snrs.npy"), mmap_mode=mode)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.xs[idx].copy()),
            torch.from_numpy(self.ys[idx].copy()),
            torch.tensor(float(self.snrs[idx])),
        )


# ── LR scheduler with warmup ───────────────────────────────────────────────

def make_warmup_cosine(opt, warmup_steps: int, total_steps: int, eta_min_ratio: float):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ── One-epoch runner ───────────────────────────────────────────────────────

def run_epoch(model, loader, opt, sched_step_fn,
              slope, intercept, is_train: bool, use_gt_snr: bool):
    """Returns (avg_loss, avg_ber). sched_step_fn called after each train step."""
    model.train(is_train)
    tot_loss = tot_ber = n = 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            if use_gt_snr and len(batch) == 3:
                snr = batch[2].to(DEVICE)
            else:
                snr = estimate_snr(x, slope, intercept)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                logits, snr_pred = model(x, snr)

            snr_norm = _norm_snr(snr)
            loss, _ = v5_loss(logits.float(), y, snr_pred.float(), snr_norm)

            if is_train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sched_step_fn is not None:
                    sched_step_fn()

            B = len(x)
            tot_loss += loss.item() * B
            with torch.no_grad():
                ber = ((torch.sigmoid(logits.float()) > 0.5).float() != y).float().mean().item()
            tot_ber += ber * B
            n += B

    return tot_loss / n, tot_ber / n


# ── Pretrain ───────────────────────────────────────────────────────────────

def pretrain(size_label: str, seed: int, snr_source: str = "linear") -> float:
    """Pretrain mambanet_2ch from random init on synthetic corpus.

    Returns best val BER achieved.
    """
    set_seed(seed)
    data_dir = DATA_B3 / size_label
    n_samples = SIZE_MAP[size_label]
    use_mmap  = n_samples >= 2_000_000

    print(f"\n{'='*70}")
    print(f"[{ts()}] PRETRAIN  size={size_label}  seed={seed}  snr={snr_source}  {DEVICE}")
    print(f"{'='*70}")

    # SNR estimator calibration
    slope, intercept = _calibrate_snr_estimator()

    # Synthetic data
    synth_ds = SynthNpyDataset(data_dir, mmap=use_mmap)
    g = torch.Generator(); g.manual_seed(seed)
    synth_loader = DataLoader(
        synth_ds, batch_size=PRE_BATCH, shuffle=True,
        num_workers=4, pin_memory=True,
        generator=g, worker_init_fn=worker_init_fn,
    )

    # Zhu validation data
    _, zhu_val_raw = zhu_train_dataset(val_frac=0.111, seed=seed)
    zhu_val = _preload_zhu(zhu_val_raw, "Zhu-val")
    val_loader = DataLoader(zhu_val, batch_size=PRE_BATCH, shuffle=False, pin_memory=True)

    model = build_model("mambanet_2ch").to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  mambanet_2ch  params={n_params:,}  data={n_samples:,}  mmap={use_mmap}")

    opt = torch.optim.AdamW(model.parameters(), lr=PRE_LR, weight_decay=1e-4)
    steps_per_epoch = math.ceil(n_samples / PRE_BATCH)
    total_steps     = PRE_MAX_EPOCHS * steps_per_epoch
    sched = make_warmup_cosine(opt, PRE_WARMUP, total_steps, PRE_LR_MIN / PRE_LR)

    log_path = RESULT_DIR / f"v6b3_pre_{size_label}_s{seed}_log.csv"
    log_fh   = open(log_path, "w", newline="")
    log_w    = csv.writer(log_fh)
    log_w.writerow(["epoch", "steps", "train_loss", "val_ber", "lr", "elapsed_s"])

    best_ber = float("inf")
    patience_ctr = 0
    global_step  = 0
    best_ckpt    = None

    for ep in range(1, PRE_MAX_EPOCHS + 1):
        t0 = time.time()

        # --- train ---
        model.train()
        tot_loss = n = 0
        for batch in synth_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            snr  = batch[2].to(DEVICE)   # GT SNR from synthetic data

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                logits, snr_pred = model(x, snr)
            snr_norm = _norm_snr(snr)
            loss, _ = v5_loss(logits.float(), y, snr_pred.float(), snr_norm)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            global_step += 1

            tot_loss += loss.item() * len(x)
            n        += len(x)

            # Step-level checkpoint
            if global_step % 5000 == 0:
                step_k = global_step // 1000
                p = CKPT_DIR / f"v6b3_pre_{size_label}_step{step_k}k_s{seed}.pt"
                torch.save({"step": global_step, "model": model.state_dict()}, p)

        train_loss = tot_loss / n

        # --- val ---
        _, val_ber = run_epoch(model, val_loader, None, None,
                               slope, intercept, is_train=False, use_gt_snr=False)

        lr      = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"[{ts()}] epoch={ep:02d} steps={global_step} "
              f"train_loss={train_loss:.4f} val_ber={val_ber:.4f} lr={lr:.2e} {elapsed:.0f}s")
        log_w.writerow([ep, global_step, round(train_loss, 6), round(val_ber, 6),
                        f"{lr:.2e}", round(elapsed, 1)])
        log_fh.flush()

        if val_ber < best_ber:
            best_ber     = val_ber
            patience_ctr = 0
            best_ckpt    = CKPT_DIR / f"v6b3_pre_{size_label}_s{seed}.pt"
            torch.save({"model": model.state_dict(), "val_ber": best_ber,
                        "epoch": ep, "step": global_step}, best_ckpt)
            print(f"  *** new best val_ber={best_ber:.4f}  → {best_ckpt.name}")
        else:
            patience_ctr += 1
            print(f"  patience {patience_ctr}/{PRE_PATIENCE}")
            if patience_ctr >= PRE_PATIENCE:
                print(f"  Early stop at epoch {ep}.")
                break

    log_fh.close()
    print(f"  Pretrain done.  best_val_ber={best_ber:.4f}  converged_ep={ep}  ckpt={best_ckpt}")
    return best_ber


# ── Fine-tune ──────────────────────────────────────────────────────────────

def finetune(size_label: str, seed: int, from_seed: int | None = None,
             snr_source: str = "linear") -> float:
    """Fine-tune on Zhu data from pretrain checkpoint. Returns best val BER."""
    set_seed(seed)
    pretrain_seed = from_seed if from_seed is not None else seed
    pre_ckpt = CKPT_DIR / f"v6b3_pre_{size_label}_s{pretrain_seed}.pt"

    print(f"\n{'='*70}")
    print(f"[{ts()}] FINETUNE  size={size_label}  seed={seed}  "
          f"pre_seed={pretrain_seed}  snr={snr_source}  {DEVICE}")
    print(f"{'='*70}")

    slope, intercept = _calibrate_snr_estimator()

    zhu_tr_raw, zhu_val_raw = zhu_train_dataset(val_frac=0.111, seed=seed)
    zhu_tr  = _preload_zhu(zhu_tr_raw,  "Zhu-train")
    zhu_val = _preload_zhu(zhu_val_raw, "Zhu-val  ")
    ft_loader  = DataLoader(zhu_tr,  batch_size=FT_BATCH, shuffle=True,  pin_memory=True)
    val_loader = DataLoader(zhu_val, batch_size=FT_BATCH, shuffle=False, pin_memory=True)

    model = build_model("mambanet_2ch").to(DEVICE)
    ck = torch.load(pre_ckpt, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    print(f"  Loaded pretrain ckpt: {pre_ckpt.name}  (val_ber={ck.get('val_ber', '?'):.4f})")

    opt   = torch.optim.Adam(model.parameters(), lr=FT_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS, eta_min=FT_LR_MIN)

    log_path = RESULT_DIR / f"v6b3_{size_label}_s{seed}_ft_log.csv"
    log_fh   = open(log_path, "w", newline="")
    log_w    = csv.writer(log_fh)
    log_w.writerow(["epoch", "train_loss", "val_ber", "lr", "elapsed_s"])

    best_ber     = float("inf")
    patience_ctr = 0
    best_ckpt    = None
    diverge_lr_applied = False

    for ep in range(1, FT_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_ber = run_epoch(
            model, ft_loader, opt, None,
            slope, intercept, is_train=True, use_gt_snr=False)
        _, val_ber = run_epoch(
            model, val_loader, None, None,
            slope, intercept, is_train=False, use_gt_snr=False)
        sched.step()

        lr      = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"[{ts()}] epoch={ep:02d} train_loss={train_loss:.4f} "
              f"val_ber={val_ber:.4f} lr={lr:.2e} {elapsed:.0f}s")
        log_w.writerow([ep, round(train_loss, 6), round(val_ber, 6),
                        f"{lr:.2e}", round(elapsed, 1)])
        log_fh.flush()

        # Divergence check: val BER > 30% after 10 epochs
        if ep == 10 and val_ber > 0.30 and not diverge_lr_applied:
            print(f"  WARNING: val_ber={val_ber:.3f} > 30% — applying LR × 0.3")
            for pg in opt.param_groups:
                pg["lr"] *= 0.3
            diverge_lr_applied = True

        if val_ber < best_ber:
            best_ber     = val_ber
            patience_ctr = 0
            best_ckpt    = CKPT_DIR / f"v6b3_{size_label}_s{seed}_ft.pt"
            torch.save({"model": model.state_dict(), "val_ber": best_ber, "epoch": ep}, best_ckpt)
            print(f"    *** new best val_ber={best_ber:.4f}  → {best_ckpt.name}")
        else:
            patience_ctr += 1
            if patience_ctr >= FT_PATIENCE:
                print(f"  Early stop at epoch {ep}.")
                break

    # Second divergence check: still > 30% after LR reduction
    if best_ber > 0.30:
        print(f"  ERROR: seed {seed} diverged (best_val_ber={best_ber:.3f}). Skipping.")
        log_fh.close()
        return float("inf")

    log_fh.close()
    print(f"  Finetune done.  best_val_ber={best_ber:.4f}  ckpt={best_ckpt}")
    return best_ber


# ── Evaluate ───────────────────────────────────────────────────────────────

def evaluate(size_label: str, seed: int, snr_source: str = "linear") -> dict:
    """Evaluate fine-tuned checkpoint on Zhu test set.

    Returns dict: condition → (n_bits, n_errors).
    Writes results/v6b3_<size>_s<seed>_test.csv.
    """
    set_seed(seed)
    ft_ckpt = CKPT_DIR / f"v6b3_{size_label}_s{seed}_ft.pt"

    print(f"\n[{ts()}] EVAL  size={size_label}  seed={seed}")

    slope, intercept = _calibrate_snr_estimator()

    model = build_model("mambanet_2ch").to(DEVICE)
    ck = torch.load(ft_ckpt, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()

    rows    = []
    results = {}   # condition → (n_bits, n_errors)

    for cond in TEST_CONDITIONS:
        ds  = zhu_test_dataset(cond)
        sub = torch.utils.data.Subset(ds, list(range(TEST_N)))
        tl  = DataLoader(sub, batch_size=256, shuffle=False, num_workers=2)

        n_bits = n_errors = 0
        with torch.no_grad():
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                snr = estimate_snr(x, slope, intercept)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                    logits, _ = model(x, snr)
                preds = (torch.sigmoid(logits.float()) > 0.5).float()
                errs  = (preds != y).sum().item()
                n_bits   += y.numel()
                n_errors += errs

        ber = n_errors / n_bits
        bt, m = parse_condition(cond)
        results[cond] = (n_bits, n_errors)
        rows.append((cond, bt, m, TEST_N, n_bits, n_errors, round(ber, 6)))
        print(f"  {cond:<26} BER={ber:.4f}")

    overall_bits   = sum(r[4] for r in rows)
    overall_errors = sum(r[5] for r in rows)
    overall_ber    = overall_errors / overall_bits
    rows.append(("OVERALL", "", "", sum(r[3] for r in rows),
                 overall_bits, overall_errors, round(overall_ber, 6)))
    print(f"  {'OVERALL':<26} BER={overall_ber:.4f}")

    csv_path = RESULT_DIR / f"v6b3_{size_label}_s{seed}_test.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "bt", "m", "n_frames", "n_bits", "n_errors", "ber"])
        w.writerows(rows)

    return results


def ensemble_eval(size_label: str, seed_results: dict[int, dict]) -> float:
    """Compute 3-seed ensemble BER and write ensemble CSV.

    seed_results: {seed: {condition: (n_bits, n_errors)}}
    Returns overall ensemble BER.
    """
    all_seeds = sorted(seed_results.keys())
    rows = []

    total_bits = total_errors = 0

    for cond in TEST_CONDITIONS:
        cond_bits   = sum(seed_results[s][cond][0] for s in all_seeds)
        cond_errors = sum(seed_results[s][cond][1] for s in all_seeds)
        ber = cond_errors / cond_bits
        bt, m = parse_condition(cond)
        rows.append((cond, bt, m,
                     TEST_N * len(all_seeds),
                     cond_bits, cond_errors, round(ber, 6)))
        total_bits   += cond_bits
        total_errors += cond_errors

    ensemble_ber = total_errors / total_bits
    rows.append(("OVERALL", "", "",
                 TEST_N * len(TEST_CONDITIONS) * len(all_seeds),
                 total_bits, total_errors, round(ensemble_ber, 6)))

    csv_path = RESULT_DIR / f"v6b3_{size_label}_ensemble_test.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "bt", "m", "n_frames", "n_bits", "n_errors", "ber"])
        w.writerows(rows)

    print(f"  Ensemble ({size_label}) overall BER={ensemble_ber*100:.4f}%  → {csv_path.name}")
    return ensemble_ber


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    for cmd in ("pretrain", "finetune", "eval"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--size",       required=True, choices=list(SIZE_MAP))
        sp.add_argument("--seed",       type=int, required=True)
        sp.add_argument("--snr-source", default="linear", choices=["linear", "gt", "neural"])
        if cmd == "finetune":
            sp.add_argument("--from-seed", type=int, default=None)

    args = ap.parse_args()

    if args.cmd == "pretrain":
        pretrain(args.size, args.seed, args.snr_source)
    elif args.cmd == "finetune":
        fs = getattr(args, "from_seed", None)
        finetune(args.size, args.seed, from_seed=fs, snr_source=args.snr_source)
    elif args.cmd == "eval":
        evaluate(args.size, args.seed, args.snr_source)


if __name__ == "__main__":
    main()
