"""V6 Synth Fine-Tune Experiment.

Hypothesis: the 37K Zhu fine-tune corpus is the bottleneck.
Test: start from v6b3_canonical_pretrain.pt, fine-tune on
      37K real Zhu + 100K synthetic AWGN + 100K synthetic KB2
      (KB2 frames pre-scaled by 0.268 to match Zhu's amplitude statistics).

Compare ensemble BER against V6B3_CANONICAL_BER = 2.2820%.
Gate: delta >= 0.05pp → fine-tune data IS the bottleneck.
      delta <  0.05pp → architecture / task ceiling confirmed.

CLI:
  python src/train/train_v6_synthft.py finetune --seed 0
  python src/train/train_v6_synthft.py finetune --seed 1
  python src/train/train_v6_synthft.py finetune --seed 2
  python src/train/train_v6_synthft.py eval     --seed 0
  python src/train/train_v6_synthft.py ensemble
"""

import sys, csv, time, math, argparse, json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from src.models.competitors import build_model
from src.models.v5_model import v5_loss, _SNR_MIN, _SNR_RANGE
from src.data_zhu import zhu_train_dataset, zhu_test_dataset, TEST_CONDITIONS
from src.train.train_v5 import (
    _calibrate_snr_estimator, estimate_snr, _preload_zhu, _norm_snr,
)
from src.train.train_v6b3 import (
    make_warmup_cosine, set_seed, worker_init_fn, ts,
    FT_BATCH, FT_LR, FT_LR_MIN, FT_EPOCHS, FT_PATIENCE, TEST_N,
    parse_condition, ensemble_eval,
)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR   = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results"
DATA_SYNTH = ROOT / "data" / "synth_zhu_equiv"

CKPT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

CANONICAL_PRETRAIN = CKPT_DIR / "v6b3_canonical_pretrain.pt"
KB2_SCALE          = 0.268   # empirically derived: sqrt(0.0717), matches Zhu's KB2 amplitude


# ── Dataset ───────────────────────────────────────────────────────────────────

class _XYOnly(Dataset):
    """Wraps a (x, y, snr) dataset and drops the snr. Returns (x, y)."""
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        return item[0], item[1]


class SynthNpyXYDataset(Dataset):
    """Loads pre-generated .npy files. Optionally scales x by a fixed factor.
    Returns (x, y) only — no GT SNR needed (fine-tune uses linear estimator).
    """
    def __init__(self, data_dir: Path, x_scale: float = 1.0, mmap: bool = True):
        mode = "r" if mmap else None
        self.xs    = np.load(str(data_dir / "xs.npy"),  mmap_mode=mode)
        self.ys    = np.load(str(data_dir / "ys.npy"),  mmap_mode=mode)
        self.scale = np.float32(x_scale)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.xs[idx].copy()) * self.scale
        y = torch.from_numpy(self.ys[idx].copy())
        return x, y


def _build_combined_dataset(zhu_train_preloaded, synth_awgn_dir: Path, synth_kb2_dir: Path):
    """Concatenate Zhu real train + synthetic AWGN + synthetic KB2 (scaled).
    KB2 xs.npy was already scaled in-place by KB2_SCALE during data generation,
    so x_scale=1.0 here avoids double-scaling (0.268² = 0.0718 ≈ −22.9 dB).
    """
    synth_awgn = SynthNpyXYDataset(synth_awgn_dir, x_scale=1.0, mmap=True)
    synth_kb2  = SynthNpyXYDataset(synth_kb2_dir,  x_scale=1.0, mmap=True)
    zhu_xy     = _XYOnly(zhu_train_preloaded)
    combined   = ConcatDataset([zhu_xy, synth_awgn, synth_kb2])
    return combined, len(zhu_xy), len(synth_awgn), len(synth_kb2)


# ── One-epoch runner (identical logic to v6b3) ────────────────────────────────

def run_epoch(model, loader, opt, sched_step_fn,
              slope, intercept, is_train: bool):
    model.train(is_train)
    tot_loss = tot_ber = n = 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            snr  = estimate_snr(x, slope, intercept)

            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(DEVICE.type == "cuda")):
                logits, snr_pred = model(x, snr)

            snr_norm = _norm_snr(snr)
            loss, _  = v5_loss(logits.float(), y, snr_pred.float(), snr_norm)

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


# ── Fine-tune ─────────────────────────────────────────────────────────────────

def finetune(seed: int) -> float:
    """Fine-tune from canonical pretrain on Zhu + synthetic combined corpus.
    Returns best val BER.
    """
    set_seed(seed)
    print(f"\n{'='*70}")
    print(f"[{ts()}] SYNTH-FINETUNE  seed={seed}  device={DEVICE}")
    print(f"  Starting from: {CANONICAL_PRETRAIN.name}")
    print(f"  KB2 amplitude scale: {KB2_SCALE}")
    print(f"{'='*70}")

    # Verify data dirs exist
    awgn_dir = DATA_SYNTH / "awgn"
    kb2_dir  = DATA_SYNTH / "kb2"
    for d in (awgn_dir, kb2_dir):
        if not (d / "xs.npy").exists():
            raise FileNotFoundError(
                f"Synth data not found: {d}/xs.npy\n"
                "Run: scripts/gen_synth_zhu_equiv.sh first."
            )

    slope, intercept = _calibrate_snr_estimator()

    # Build datasets
    zhu_tr_raw, zhu_val_raw = zhu_train_dataset(val_frac=0.111, seed=seed)
    zhu_tr  = _preload_zhu(zhu_tr_raw,  "Zhu-train")
    zhu_val = _preload_zhu(zhu_val_raw, "Zhu-val  ")

    combined, n_zhu, n_awgn, n_kb2 = _build_combined_dataset(zhu_tr, awgn_dir, kb2_dir)
    print(f"  Combined dataset: {n_zhu} Zhu + {n_awgn} synth-AWGN + "
          f"{n_kb2} synth-KB2(pre-scaled ×{KB2_SCALE}) = {len(combined)} total")

    g = torch.Generator(); g.manual_seed(seed)
    ft_loader  = DataLoader(combined,  batch_size=FT_BATCH, shuffle=True,
                            num_workers=4, pin_memory=True,
                            generator=g, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(zhu_val, batch_size=FT_BATCH,
                            shuffle=False, pin_memory=True)

    # Load canonical pretrain
    model = build_model("mambanet_2ch").to(DEVICE)
    ck = torch.load(CANONICAL_PRETRAIN, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    print(f"  Loaded pretrain ckpt (val_loss={ck.get('val_loss', '?')})")

    opt   = torch.optim.Adam(model.parameters(), lr=FT_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=FT_EPOCHS, eta_min=FT_LR_MIN)

    log_path = RESULT_DIR / f"v6_synthft_s{seed}_ft_log.csv"
    log_fh   = open(log_path, "w", newline="")
    log_w    = csv.writer(log_fh)
    log_w.writerow(["epoch", "train_loss", "val_ber", "lr", "elapsed_s"])

    best_ber     = float("inf")
    patience_ctr = 0
    best_ckpt    = None
    diverge_lr_applied = False

    for ep in range(1, FT_EPOCHS + 1):
        t0 = time.time()
        train_loss, _ = run_epoch(
            model, ft_loader, opt, None,
            slope, intercept, is_train=True)
        _, val_ber = run_epoch(
            model, val_loader, None, None,
            slope, intercept, is_train=False)
        sched.step()

        lr      = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"[{ts()}] ep={ep:02d} train_loss={train_loss:.4f} "
              f"val_ber={val_ber:.4f} lr={lr:.2e} {elapsed:.0f}s")
        log_w.writerow([ep, round(train_loss, 6), round(val_ber, 6),
                        f"{lr:.2e}", round(elapsed, 1)])
        log_fh.flush()

        if ep == 10 and val_ber > 0.30 and not diverge_lr_applied:
            print(f"  WARNING: val_ber={val_ber:.3f} > 30% — LR × 0.3")
            for pg in opt.param_groups:
                pg["lr"] *= 0.3
            diverge_lr_applied = True

        if val_ber < best_ber:
            best_ber     = val_ber
            patience_ctr = 0
            best_ckpt    = CKPT_DIR / f"v6_synthft_s{seed}_ft.pt"
            torch.save({"model": model.state_dict(),
                        "val_ber": best_ber, "epoch": ep}, best_ckpt)
            print(f"    *** new best val_ber={best_ber:.4f}  → {best_ckpt.name}")
        else:
            patience_ctr += 1
            if patience_ctr >= FT_PATIENCE:
                print(f"  Early stop at epoch {ep}.")
                break

    if best_ber > 0.30:
        print(f"  ERROR: seed {seed} diverged (best_val_ber={best_ber:.3f})")
        log_fh.close()
        return float("inf")

    log_fh.close()
    print(f"  Finetune done. best_val_ber={best_ber:.4f}  ckpt={best_ckpt}")
    return best_ber


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(seed: int) -> dict:
    """Evaluate fine-tuned checkpoint on Zhu test set. Returns condition→(nbits, nerrs)."""
    set_seed(seed)
    ft_ckpt = CKPT_DIR / f"v6_synthft_s{seed}_ft.pt"
    print(f"\n[{ts()}] EVAL  seed={seed}  ckpt={ft_ckpt.name}")

    slope, intercept = _calibrate_snr_estimator()

    model = build_model("mambanet_2ch").to(DEVICE)
    ck = torch.load(ft_ckpt, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()

    rows    = []
    results = {}

    for cond in TEST_CONDITIONS:
        ds  = zhu_test_dataset(cond)
        sub = torch.utils.data.Subset(ds, list(range(TEST_N)))
        tl  = DataLoader(sub, batch_size=256, shuffle=False, num_workers=2)

        n_bits = n_errors = 0
        with torch.no_grad():
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                snr = estimate_snr(x, slope, intercept)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(DEVICE.type == "cuda")):
                    logits, _ = model(x, snr)
                preds     = (torch.sigmoid(logits.float()) > 0.5).float()
                n_bits   += y.numel()
                n_errors += (preds != y).sum().item()

        ber = n_errors / n_bits
        bt, m = parse_condition(cond)
        results[cond] = (n_bits, n_errors)
        rows.append((cond, bt, m, TEST_N, n_bits, n_errors, round(ber, 6)))
        print(f"  {cond:<28} BER={ber:.4f}")

    overall_bits   = sum(r[4] for r in rows)
    overall_errors = sum(r[5] for r in rows)
    overall_ber    = overall_errors / overall_bits
    rows.append(("OVERALL", "", "", sum(r[3] for r in rows),
                 overall_bits, overall_errors, round(overall_ber, 6)))
    print(f"  {'OVERALL':<28} BER={overall_ber:.4f}")

    csv_path = RESULT_DIR / f"v6_synthft_s{seed}_test.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition","bt","m","n_frames","n_bits","n_errors","ber"])
        w.writerows(rows)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap  = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    for cmd in ("finetune", "eval"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--seed", type=int, required=True)

    sp_ens = sub.add_parser("ensemble")
    sp_ens.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    args = ap.parse_args()

    if args.cmd == "finetune":
        finetune(args.seed)
    elif args.cmd == "eval":
        evaluate(args.seed)
    elif args.cmd == "ensemble":
        seed_results = {}
        for s in args.seeds:
            seed_results[s] = evaluate(s)
        ber = ensemble_eval("synthft", seed_results)
        canonical = 2.2820
        delta = (canonical - ber * 100)
        print(f"\nSynth fine-tune ensemble BER = {ber*100:.4f}%")
        print(f"Canonical BER                = {canonical:.4f}%")
        print(f"Delta                        = {delta:+.4f} pp")
        verdict = "BOTTLENECK CONFIRMED (data)" if delta >= 0.05 else "FLAT — architecture/task ceiling"
        print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
