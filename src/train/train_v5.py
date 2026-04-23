"""Phase 4: V5 main training — pre-train on synthetic, fine-tune on Zhu.

Workflow (single seed):
  1. Generate 500K synthetic samples (mixed AWGN+K-dist, SNR in [-4,8])
  2. Pre-train for --pretrain-epochs (Adam, LR cosine 1e-3→1e-5)
     Val: Zhu 11.1% split with power-based SNR estimate
  3. Fine-tune on Zhu 88.9% for --finetune-epochs (LR cosine 1e-4→1e-6)
  4. Final eval on Zhu test set → results/v5_s{seed}_test.csv

Usage:
  python src/train/train_v5.py --seed 0
  python src/train/train_v5.py --seed 1 --pretrain-epochs 25
  python src/train/train_v5.py --seed 0 --skip-pretrain --resume checkpoints/v5_s0_pre_best.pt

Outputs:
  checkpoints/v5_s{seed}_pre_ep{N:02d}.pt  (+ v5_s{seed}_pre_best.pt)
  checkpoints/v5_s{seed}_ft_ep{N:02d}.pt   (+ v5_s{seed}_ft_best.pt)
  results/v5_s{seed}_log.csv
  results/v5_s{seed}_test.csv
"""

import sys, os, csv, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.v5_model import V5Model, v5_loss, _SNR_MIN, _SNR_RANGE
from src.data_zhu import zhu_train_dataset, zhu_test_dataset, TEST_CONDITIONS

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR   = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results"
CKPT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

SNR_LO  = _SNR_MIN
SNR_HI  = _SNR_MIN + _SNR_RANGE
TEST_N  = 700   # per condition, matching paper


# ── SNR estimator ─────────────────────────────────────────────────────────────

def _calibrate_snr_estimator(n_per=150):
    """Fit linear model: snr_db = slope * log10(rx_power) + intercept."""
    from src.synth_gen import generate_sample
    snr_levels = np.arange(SNR_LO, SNR_HI + 0.1, 2.0)
    log_pw, snrs = [], []
    rng = np.random.default_rng(777)
    for snr in snr_levels:
        for _ in range(n_per):
            x, _, _ = generate_sample(rng, 0.3, float(snr), "awgn")
            log_pw.append(float(np.log10(np.mean(x**2) + 1e-8)))
            snrs.append(float(snr))
    A = np.column_stack([log_pw, np.ones(len(log_pw))])
    coef = np.linalg.lstsq(A, snrs, rcond=None)[0]
    print(f"  SNR estimator calibrated: snr = {coef[0]:.3f}*log10(pwr) + {coef[1]:.3f}")
    return float(coef[0]), float(coef[1])


def estimate_snr(iq: torch.Tensor, slope: float, intercept: float) -> torch.Tensor:
    log_pwr = torch.log10(iq.pow(2).mean(dim=(1, 2)).clamp(min=1e-8))
    return (slope * log_pwr + intercept).clamp(SNR_LO, SNR_HI)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _preload_zhu(ds, tag="Zhu", batch=512):
    """CSV-per-sample → TensorDataset in RAM (avoids repeated disk I/O)."""
    t0 = time.time()
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)
    xs = torch.cat(xs); ys = torch.cat(ys)
    print(f"  {tag}: {len(xs):,} samples pre-loaded in {time.time()-t0:.1f}s")
    return TensorDataset(xs, ys)


def _norm_snr(snr_db: torch.Tensor) -> torch.Tensor:
    return (snr_db - SNR_LO) / _SNR_RANGE


# ── Train / val loops ─────────────────────────────────────────────────────────

def _run(model, loader, opt, train: bool, slope, intercept, synth_snr=True):
    """One epoch.  synth_snr=True → batch yields (x,y,snr); False → estimate."""
    model.train(train)
    tot_loss = tot_bce = tot_ber = n = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            if synth_snr:
                x, y, snr = batch
                snr = snr.to(DEVICE)
            else:
                x, y = batch
                snr = None
            x, y = x.to(DEVICE), y.to(DEVICE)
            if snr is None:
                snr = estimate_snr(x, slope, intercept)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                logits, snr_pred = model(x, snr)
            snr_norm = _norm_snr(snr)
            loss, ld = v5_loss(logits.float(), y, snr_pred.float(), snr_norm)

            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            B = len(x)
            tot_loss += loss.item() * B
            tot_bce  += ld["bce"]  * B
            ber = ((torch.sigmoid(logits.float().detach()) > 0.5).float() != y).float().mean().item()
            tot_ber  += ber * B
            n += B

    return tot_loss / n, tot_bce / n, tot_ber / n


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save(model, opt, sched, epoch, phase, seed, val_ber, extra=""):
    stem = f"v5_s{seed}_{phase}_ep{epoch:02d}{extra}.pt"
    path = CKPT_DIR / stem
    torch.save({
        "model": model.state_dict(),
        "opt":   opt.state_dict(),
        "sched": sched.state_dict() if sched else None,
        "epoch": epoch,
        "phase": phase,
        "seed":  seed,
        "val_ber": val_ber,
    }, path)
    return path


def _load(path, model, opt=None, sched=None):
    ck = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    if opt   and ck.get("opt"):   opt.load_state_dict(ck["opt"])
    if sched and ck.get("sched"): sched.load_state_dict(ck["sched"])
    return ck.get("epoch", 0), ck.get("val_ber", 1.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",             type=int,   default=0)
    ap.add_argument("--n-synth",          type=int,   default=500_000)
    ap.add_argument("--pretrain-epochs",  type=int,   default=20)
    ap.add_argument("--finetune-epochs",  type=int,   default=30)
    ap.add_argument("--batch",            type=int,   default=512)
    ap.add_argument("--pretrain-lr",      type=float, default=1e-3)
    ap.add_argument("--finetune-lr",      type=float, default=3e-4)
    ap.add_argument("--resume",           type=str,   default=None)
    ap.add_argument("--skip-pretrain",    action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_path = RESULT_DIR / f"v5_s{args.seed}_log.csv"
    log_fh   = open(log_path, "w", newline="")
    log_w    = csv.writer(log_fh)
    log_w.writerow(["phase","epoch","train_loss","train_bce","train_ber",
                    "val_loss","val_bce","val_ber","lr","elapsed_s"])

    print(f"\n=== Phase 4 — V5 seed={args.seed} | {DEVICE} ===")

    # ── SNR estimator calibration ────────────────────────────────────────────
    print("\nCalibrating SNR estimator ...")
    slope, intercept = _calibrate_snr_estimator()

    # ── Zhu data (pre-load into RAM once) ───────────────────────────────────
    print("\nPre-loading Zhu data ...")
    zhu_tr_raw, zhu_val_raw = zhu_train_dataset(val_frac=0.111, seed=args.seed)
    zhu_tr  = _preload_zhu(zhu_tr_raw,  "Zhu-train")
    zhu_val = _preload_zhu(zhu_val_raw, "Zhu-val  ")

    val_loader = DataLoader(zhu_val, batch_size=args.batch, shuffle=False, pin_memory=True)

    # ── Build model ──────────────────────────────────────────────────────────
    model = V5Model().to(DEVICE)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase A — pre-train on synthetic
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_pretrain:
        print(f"\n--- Phase A: pre-train {args.pretrain_epochs} epochs on "
              f"{args.n_synth:,} synthetic ---")

        from src.synth_gen import SynthDataset
        print("  Generating synthetic dataset (this takes ~30s) ...")
        t0 = time.time()
        synth_ds = SynthDataset(
            n_samples=args.n_synth,
            channel="mixed",
            snr_range=(SNR_LO, SNR_HI),
            seed=args.seed,
        )
        print(f"  Done in {time.time()-t0:.1f}s")
        synth_loader = DataLoader(synth_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=4, pin_memory=True)

        opt_pre  = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
        sched_pre = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_pre, T_max=args.pretrain_epochs, eta_min=1e-5)

        best_ber, best_pre = 1.0, None
        start_ep = 0

        if args.resume and Path(args.resume).exists():
            ep0, _ = _load(args.resume, model, opt_pre, sched_pre)
            start_ep = ep0

        for ep in range(start_ep + 1, args.pretrain_epochs + 1):
            t0  = time.time()
            tr  = _run(model, synth_loader, opt_pre, True,  slope, intercept, synth_snr=True)
            val = _run(model, val_loader,   None,    False, slope, intercept, synth_snr=False)
            sched_pre.step()
            lr  = opt_pre.param_groups[0]["lr"]
            elapsed = time.time() - t0

            print(f"  [pre ep{ep:02d}] tr_ber={tr[2]:.4f} val_ber={val[2]:.4f} "
                  f"val_bce={val[1]:.4f} lr={lr:.2e} {elapsed:.0f}s")
            log_w.writerow(["pretrain", ep, *tr, *val, lr, round(elapsed, 1)])
            log_fh.flush()

            ckpt = _save(model, opt_pre, sched_pre, ep, "pre", args.seed, val[2])
            if val[2] < best_ber:
                best_ber = val[2]

        # For fine-tuning: use LAST pretrain epoch (most-trained backbone),
        # not best Zhu val_ber (which is lowest at ep01 before specialisation).
        last_pre = CKPT_DIR / f"v5_s{args.seed}_pre_best.pt"
        torch.save(torch.load(ckpt), last_pre)   # ckpt = last epoch
        best_pre = last_pre
        print(f"  Pretrain done. Best Zhu val_ber={best_ber:.4f}  "
              f"(last-epoch checkpoint used for fine-tune)")
    else:
        best_pre = Path(args.resume) if args.resume else None

    # ══════════════════════════════════════════════════════════════════════════
    # Phase B — fine-tune on Zhu
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n--- Phase B: fine-tune {args.finetune_epochs} epochs on Zhu-train ---")

    # Load best pre-train checkpoint
    if best_pre and best_pre.exists():
        print(f"  Loading {best_pre}")
        _load(best_pre, model)

    ft_loader = DataLoader(zhu_tr, batch_size=args.batch, shuffle=True, pin_memory=True)
    opt_ft    = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
    sched_ft  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_ft, T_max=args.finetune_epochs, eta_min=1e-6)

    best_ber_ft, best_ft = 1.0, None
    for ep in range(1, args.finetune_epochs + 1):
        t0  = time.time()
        tr  = _run(model, ft_loader,  opt_ft, True,  slope, intercept, synth_snr=False)
        val = _run(model, val_loader, None,   False, slope, intercept, synth_snr=False)
        sched_ft.step()
        lr  = opt_ft.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"  [ft  ep{ep:02d}] tr_ber={tr[2]:.4f} val_ber={val[2]:.4f} "
              f"val_bce={val[1]:.4f} lr={lr:.2e} {elapsed:.0f}s")
        log_w.writerow(["finetune", ep, *tr, *val, lr, round(elapsed, 1)])
        log_fh.flush()

        ckpt = _save(model, opt_ft, sched_ft, ep, "ft", args.seed, val[2])
        if val[2] < best_ber_ft:
            best_ber_ft = val[2]
            best_ft = CKPT_DIR / f"v5_s{args.seed}_ft_best.pt"
            torch.save(torch.load(ckpt), best_ft)
            print(f"    *** new best finetune val_ber={best_ber_ft:.4f}")

    log_fh.close()
    print(f"\nBest fine-tune val_ber={best_ber_ft:.4f}  → {best_ft}")

    # ══════════════════════════════════════════════════════════════════════════
    # Final eval on Zhu test set (held-out, single evaluation)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n--- Final test evaluation (held-out, first {TEST_N} per condition) ---")
    if best_ft and best_ft.exists():
        _load(best_ft, model)
    model.eval()

    test_rows = []
    all_ber = []
    for cond in TEST_CONDITIONS:
        ds   = zhu_test_dataset(cond)
        sub  = torch.utils.data.Subset(ds, list(range(TEST_N)))
        tl   = DataLoader(sub, batch_size=256, shuffle=False, num_workers=2)
        ber_sum = n = 0
        with torch.no_grad():
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                snr  = estimate_snr(x, slope, intercept)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                    logits, _ = model(x, snr)
                ber = ((torch.sigmoid(logits.float()) > 0.5).float() != y).float().mean().item()
                ber_sum += ber * len(x)
                n += len(x)
        cber = ber_sum / n
        all_ber.append(cber)
        print(f"  {cond:<25} BER={cber:.4f}")
        test_rows.append((cond, round(cber, 6)))

    overall = float(np.mean(all_ber))
    print(f"  {'OVERALL':<25} BER={overall:.4f}")
    test_rows.append(("OVERALL", round(overall, 6)))

    csv_path = RESULT_DIR / f"v5_s{args.seed}_test.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "ber"])
        w.writerows(test_rows)
    print(f"\nTest results → {csv_path}")
    print(f"Baseline BER=3.12%  →  V5 seed={args.seed} BER={overall*100:.2f}%")


if __name__ == "__main__":
    main()
