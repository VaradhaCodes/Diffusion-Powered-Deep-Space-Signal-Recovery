"""Phase 5: Train a competitor baseline model.

Identical training procedure to train_v5.py — same CNN stem, FiLM, pretrain
strategy, fine-tuning split, and evaluation. Only the sequence backbone differs.

Usage:
  python src/train/train_competitor.py --model bi_transformer --seed 0
  python src/train/train_competitor.py --model bi_mamba2      --seed 1
  python src/train/train_competitor.py --model mambanet       --seed 2

Models: bi_transformer | bi_mamba2 | mambanet

Outputs:
  checkpoints/{model}_s{seed}_pre_best.pt
  checkpoints/{model}_s{seed}_ft_best.pt
  results/{model}_s{seed}_log.csv
  results/{model}_s{seed}_test.csv
"""

import sys, csv, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.competitors import build_model, MODELS
from src.models.v5_model import v5_loss, _SNR_MIN, _SNR_RANGE
from src.data_zhu import zhu_train_dataset, zhu_test_dataset, TEST_CONDITIONS
from src.train.train_v5 import (
    _calibrate_snr_estimator, estimate_snr, _preload_zhu,
    _norm_snr, _save, _load,
)

NEURAL_CKPT = ROOT / "checkpoints" / "v6b2_snr_estimator.pt"


def _run(model, loader, opt, train: bool, slope, intercept,
         synth_snr=True, snr_fn=None, gt_snr_tensor=None):
    """One epoch. snr_fn overrides linear estimator when provided."""
    model.train(train)
    tot_loss = tot_bce = tot_ber = n = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            DEVICE = next(model.parameters()).device
            if synth_snr and len(batch) == 3:
                x, y, snr = batch
                snr = snr.to(DEVICE)
            else:
                x, y = batch[0], batch[1]
                snr = None
            x, y = x.to(DEVICE), y.to(DEVICE)

            if snr is None:
                if snr_fn is not None:
                    snr = snr_fn(x)
                else:
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
            tot_bce  += ld["bce"] * B
            ber = ((torch.sigmoid(logits.float().detach()) > 0.5).float() != y).float().mean().item()
            tot_ber  += ber * B
            n += B

    return tot_loss / n, tot_bce / n, tot_ber / n

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR   = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results"
CKPT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

SNR_LO = _SNR_MIN
SNR_HI = _SNR_MIN + _SNR_RANGE
TEST_N = 700


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",           type=str,   required=True, choices=list(MODELS))
    ap.add_argument("--seed",            type=int,   default=0)
    ap.add_argument("--n-synth",         type=int,   default=500_000)
    ap.add_argument("--pretrain-epochs", type=int,   default=20)
    ap.add_argument("--finetune-epochs", type=int,   default=30)
    ap.add_argument("--batch",           type=int,   default=512)
    ap.add_argument("--pretrain-lr",     type=float, default=1e-3)
    ap.add_argument("--finetune-lr",     type=float, default=3e-4)
    ap.add_argument("--skip-pretrain",   action="store_true")
    ap.add_argument("--resume",          type=str,   default=None)
    ap.add_argument("--snr-source",      type=str,   default="linear",
                    choices=["gt", "linear", "neural"],
                    help="gt=ground truth (oracle), linear=V5 power estimator, neural=v6b2 neural")
    # V6 Batch 4 architecture sweep flags
    ap.add_argument("--depth",           type=int,   default=1,
                    help="number of (MHA+BiMamba2) blocks (mambanet_2ch_cfg only)")
    ap.add_argument("--width",           type=int,   default=128,
                    help="d_model embedding dimension (mambanet_2ch_cfg only)")
    ap.add_argument("--kernel-size",     type=int,   default=7,
                    help="first CNN kernel size (mambanet_2ch_cfg only)")
    ap.add_argument("--block-type",      type=str,   default="serial",
                    choices=["serial", "parallel"],
                    help="serial = MHA→BiMamba2 chain; parallel = Mcformer-style additive")
    ap.add_argument("--loss",            type=str,   default="bce",
                    choices=["bce", "bce_ls", "focal", "focal_ls"],
                    help="loss variant: bce | bce_ls (label smooth 0.05) | focal (γ=2) | focal_ls")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag      = f"{args.model}_s{args.seed}"
    log_path = RESULT_DIR / f"{tag}_log.csv"
    log_fh   = open(log_path, "w", newline="")
    log_w    = csv.writer(log_fh)
    log_w.writerow(["phase","epoch","train_loss","train_bce","train_ber",
                    "val_loss","val_bce","val_ber","lr","elapsed_s"])

    print(f"\n=== Phase 5 — {args.model} seed={args.seed} snr_source={args.snr_source} | {DEVICE} ===")

    # ── SNR estimator ────────────────────────────────────────────────────────
    snr_fn = None   # None → use linear slope/intercept (default / backward compat)
    if args.snr_source == "neural":
        from src.infer.snr_helper import estimate_snr_db as _neural_snr
        _ckpt = NEURAL_CKPT
        snr_fn = lambda x: _neural_snr(x, _ckpt, DEVICE).clamp(_SNR_MIN, _SNR_MIN + _SNR_RANGE)
        slope, intercept = 0.0, 0.0   # unused when snr_fn is set
        print(f"Neural SNR estimator: {_ckpt}")
    elif args.snr_source == "gt":
        # Ground truth used only when synth_snr=True; for Zhu data we fall back to linear
        slope, intercept = _calibrate_snr_estimator()
        print("SNR source: gt (oracle — uses dataset label for synth, linear for Zhu)")
    else:
        print("\nCalibrating SNR estimator ...")
        slope, intercept = _calibrate_snr_estimator()

    # ── Zhu data ─────────────────────────────────────────────────────────────
    print("\nPre-loading Zhu data ...")
    zhu_tr_raw, zhu_val_raw = zhu_train_dataset(val_frac=0.111, seed=args.seed)
    zhu_tr  = _preload_zhu(zhu_tr_raw,  "Zhu-train")
    zhu_val = _preload_zhu(zhu_val_raw, "Zhu-val  ")
    val_loader = DataLoader(zhu_val, batch_size=args.batch, shuffle=False, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_kwargs = {}
    if args.model == "mambanet_2ch_cfg":
        model_kwargs = dict(
            d_model=args.width, n_blocks=args.depth,
            cnn_k1=args.kernel_size, parallel=(args.block_type == "parallel"),
        )
    model = build_model(args.model, **model_kwargs).to(DEVICE)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase A — pre-train on synthetic
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_pretrain:
        print(f"\n--- Phase A: pre-train {args.pretrain_epochs} epochs on "
              f"{args.n_synth:,} synthetic ---")
        from src.synth_gen import SynthDataset
        print("  Generating synthetic dataset ...")
        t0 = time.time()
        synth_ds = SynthDataset(n_samples=args.n_synth, channel="mixed",
                                snr_range=(SNR_LO, SNR_HI), seed=args.seed)
        print(f"  Done in {time.time()-t0:.1f}s")
        synth_loader = DataLoader(synth_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=4, pin_memory=True)

        opt_pre   = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
        sched_pre = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_pre, T_max=args.pretrain_epochs, eta_min=1e-5)

        best_ber, start_ep = 1.0, 0
        if args.resume and Path(args.resume).exists():
            start_ep, _ = _load(args.resume, model, opt_pre, sched_pre)

        for ep in range(start_ep + 1, args.pretrain_epochs + 1):
            t0  = time.time()
            tr  = _run(model, synth_loader, opt_pre, True,  slope, intercept, synth_snr=True,  snr_fn=snr_fn)
            val = _run(model, val_loader,   None,    False, slope, intercept, synth_snr=False, snr_fn=snr_fn)
            sched_pre.step()
            lr  = opt_pre.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"  [pre ep{ep:02d}] tr_ber={tr[2]:.4f} val_ber={val[2]:.4f} "
                  f"val_bce={val[1]:.4f} lr={lr:.2e} {elapsed:.0f}s")
            log_w.writerow(["pretrain", ep, *tr, *val, lr, round(elapsed, 1)])
            log_fh.flush()
            if val[2] < best_ber:
                best_ber = val[2]
            ckpt = _save(model, opt_pre, sched_pre, ep, f"{args.model}_pre", args.seed, val[2])

        # Use last pretrain epoch for fine-tuning
        best_pre = CKPT_DIR / f"{tag}_pre_best.pt"
        torch.save(torch.load(ckpt), best_pre)
        print(f"  Pretrain done. Best Zhu val_ber={best_ber:.4f}  (using last ckpt for fine-tune)")
    else:
        best_pre = Path(args.resume) if args.resume else None

    # ══════════════════════════════════════════════════════════════════════════
    # Phase B — fine-tune on Zhu
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n--- Phase B: fine-tune {args.finetune_epochs} epochs on Zhu-train ---")
    if best_pre and best_pre.exists():
        print(f"  Loading {best_pre.name}")
        _load(best_pre, model)

    ft_loader = DataLoader(zhu_tr, batch_size=args.batch, shuffle=True, pin_memory=True)
    opt_ft    = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
    sched_ft  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_ft, T_max=args.finetune_epochs, eta_min=1e-6)

    best_ber_ft, best_ft = 1.0, None
    for ep in range(1, args.finetune_epochs + 1):
        t0  = time.time()
        tr  = _run(model, ft_loader,  opt_ft, True,  slope, intercept, synth_snr=False, snr_fn=snr_fn)
        val = _run(model, val_loader, None,   False, slope, intercept, synth_snr=False, snr_fn=snr_fn)
        sched_ft.step()
        lr  = opt_ft.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"  [ft  ep{ep:02d}] tr_ber={tr[2]:.4f} val_ber={val[2]:.4f} "
              f"val_bce={val[1]:.4f} lr={lr:.2e} {elapsed:.0f}s")
        log_w.writerow(["finetune", ep, *tr, *val, lr, round(elapsed, 1)])
        log_fh.flush()

        ckpt = _save(model, opt_ft, sched_ft, ep, f"{args.model}_ft", args.seed, val[2])
        if val[2] < best_ber_ft:
            best_ber_ft = val[2]
            best_ft = CKPT_DIR / f"{tag}_ft_best.pt"
            torch.save(torch.load(ckpt), best_ft)
            print(f"    *** new best val_ber={best_ber_ft:.4f}")

    log_fh.close()
    print(f"\nBest fine-tune val_ber={best_ber_ft:.4f}  → {best_ft}")

    # ══════════════════════════════════════════════════════════════════════════
    # Final eval on Zhu test set
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n--- Final test evaluation ---")
    if best_ft and best_ft.exists():
        _load(best_ft, model)
    model.eval()

    test_rows, all_ber = [], []
    for cond in TEST_CONDITIONS:
        ds  = zhu_test_dataset(cond)
        sub = torch.utils.data.Subset(ds, list(range(TEST_N)))
        tl  = DataLoader(sub, batch_size=256, shuffle=False, num_workers=2)
        ber_sum = n = 0
        with torch.no_grad():
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if snr_fn is not None:
                    snr = snr_fn(x)
                else:
                    snr = estimate_snr(x, slope, intercept)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                    logits, _ = model(x, snr)
                ber = ((torch.sigmoid(logits.float()) > 0.5).float() != y).float().mean().item()
                ber_sum += ber * len(x); n += len(x)
        cber = ber_sum / n
        all_ber.append(cber)
        print(f"  {cond:<25} BER={cber:.4f}")
        test_rows.append((cond, round(cber, 6)))

    overall = float(np.mean(all_ber))
    print(f"  {'OVERALL':<25} BER={overall:.4f}")
    test_rows.append(("OVERALL", round(overall, 6)))

    csv_path = RESULT_DIR / f"{tag}_test.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows([["condition","ber"]] + test_rows)

    print(f"\nBaseline BER=3.12%  V5-ensemble=2.76%  →  {args.model} s={args.seed} BER={overall*100:.2f}%")
    print(f"Test results → {csv_path}")


if __name__ == "__main__":
    main()
