"""Train Zhu baseline.  Resumable, checkpoints every epoch.

Usage:
  python src/train/train_baseline.py [--epochs 40] [--seed 0] [--resume]

Outputs (per seed):
  checkpoints/baseline_s{seed}_ep{N:02d}.pt
  results/baseline_s{seed}_log.csv

Legacy (no --seed): checkpoints/baseline_ep{N:02d}.pt + results/baseline_train_log.csv
"""

import sys, os, csv, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_zhu import zhu_train_dataset, zhu_test_dataset, TEST_CONDITIONS
from models.zhu_baseline import ZhuBaseline
from torch.utils.data import Subset

# ── Config (matches Zhu exactly) ─────────────────────────────────────────────
BATCH    = 512
LR       = 1e-3
EPOCHS   = 40
VAL_FRAC = 0.111  # paper: 63000 train / 7875 val ≈ 88.9/11.1 split
TEST_N   = 700

CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ber(pred: torch.Tensor, target: torch.Tensor) -> float:
    """BER: fraction of wrongly decoded bits (threshold at 0.5)."""
    return ((pred > 0.5).float() != target).float().mean().item()


def run_epoch(model, loader, optim, criterion, train: bool):
    model.train(train)
    total_loss = total_ber = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            if train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            total_loss += loss.item() * len(x)
            total_ber  += ber(pred.detach(), y) * len(x)
            n += len(x)
    return total_loss / n, total_ber / n


def eval_on_test(model):
    """Evaluate model on held-out test set; returns per-condition rows + overall BER."""
    rows = []
    for cond in TEST_CONDITIONS:
        ds = Subset(zhu_test_dataset(cond), list(range(TEST_N)))
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
        mse_sum = ber_sum = n = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                mse_sum += ((pred - y) ** 2).mean().item() * len(x)
                ber_sum += ((pred > 0.5).float() != y).float().mean().item() * len(x)
                n += len(x)
        rows.append((cond, mse_sum / n, ber_sum / n, n))
    overall_ber = sum(r[2] * r[3] for r in rows) / sum(r[3] for r in rows)
    rows.append(("ALL", sum(r[1] * r[3] for r in rows) / sum(r[3] for r in rows), overall_ber,
                 sum(r[3] for r in rows)))
    return rows, overall_ber


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed",   type=int, default=None,
                        help="Random seed. Enables per-seed file naming.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Seeded run → per-seed naming; legacy run (no --seed) → original naming.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        tag        = f"baseline_s{args.seed}"
        result_csv = f"results/baseline_s{args.seed}_log.csv"
        test_csv   = f"results/baseline_s{args.seed}_test.csv"
    else:
        tag        = "baseline"
        result_csv = "results/baseline_train_log.csv"
        test_csv   = None

    print(f"Device: {DEVICE}  tag={tag}")
    train_ds, val_ds = zhu_train_dataset(val_frac=VAL_FRAC, seed=args.seed if args.seed is not None else 42)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = ZhuBaseline().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    start_epoch = 1
    log_rows = []
    best_val_loss = float("inf")
    best_ckpt_path = None

    # Resume from latest checkpoint for this tag
    if args.resume:
        ckpts = sorted(f for f in os.listdir(CKPT_DIR) if f.startswith(f"{tag}_ep"))
        if ckpts:
            ckpt_path = os.path.join(CKPT_DIR, ckpts[-1])
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model"])
            optim.load_state_dict(ckpt["optim"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            log_rows = ckpt.get("log", [])
            print(f"Resumed from {ckpt_path} (epoch {ckpt['epoch']})")

    print(f"\nTraining epochs {start_epoch}–{args.epochs}")
    print(f"{'Ep':>3}  {'trn_loss':>9}  {'trn_ber':>8}  {'val_loss':>9}  {'val_ber':>8}  {'time':>6}")

    for ep in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        trn_loss, trn_ber = run_epoch(model, train_loader, optim, criterion, train=True)
        val_loss, val_ber = run_epoch(model, val_loader,   optim, criterion, train=False)
        elapsed = time.time() - t0

        row = (ep, trn_loss, trn_ber, val_loss, val_ber, elapsed)
        log_rows.append(row)
        print(f"{ep:>3}  {trn_loss:>9.5f}  {trn_ber:>8.5f}  {val_loss:>9.5f}  {val_ber:>8.5f}  {elapsed:>5.1f}s")

        ckpt_data = {
            "epoch": ep, "model": model.state_dict(), "optim": optim.state_dict(),
            "val_loss": val_loss, "log": log_rows,
        }
        ep_path = os.path.join(CKPT_DIR, f"{tag}_ep{ep:02d}.pt")
        torch.save(ckpt_data, ep_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = ep_path

    # Save training log CSV
    with open(result_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "trn_loss", "trn_ber", "val_loss", "val_ber", "seconds"])
        w.writerows(log_rows)
    print(f"\nFinal val_loss={val_loss:.5f}  val_ber={val_ber:.5f}")
    print(f"Log saved to {result_csv}")

    # Per-seed run: also evaluate on test set using best checkpoint
    if test_csv is not None:
        print(f"\nEvaluating best checkpoint ({best_ckpt_path}) on test set...")
        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(best_ckpt["model"])
        model.eval()
        test_rows, overall_ber = eval_on_test(model)

        with open(test_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["condition", "mse", "ber", "n_samples"])
            w.writerows(test_rows)

        print(f"\n{'Condition':<22}  {'BER':>8}")
        for cond, mse, ber_v, n in test_rows:
            print(f"{cond:<22}  {ber_v:>8.5f}")
        print(f"\nTest BER={overall_ber*100:.3f}%  saved to {test_csv}")


if __name__ == "__main__":
    main()
