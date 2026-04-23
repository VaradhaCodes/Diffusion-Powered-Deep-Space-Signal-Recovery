"""V6 Batch 2 Part B3 — Train neural SNR estimator.

Saves: checkpoints/v6b2_snr_estimator.pt (best val MAE)
       results/v6b2_snr_estimator_log.csv
"""

import sys, csv, time, math, threading
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.snr_estimator import SNREstimator

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR  = ROOT / "checkpoints"
RES_DIR   = ROOT / "results"
DATA_DIR  = ROOT / "data" / "v6b2_snr_estimator"
CKPT_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

EPOCHS    = 40
BATCH     = 256
LR_MAX    = 1e-3
LR_MIN    = 1e-5
WARMUP    = 500   # steps
SEED      = 42

# Self-kill: terminate if no log line in 15 min
_last_log_time = time.time()
_KILL_TIMEOUT  = 15 * 60


def _watchdog():
    while True:
        time.sleep(30)
        if time.time() - _last_log_time > _KILL_TIMEOUT:
            import traceback
            traceback.print_stack()
            raise RuntimeError("WATCHDOG: no log line for 15 min — terminating")


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    tr = np.load(DATA_DIR / "train" / "data.npz")
    va = np.load(DATA_DIR / "val"   / "data.npz")
    tr_ds = TensorDataset(
        torch.from_numpy(tr["x"]),
        torch.from_numpy(tr["snr"]),
    )
    va_ds = TensorDataset(
        torch.from_numpy(va["x"]),
        torch.from_numpy(va["snr"]),
    )
    return tr_ds, va_ds


def get_lr(step: int, total_steps: int) -> float:
    if step < WARMUP:
        return LR_MAX * step / max(WARMUP, 1)
    progress = (step - WARMUP) / max(total_steps - WARMUP, 1)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))


def main():
    global _last_log_time
    set_seed(SEED)

    # Start watchdog
    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()

    tr_ds, va_ds = load_data()
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                       num_workers=2, pin_memory=True,
                       generator=torch.Generator().manual_seed(SEED))
    va_dl = DataLoader(va_ds, batch_size=BATCH, shuffle=False,
                       num_workers=2, pin_memory=True)

    model = SNREstimator().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SNREstimator params={n_params:,}  device={DEVICE}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-4)
    total_steps = EPOCHS * len(tr_dl)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: get_lr(s, total_steps) / LR_MAX
    )

    log_path = RES_DIR / "v6b2_snr_estimator_log.csv"
    log_fh = open(log_path, "w", newline="")
    log_w  = csv.writer(log_fh)
    log_w.writerow(["epoch","train_loss","val_mae","lr","elapsed_s"])

    best_mae, best_path = float("inf"), CKPT_DIR / "v6b2_snr_estimator.pt"

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        tr_loss = 0.0
        for x, snr in tr_dl:
            x, snr = x.to(DEVICE), snr.to(DEVICE)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, snr)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            tr_loss += loss.item() * len(x)
        tr_loss /= len(tr_ds)

        # Val
        model.eval()
        mae_sum = 0.0
        with torch.no_grad():
            for x, snr in va_dl:
                x, snr = x.to(DEVICE), snr.to(DEVICE)
                pred = model(x)
                mae_sum += (pred - snr).abs().sum().item()
        val_mae = mae_sum / len(va_ds)

        lr = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] epoch={ep} train_loss={tr_loss:.4f} val_mae={val_mae:.4f} "
              f"lr={lr:.2e} elapsed={elapsed:.0f}s")
        log_w.writerow([ep, round(tr_loss, 6), round(val_mae, 6), f"{lr:.6e}", round(elapsed, 1)])
        log_fh.flush()
        _last_log_time = time.time()

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({"model": model.state_dict(), "epoch": ep, "val_mae": val_mae}, best_path)
            print(f"  *** new best val_mae={best_mae:.4f}")

    log_fh.close()
    print(f"\nTraining done. Best val_mae={best_mae:.4f} -> {best_path}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
