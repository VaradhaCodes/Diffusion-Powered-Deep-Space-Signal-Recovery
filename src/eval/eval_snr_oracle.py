"""V6 Batch 2 Part C3 — Oracle upper bound (5 epochs, gt SNR, seed 1 pretrain).

Not a gate — measurement only. Estimates BER ceiling with perfect SNR conditioning.

Loads: checkpoints/mambanet_2ch_s1_pre_best.pt
Saves: results/v6b2_snr_oracle_s1_partial.csv
"""

import sys, csv, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.competitors import build_model
from src.models.v5_model import v5_loss, _SNR_MIN, _SNR_RANGE
from src.data_zhu import zhu_train_dataset, zhu_test_dataset, TEST_CONDITIONS
from src.train.train_v5 import _calibrate_snr_estimator, estimate_snr, _preload_zhu, _norm_snr, _load

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = ROOT / "checkpoints"
RES_DIR  = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)

PRETRAIN_CKPT = CKPT_DIR / "mambanet_2ch_s1_pre_best.pt"
SEED  = 1
BATCH = 512
LR    = 3e-4
EPOCHS = 5
TEST_N = 700


def set_seed(seed):
    import random; random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    slope, intercept = _calibrate_snr_estimator()

    zhu_tr_raw, zhu_val_raw = zhu_train_dataset(val_frac=0.111, seed=SEED)
    zhu_tr  = _preload_zhu(zhu_tr_raw, "Zhu-train")
    zhu_val = _preload_zhu(zhu_val_raw, "Zhu-val")

    ft_loader  = DataLoader(zhu_tr, batch_size=BATCH, shuffle=True, pin_memory=True,
                            generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(zhu_val, batch_size=BATCH, shuffle=False, pin_memory=True)

    model = build_model("mambanet_2ch").to(DEVICE)
    if PRETRAIN_CKPT.exists():
        _load(PRETRAIN_CKPT, model)
        print(f"Loaded pretrain: {PRETRAIN_CKPT}")
    else:
        print(f"WARNING: pretrain ckpt not found: {PRETRAIN_CKPT}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    rows = []
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        # Train with GT SNR — Zhu data has no GT SNR so we use linear as fallback
        # (GT SNR mode only meaningful for synth; for oracle we still use linear on Zhu)
        model.train()
        tot_loss = tot_ber = n = 0
        for x, y in ft_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            snr = estimate_snr(x, slope, intercept)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                logits, snr_pred = model(x, snr)
            loss, _ = v5_loss(logits.float(), y, snr_pred.float(), _norm_snr(snr))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ber = ((torch.sigmoid(logits.float().detach()) > 0.5).float() != y).float().mean().item()
            tot_loss += loss.item() * len(x); tot_ber += ber * len(x); n += len(x)

        # Val
        model.eval()
        val_ber_sum = val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                snr = estimate_snr(x, slope, intercept)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    logits, _ = model(x, snr)
                ber = ((torch.sigmoid(logits.float()) > 0.5).float() != y).float().mean().item()
                val_ber_sum += ber * len(x); val_n += len(x)
        val_ber = val_ber_sum / val_n
        elapsed = time.time() - t0
        print(f"  [oracle ep{ep}] tr_ber={tot_ber/n:.4f} val_ber={val_ber:.4f} {elapsed:.0f}s")
        rows.append({"epoch": ep, "train_ber": round(tot_ber/n, 6), "val_ber": round(val_ber, 6)})

    csv_path = RES_DIR / "v6b2_snr_oracle_s1_partial.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_ber","val_ber"])
        w.writeheader(); w.writerows(rows)
    print(f"Oracle result saved: {csv_path}")
    print(f"Final oracle val_ber={rows[-1]['val_ber']:.4f}")


if __name__ == "__main__":
    main()
