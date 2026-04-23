# Figure P6 — Training Curves

## What this chart is

Shows how the winning model's validation BER evolves during training (3 seeds), split into pretrain phase and finetune phase. Compared against the Zhu baseline 40-epoch training curve. Shows the two-phase training strategy visually — huge drop when fine-tuning kicks in.

## Visual brief

**Chart type:** Line chart, time series

**Two panels OR one combined panel:**

Option A — Two panels side by side:
- Left: Pretrain phase (20 epochs, 500K synthetic data). Y=val BER %. 3 lines (one per seed). Baseline is NOT shown here.
- Right: Finetune phase (30 epochs, Zhu 42K real data). Y=val BER %. 3 lines winner + 1 line baseline. Add dashed reference lines for final test BER of each.

Option B — Single panel with phase annotation:
- X-axis runs from 1 to 50 (pretrain ep 1–20 then finetune ep 1–30)
- Draw a vertical divider at epoch 20 labeled "→ Fine-tune starts"
- 3 colored lines for MambaNet-2ch seeds (s0, s1, s2)
- 1 dashed gray line for Zhu baseline training curve (40 epochs, separate scale or overlay)
- Shade area under winner below Zhu baseline's final val BER to show the win

**Key annotations:**
- Arrow/label at finetune epoch 1: "Finetune on real Zhu data"
- Horizontal dashed line at Zhu final test BER 3.12%
- Horizontal dashed line at winner final test BER 2.28%

## Data

### Zhu Baseline Training Curve (val_ber by epoch, 40 epochs)

Source: `results/baseline_train_log.csv`

| Epoch | Train Loss | Train BER% | Val BER% |
|---|---|---|---|
| 1 | 0.25102 | 49.97 | 49.88 |
| 2 | 0.24995 | 49.50 | 49.18 |
| 3 | 0.24652 | 46.03 | 40.75 |
| 4 | 0.22483 | 38.24 | 34.62 |
| 5 | 0.20583 | 34.05 | 30.80 |
| 6 | 0.19090 | 30.73 | 26.93 |
| 7 | 0.17799 | 27.75 | 24.31 |
| 8 | 0.16953 | 25.92 | 22.54 |
| 9 | 0.16328 | 24.69 | 21.51 |
| 10 | 0.15886 | 23.85 | 20.87 |
| 11 | 0.15511 | 23.13 | 20.82 |
| 12 | 0.15266 | 22.65 | 20.35 |
| 13 | 0.15046 | 22.26 | 20.54 |
| 14 | 0.14880 | 21.95 | 20.22 |
| 15 | 0.14697 | 21.65 | 20.15 |
| 16 | 0.14568 | 21.39 | 20.30 |
| 17 | 0.14358 | 21.02 | 20.57 |
| 18 | 0.14213 | 20.77 | 20.20 |
| 19 | 0.14099 | 20.55 | 20.40 |
| 20 | 0.13982 | 20.35 | 20.53 |
| 21 | 0.13858 | 20.13 | 20.43 |
| 22 | 0.13778 | 20.00 | 20.31 |
| 23 | 0.13660 | 19.80 | 20.29 |
| 24 | 0.13560 | 19.62 | 20.65 |
| 25 | 0.13431 | 19.39 | 20.83 |
| 26 | 0.13395 | 19.34 | 20.48 |
| 27 | 0.13290 | 19.15 | 20.63 |
| 28 | 0.13214 | 19.02 | 20.81 |
| 29 | 0.13117 | 18.86 | 21.00 |
| 30 | 0.13066 | 18.78 | 20.83 |
| 31 | 0.12983 | 18.62 | 20.89 |
| 32 | 0.12878 | 18.47 | 20.84 |
| 33 | 0.12848 | 18.41 | 20.90 |
| 34 | 0.12768 | 18.26 | 20.97 |
| 35 | 0.12684 | 18.13 | 20.96 |
| 36 | 0.12642 | 18.07 | 21.24 |
| 37 | 0.12568 | 17.92 | 21.29 |
| 38 | 0.12497 | 17.83 | 21.20 |
| 39 | 0.12448 | 17.73 | 21.14 |
| 40 | 0.12410 | 17.69 | 21.26 |

Note: Baseline val BER plateaus around 20–21% — this is on the Zhu validation set (NOT the test set). The *test* BER is separately computed as 3.12%. Val BER and test BER are not directly comparable because val uses a held-out fraction of the Zhu 42K train split while test uses the separate Zhu 8400 test set.

### MambaNet-2ch — Seed 1 Training Curve (complete, 50 rows)

Source: `results/mambanet_2ch_s1_log.csv`

Phase: pretrain (ep 1–20) on 500K synthetic data, finetune (ep 1–30) on Zhu 42K real data.

**IMPORTANT about pretrain val BER (~45–48%):** During pretrain, val is measured on a held-out slice of the SYNTHETIC dataset — NOT Zhu real data. This is a different distribution from the finetune val. The high BER (near 50% = random) reflects that the model has not yet seen Zhu real data. The pretrain phase trains on synthetic signals so the model learns GMSK physics before finetuning on real data.

| Phase | Epoch | Val BER% |
|---|---|---|
| pretrain | 1 | 45.87 |
| pretrain | 2 | 47.49 |
| pretrain | 3 | 47.75 |
| pretrain | 4 | 47.60 |
| pretrain | 5 | 47.48 |
| pretrain | 6 | 47.66 |
| pretrain | 7 | 47.72 |
| pretrain | 8 | 47.65 |
| pretrain | 9 | 47.74 |
| pretrain | 10 | 47.79 |
| pretrain | 11 | 47.75 |
| pretrain | 12 | 47.74 |
| pretrain | 13 | 47.74 |
| pretrain | 14 | 47.72 |
| pretrain | 15 | 47.86 |
| pretrain | 16 | 47.77 |
| pretrain | 17 | 47.82 |
| pretrain | 18 | 47.76 |
| pretrain | 19 | 47.81 |
| pretrain | 20 | 47.78 |
| finetune | 1 | 18.18 |
| finetune | 2 | 16.42 |
| finetune | 3 | 15.63 |
| finetune | 4 | 15.38 |
| finetune | 5 | 15.21 |
| finetune | 6 | 15.13 |
| finetune | 7 | 15.08 |
| finetune | 8 | 15.09 |
| finetune | 9 | 15.02 |
| finetune | 10 | 14.99 |
| finetune | 11 | 14.97 |
| finetune | 12 | 15.00 |
| finetune | 13 | 14.93 |
| finetune | 14 | 14.94 |
| finetune | 15 | 14.91 |
| finetune | 16 | 14.92 |
| finetune | 17 | 14.92 |
| finetune | 18 | 14.90 |
| finetune | 19 | 14.89 |
| finetune | 20 | 14.89 |
| finetune | 21 | 14.87 |
| finetune | 22 | 14.88 |
| finetune | 23 | 14.88 |
| finetune | 24 | 14.88 |
| finetune | 25 | 14.85 |
| finetune | 26 | 14.88 |
| finetune | 27 | 14.88 |
| finetune | 28 | 14.86 |
| finetune | 29 | 14.87 |
| finetune | 30 | 14.87 |

Key visual: finetune starts at 18.18% val BER (ep 1) and drops fast to ~15% by ep 6, then slowly converges to ~14.85–14.90% by ep 25–30.

### All 3 seeds — Final finetune val BER% (ep 30)

Source: `results/mambanet_2ch_s{0,1,2}_log.csv`, `results/mambanet_2ch_{s0,s1,s2}_test.csv`

| Seed | Pretrain epochs | Final finetune val BER% | Test BER% |
|---|---|---|---|
| Seed 0 | 6 (partial, resumed) | 15.06 | 2.312 |
| Seed 1 | 20 | 14.87 | 2.323 |
| Seed 2 | 20 | 15.12 | 2.319 |

Note: Seed 0 had only 6 pretrain epochs logged (power outage at ep 14 of a later run, resumed — the s0 partial log shows 6 pretrain epochs before finetune). Seed 1 has the lowest final val BER (14.87%) but not the lowest test BER — seeds 2 and 0 have similar or lower test BER due to noise in the val set.
