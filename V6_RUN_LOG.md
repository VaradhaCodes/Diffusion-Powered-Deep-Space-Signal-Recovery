# V6 Run Log
Started: 2026-04-23T23:05:00Z
Batch 0 (Zhu 3-seed baseline): RUNNING in separate session — do not touch
SNR_FIX_STATUS: FAILED
V6B3_CANONICAL_BER: 2.2820%  (500K pretrain, 3-seed ensemble, 2026-04-24)

---

## Batch 2 preflight answers

**Q1: Which function generates K-distributed complex channel gains h(t)?**
`kdist_fade` — src/synth_gen.py line 62.
```python
def kdist_fade(n_frames: int, m: float, b: float, rng: np.random.Generator) -> np.ndarray:
```

**Q2: What are b, alpha, m literally called in the code?**
- `b` is literally `b` (scale parameter, E[|h|²]=b²)
- `m` is literally `m` (shape/order parameter; Zhu paper calls this α)
- `alpha` from Zhu notation is NOT in the code — Zhu's α maps to the code's `m`

**Q3: Is fading i.i.d. per frame, per symbol, or correlated? Quote the exact line.**
I.i.d. per frame — one complex scalar h per frame:
```python
h = kdist_fade(1, m, b, rng)[0]   # src/synth_gen.py line 103
```
n_frames=1 → single draw, then applied to all 800 samples of that frame.

**Q4: How is noise variance set from Eb/N0? Is signal power pre- or post-fading? Quote the formula.**
POST-fading. From _awgn_sigma (lines 78-83):
```python
Ps = np.mean(np.abs(signal) ** 2)   # signal here is post-fading (sig * h)
Eb = Ps * sps
N0 = Eb / (10 ** (snr_db / 10))
return np.sqrt(N0 / 2)
```
Called after `sig = sig * h`, so `snr_db` IS the instantaneous received Eb/N0.

**Q5: How are the 6 test conditions tagged per frame — TX-side or instantaneous?**
TX-side Eb/N0 conditions only. data_zhu.py returns (x, y) — no per-frame SNR tag.
Condition names (e.g. "Awgn_Tb0d3", "kb2_Tb0d3_m1d2") identify channel config but no
instantaneous received SNR is stored per frame.

**Q6: Where is the SNR estimator called? Scalar per batch or per frame?**
Per frame (tensor shape B). In train_v5.py line 105:
```python
snr = estimate_snr(x, slope, intercept)
```
estimate_snr: `torch.log10(iq.pow(2).mean(dim=(1, 2)).clamp(min=1e-8))` → (B,) tensor.

**Q7: Is snr_norm = (snr_db - (-4.0)) / 12.0 literally in the code? Quote the line.**
NOT literally. The equivalent is in v5_model.py line 129 and competitors.py line 84/108/145:
```python
snr_norm = ((snr_db - _SNR_MIN) / _SNR_RANGE).unsqueeze(-1)
```
where `_SNR_MIN = -4.0` and `_SNR_RANGE = 12.0`.

**Q8: Quote the FiLM forward() signature. Confirm it expects one scalar per frame in [0,1].**
From v5_model.py lines 76-85:
```python
def forward(self, h: torch.Tensor, snr_norm: torch.Tensor) -> torch.Tensor:
    """
    h         : (B, T, D)
    snr_norm  : (B, 1)  values in [0, 1]
    returns   : (B, T, D)
    """
```
Yes — expects one scalar per frame (B, 1) in [0, 1].

---

## Batch 2 SNR target

Target = instantaneous received Eb/N0 = 10*log10(|h|² * Eb/N0_tx) in dB.

Rationale: FiLM should condition on what the decoder physically faces on this specific
frame. V5's linear estimator failed on KB2 because it was calibrated on AWGN data only
(AWGN received powers near 1), so under KB2 fading where E[|h|²]=4, the estimator
sees large received powers and maps them incorrectly.

Note: in synth_gen.py, `snr_db` already equals instantaneous received Eb/N0 since
_awgn_sigma computes noise from post-fading signal power. The SNR estimator training
label is therefore just `snr_db` from the generator.

---

---

## Batch 2 Part B — SNR estimator results

Gates (all 5 PASS):
- G1 Overall MAE:   0.585 dB  ≤1.0 dB  PASS
- G2 AWGN MAE:      0.495 dB  ≤0.5 dB  PASS (marginal)
- G3 KB2 m=1.4 MAE: 0.585 dB  ≤1.5 dB  PASS
- G4 Max bin bias:  0.889 dB  ≤2.0 dB  PASS
- G5 Decile cal:    max=0.04 dB  ≤1.0 dB  PASS

---

## Batch 2 Part C — SNR fix integration results

C3 Oracle (5 epochs, linear SNR on Zhu, measurement):
  Final val_ber = 15.20% (5 epochs from pretrain — still converging)

C4 Main retrain (30 epochs, neural SNR, seed 1):
  Overall BER = 2.3136% vs V5 seed-1 = 2.323% (spec)

Gates G6-G8:
- G6: delta=0.0094pp (need 0.05pp)  FAIL
- G7: KB2 m=1.4 delta=-0.0071pp (need 0.10pp improvement)  FAIL
- G8: max regression=0.044pp  PASS

Root cause analysis:
  - Linear estimator saturates at 8 dB for ALL KB2 frames (no variance at all)
  - Neural estimator gives 2-4 dB for KB2 frames (correct range, meaningful signal)
  - But BER improvement matches Phase 6 FiLM ablation noise floor (FiLM ~0.009pp)
  - FiLM SNR conditioning contributes only ~0.009pp regardless of SNR quality
  - Fundamental limit: the model doesn't lean on FiLM heavily enough for neural SNR
    to matter. Root fix requires wider synthetic SNR range + retrain from scratch.

SNR_FIX_STATUS: FAILED → all downstream batches use --snr-source=linear

Wall-clock: partA ~5min, partB ~25min (data gen 14s + training 3min + gates 2min),
           partC ~45min (oracle 5min + retrain 3min + debug 35min). Total ~75min.

---

## Preflight hardware

- GPU: NVIDIA RTX 5070, 12227 MiB VRAM, sm_120 ✓
- GPU at preflight: 10099 MiB used (baseline Batch 0 seed 2 training)
- Disk: 834 GB free ✓ (>80 GB required)
- Python: 3.12.3 ✓
- PyTorch: 2.11.0+cu130, CUDA 13.0 ✓
- mamba_ssm: OK ✓

---

---

## Batch 3 preflight

**Date**: 2026-04-24 09:04:36

**Predecessor check**:
- SNR_FIX_STATUS read from V6_RUN_LOG.md: FAILED
- All Batch 3 runs use --snr-source=linear (WARNING: neural SNR integration failed in Batch 2)

**Environment**:
- GPU: NVIDIA RTX 5070, 12227 MiB VRAM ✓
- mamba_ssm: OK (inside .venv) ✓
- Disk: 824 GB free ✓ (all sizes including 5M can run; peak ~63 GB)

**Batch 3 preflight answers**:

Q1: CLI entry point in synth_gen.py?
  No prior CLI existed. Added `if __name__ == "__main__"` block with --num-samples, --seed,
  --output-dir, --channel, --snr-lo, --snr-hi, --chunk flags. Generates in chunks of 100K to
  avoid RAM spike. Pre-allocates np.lib.format.open_memmap .npy files on disk.

  Programmatic signature (unchanged):
  ```python
  SynthDataset(n_samples, channel='mixed', snr_range=(-3.0,20.0),
               BT_choices=None, m_choices=None, b=2.0, seed=0)
  ```

Q2: Deterministically seeded?
  YES. `rng = np.random.default_rng(seed)` at line 153.
  Verified: MD5 run1=adb71537843ea01f0eb88d921135b094
            MD5 run2=adb71537843ea01f0eb88d921135b094  → MATCH ✓

Q3: V5 pretrain optimizer/scheduler/batch/grad-clip (from train_competitor.py):
  - Optimizer:  `torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)`  (pretrain_lr=1e-3)
  - Scheduler:  `CosineAnnealingLR(opt_pre, T_max=pretrain_epochs, eta_min=1e-5)`
  - Batch size: 512
  - Grad clip:  `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` (in _run, line 73)

Q4: Early stopping support?
  No — train_competitor.py runs all epochs unconditionally.
  Added: train_v6b3.py implements early stopping via patience counter in both
  pretrain (patience=5) and finetune (patience=10) functions.

**Batch 3 adaptations**:
  1. synth_gen.py: added CLI entry point (--num-samples, --seed, --output-dir, chunked memmap)
  2. train_v6b3.py: new script with AdamW+warmup pretrain, early stopping, v6b3_* naming,
     step-level checkpoints, SynthNpyDataset, set_seed/worker_init_fn seeding
  3. sweep_v6b3.py: sweep orchestration with sweep-level early stopping, disk cleanup
  4. v6b3_scaling.py: power-law fit + scaling curve figure

**Disk budget**:
  Free: 824 GB. All sizes (500K/1M/2M/5M) can run simultaneously (~63 GB peak).
  Deletion policy applied after each successor corpus is verified.

**Sweep**: 500K → 1M → 2M → 5M  |  snr-source=linear

---

## Batch 3 preflight

**Date**: 2026-04-24 09:05:06

**Predecessor check**:
- SNR_FIX_STATUS read from V6_RUN_LOG.md: FAILED
- All Batch 3 runs use --snr-source=linear (WARNING: neural SNR integration failed in Batch 2)

**Environment**:
- GPU: NVIDIA RTX 5070, 12227 MiB VRAM ✓
- mamba_ssm: OK (inside .venv) ✓
- Disk: 824 GB free ✓ (all sizes including 5M can run; peak ~63 GB)

**Batch 3 preflight answers**:

Q1: CLI entry point in synth_gen.py?
  No prior CLI existed. Added `if __name__ == "__main__"` block with --num-samples, --seed,
  --output-dir, --channel, --snr-lo, --snr-hi, --chunk flags. Generates in chunks of 100K to
  avoid RAM spike. Pre-allocates np.lib.format.open_memmap .npy files on disk.

  Programmatic signature (unchanged):
  ```python
  SynthDataset(n_samples, channel='mixed', snr_range=(-3.0,20.0),
               BT_choices=None, m_choices=None, b=2.0, seed=0)
  ```

Q2: Deterministically seeded?
  YES. `rng = np.random.default_rng(seed)` at line 153.
  Verified: MD5 run1=adb71537843ea01f0eb88d921135b094
            MD5 run2=adb71537843ea01f0eb88d921135b094  → MATCH ✓

Q3: V5 pretrain optimizer/scheduler/batch/grad-clip (from train_competitor.py):
  - Optimizer:  `torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)`  (pretrain_lr=1e-3)
  - Scheduler:  `CosineAnnealingLR(opt_pre, T_max=pretrain_epochs, eta_min=1e-5)`
  - Batch size: 512
  - Grad clip:  `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` (in _run, line 73)

Q4: Early stopping support?
  No — train_competitor.py runs all epochs unconditionally.
  Added: train_v6b3.py implements early stopping via patience counter in both
  pretrain (patience=5) and finetune (patience=10) functions.

**Batch 3 adaptations**:
  1. synth_gen.py: added CLI entry point (--num-samples, --seed, --output-dir, chunked memmap)
  2. train_v6b3.py: new script with AdamW+warmup pretrain, early stopping, v6b3_* naming,
     step-level checkpoints, SynthNpyDataset, set_seed/worker_init_fn seeding
  3. sweep_v6b3.py: sweep orchestration with sweep-level early stopping, disk cleanup
  4. v6b3_scaling.py: power-law fit + scaling curve figure

**Disk budget**:
  Free: 824 GB. All sizes (500K/1M/2M/5M) can run simultaneously (~63 GB peak).
  Deletion policy applied after each successor corpus is verified.

**Sweep**: 500K → 1M → 2M → 5M  |  snr-source=linear

---

## Batch 3 preflight

**Date**: 2026-04-24 09:22:02

**Predecessor check**:
- SNR_FIX_STATUS read from V6_RUN_LOG.md: FAILED
- All Batch 3 runs use --snr-source=linear (WARNING: neural SNR integration failed in Batch 2)

**Environment**:
- GPU: NVIDIA RTX 5070, 12227 MiB VRAM ✓
- mamba_ssm: OK (inside .venv) ✓
- Disk: 824 GB free ✓ (all sizes including 5M can run; peak ~63 GB)

**Batch 3 preflight answers**:

Q1: CLI entry point in synth_gen.py?
  No prior CLI existed. Added `if __name__ == "__main__"` block with --num-samples, --seed,
  --output-dir, --channel, --snr-lo, --snr-hi, --chunk flags. Generates in chunks of 100K to
  avoid RAM spike. Pre-allocates np.lib.format.open_memmap .npy files on disk.

  Programmatic signature (unchanged):
  ```python
  SynthDataset(n_samples, channel='mixed', snr_range=(-3.0,20.0),
               BT_choices=None, m_choices=None, b=2.0, seed=0)
  ```

Q2: Deterministically seeded?
  YES. `rng = np.random.default_rng(seed)` at line 153.
  Verified: MD5 run1=adb71537843ea01f0eb88d921135b094
            MD5 run2=adb71537843ea01f0eb88d921135b094  → MATCH ✓

Q3: V5 pretrain optimizer/scheduler/batch/grad-clip (from train_competitor.py):
  - Optimizer:  `torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)`  (pretrain_lr=1e-3)
  - Scheduler:  `CosineAnnealingLR(opt_pre, T_max=pretrain_epochs, eta_min=1e-5)`
  - Batch size: 512
  - Grad clip:  `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` (in _run, line 73)

Q4: Early stopping support?
  No — train_competitor.py runs all epochs unconditionally.
  Added: train_v6b3.py implements early stopping via patience counter in both
  pretrain (patience=5) and finetune (patience=10) functions.

**Batch 3 adaptations**:
  1. synth_gen.py: added CLI entry point (--num-samples, --seed, --output-dir, chunked memmap)
  2. train_v6b3.py: new script with AdamW+warmup pretrain, early stopping, v6b3_* naming,
     step-level checkpoints, SynthNpyDataset, set_seed/worker_init_fn seeding
  3. sweep_v6b3.py: sweep orchestration with sweep-level early stopping, disk cleanup
  4. v6b3_scaling.py: power-law fit + scaling curve figure

**Disk budget**:
  Free: 824 GB. All sizes (500K/1M/2M/5M) can run simultaneously (~63 GB peak).
  Deletion policy applied after each successor corpus is verified.

**Sweep**: 500K → 1M → 2M → 5M  |  snr-source=linear

**Sweep**: 500K → 1M → 2M → 5M  |  snr-source=linear

---

## Batch 3 — 500K pretrain recovery (2026-04-24, resumed after power outage ~15:00 IST)

**Recovery state**:
- 500K corpus: verified present (3.2 GB, md5=5b87f83d2a476be1af4312c2ef32463b)
- Pretrain s0: COMPLETE — early-stopped ep28, best at ep23 (synth_val_loss=0.222284), ckpt=v6b3_pre_500K_s0.pt
- Pretrain s1: INTERRUPTED by power outage at ep32, patience=3/5 — best checkpoint saved at ep29 (synth_val_loss=0.221824), ckpt=v6b3_pre_500K_s1.pt. Remaining improvement negligible (LR=1.07e-04 at ep32, delta since best=0.000009). Proceeding with ep29 checkpoint.
- Fine-tuning: NOT STARTED (resuming here)
- snr-source=linear (WARNING: SNR_FIX_STATUS=FAILED)

---

## Batch 3 — 500K fine-tune + eval results

**Date**: 2026-04-24 12:45 – 12:54 IST

### Pretrain summary
| seed | conv_epoch | best_synth_val_loss |
|------|-----------|---------------------|
| 0    | 28 (early stop) | 0.222284 |
| 1    | 32 (power outage, patience=3/5, best@ep29) | 0.221824 |

### Fine-tune summary (30 max epochs, patience=10)
| seed | conv_epoch | best_val_ber |
|------|-----------|-------------|
| 0    | 25 (early stop) | 0.1502 |
| 1    | 27 (early stop) | 0.1469 |
| 2    | 25 (early stop) | 0.1495 |

### Per-seed test BER (4200 frames, 6 conditions)
| seed | AWGN_Tb0d3 | AWGN_Tb0d5 | KB2_Tb0d3_m1d2 | KB2_Tb0d3_m1d4 | KB2_Tb0d5_m1d2 | KB2_Tb0d5_m1d4 | OVERALL |
|------|-----------|-----------|---------------|---------------|---------------|---------------|---------|
| 0    | 1.08%     | 1.19%     | 1.92%         | 3.81%         | 1.89%         | 3.92%         | 2.30%   |
| 1    | 1.03%     | 1.17%     | 1.89%         | 3.75%         | 1.85%         | 3.94%         | 2.27%   |
| 2    | 1.02%     | 1.17%     | 1.89%         | 3.81%         | 1.84%         | 3.92%         | 2.28%   |

### 3-seed ensemble BER: 2.2820%

### Sweep-level early stopping decision
Gain vs V5 baseline (2.2750%): -0.007 pp (500K is 0.007pp WORSE than V5, not better)
Rule: "If 500K fails to beat V5 baseline by 0.05pp → winner = 500K"
Decision: **SWEEP STOPPED AFTER 500K. Winner = 500K.**
Sizes 1M / 2M / 5M: NOT RUN.

### BRUTAL HONESTY CLAUSE applied
V6b3 500K (2.282%) ≅ V5 mambanet_2ch (2.275%). Delta = 0.007pp, within noise.
Finding: **Data size is NOT the bottleneck for this task. The architecture/training regime is already saturated at 500K synthetic samples for the Zhu fine-tune target.**
Power-law fit: skipped (only 1 data point).

### Canonical files promoted
- checkpoints/v6b3_canonical_pretrain.pt ← v6b3_pre_500K_s1.pt (seed 1, ep29)
- checkpoints/v6b3_canonical_s0.pt       ← v6b3_500K_s0_ft.pt
- checkpoints/v6b3_canonical_s1.pt       ← v6b3_500K_s1_ft.pt
- checkpoints/v6b3_canonical_s2.pt       ← v6b3_500K_s2_ft.pt

**V6B3_CANONICAL_BER = 2.2820%**

---

## Batch 3 — 1M run (2026-04-24, manual override of sweep-stop)

**Rationale**: Sweep stopped at 500K because gain vs V5 baseline was -0.007pp. However, V5 pretrain used old synth_gen.py (no CLI, different chunking) so comparison wasn't clean. User decided to run 1M to get a real data point.

### Pretrain summary
| seed | conv_epoch | best_synth_val_loss |
|------|-----------|---------------------|
| 0    | 40 (hit max, no early stop) | 0.2204 |
| 1    | 39 (early stop) | 0.2194 |

1M pretrain reached lower loss than 500K (0.2204/0.2194 vs 0.2222/0.2218) — the model IS learning more from more data during pretrain.

### Fine-tune summary
| seed | conv_epoch | best_val_ber |
|------|-----------|-------------|
| 0    | 30 (hit max) | 0.1497 |
| 1    | 27 (early stop) | 0.1448 |
| 2    | 25 (early stop) | 0.1478 |

### Per-seed test BER
| seed | AWGN_Tb0d3 | AWGN_Tb0d5 | KB2_Tb0d3_m1d2 | KB2_Tb0d3_m1d4 | KB2_Tb0d5_m1d2 | KB2_Tb0d5_m1d4 | OVERALL |
|------|-----------|-----------|---------------|---------------|---------------|---------------|---------|
| 0    | 1.10%     | 1.19%     | 1.86%         | 3.75%         | 1.88%         | 3.93%         | 2.28%   |
| 1    | 1.10%     | 1.14%     | 1.87%         | 3.70%         | 1.88%         | 3.89%         | 2.26%   |
| 2    | 1.09%     | 1.14%     | 1.87%         | 3.76%         | 1.83%         | 3.86%         | 2.26%   |

### 3-seed ensemble BER: 2.2692%

### Sweep-level early stopping decision
| Size | Ensemble BER | Gain vs previous |
|------|-------------|-----------------|
| V5 baseline | 2.2750% | — |
| 500K | 2.2820% | -0.007pp (worse) |
| 1M   | 2.2692% | +0.013pp vs 500K |

Gain 500K→1M = 0.013pp < 0.05pp threshold → **SWEEP STOPPED**.
Tiebreaker: 500K and 1M within 0.02pp → smaller size wins = **500K**.

### BRUTAL HONESTY CLAUSE
Range across all sizes: 2.2692% – 2.2820% = 0.013pp < 0.15pp.
The curve is FLAT. **Data size is not the bottleneck.**
More data does improve pretrain quality (lower synth_val_loss) but this does NOT translate to meaningful test BER improvement after Zhu fine-tune.
Bottleneck is likely the fine-tune data size (only 37K samples) or architecture capacity.

**Winner = 500K. V6B3_CANONICAL_BER = 2.2820%** (unchanged).

Scaling curve updated with 2 data points → figures/v6b3_scaling_curve.png

Scaling curve updated with 2 data points → figures/v6b3_scaling_curve.png

---

## Batch 4 preflight

**Date**: 2026-04-24

### Q1 — MambaNet2ch exact architecture

**CNN stem (2-channel, raw IQ input — no FeatureExtractor)**:
| Layer | Type | Config |
|-------|------|--------|
| 0 | Conv1d | in=2, out=32, k=7, pad=3 |
| 1 | BatchNorm1d | 32 |
| 2 | GELU | — |
| 3 | Conv1d | in=32, out=64, k=7, pad=3 |
| 4 | BatchNorm1d | 64 |
| 5 | GELU | — |
| 6 | Conv1d | in=64, out=128, k=8, stride=8 | ← 800→100 downsampling |
| 7 | BatchNorm1d | 128 |
| 8 | GELU | — |

**MHA block**: `d_model=128, num_heads=8, dropout=0.0, batch_first=True`
- Pre-norm residual: `h = norm1(h + attn(h, h, h, need_weights=False)[0])`
- **NO FFN** (bare attention only)

**BiMamba2 block**: `Mamba2(d_model=128, d_state=64, headdim=64, chunk_size=64)`
- Bidirectional: `h_f + flip(bwd(flip(h)))` — sum merge
- Pre-norm residual: `h = norm2(h + bi_m2(h))`

**FiLM**: `snr_norm = (snr_db − (−4.0)) / 12.0` → MLP(1→64 GELU→256) → gamma+beta
- Applied per-token after MHA+BiMamba2 blocks: `(1+γ)·h + β`

**Bit head**: `Linear(128, 1)` per symbol token
**SNR head**: `Linear(128, 1)` on `mean(h, dim=1)`

### Q2 — Param count print

`train_competitor.py` already prints params at model creation (confirmed at line ~163).
Added param count print to `build_model()` in `competitors.py` (now prints every call).
No action needed beyond that.

### Q3 — CLI flags in train_competitor.py

Previously missing: `--depth`, `--width`, `--kernel-size`, `--block-type`, `--loss`.
**Added all 5 flags** (map to `mambanet_2ch_cfg` architecture kwargs when `--model mambanet_2ch_cfg`).

### SNR source
`SNR_FIX_STATUS = FAILED` (line 4 of this log).
All V6 Batch 4 fine-tunes: `--snr-source=linear` (power estimator fallback).

### Param counts for sweep configs (verified by forward pass)
| SB | Config | Params |
|----|--------|--------|
| SB1 | d=128 n=1 k=7 serial | 399,354 |
| SB2 | d=128 n=1 k=31 serial | 400,890 |
| SB3 | d=128 n=2 k=31 serial | 702,226 |
| SB4 | d=128 n=4 k=31 serial | 1,304,898 |
| SB5 | d=192 n=2 k=31 serial | 1,438,442 |
| SB6 | d=256 n=2 k=31 serial | 2,437,826 |
| SB7 | d=192 n=2 k=31 parallel | 1,437,674 |

All ≤ 2.5M (C4 gate). SB5 best_depth TBD pending SB3/SB4 results.

### Pretrain key transfer (strict=False, shape-safe filter)
- SB1 (same arch): 53/53 tensors loaded (100%)
- SB2 (k=31): 52/53 (first conv shape mismatch → random init for cnn.0)
- SB3 (depth=2): 52/77 (first conv + second block random init)
- SB5/SB7 (d=192): 18/73-77 (d_model change → only first 2 CNN layers transfer)


---

## Batch 4 preflight (auto-written 2026-04-24 18:11:27)

### Q1 — MambaNet2ch exact architecture
**CNN stem (2-channel)**:
- Conv1d(2→32, k=7, pad=3) → BN → GELU
- Conv1d(32→64, k=7, pad=3) → BN → GELU
- Conv1d(64→128, k=8, stride=8) → BN → GELU  [800→100 downsampling]

**MHA block**: d_model=128, num_heads=8, dropout=0.0, batch_first=True
- pre-norm residual: `h = norm1(h + attn(h,h,h)[0])`
- NO FFN (attention only)

**BiMamba2 block**: d_state=64, headdim=64, chunk_size=64
- fwd pass + flip(bwd(flip(h))) — sum merge
- pre-norm residual: `h = norm2(h + bi_m2(h))`

**FiLM**: snr_norm = (snr_db - (-4.0)) / 12.0; MLP(1→64→256) → gamma+beta
- Applied after MHA+BiMamba2 blocks (per-token scale+shift)

**Bit head**: Linear(128, 1) per token
**SNR head**: Linear(128, 1) on mean-pooled h

### Q2 — Param count print
train_competitor.py already prints params at line ~163. build_model() now also
prints params (added in competitors.py). No additional changes needed.

### Q3 — CLI flags in train_competitor.py
Added: --depth, --width, --kernel-size, --block-type, --loss
(All map to mambanet_2ch_cfg architecture kwargs.)

### SNR source
SNR_FIX_STATUS=FAILED → --snr-source=linear for all fine-tunes.

### PRE_B4_CANONICAL_BER = 2.2820%

### SB1 — LR warmup
seed0=2.2850%  seed1=2.2774%
mean=2.2812%  delta_vs_pre=0.0008pp

### SB2 — Wider CNN k=31
seed0=2.2981%  seed1=2.2748%
mean=2.2865%  delta_vs_pre=-0.0045pp

### SB3 — Depth 2x
seed0=2.2581%  seed1=2.2102%
mean=2.2342%  delta_vs_pre=0.0478pp

### SB4 — Depth 4x (grad_ckpt)
seed0=2.2550%  seed1=2.2407%
mean=2.2478%  delta_vs_pre=0.0341pp

**Depth decision**: SB3=2.2342% SB4=2.2478% → best_depth=2

### SB5 — Width 192 depth=2
seed0=2.2271%  seed1=2.2026%
mean=2.2149%  delta_vs_pre=0.0671pp

### SB6 — Width 256 depth=2 (grad_ckpt)
seed0=2.2152%  seed1=2.2188%
mean=2.2170%  delta_vs_pre=0.0650pp

**Width decision**: SB5=2.2149% SB6=2.2170% → sb7_d_model=192

### SB7 — Parallel Mcformer d=192 depth=2
seed0=2.2169%  seed1=2.2300%
mean=2.2235%  delta_vs_pre=0.0585pp

### Winner selection
Winner: sb7  mean_BER=2.2235%  delta_vs_pre=0.0585pp
Params: 1,437,674

### 3-seed promotion
Winner seed=2 BER=2.2105%
3-seed ensemble BER = 2.2191%
Improvement vs pre-B4 canonical: 0.0629pp
Paired t-test (n=6 conditions) p=0.1685
Decision: NO-OP (insufficient improvement). v6b3 stays final.

### SB8 — Loss ablation
| variant | seed | BER |
|---------|------|-----|
| bce | 0 | 2.2126% |
| bce_ls | 0 | 2.2136% |
| focal | 0 | 2.2183% |
| focal_ls | 0 | 2.2045% |

**SB8 winner**: bce (BER=2.2126%)

### Final promotion
V6_FINAL_BER = 2.2191%  (config: sb7)
PRE_B4_CANONICAL_BER = 2.2820%
Delta = 0.0629pp
Checkpoints: v6_final_s{0,1,2}.pt

### Sweep summary (2-seed mean BER)
| sb | mean_BER |
|----|---------|
| sb1 | 2.2812% |
| sb2 | 2.2865% |
| sb3 | 2.2342% |
| sb4 | 2.2478% |
| sb5 | 2.2149% |
| sb6 | 2.2170% |
| sb7 | 2.2235% |

---

## Batch 4 — Architecture sweep results

**Date**: 2026-04-24 18:10 – 19:22 IST
**PRE_B4_CANONICAL_BER = 2.2820%** (v6b3_canonical 3-seed ensemble, confirmed fresh)

### Sweep summary (2-seed mean BER, seeds 0 and 1)

| SB | Description | d_model | n_blocks | cnn_k1 | block | mean BER | Δ vs PRE | C1 (≥0.05pp) | Notes |
|----|-------------|---------|---------|--------|-------|----------|---------|-------------|-------|
| SB1 | warmup only | 128 | 1 | 7 | serial | 2.2812% | +0.001pp | ✗ | Warmup ~neutral |
| SB2 | wider k=31 | 128 | 1 | 31 | serial | 2.2865% | −0.004pp | ✗ | Wider CNN hurt |
| SB3 | depth=2 | 128 | 2 | 31 | serial | 2.2342% | +0.048pp | ✗ (0.002pp short) | Best depth |
| SB4 | depth=4 | 128 | 4 | 31 | serial | 2.2478% | +0.034pp | ✗ | Overfits, depth=2 wins |
| SB5 | width=192 | 192 | 2 | 31 | serial | 2.2149% | +0.067pp | **✓** | First gate-passer |
| SB6 | width=256 | 256 | 2 | 31 | serial | 2.2170% | +0.065pp | **✓** | SB5 wins on params |
| SB7 | parallel | 192 | 2 | 31 | parallel | 2.2235% | +0.059pp | **✓** | **WINNER** (smallest params among candidates) |

**Depth decision**: SB3 (2.2342%) < SB4 (2.2478%) → best_depth = 2
**Width decision**: SB5 (2.2149%) < SB6 (2.2170%) → sb7_d_model = 192

### Winner selection
Candidates (pass all C1-C4): SB5, SB6, SB7
All three within 0.03pp tiebreak window (range = 0.009pp):
- SB7: 1,437,674 params ← smallest → **WINNER**
- SB5: 1,438,442 params (768 params more)
- SB6: 2,437,826 params

### 3-seed promotion (winner = SB7)
| seed | BER |
|------|-----|
| 0 | 2.217% |
| 1 | 2.230% |
| 2 | 2.211% |

**3-seed ensemble BER = 2.2191%**
Improvement vs PRE: **0.0629pp**
Paired t-test (n=6 conditions): **p = 0.1685**

**Decision: NO-OP** — p > 0.10. Improvement is real (0.063pp) but not statistically significant across 6 conditions (n=6 gives very low test power). **v6b3_canonical stays as V6 final.**

Note: code bug in final promotion path caused SB7 checkpoints to be copied to v6_final — this was corrected post-run (v6_final_s{0,1,2}.pt re-set to v6b3_canonical).

### SB8 — Loss ablation (SB7 arch, seed=0)
| Variant | BER (s0) |
|---------|---------|
| bce (baseline) | 2.2126% |
| bce + label smooth 0.05 | 2.2136% |
| focal BCE γ=2 | 2.2183% |
| focal + label smooth | 2.2045% |

Range = 0.014pp < 0.03pp threshold → **plain BCE wins** (simplest).

### Final state
**V6_FINAL_BER = 2.2820%** (v6b3_canonical, unchanged per NO-OP rule)
**V6_FINAL = checkpoints/v6_final_s{0,1,2}.pt = v6b3_canonical**

### Key findings from Batch 4
1. **Architecture capacity IS a bottleneck**: wider model (d=192) gains +0.067pp over baseline (d=128). More capacity helps.
2. **Statistical gate conservative**: the n=6 condition t-test lacks power to confirm 0.063pp gain. A true 3-seed × 6-condition evaluation would need more conditions or seeds to reach p<0.10.
3. **Depth ceiling at 2**: depth=4 overfits on 37K Zhu fine-tune samples. Fine-tune data, not capacity, limits deeper models.
4. **Parallel vs serial negligible**: SB7 parallel ≈ SB5 serial (0.009pp difference). Mcformer-style fusion provides no benefit here.
5. **Loss function irrelevant**: all 4 variants within 0.014pp. BCE is fine.
6. **Next bottleneck**: fine-tune data size (only 37K Zhu frames). SB7 overfits from ep8 onwards. Generating synthetic Zhu-equivalent fine-tune data is the clear next step.

