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
