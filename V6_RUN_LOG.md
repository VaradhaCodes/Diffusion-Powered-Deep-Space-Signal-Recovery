# V6 Run Log
Started: 2026-04-23T23:05:00Z
Batch 0 (Zhu 3-seed baseline): RUNNING in separate session — do not touch
SNR_FIX_STATUS: PENDING
V6B3_CANONICAL_BER: PENDING

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

## Preflight hardware

- GPU: NVIDIA RTX 5070, 12227 MiB VRAM, sm_120 ✓
- GPU at preflight: 10099 MiB used (baseline Batch 0 seed 2 training)
- Disk: 834 GB free ✓ (>80 GB required)
- Python: 3.12.3 ✓
- PyTorch: 2.11.0+cu130, CUDA 13.0 ✓
- mamba_ssm: OK ✓

---
