# Mission

Build a receiver that beats Zhu et al. 2023's 1D-CNN+Bi-LSTM on *their own* public Zenodo dataset, by a large enough margin to be publishable and not explainable as noise. This is a 7-day wall-clock group project with optional paper follow-through.

The architectural headline is **Bidirectional Mamba-3 with SNR-conditional FiLM modulation, trained on a 10× augmented synthetic dataset, with Viterbi trellis post-processing**. The *compound* of these, not any one of them alone, is the bet.

This project is `v5`. Versions 1–3 lost to vibe-coding. Version 4 was a single-axis bet on "swap Bi-LSTM for Mamba-3" that autopsied badly. **This version compounds ~8 independent 0.2–1.0 dB wins so that even if half fail we still win.**

---

# Rules of engagement (non-negotiable)

1. **No silent fallbacks.** If `from mamba_ssm import Mamba3` fails, or `Mamba3(...)` errors on a CUDA tensor, or forward-pass NaNs, you STOP and log. You do not substitute Mamba-2, minimal-mamba, GRU, or LSTM. A failed Mamba-3 means diagnostic mode, not fake mode.
2. **No invented numbers.** Every number in logs or reports traces to a file on disk. "Not measured" is a valid entry. Fabrication is not.
3. **No "day 1, day 2" structure.** You will run continuously. Progress is gated by phase completion, not by time.
4. **Append, never rewrite `RUN_LOG.md`.** Every non-trivial step gets a timestamped entry with a one-line summary, a pointer to the artifact produced, and the next intended step.
5. **Idempotent phases.** Every phase has an early-exit: if its output artifact exists and is valid, skip. This is so you can be restarted safely after any crash.
6. **Read `CLAUDE.md` first, every time context resets.** If you lose context mid-run, re-reading `CLAUDE.md` must be enough to continue.
7. **If you find yourself tuning to make a number look better: stop.** Honest loss with a clean recipe is worth infinitely more than cherry-picked win.
8. **Pace yourself. User has wellbeing history around hero-mode. Don't design scripts that assume 3 AM supervision.** Checkpoint every epoch, resumable training, single-command resume.

---

# Environment assumptions

- **OS**: Ubuntu via WSL2 on Windows 11. Bare Linux equivalent.
- **GPU**: NVIDIA RTX 5070 12GB, Blackwell, compute capability sm_120 (12.0).
- **Toolchain**: CUDA 12.8+ driver. PyTorch 2.10 or 2.11 stable cu128 wheels natively support sm_120 (Jan–March 2026 releases). **No building PyTorch from source needed.**
- **mamba-ssm 2.3.1** (March 10, 2026 PyPI release) ships the real `Mamba3` API. Install from PyPI, fall back to source build if wheel resolution fails.
- **Python**: 3.11 or 3.12.
- **User data**:
  - Zhu's Zenodo dataset is already at `/mnt/c/EM Project/data_set/data_set/` (check this first, Zenodo download is fallback).
  - Zhu's reference notebook is at `/mnt/c/EM Project/AI_Demodulator-main/ai_demodulator.ipynb`. Parse it to cross-check baseline hyperparameters.
- **Disk**: assume 80 GB free minimum.

If any of these is false, write `ENV_MISMATCH.md` and STOP.

---

# First action (literally the first thing you do)

Create `CLAUDE.md` with this content exactly. This is re-read on every context reset.

````markdown
# CLAUDE.md — Deep Space Signal Recovery v5

## What this project is
Build a receiver that beats Zhu et al. 2023 on their Zenodo GMSK dataset.
Architecture: CNN stem + Bidirectional Mamba-3 + FiLM(SNR) + multi-task loss + Viterbi post.
Training data: 500K–1M synthetic samples from our own K-dist simulator, Zhu's 63K as fine-tune/validation, Zhu's test set as held-out.

## Current phase
Update this whenever you advance a phase. Format: `Phase N — <label> — in progress | DONE | BLOCKED`.

- Phase 0 — Environment + Mamba-3 Blackwell gate — <UPDATE>
- Phase 1 — Data (Zhu local + synth generator + generator validation) — <UPDATE>
- Phase 2 — Zhu baseline reproduction — <UPDATE>
- Phase 3 — Features + V5 model smoke test — <UPDATE>
- Phase 4 — V5 main training (3 seeds) — <UPDATE>
- Phase 5 — Competitor baselines (Bi-Transformer, Mamba-2, MambaNet-style) — <UPDATE>
- Phase 6 — Ablations — <UPDATE>
- Phase 7 — Evaluation (TTA, ensemble, Viterbi post, optional coded BER) — <UPDATE>
- Phase 8 — Figures + reports — <UPDATE>

## Key non-negotiables (re-read these)
- No silent fallbacks. If Mamba-3 fails, stop — don't substitute.
- Every number in reports must trace to a CSV/artifact. No fabrication.
- Soft gate (not hard) on baseline reproduction: within 1.0 dB average AND qualitative trends match.
- fp32 SSM parameters, bf16 autocast. Gradient clip 1.0.
- Checkpoint every epoch. Resumable.

## Paths
- Zhu data (local): `/mnt/c/EM Project/data_set/data_set/`
- Zhu reference notebook: `/mnt/c/EM Project/AI_Demodulator-main/ai_demodulator.ipynb`
- Working dir: `$(pwd)` (probably `~/deepspace_v5`)
- Synthetic data: `./synth_data/`
- Checkpoints: `./checkpoints/`
- Results CSVs: `./results/`
- Figures: `./figures/`

## Critical hyperparameters (from Zhu Table 1, 2)
- Input: 2×800 (I/Q, 100 symbols × 8 samples)
- Labels: 100 bits per frame
- Zhu loss: MSE (we reproduce this for baseline; our main model uses BCE)
- Zhu optimizer: Adam, lr=1e-3 assumed (not explicit in paper; confirm via ai_demodulator.ipynb)
- Zhu batch size: 512, epochs: 40, dropout: 0.08 FC / 0.2 Bi-LSTM
- Zhu train/val/test: 63000 / 7875 / 4200
- Scintillation indices: m ∈ {1.2, 1.4}. Channel params: α=10 or 5, b=2.
- BT product: {0.3, 0.5}

## If you lose context mid-run
1. `cat CLAUDE.md` — you are here.
2. `tail -50 RUN_LOG.md` — recent actions.
3. Find latest "in progress" phase in CLAUDE.md. Open the scripts for that phase.
4. Check if their output artifacts exist under `./results/` or `./checkpoints/`.
5. If they exist and look valid — advance phase status, continue next phase.
6. If partial — resume training from latest checkpoint.
7. If unclear — `git log --oneline -20`.
````

Commit this file. Initialize a git repo. `git add CLAUDE.md PROMPT.md && git commit -m "project init"`. Commit at the end of every phase.

---

# Directory layout (create this once)

```
deepspace_v5/
├── CLAUDE.md
├── PROMPT.md                         # this file
├── RUN_LOG.md                        # append-only
├── INSTALL_LOG.md
├── env.yml                           # conda/micromamba env
├── requirements.txt                  # pip deltas
├── src/
│   ├── env_check.py
│   ├── data/
│   │   ├── zhu_loader.py             # PyTorch Dataset, local-first path
│   │   ├── zhu_notebook_parse.py     # extract hparams from ai_demodulator.ipynb
│   │   └── synth_generator.py        # our K-dist GMSK simulator
│   ├── validate/
│   │   ├── validate_synth_vs_zhu.py  # simulator must match Zhu Fig 3 classical Viterbi
│   │   └── sanity_plots.py
│   ├── features/
│   │   └── preprocess.py             # 6-channel input pipeline
│   ├── baselines/
│   │   ├── zhu_cnn_bilstm.py
│   │   ├── bilstm_only.py
│   │   ├── cnn_only.py
│   │   ├── bi_transformer.py
│   │   ├── mamba2_receiver.py
│   │   └── mambanet_style.py
│   ├── models/
│   │   ├── mamba3_bi_receiver.py     # THE MAIN MODEL
│   │   └── viterbi_post.py           # soft-input Viterbi refinement
│   ├── train/
│   │   ├── common.py                 # AMP, EMA, curriculum, mixup, clip, resume
│   │   ├── train_baseline.py
│   │   ├── train_main.py
│   │   ├── train_competitor.py
│   │   └── train_ablation.py
│   ├── eval/
│   │   ├── sweep_ber.py
│   │   ├── stats.py                  # Wilson CI, paired t-test
│   │   ├── tta.py
│   │   ├── ensemble.py
│   │   └── coded_ber.py              # OPTIONAL, LDPC CCSDS
│   └── figures/
│       ├── fig1_geometry.py
│       ├── fig2_ber_ebn0.py
│       ├── fig3_ber_vs_m.py
│       ├── fig4_constellations.py
│       └── fig5_ablations.py
├── data/
│   └── zhu/                          # symlink to user's local copy
├── synth_data/                       # our generated data
├── checkpoints/
├── figures/
├── results/
│   ├── baseline_ber.csv
│   ├── v5_ber.csv
│   ├── competitor_ber.csv
│   └── ablation_ber.csv
└── reports/
    ├── group_project.md
    └── paper_draft.md
```

---

# Phase 0 — Environment + Blackwell + Mamba-3 gate (HARD)

## 0.1 System check

```bash
nvidia-smi
nvcc --version || echo "nvcc not in path; ok if torch bundles its own"
python --version   # expect 3.11 or 3.12
```

Log all three outputs verbatim in `INSTALL_LOG.md`.

## 0.2 Environment

Use `micromamba` if available; else `conda`; else `python -m venv`. Create `env.yml`:

```yaml
name: dsv5
channels: [conda-forge]
dependencies:
  - python=3.11
  - pip
  - ninja
  - cmake
  - git
  - pip:
      - numpy<2.0
      - scipy
      - pandas
      - matplotlib
      - seaborn
      - tqdm
      - scikit-learn
      - einops
      - packaging
      - commpy                # GMSK Viterbi, convolutional codes
      - nbformat              # parse .ipynb
      - jupyter               # inspect Zhu notebook
      - pyldpc                # OPTIONAL: LDPC coded BER
      - requests
```

Activate. `pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.* torchvision torchaudio`.

If 2.11 not resolvable, try `2.10.*`. If that fails, `--pre --index-url .../nightly/cu128`. Document every attempt in `INSTALL_LOG.md`.

## 0.3 PyTorch + Blackwell verification

Write `src/env_check.py`:

```python
import sys, torch, platform
print("Python:", sys.version)
print("Platform:", platform.platform())
assert torch.cuda.is_available(), "CUDA not available"
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"Device: {name}  cap={cap}  torch={torch.__version__}  cuda_rt={torch.version.cuda}")
assert cap == (12, 0), f"Expected Blackwell sm_120 (12,0), got {cap}"

# Smoke: a real matmul in bf16 must run without 'no kernel' error
x = torch.randn(32, 128, 128, device="cuda", dtype=torch.bfloat16)
y = torch.nn.functional.gelu(x) @ torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
assert y.isfinite().all(), "bf16 matmul produced non-finite values"
print("PyTorch + Blackwell bf16 matmul OK")
```

Run it. If fail, STOP, log.

## 0.4 mamba-ssm

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm==2.3.1 --no-build-isolation
```

If `mamba-ssm` wheel resolution fails on sm_120, build from source:

```bash
git clone https://github.com/state-spaces/mamba.git _mamba_src
cd _mamba_src
# setup.py cc_flag list: ensure cc_flag.append("arch=compute_120,code=sm_120") is present.
# Locate by searching for "arch=compute" — DO NOT rely on a line number.
grep -n "arch=compute" setup.py
# Patch if missing, then:
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
pip install -e . --no-build-isolation -v
cd ..
```

## 0.5 Mamba-3 forward + backward gate (HARD)

Append to `src/env_check.py`:

```python
import torch
from mamba_ssm import Mamba3

for label, kwargs in [
    ("SISO", dict(d_model=128, d_state=128, headdim=64,
                  is_mimo=False, chunk_size=64, dtype=torch.bfloat16)),
    ("MIMO", dict(d_model=128, d_state=128, headdim=64,
                  is_mimo=True, mimo_rank=4, chunk_size=16,
                  is_outproj_norm=False, dtype=torch.bfloat16)),
]:
    m = Mamba3(**kwargs).cuda()
    x = torch.randn(2, 800, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape and y.isfinite().all(), f"{label} forward failed"
    loss = y.float().pow(2).mean()
    loss.backward()
    grads_ok = all(
        p.grad is not None and p.grad.isfinite().all()
        for p in m.parameters() if p.requires_grad
    )
    assert grads_ok, f"{label} backward produced bad grads"
    print(f"Mamba-3 {label} forward+backward on Blackwell OK")

# Bidirectional sanity: forward on flipped, flip result, add to forward on original.
# This is what our BiMamba3Block will do.
m = Mamba3(d_model=128, d_state=128, headdim=64, is_mimo=False,
           chunk_size=64, dtype=torch.bfloat16).cuda()
x = torch.randn(2, 800, 128, device="cuda", dtype=torch.bfloat16)
y_fwd = m(x)
y_rev = torch.flip(m(torch.flip(x, dims=[1])), dims=[1])
assert (y_fwd + y_rev).isfinite().all()
print("Bi-Mamba-3 composition OK")
```

Run. If all prints appear, set `CLAUDE.md` Phase 0 to DONE and commit. If not, write `PHASE0_BLOCKED.md` with the full error trace, `nvidia-smi` output, `pip freeze`, and the failing command. STOP.

---

# Phase 1 — Data: Zhu loader + synthetic generator

## 1.1 Locate Zhu's dataset locally first

```python
# src/data/zhu_loader.py — partial sketch
ZHU_LOCAL = "/mnt/c/EM Project/data_set/data_set"
ZHU_FALLBACK_ZENODO = "https://zenodo.org/api/records/5781913"

def locate_zhu_root():
    if os.path.isdir(ZHU_LOCAL):
        # Verify expected folder structure is present
        expected = [
            "train_dataset/data_awgn/mod_data",
            "train_dataset/data_awgn/label_data",
            "train_dataset/data_kb2/mod_data",
            "train_dataset/data_kb2/label_data",
            "test_dataset/test_data/Awgn_Tb0d3",
            "test_dataset/test_data/Awgn_Tb0d5",
            "test_dataset/test_data/kb2_Tb0d3_m1d2",
            "test_dataset/test_data/kb2_Tb0d3_m1d4",
            "test_dataset/test_data/kb2_Tb0d5_m1d2",
            "test_dataset/test_data/kb2_Tb0d5_m1d4",
            "test_dataset/test_label/Awgn_Tb0d3",
            # ...matching test_label set
        ]
        missing = [p for p in expected if not os.path.isdir(f"{ZHU_LOCAL}/{p}")]
        if not missing:
            return ZHU_LOCAL
        else:
            print(f"Local Zhu path exists but missing: {missing[:3]}...")
    # Fallback: download from Zenodo
    return download_zenodo_to("./data/zhu_download")
```

Symlink the chosen root to `./data/zhu/` for consistent internal paths:
```bash
ln -sfn "/mnt/c/EM Project/data_set/data_set" ./data/zhu
```

## 1.2 Dataset class

Implement `ZhuGMSKDataset` that:
- Reads CSVs under `data/zhu/{train,test}_dataset/.../mod_data/*.csv` and `label_data/*.csv`.
- Returns `(iq[2,800], bits[100])` per item as torch float32.
- On first pass, **memoizes each subdir's full contents to a single `.pt` file** under `data/zhu_cache/`. Subsequent loads are fast.
- Supports `split={train,val,test}`, `channel={awgn,kb2}`, `tb={0.3,0.5}`, `m={1.2,1.4}` filters (for test); train/val by 90/10 seeded random split of the combined train_dataset.

**Critical**: the exact CSV schema needs to be inspected. Do this as a first step:

```python
import pandas as pd, glob
sample = glob.glob("./data/zhu/train_dataset/data_kb2/mod_data/*.csv")[0]
df = pd.read_csv(sample, header=None)
print(df.shape, df.dtypes, df.head())
# Inspect a label CSV similarly. Determine:
# - Is I/Q stored as 2 rows × 800 cols, or 800 rows × 2 cols, or 1600 rows × 1 col?
# - Is the label 100 rows × 1 col or 1 row × 100 cols?
# - Are values already normalized?
```

Document findings in `RUN_LOG.md`. Your Dataset class must match the actual format, not a guessed format.

**Also read `/mnt/c/EM Project/AI_Demodulator-main/ai_demodulator.ipynb`** via `nbformat` and extract Zhu's exact:
- DataLoader construction (how they index the CSVs, what order, what normalization).
- Model code (to cross-check against our reimplementation).
- Optimizer / loss exact lines.

Save extracted hyperparameters to `src/data/zhu_notebook_parse.py` output as `ZHU_HPARAMS` dict.

## 1.3 File-to-SNR mapping

Critical question: within each test subdirectory (e.g. `kb2_Tb0d3_m1d2/`), how are SNRs encoded? Possibilities:
- Filename encodes SNR (e.g. `mod_signal_0001_snr-4.csv`).
- Fixed number of files per SNR, ordered numerically.
- All SNRs mixed within files.

Inspect the notebook's data-loading code to resolve this. Document the mapping in a dictionary and use it consistently. **If this cannot be resolved definitively, STOP before training** — incorrect SNR mapping is catastrophic for evaluation.

## 1.4 Sanity plots

Generate `figures/sanity_zhu_traces.png`: 6 subplots, one per test subdir, showing the first I/Q trace each. K-dist traces should visibly fade in amplitude; AWGN traces should not.

Generate `figures/sanity_zhu_snr_dist.png`: if SNR is encoded per file, bar chart of sample counts per SNR per subdir.

## 1.5 Synthetic generator (THE KEY ADDITION)

`src/data/synth_generator.py` implements Zhu's simulation faithfully:

```python
# Zhu eq (2): r(t) = sqrt(p) * h(t) * x(t) + n(t)
#   x(t): GMSK-modulated bit sequence (complex)
#   h(t): K-distributed fading channel gain, complex with random phase
#   n(t): AWGN ~ CN(0, sigma^2)
#
# K-distribution PDF (eq 3): f(h) = (4 * b^((alpha+1)/2) / Gamma(alpha))
#                                  * h^alpha * K_{alpha-1}(2*sqrt(b*h))
#   with alpha = 2/(m-1), b from eq 4
#
# For synthesis: sample amplitude envelopes from K-dist via
#   |h|^2 = Gamma_shape(alpha) * Exponential(1/b), which is the
#   known compound-Gamma representation of K. Validate empirically.

def generate_sample(
    *, bits: np.ndarray,      # (100,) binary
    BT: float,                 # 0.3 or 0.5
    samples_per_symbol: int = 8,
    snr_db: float,
    channel: str,              # "awgn" or "kdist"
    m: float | None = None,    # scintillation index, required if kdist
    b: float = 2.0,
    cfo_hz: float = 0.0,       # optional impairment
    timing_jitter: float = 0.0,
    amplitude_jitter: float = 0.0,
) -> np.ndarray:               # returns (2, 800) I/Q
    ...
```

Implementation steps:
1. GMSK modulator via `commpy.modulation.GMSKModem(bt=BT, samples_per_symbol=8)` — or implement manually (integrate Gaussian-filtered NRZ bits, take cos/sin of phase). Validate shape: 100 bits → 800 complex samples.
2. K-dist channel: sample `|h|^2` as compound Gamma, apply random uniform phase, interpolate across time (frame-constant is fine for a 100-symbol frame given the scintillation timescale; confirm with Zhu's assumptions).
3. AWGN: `n = (randn + 1j*randn) * sigma / sqrt(2)` with `sigma^2 = signal_power / 10^(snr_db/10)`.
4. Split complex output into real/imag rows → (2, 800).

## 1.6 Generator validation (HARD GATE)

Before using the synthetic generator for training, it MUST reproduce Zhu's classical Viterbi BER curve to within **0.3 dB** at 3 reference operating points.

Pick: (m=1.2, Tb=0.3, SNR=4 dB), (m=1.4, Tb=0.5, SNR=6 dB), (AWGN, Tb=0.3, SNR=0 dB).

For each:
- Generate 10,000 frames with our simulator.
- Demodulate each frame with classical Viterbi (use `commpy` GMSK demod or reference implementation).
- Compute BER.
- Compare to Zhu's Fig 3 numbers (eyeball from the paper: at AWGN SNR=0 dB GMSK is around ~8e-2; at K-dist m=1.2 SNR=4 dB around ~7e-2 for b=1; etc.)

**If our simulated classical BER is >0.5 dB off from Zhu's curve at any of the 3 points:** STOP. Our simulator is wrong. Do NOT proceed to training with bad synthetic data.

Log outputs to `results/synth_validation.csv` and `figures/synth_vs_zhu_fig3.png`.

## 1.7 Generate production synthetic dataset

Once validated, generate:
- **500,000 training frames** across:
  - Channel: 30% AWGN, 70% K-dist.
  - m values (K-dist only): uniform over {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8}.
  - BT: uniform over {0.3, 0.4, 0.5}.
  - SNR: uniform over {-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14} dB.
  - Impairments (randomly applied, 50% of samples get one):
    - residual CFO uniform [-200, 200] Hz at 5 MHz symbol rate.
    - timing jitter Gaussian σ=0.02 Tb.
    - amplitude jitter ±3% envelope.
- **50,000 validation frames** across same distribution.

Store as sharded `.pt` files in `synth_data/train/shard_{000..099}.pt` (5000 frames per shard) to keep I/O fast. Each file stores a tensor (5000, 2, 800) and a tensor (5000, 100) and a dict of metadata (SNR, m, Tb, impairment flags) per sample.

Approximate generation time: 30–90 minutes on CPU (8 workers via `multiprocessing`). Log progress every 10 shards.

Set `CLAUDE.md` Phase 1 to DONE. Commit.

---

# Phase 2 — Zhu baseline reproduction (soft gate)

## 2.1 Architecture

Reimplement Zhu's combined CNN+Bi-LSTM exactly per Fig 7 and Table 1. Use the hyperparameters extracted from `ai_demodulator.ipynb` (not just the paper text — the notebook is ground truth).

Key decisions to double-check from the notebook:
- Exact conv padding (is input preserved at 800? or reduced?).
- Exact pooling kernel/stride on each branch (paper says 2×1 pool).
- Exact flatten dimensions entering the FC stack (paper math: CNN branch flatten 6400 or 3200? Bi-LSTM branch 25600? Concat 28800 or 32000? Verify from notebook).
- Loss: MSE per Zhu. Keep MSE for the baseline reproduction.
- Optimizer: Adam. Confirm lr from notebook (paper doesn't give it explicitly).

## 2.2 Train

Train **one** model (seed=0) on Zhu's 63K+7,875 train set (or whatever the notebook's split is — match exactly). 40 epochs, batch size 512, per Zhu Table 2.

Checkpoint every epoch to `checkpoints/zhu_baseline/epoch_{N}.pt`. Training should converge to near-zero loss by epoch 30 per Zhu Fig 8.

## 2.3 Baseline BER sweep

Evaluate on all 6 test subdirs × all SNR values encoded therein → produces `results/baseline_ber.csv`:

```
channel, tb, m, snr_db, n_bits_tested, n_bit_errors, ber, wilson_lo, wilson_hi, seed
```

## 2.4 Soft gate

**Criteria (both must hold):**

(a) **Quantitative**: mean absolute delta between our reproduced BER (in dB of SNR, read from the BER-vs-SNR curves) and Zhu's published numbers (eyeballed from Fig 9, 10, 11) is **≤ 1.0 dB**, averaged over all operating points.

(b) **Qualitative trends match:**
- AWGN BER < K-dist BER at every SNR.
- m=1.2 BER < m=1.4 BER at every SNR (harder scintillation → worse BER).
- At low SNR (< 2 dB), Tb=0.3 and Tb=0.5 BER are comparable (Zhu Fig 10).
- At high SNR (≥ 6 dB), Tb=0.5 BER < Tb=0.3 BER.
- Combined CNN+Bi-LSTM BER < classical Viterbi BER at every operating point (Zhu Fig 11).

**If both pass:** write `BASELINE_GATE_PASSED.md` with tables comparing our numbers vs Zhu's. Commit. Proceed.

**If only qualitative passes but quantitative is 1.0–1.5 dB off:** write `BASELINE_SOFT_FAIL.md` documenting the gap, hypothesize causes (likely suspects: MSE vs BCE interpretation, per-channel normalization, train/val split, LR mismatch). Proceed cautiously to Phase 3, but note in the final report that our reproduction had a ~X dB offset from Zhu's published curves.

**If qualitative trends fail:** STOP. Write `BASELINE_HARD_FAIL.md`. Something is fundamentally wrong with the data pipeline or model. Do not proceed to Phase 3.

Set `CLAUDE.md` Phase 2 status. Commit.

---

# Phase 3 — Feature pipeline + V5 model smoke test

## 3.1 Feature pipeline

`src/features/preprocess.py` takes raw (B, 2, 800) and produces (B, 6, 800):

```python
def preprocess(iq_raw: torch.Tensor) -> torch.Tensor:
    """
    iq_raw: (B, 2, 800) float32, channel 0 = I, channel 1 = Q.
    returns: (B, 6, 800) float32 with channels:
        0: I, normalized per-frame by RMS of |y|
        1: Q, normalized per-frame by RMS of |y|
        2: envelope |y(t)|^2, log-compressed
        3: phase derivative ∂θ/∂t (unwrapped, smoothed)
        4: matched filter output I (convolve with GMSK pulse)
        5: matched filter output Q
    """
```

Unit-test this on one Zhu sample: print the shape, confirm no NaNs, plot each channel for one sample to `figures/preprocess_example.png`. Sanity: envelope should show K-dist fades for kb2 samples.

## 3.2 V5 main model

`src/models/mamba3_bi_receiver.py`:

```python
class BiMamba3Block(nn.Module):
    """Bidirectional Mamba-3 with residual + FFN + FiLM conditioning."""
    def __init__(self, d_model, d_state=128, headdim=64, ffn_expand=4,
                 dtype=torch.bfloat16):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        # fp32 parameters, bf16 autocast recommended by Mamba authors
        self.mamba_fwd = Mamba3(d_model=d_model, d_state=d_state, headdim=headdim,
                                 is_mimo=False, chunk_size=64, dtype=dtype)
        self.mamba_rev = Mamba3(d_model=d_model, d_state=d_state, headdim=headdim,
                                 is_mimo=False, chunk_size=64, dtype=dtype)
        self.proj = nn.Linear(2 * d_model, d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_expand * d_model),
            nn.SiLU(),
            nn.Linear(ffn_expand * d_model, d_model),
        )
    def forward(self, x, gamma, beta):
        # FiLM on the normalized input to the SSM block
        h = self.norm1(x)
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)   # broadcast (B,1,D)
        fwd = self.mamba_fwd(h)
        rev = torch.flip(self.mamba_rev(torch.flip(h, dims=[1])), dims=[1])
        x = x + self.proj(torch.cat([fwd, rev], dim=-1))
        x = x + self.ffn(self.norm2(x))
        return x


class V5Receiver(nn.Module):
    def __init__(self, n_in_ch=6, d_model=128, n_layers=4, n_snr_bins=12,
                 n_bits=100, samples_per_bit=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_in_ch, d_model, kernel_size=11, padding=5),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=11, padding=5),
            nn.GELU(),
        )
        self.snr_embed = nn.Embedding(n_snr_bins, d_model)
        self.film_generators = nn.ModuleList([
            nn.Linear(d_model, 2 * d_model) for _ in range(n_layers)
        ])
        self.blocks = nn.ModuleList([BiMamba3Block(d_model) for _ in range(n_layers)])
        self.pool = nn.AvgPool1d(kernel_size=samples_per_bit,
                                  stride=samples_per_bit)
        self.bit_head = nn.Linear(d_model, 1)
        # Aux heads (only used at train time)
        self.aux_gain_head = nn.Linear(d_model, 1)     # per-symbol |h(t)|
        self.aux_snr_head = nn.Linear(d_model, 1)      # global SNR regression

    def forward(self, x_6ch, snr_bin_id):
        # x_6ch: (B, 6, 800)
        h = self.stem(x_6ch)                           # (B, D, 800)
        h = h.transpose(1, 2)                          # (B, 800, D)
        snr_vec = self.snr_embed(snr_bin_id)           # (B, D)
        for block, film_gen in zip(self.blocks, self.film_generators):
            gb = film_gen(snr_vec)                     # (B, 2D)
            gamma, beta = gb.chunk(2, dim=-1)          # each (B, D)
            h = block(h, gamma, beta)
        # Pool to (B, 100, D)
        h_pooled = self.pool(h.transpose(1, 2)).transpose(1, 2)
        bit_logits = self.bit_head(h_pooled).squeeze(-1)                 # (B, 100)
        aux_gain = self.aux_gain_head(h_pooled).squeeze(-1)              # (B, 100)
        aux_snr = self.aux_snr_head(h_pooled.mean(dim=1)).squeeze(-1)    # (B,)
        return dict(bit_logits=bit_logits, aux_gain=aux_gain, aux_snr=aux_snr)
```

## 3.3 Model smoke test

```python
model = V5Receiver().cuda()
x = torch.randn(8, 6, 800, device="cuda")
snr_bin = torch.randint(0, 12, (8,), device="cuda")
out = model(x, snr_bin)
for k, v in out.items():
    assert v.isfinite().all(), f"{k} has non-finite"
loss = out["bit_logits"].pow(2).mean()
loss.backward()
print("V5 forward+backward OK.  Params:", sum(p.numel() for p in model.parameters()))
```

Expect ~2-5M parameters. Memory footprint at batch=128: should fit in 12 GB; confirm with `nvidia-smi` during a training step.

Set `CLAUDE.md` Phase 3 to DONE. Commit.

---

# Phase 4 — V5 main training (3 seeds)

## 4.1 Training loop (`src/train/common.py`)

Single unified training loop used for baseline, V5, competitors, ablations. Features:

1. **Mixed precision**: `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)` for forward; optimizer state in fp32; gradient scaler not needed for bf16.
2. **Gradient clip** 1.0 (mandatory with SSMs).
3. **EMA**: exponential moving average of weights with decay 0.9995; save both raw and EMA checkpoints.
4. **AdamW**: lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95).
5. **Schedule**: linear warmup 1000 steps → cosine decay to 1e-5 over total steps.
6. **Curriculum**:
   - Epochs 1–10: train only on samples with SNR ≥ 0 dB.
   - Epochs 11–25: expand to SNR ≥ -2 dB.
   - Epochs 26+: full range.
7. **Mixup**: starting epoch 15, with α=0.2, applied only between samples in the same SNR bin.
8. **Label smoothing** 0.05 on bit targets.
9. **Multi-task loss**:
   ```
   L = BCE_smooth(bit_logits, bits) +
       0.1 * MSE(aux_gain, channel_gain_magnitude) +   # from metadata
       0.1 * MSE(aux_snr, snr_db / 10.0)               # normalized to unit scale
   ```
10. **Resumable**: at start of `train_*.py`, look for latest epoch checkpoint and resume. Single command to restart.
11. **Checkpointing**: save every epoch. Keep only last 3 epoch checkpoints + best-val checkpoint.

## 4.2 Train V5 (3 seeds)

```bash
for seed in 0 1 2; do
    python -m src.train.train_main \
        --config configs/v5_main.yaml \
        --seed $seed \
        --train-data ./synth_data/train \
        --val-data ./synth_data/val \
        --out ./checkpoints/v5_main/seed_$seed \
        --epochs 60 \
        --batch-size 128 \
        --grad-accum 4
done
```

Training time estimate: ~45–90 min per seed on a 5070. Three seeds = 3-5 hours.

Per-seed output: `metrics.csv` (per-epoch train loss, val loss, val BER on held-out synth val), `best.pt`, `ema_best.pt`, `last.pt`.

## 4.3 V5 BER sweep on Zhu test set

```bash
python -m src.eval.sweep_ber \
    --checkpoint ./checkpoints/v5_main/seed_0/ema_best.pt \
    --model-name v5_main_seed0 \
    --test-root ./data/zhu/test_dataset \
    --out ./results/v5_ber.csv
```

Run for all 3 seeds. The output CSV has:
```
model, seed, channel, tb, m, snr_db, n_bits, n_errors, ber, wilson_lo, wilson_hi
```

Set `CLAUDE.md` Phase 4 to DONE. Commit.

---

# Phase 5 — Competitor baselines

Train each at 2 seeds (not 3, save compute):

1. **Bi-LSTM alone** — Zhu's Bi-LSTM branch only.
2. **1D CNN alone** — Zhu's CNN branch only.
3. **Bi-Transformer** — 4 layers, 8 heads, d=128, standard pre-LN Transformer with relative position encoding. Bidirectional by design (no causal mask for this discriminative task).
4. **Mamba-2 receiver** — same V5 architecture skeleton (CNN stem + FiLM + 4 bi-directional blocks + heads) but `Mamba2` instead of `Mamba3`. This isolates whether Mamba-3's specific innovations matter here.
5. **MambaNet-style hybrid** — CNN stem + multi-head self-attention layer + bidirectional Mamba-2, per Luan 2026's architecture but adapted to our setup.

All trained on the same 500K synthetic training set, same recipe as V5, same evaluation on Zhu's test set. Output to `results/competitor_ber.csv`.

Set `CLAUDE.md` Phase 5 to DONE. Commit.

---

# Phase 6 — Ablations of V5

Train 2 seeds each, same recipe as V5 main except for the variant change:

- **Ablation A**: no synthetic augmentation (train only on Zhu's 63K). Isolates the data scaling contribution.
- **Ablation B**: no FiLM SNR conditioning. Isolates conditioning contribution.
- **Ablation C**: no multi-task aux losses (bit loss only).
- **Ablation D**: unidirectional Mamba-3 (no reverse pass).
- **Ablation E**: no Viterbi post-processor (raw soft outputs → threshold). Evaluated in Phase 7.
- **Ablation F**: no curriculum (full SNR range from epoch 1).

Write results to `results/ablation_ber.csv` with an extra `variant` column.

Set `CLAUDE.md` Phase 6 to DONE. Commit.

---

# Phase 7 — Evaluation: TTA + ensemble + Viterbi post + optional coded BER

**TARGET MODEL: mambanet_2ch** (Phase 6 winner, 2.274% BER; 0.001pp over mambanet — noise-level margin but best result).
All references to "V5" in this phase now mean mambanet_2ch. Checkpoints: `checkpoints/mambanet_2ch_s{0,1,2}_ft_best.pt`.

## 7.1 Test-time augmentation

For GMSK the classical full-phase-rotation TTA is delicate (GMSK encodes data in instantaneous phase). Safer TTA options:
- **Time-reversal**: feed reversed signal, reverse output, average. Confirms bidirectionality works.
- **Small circular shifts** of ±1, ±2 samples; un-shift output.

Implement both; confirm empirically that each gives a BER improvement on the val set. If either makes BER worse, drop it.

## 7.2 Ensembling

Average sigmoid outputs across the 3 mambanet_2ch seeds before thresholding. Expected gain: 0.2–0.5 dB.
(Baseline ensemble already computed: results/mambanet_2ch_ensemble_test.csv = 2.274%. TTA+ensemble is the new target.)

Generate `results/mambanet_2ch_tta_ensemble_test.csv`.

## 7.3 Viterbi post-processor

`src/models/viterbi_post.py`. **This is the scariest module. Isolate it.**

```python
def viterbi_refine(
    bit_probs: np.ndarray,        # (B, 100) in (0, 1), our model's outputs
    bt_product: float,             # 0.3 or 0.5
) -> np.ndarray:                   # (B, 100) refined hard decisions
    """
    Treat our model's per-bit probabilities as branch metrics in a GMSK trellis
    Viterbi decoder. Trellis has 2^L states where L is the GMSK memory length
    (approximately ceil(1/BT) symbols — typically 3 for BT=0.3, 2 for BT=0.5).

    We use our model's probabilities as the observation likelihood at each symbol,
    NOT as final decisions. The trellis provides a sequence-level prior that
    enforces physical consistency with the GMSK modulation structure.
    """
```

Implementation strategy:
- Simple bit-wise trellis (no channel model, just GMSK memory constraints).
- Branch metric at step i for state s and input bit b: `log P(y_i | b, s)` where `P(y_i | b=1) = bit_probs[i]`, `P(y_i | b=0) = 1 - bit_probs[i]`.
- Forward dynamic programming, traceback.
- If this is too complex: **simpler alternative** — 1D CRF with learned pairwise transition matrix (trained post-hoc on model outputs). This is strictly a 2×2 transition matrix over adjacent bits; ~10 lines of code with `torchcrf` or manual.

**Gate on Viterbi post**: evaluate on val set. If it improves BER, use it. If it doesn't (possible — the model may already capture bit-to-bit constraints implicitly), drop it and note in the report. **Don't force a win that isn't there.**

## 7.4 Optional: LDPC-coded BER

Time-permitting: show coded BER on synthetic LDPC-coded stream. Use `pyldpc` with a (1/2)-rate CCSDS-like code.

Pipeline:
- Generate 10,000 random 50-bit payloads.
- LDPC encode to 100 bits.
- GMSK modulate.
- Pass through K-dist channel at SNR range [0, 8] dB.
- Run our mambanet_2ch+ensemble+Viterbi receiver → soft bit LLRs.
- LDPC decode.
- Measure post-decoding payload BER.

Compare to baseline Bi-LSTM receiver with same LDPC decoder. Expect 2–10× amplification of BER difference.

If this phase takes more than half a day, skip it and note in the report as "future work".

Set `CLAUDE.md` Phase 7 to DONE. Commit.

---

# Phase 8 — Figures + reports

## 8.1 Figures (300 DPI PNG, generated from CSVs)

- **Fig 1** — Channel geometry. Two panels: (a) SEP angle diagram with Earth, Sun, probe, Voyager 1 at ~1 light-day (Nov 2026), Mars DSOC, Juno. (b) Signal chain block diagram: TX → GMSK mod → K-dist channel → AWGN → our V5 RX.
- **Fig 2** — BER vs Eb/N0, 6 panels (one per {channel} × {Tb} combo), traces: Zhu reproduction, V5 main, V5 ensemble, V5+Viterbi, Mamba-2 variant, Bi-Transformer, classical Viterbi. 95% Wilson CIs shaded.
- **Fig 3** — BER vs scintillation index m at SNR=4 dB, Tb=0.3. Bar chart with error bars for all models.
- **Fig 4** — I/Q constellation diagrams pre/post demod for one sample at each (Tb, m) combo, SNR=4 dB.
- **Fig 5** — Ablation waterfall: start from Zhu baseline BER at (m=1.4, Tb=0.3, SNR=4 dB), add each V5 improvement one at a time, show cumulative BER reduction.

## 8.2 Reports

### `reports/group_project.md` (≤ 3500 words)

Voyager-heavy framing. Structure:
1. **Why deep-space comms is hard**: Voyager 1 at one light-day in November 2026, Mars Psyche DSOC, Juno. Friis + plasma scintillation.
2. **Channel physics**: SEP geometry, K-distribution model, Zhu's formalism.
3. **Prior work**: classical Viterbi, Zhu 2023.
4. **Our approach**: compound receiver = Bi-Mamba-3 + FiLM + synth-aug + multi-task + Viterbi post. One paragraph per ingredient.
5. **Experiments**: dataset, baseline reproduction, V5 results, ablations, statistical significance.
6. **Discussion**: what each component contributes, limits, next steps.
7. **All 5 figures inline.**
8. **References**: Zhu 2023, Mamba-3, MambaNet, IQUMamba-1D, Cai 2025, CCSDS 401.0-B-32, Morabito 2003.

### `reports/paper_draft.md` (≤ 4000 words)

Benchmark-heavy framing. Voyager in one paragraph of intro only.

Structure:
1. **Abstract** (150 words).
2. **Intro**: deep-space receivers, Zhu 2023 SOTA, we propose X, achieve Y dB.
3. **Related work**: Mamba family (3 paragraphs), comms receivers (3 paragraphs).
4. **Method**: architecture + training + post-processing.
5. **Experiments**: Zhu benchmark, baseline reproduction at 1 dB offset, V5 main + ablations + competitors.
6. **Discussion + limitations**.
7. **References**.

Explicit novelty claims to defend:
- SNR-conditional FiLM for deep-space receivers on K-dist channels.
- 10× synthetic data scaling across intermediate (m, Tb) points.
- Hybrid neural-Viterbi receiver for GMSK under scintillation.
- First bidirectional Mamba-3 application with head-to-head benchmark win.

Set `CLAUDE.md` Phase 8 to DONE. Commit.

---

# `run_all.sh` — top-level orchestrator

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "== Phase 0: env check =="
python -m src.env_check                             || exit 10

echo "== Phase 1: data =="
python -m src.data.zhu_loader --smoke-test          || exit 11
python -m src.data.zhu_notebook_parse               || exit 12
python -m src.data.synth_generator --validate       || exit 13
python -m src.data.synth_generator --produce \
    --n-train 500000 --n-val 50000                  || exit 14

echo "== Phase 2: baseline =="
python -m src.train.train_baseline --seed 0         || exit 20
python -m src.eval.sweep_ber --model baseline \
    --checkpoint checkpoints/zhu_baseline/best.pt \
    --out results/baseline_ber.csv                  || exit 21
python -m src.eval.baseline_gate                    || exit 22

echo "== Phase 3: features + V5 smoke =="
python -m src.features.preprocess --smoke           || exit 30
python -m src.models.mamba3_bi_receiver --smoke     || exit 31

echo "== Phase 4: V5 main training =="
for seed in 0 1 2; do
    python -m src.train.train_main --seed $seed     || exit 40
done
python -m src.eval.sweep_ber --model v5_main \
    --out results/v5_ber.csv                        || exit 41

echo "== Phase 5: competitors =="
for m in bilstm_only cnn_only bi_transformer mamba2 mambanet; do
    for seed in 0 1; do
        python -m src.train.train_competitor \
            --variant $m --seed $seed               || exit 50
    done
done
python -m src.eval.sweep_ber --model competitors \
    --out results/competitor_ber.csv                || exit 51

echo "== Phase 6: ablations =="
for v in no_synth no_film no_multitask no_bidir no_curriculum; do
    for seed in 0 1; do
        python -m src.train.train_ablation \
            --variant $v --seed $seed               || exit 60
    done
done
python -m src.eval.sweep_ber --model ablations \
    --out results/ablation_ber.csv                  || exit 61

echo "== Phase 7: evaluation =="
python -m src.eval.ensemble                         || exit 70
python -m src.eval.tta                              || exit 71
python -m src.eval.viterbi_post_eval                || exit 72
python -m src.eval.coded_ber 2>/dev/null || echo "coded BER skipped (optional)"

echo "== Phase 8: figures + reports =="
python -m src.figures.fig1_geometry                 || exit 80
python -m src.figures.fig2_ber_ebn0                 || exit 81
python -m src.figures.fig3_ber_vs_m                 || exit 82
python -m src.figures.fig4_constellations           || exit 83
python -m src.figures.fig5_ablations                || exit 84
python -m src.reports.build_reports                 || exit 85

echo "== DONE =="
```

Each script checks for existing valid outputs and early-exits if present, so `run_all.sh` is idempotent and safely re-runnable after any crash.

---

# Failure playbook

**If at any point something is wrong and you don't know what:**
1. `tail -100 RUN_LOG.md` — what were you just doing.
2. Check `nvidia-smi` (maybe the GPU hung).
3. Check `df -h` (maybe disk is full).
4. Check last 3 checkpoint files exist and are non-zero bytes.
5. Try re-running the last command with `--verbose`.
6. If still stuck after 30 minutes of diagnosis, write `BLOCKED_AT_PHASE_<N>.md` with:
   - Exact error trace.
   - `pip freeze` output.
   - Files at the relevant paths (`ls -la`).
   - The last 3 things you tried and why they didn't work.
   - Your hypothesis about root cause.
   Then STOP. Human decision required.

**If your V5 model is training but BER isn't improving:**
- Check: is the aux loss scale drowning out the bit loss? (`0.1×` factor — reduce to `0.01×` if so).
- Check: is SNR conditioning actually being used? Mask `snr_bin_id` to 0 for all samples and see if BER changes. If not, FiLM is broken.
- Check: synthetic data statistics — are your generated SNRs actually uniform as intended, or biased?
- Check: is validation metric being computed on EMA weights? Use `--use-ema`.

**If the baseline reproduction doesn't cross the soft gate:**
- First suspect: SNR-to-file mapping. Re-verify from `ai_demodulator.ipynb`.
- Second suspect: loss function misreading. Zhu uses MSE on sigmoid outputs. Not BCE. Not MSE on logits. `((sigmoid(logits) - labels) ** 2).mean()`.
- Third suspect: normalization. Zhu doesn't explicitly state input normalization; try with and without per-frame RMS norm.
- Fourth suspect: train/val/test data leakage. Confirm the 90/10 split on the notebook's logic.

**If Mamba-3 NaN's during training:**
- Reduce lr to 1e-4.
- Ensure `fp32` parameters, not `bf16` parameters.
- Reduce `d_state` to 64 from 128.
- Add pre-activation RMSNorm.
- As last resort, temporarily swap to `Mamba2` and document. **This is an engineering fallback, not a research one — the final V5 must use Mamba-3.**

---

# Acceptance criteria — project is DONE when:

1. `INSTALL_LOG.md`: Blackwell sm_120 + Mamba-3 forward+backward verified.
2. `figures/synth_vs_zhu_fig3.png` + `results/synth_validation.csv`: synthetic generator matches Zhu's classical Viterbi to within 0.3 dB.
3. `BASELINE_GATE_PASSED.md` or `BASELINE_SOFT_FAIL.md` exists with numerical receipts.
4. `results/{baseline,v5,competitor,ablation}_ber.csv` all exist, no NaN values, one row per (operating point × seed × model/variant).
5. Figures 1–5 exist in `figures/` at 300 DPI, every number traces to a CSV row.
6. Both reports exist, every numeric claim backed, every figure reference resolves.
7. `RUN_LOG.md` is continuous, timestamped, tells a coherent story from `Phase 0 start` to `Phase 8 done`.
8. `CLAUDE.md` has all phases marked DONE.
9. **The headline result**: V5 main + ensemble + Viterbi post beats Zhu's reproduced baseline by ≥ 1.0 dB averaged across operating points, with p<0.05 paired t-test across seeds.

If criterion 9 is not met, the project is still DONE — just with a different conclusion. Report honestly.

---

# Start

Your first action:
1. Create the directory structure from scratch.
2. Create `CLAUDE.md` with the content specified above.
3. Initialize `RUN_LOG.md` with an opening entry: `[timestamp] Phase 0 start`.
4. Write `src/env_check.py` as specified in section 0.3 + 0.5.
5. Create `env.yml`, install the conda env, install PyTorch cu128, install causal-conv1d and mamba-ssm.
6. Run `python -m src.env_check`.
7. Report what you see.

Go.
