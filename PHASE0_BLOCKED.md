# PHASE0_BLOCKED.md

**Date:** 2026-04-21T13:09Z
**Phase:** 0.5 — Mamba-3 forward + backward gate
**Status:** HARD BLOCK — Mamba3 does not exist in mamba-ssm 2.3.1

---

## Failing command

```
source .venv/bin/activate && python3 -m src.env_check
```

## Full error

```
Python: 3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0]
Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
Device: NVIDIA GeForce RTX 5070  cap=(12, 0)  torch=2.11.0+cu128  cuda_rt=12.8
PyTorch + Blackwell bf16 matmul OK

Traceback (most recent call last):
  File "src/env_check.py", line 19, in <module>
    from mamba_ssm import Mamba3
ImportError: cannot import name 'Mamba3' from 'mamba_ssm'
  (/home/itsva/deepspace_v5/.venv/lib/python3.12/site-packages/mamba_ssm/__init__.py).
  Did you mean: 'Mamba'?
```

## What was verified

- PyTorch 2.11.0+cu128 — OK
- CUDA device: RTX 5070, cap=(12, 0) — OK (Blackwell sm_120 confirmed)
- bf16 matmul on CUDA — OK
- causal-conv1d 1.6.1 — built from source, imports OK
- mamba-ssm 2.3.1 — installed, imports as `mamba_ssm`

## What is NOT there

`mamba_ssm` 2.3.1 exports: `Mamba`, `Mamba2`, `MambaLMHeadModel`, `mamba_inner_fn`, `selective_scan_fn`, `distributed`, `models`, `modules`, `ops`, `utils`.

**`Mamba3` is absent.** No file in the mamba_ssm package contains the string "Mamba3" or "class Mamba3".

All PyPI versions checked (1.0.1 through 2.3.1): 2.3.1 is the latest and it does not include Mamba3.

## Root cause hypothesis

PROMPT.md was written in anticipation of a "mamba-ssm 2.3.1 (March 10, 2026 release)" that would include a `Mamba3` API. As of 2026-04-21, mamba-ssm 2.3.1 is indeed the latest release but it only provides `Mamba` and `Mamba2`. Either:

1. Mamba3 was not shipped in this release cycle, or
2. Mamba3 exists under a different import path or package name.

## nvidia-smi output

```
Tue Apr 21 13:09:43 2026
NVIDIA-SMI 590.57   Driver Version: 591.86   CUDA Version: 13.1
GPU: NVIDIA GeForce RTX 5070   3243MiB / 12227MiB
```

## pip freeze (relevant packages)

```
causal_conv1d==1.6.1
mamba_ssm==2.3.1
torch==2.11.0+cu128
torchaudio==2.11.0+cu128
torchvision==0.26.0+cu128
cuda-toolkit==12.8.1
```

## Full pip freeze

```
annotated-doc==0.0.4
anyio==4.13.0
attrs==26.1.0
causal_conv1d==1.6.1
certifi==2026.2.25
charset-normalizer==3.4.7
click==8.3.2
commpy==0.1.2
contourpy==1.3.3
cuda-bindings==12.9.4
cuda-pathfinder==1.2.2
cuda-toolkit==12.8.1
cycler==0.12.1
einops==0.8.2
fastjsonschema==2.21.2
filelock==3.25.2
fonttools==4.62.1
fsspec==2026.2.0
h11==0.16.0
hf-xet==1.4.3
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.11.0
idna==3.11
Jinja2==3.1.6
joblib==1.5.3
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter_core==5.9.1
kiwisolver==1.5.0
mamba_ssm==2.3.1
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.8
mdurl==0.1.2
mpmath==1.3.0
nbformat==5.10.4
networkx==3.6.1
ninja==1.13.0
numpy==1.26.4
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.19.0.56
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.28.9
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.8.90
packaging==26.1
pandas==3.0.2
pillow==12.2.0
platformdirs==4.9.6
Pygments==2.20.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
PyYAML==6.0.3
referencing==0.37.0
regex==2026.4.4
requests==2.33.1
rich==15.0.0
rpds-py==0.30.0
safetensors==0.7.0
scikit-learn==1.8.0
scipy==1.17.1
seaborn==0.13.2
setuptools==70.2.0
shellingham==1.5.4
six==1.17.0
sympy==1.14.0
threadpoolctl==3.6.0
tokenizers==0.22.2
torch==2.11.0+cu128
torchaudio==2.11.0+cu128
torchvision==0.26.0+cu128
tqdm==4.67.3
traitlets==5.14.3
transformers==5.5.4
triton==3.6.0
typer==0.24.1
typing_extensions==4.15.0
urllib3==2.6.3
wheel==0.46.3
```

## What passed

- Phase 0.1–0.3: Python 3.12, RTX 5070 Blackwell sm_120, PyTorch 2.11+cu128, bf16 matmul — ALL PASSED
- Phase 0.4: causal-conv1d 1.6.1 built and installed — PASSED
- Phase 0.5: `from mamba_ssm import Mamba3` — **FAILED**

## Human decision required

Options for the project lead to decide:

1. **Check mamba-ssm GitHub for a branch/commit with Mamba3** — it may exist in a dev branch not yet released to PyPI. Install from that commit.
2. **Check if Mamba3 = Mamba2 with MIMO=True** — the PROMPT.md Mamba3 API signature has `is_mimo=True/False` and `mimo_rank` kwargs that don't exist in `Mamba2`. If Mamba3 was renamed or merged, clarify the API.
3. **Accept Mamba2 as the architectural backbone** — this is a non-trivial research decision and violates Rule #1. Not recommended without explicit authorization.
4. **Implement Mamba3 ourselves** on top of Mamba2 if the architectural difference is documented in a preprint.
