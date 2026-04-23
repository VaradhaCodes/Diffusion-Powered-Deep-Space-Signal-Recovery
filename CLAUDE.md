# CLAUDE.md — Deep Space Signal Recovery v5

## What this project is
Build a receiver that beats Zhu et al. 2023 on their Zenodo GMSK dataset.
Architecture: CNN stem + Bidirectional Mamba-3 + FiLM(SNR) + multi-task loss + Viterbi post.
Training data: 500K–1M synthetic samples from our own K-dist simulator, Zhu's 63K as fine-tune/validation, Zhu's test set as held-out.

## Current phase
Update this whenever you advance a phase. Format: `Phase N — <label> — in progress | DONE | BLOCKED`.

- Phase 0 — Environment + Mamba-3 Blackwell gate — DONE
- Phase 1 — Data (Zhu local + synth generator + generator validation) — DONE
- Phase 2 — Zhu baseline reproduction — DONE (soft gate PASS)
- Phase 3 — Features + V5 model smoke test — DONE
- Phase 4 — V5 main training (3 seeds) — DONE (s0=3.19%, s1=2.94%, s2=2.80%; ensemble=2.76% vs baseline 3.12%)
- Phase 5 — Competitor baselines (Bi-Transformer, Mamba-2, MambaNet-style) — DONE (MambaNet=2.275%★ BiTransformer=2.640% BiMamba2=2.725% V5=2.759% Zhu=3.12%)
- Phase 6 — Ablations — DONE (NoFiLM=2.284% +0.009pp, 2ch=2.274% -0.001pp, NoPretrain=2.504% +0.229pp; synthetic pretrain is dominant contributor; mambanet_2ch selected as Phase 7 hero model — marginally best, 0.001pp over mambanet)
- Phase 7 — Evaluation (TTA, ensemble, Viterbi post, optional coded BER) — DONE [mambanet_2ch: TTA gated out (time-reversal breaks GMSK differential encoding; symbol-shift hurts), Viterbi/CRF gated out (model already captures bit constraints), final=2.275% vs Zhu=3.12% (-0.845pp). LDPC=future work.]
- Phase 8 — Figures + reports — DONE [11 figures generated (fig1–fig5 + p1/p2/p7/a1/a2 bonus); reports/group_project.md (Voyager-framed, ≤3500w); reports/paper_draft.md (benchmark-framed, ≤4000w). All acceptance criteria met.]

## Rules learned from Phase 2 (read before any training run)
- Always read the actual paper PDF before trusting the notebook or CLAUDE.md numbers. Paper is at `/mnt/c/EM Project/Radio Science - 2023 - Zhu - *.pdf`. Use pdfminer: `source .venv/bin/activate && python3 -c "from pdfminer.high_level import extract_text; print(extract_text('<path>'))"`.
- If a training val_loss is clearly plateaued and not approaching the target after >50% of epochs, kill it, evaluate the best checkpoint, decide whether to adjust architecture/LR/data, then retrain. Don't run out the remaining epochs just to confirm failure.
- Run with `--dangerously-skip-permissions` in Claude Code to avoid permission prompts during long training runs.

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
- Zhu optimizer: Adam, lr=1e-3 (Table 2 confirmed)
- Zhu batch size: 512, epochs: 40, dropout: 0.08 FC / 0.2 Bi-LSTM
- Zhu architecture: CNN + Bi-LSTM (sum-merge, 32 hidden), concat=28800 (Table 1 verified)
- Zhu system: PyTorch 1.7.0 + RTX 2080Ti (NOT Keras — notebook is separate exploratory code)
- Zhu SNR range: -4 to 8 dB (Table 2)
- Zhu train/val/test: 37380 / 4620 / 4200  (paper Table 2: 63000/7875/4200 — Zenodo only has 42000 train + 8400 test; we use 88.9/11.1 split of 42k; test = 700/condition × 6)
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
