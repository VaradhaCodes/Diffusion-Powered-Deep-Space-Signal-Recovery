# ⚠️ PRIME DIRECTIVE: LOCAL CODEBASE IS GROUND TRUTH ⚠️

This Phase 8 prompt was generated externally. While it provides the strict structural roadmap and formatting rules for your deliverables, its specific assumptions about our architecture, file names, hyperparameters, or data structures might be outdated, hallucinated, or slightly incorrect.

**YOUR MANDATE:** Before generating any figure, writing the Bible, or drawing any Mermaid diagram, you MUST inspect the actual files in `src/`, `results/`, and `checkpoints/`. 

* If **anything** in this prompt contradicts the actual reality of the local codebase, **THE LOCAL CODEBASE WINS.** * Do not blindly follow the prompt's physical assumptions. 
* Adjust your outputs, figures, and diagrams to reflect the *actual* implemented reality. 
* Briefly note any major discrepancies you had to correct in your `PHASE8_INVENTORY.md`.

Read and acknowledge this directive before proceeding to the Mission.

# Mission (read this completely before doing anything)

Phases 0–7 are complete. The winning model is **MambaNet-2ch** (MHA → BiMamba2 hybrid, raw 2-channel I/Q, no FiLM, no feature engineering), 3-seed ensemble BER = **2.275%** on Zhu's held-out test set vs Zhu baseline 3.12% (**-0.845pp absolute, -27% relative**). The original V5 Bi-Mamba-3 story from `PROMPT.md` is **outdated** — the actual findings are:

- Attention + SSM hybrid (MambaNet-style) **beats pure SSM** stacks on this L=800 task.
- Mamba-2 ≈ Mamba-3 here — Mamba-3 gave no measurable gain.
- **Synthetic pretrain (500K samples) is the single dominant contributor (+0.229pp if removed).** FiLM SNR conditioning and 5-channel feature engineering contributed ~0 in the ensemble.
- TTA and Viterbi post-processing were **gated out** (both hurt or tied with raw model output).

Phase 8 produces three deliverables:
1. **Figures** — a curated set of publication-grade and presentation-grade figures, each as a standalone Python script + rendered 300 DPI PNG. Scripts must be editable outside Claude Code.
2. **The Bible** — `reports/BIBLE.md`. A comprehensive manual of the whole project that the user can use to (a) present to their team, (b) make the group-project slides, (c) hand off for a paper write-up later. Not a report. A guide.
3. **Architecture Diagrams** — `reports/ARCHITECTURE.md`. Block diagrams in Mermaid + structured ASCII + descriptive text the user can manually redraw in drawio/excalidraw.

**Skip the paper draft for now.** User will extract a paper from the Bible later.

---

# Rules (non-negotiable)

1. **No fabricated numbers, ever.** Every number in figures, tables, and prose must trace to a CSV in `results/`, a checkpoint in `checkpoints/`, or a script in `src/`. If a number isn't available, the caption says "not measured" or the figure is dropped.
2. **Inventory before producing.** Do not assume CSV column names, directory layout, or checkpoint structure. Read, list, and inspect actual files. You've been working on this project for a week — but a fresh context reset means you reload from disk, not memory.
3. **Figure scripts are self-contained.** Each figure's `.py` file must run standalone: reads its own CSVs, builds its own figure, saves its own PNG. No shared state, no common plotting utility module. User wants to edit them individually outside Claude Code.
4. **Matplotlib only.** No seaborn styling, no plotly. `matplotlib.pyplot` + `numpy` + `pandas`. Consistent style via an explicit top-of-file style block in each script (copy-pasted, not imported).
5. **No Voyager framing.** Group-project report doesn't need it in Phase 8. Keep it technical and concrete.
6. **Per-figure token budget.** Don't write 500-line figure scripts. Target 80–200 lines each. User will polish them, not Claude.
7. **The Bible is comprehensive but written, not code-dumped.** Include representative code snippets (20–40 lines each) for the 3–4 most important modules, not full file contents.
8. **Commit after each sub-phase.** `git add ... && git commit -m "phase 8.X: <label>"`.
9. **Update `CLAUDE.md` at the end.** Set Phase 8 to DONE with a one-line summary.

---

# Sub-phase 8.0 — Inventory (do this first, don't skip)

Before writing a single figure script or a single paragraph:

1. `ls -la results/` — list every CSV. For each file, print: filename, row count, column headers. Use `head -1` and `wc -l`.
2. `ls -la checkpoints/` — list every subdirectory.
3. `ls src/**/*.py` or `find src -name "*.py"` — list all scripts.
4. `cat CLAUDE.md` and `tail -200 RUN_LOG.md` — reload project state.
5. Open 2–3 representative result CSVs and print their full content if small, first 20 rows if large: `results/baseline_test_results.csv`, `results/v5_ensemble_test.csv`, the winning model's test CSV (probably `results/mambanet_2ch_final_test.csv`), and the ablation results.
6. Confirm the following models have complete 3-seed test data on Zhu's test set:
   - `baseline` (Zhu Bi-LSTM reproduction)
   - `v5` (BiMamba3)
   - `bi_transformer`
   - `bi_mamba2`
   - `mambanet` (MHA→BiMamba2)
   - `mambanet_2ch` (WINNER)
   - Ablations: `mambanet_no_film`, `mambanet_2ch` (this is one of the ablations AND the final model), `mambanet_no_pretrain`
7. For each of the above models, record: ensemble BER, per-seed BERs, per-condition BER breakdown. Put it all in one internal working table before you start plotting.
8. Write `PHASE8_INVENTORY.md` with: the complete table from (7), the list of available artifacts, any data that was promised in the original prompt but isn't actually available (e.g., if we didn't compute Wilson CIs per operating point, note it — we may need to compute them quickly before figures).

**Do not proceed to 8.1 until PHASE8_INVENTORY.md exists and accounts for every figure we're about to make.**

---

# Sub-phase 8.1 — Figures

## Figure set (9 figures total — more than original 5 because the original was wrong about what this project is)

Divide into two tiers. Each figure gets its own `src/figures/figN_<slug>.py` and saves to `figures/figN_<slug>.png` at 300 DPI.

### Tier 1 — Presentation / group-project (7 figures, simpler, easier to explain)

**P1 — Headline bar.** One big bar chart. Zhu baseline (3.12%) vs MambaNet-2ch final (2.275%). Two bars. Clean, annotated with the absolute delta (-0.845pp) and relative (-27%). This is the "one slide, one number" flex.

**P2 — Full model comparison.** Bar chart of all 6 trained models' ensemble BER: Zhu baseline, V5 (BiMamba3), BiTransformer, BiMamba2, MambaNet, MambaNet-2ch. Error bars = seed std across 3 seeds. X-axis sorted by performance (worst → best) so the visual trend is monotone. Winner highlighted.

**P3 — Per-condition BER, grouped bars.** 6 operating conditions on x-axis (AWGN Tb0.3, AWGN Tb0.5, KB2 Tb0.3 m1.2, KB2 Tb0.3 m1.4, KB2 Tb0.5 m1.2, KB2 Tb0.5 m1.4). Two grouped bars per condition: Zhu baseline vs MambaNet-2ch. Shows we win everywhere, with the biggest gains on the hard conditions (KB2 m=1.4).

**P4 — BER vs SNR line plot.** Three subplots (AWGN / KB2 m=1.2 / KB2 m=1.4), each showing BER vs SNR curves for baseline + MambaNet-2ch. If the Zhu test CSVs have per-SNR breakdowns (check in inventory), use that; otherwise aggregate at operating-point level only and document the limitation in the caption.

**P5 — Ablation waterfall.** Horizontal waterfall or cascade bar chart. Start with full MambaNet-2ch (2.275%), show what happens when you remove each component: no pretrain → 2.504% (+0.229), no FiLM → 2.284% (+0.009), no feature eng (already 2ch → this IS the winner, so inverse: "adding 5ch features: +0.001pp harm"). Visual story: **pretrain is the hero**, everything else is negligible.

**P6 — Training curves.** Two panels side by side: (a) synthetic pretrain val loss over epochs for the 3 MambaNet-2ch seeds; (b) Zhu finetune val loss over epochs, same seeds. If Phase 4/5/6 training logs are preserved (check `checkpoints/*/metrics.csv` or similar), use them; if not, document that training curves are not available and drop this figure.

**P7 — Architecture block diagram.** Use matplotlib to draw a clean block diagram of the MambaNet-2ch architecture: Input (2×800) → CNN stem → N × [MHA → BiMamba2 → FFN] → AvgPool → Bit head. Boxes, arrows, dimensions labeled. Keep it minimal — the detailed version goes in ARCHITECTURE.md. This is "the model in one slide."

### Tier 2 — Paper-quality / rigorous (2 additional figures)

**A1 — Per-seed variance box plot.** For each model (6 models), a box plot showing the distribution of seed-level test BERs. Demonstrates result robustness. Overlay ensemble BER as a point. Caption notes seed mean ± std.

**A2 — Statistical significance heatmap.** 6×6 matrix of paired t-test p-values for model A vs model B (paired across the 3 seeds on per-sample bit-error counts, or per-operating-point BERs if per-sample is infeasible). Color-coded (green=significant, red=not). Triangle only (upper or lower), with diagonal grayed out. Documents that our wins are not seed noise.

### Figure script template

Every figure script follows this skeleton. Copy this block to the top of each `figN_*.py`:

```python
# src/figures/figN_<slug>.py
# Phase 8 — figure N. Standalone. Edit freely outside Claude Code.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# ---- Style (copy-pasted, not imported; edit per-figure as needed) ----
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 120,          # preview DPI
    "savefig.dpi": 300,         # export DPI
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})
COLORS = {
    "baseline": "#888888",
    "v5":       "#4C72B0",
    "bimamba2": "#55A868",
    "bitf":     "#8172B2",
    "mambanet": "#CCB974",
    "winner":   "#C44E52",   # MambaNet-2ch
}

# ---- Paths (all relative to repo root) ----
REPO  = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = REPO / "figures"
OUT.mkdir(exist_ok=True)

# ---- Load (fill in actual CSV names from inventory) ----
# df = pd.read_csv(RESULTS / "...")

# ---- Plot ----
fig, ax = plt.subplots(figsize=(6, 4))
# ... plotting code ...

# ---- Save ----
out_path = OUT / "figN_<slug>.png"
fig.savefig(out_path)
print(f"Saved {out_path}")
```

Constraints:
- Figure dimensions: Tier 1 presentation figures → `figsize=(6, 4)` or `(8, 4)` for wide. Tier 2 → `figsize=(7, 5)` single-column paper width.
- Caption text: **do not embed long captions inside the figure**. Save captions as comments at the bottom of each `.py` file. The Bible quotes them.
- Each script must print the output path. Each script must exit 0 on success.
- If a figure cannot be made because data is missing, the script should print a clear `SKIP:` message and exit 0, not crash. Update the Bible to note the skip.

## Produce figures in this order

1. First run inventory (8.0), confirm data availability per figure.
2. Generate Tier 1 figures P1–P7 in order. Commit after each.
3. Generate Tier 2 figures A1–A2. Commit.
4. Produce a `figures/README.md` that lists every figure, its source script, its source CSVs, and a one-sentence "what this shows" description.

---

# Sub-phase 8.2 — The Bible (`reports/BIBLE.md`)

**Tone**: technical manual. Written so a team member who wasn't on the project can read it front-to-back in 30 minutes and come out understanding everything. Neutral first-person plural ("we did X"). No hype. No Voyager narrative.

**Length**: ~8000–12000 words. Longer than the original report budget because this is a manual, not a report. User will extract a paper / presentation from it.

**Structure (use exactly these section headings):**

### 1. Executive Summary (1–2 pages)
- One-paragraph problem: GMSK demodulation over K-distribution solar scintillation channels, public benchmark by Zhu et al. 2023.
- One-paragraph result: we built a compound neural receiver achieving 2.275% BER vs Zhu baseline 3.12% (-27% relative) on their held-out test set, with 3-seed ensembling and no post-processing needed.
- One-paragraph takeaway: the dominant contributor was **synthetic pretraining on 500K simulated frames** across broader operating conditions; the winning architecture was an attention+Mamba hybrid (MambaNet-style); Mamba-3 gave no measurable gain over Mamba-2 on this task; feature engineering and SNR conditioning contributed negligibly in the ensemble.
- One-paragraph caveats: baseline reproduction was offset ~X% from Zhu's published curves due to the Zenodo dataset being smaller than the paper's stated training set; TTA and Viterbi post-processing did not help and were dropped; our final result is on the Zenodo test split, not a field-captured deep-space signal.

### 2. Problem Setup
- What GMSK is, why it matters for deep space (CCSDS standard, 2–3 sentences).
- What K-distribution scintillation is (2–3 sentences, citation to Zhu 2023 §2).
- The Zhu 2023 benchmark: dataset provenance, splits, operating points (Tb ∈ {0.3, 0.5}, m ∈ {1.2, 1.4}, SNR -4 to 8 dB), evaluation metric (BER).
- Data reality vs paper: Zenodo archive has 42000 train + 8400 test, not the paper's claimed 63000/7875/4200. We used 88.9/11.1 split of 42k. This is **the** caveat for the baseline gap.

### 3. Approach Overview
- One-paragraph summary of our five-ingredient compound receiver as originally designed (CNN stem, BiMamba3, FiLM, multi-task aux, Viterbi post).
- What survived empirically (CNN stem, bidirectional SSM, attention+SSM hybrid, synthetic pretrain).
- What didn't survive (Mamba-3 specifically — Mamba-2 tied; FiLM — contributed ~0; feature engineering — contributed ~0; TTA — gated out; Viterbi post — gated out).
- Short paragraph: "Our original Phase 4 headline model (V5 Bi-Mamba-3) beat baseline by 0.36pp. Phase 5 competitor exploration found that MambaNet (attention+Mamba hybrid) beat V5 by another 0.48pp, becoming our final model."

### 4. Dataset
- Zhu's Zenodo dataset layout (as we actually found it on disk).
- Our PyTorch loader (`src/data_zhu.py`) — what it does, how splits work.
- Synthetic data generator (`src/synth_gen.py`): GMSK modulator, K-distribution channel, impairment models. Reference Zhu eq (2)–(4).
- Synthetic generator validation: we matched Zhu's classical Viterbi BER within X% at 3 reference operating points (cite `results/phase1_ber_awgn.csv`).
- Synthetic training distribution: 500K frames across m ∈ {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8}, Tb ∈ {0.3, 0.4, 0.5}, SNR ∈ [-8, 14] dB, with CFO/jitter/amplitude impairments on 50% of samples. **Key insight section**: synthetic pretrain extended coverage to intermediate (m, Tb, SNR) values Zhu didn't train on, making the fine-tuned model smoother across the Zhu test points.

### 5. Baseline Reproduction (Phase 2)
- What we reproduced: Zhu's Fig 7 CNN + Bi-LSTM (sum-merge, 32 hidden units, concat=28800). PyTorch, Adam lr=1e-3, 40 epochs, batch=512, MSE loss, dropout 0.08/0.2.
- What happened: val loss plateaued at ~0.108 rather than Zhu's near-zero. Our final baseline test BER = 3.12% vs Zhu's published ~1.5% at the high-SNR end.
- Honest diagnosis: our Zenodo train set is ~60% the size of Zhu's claimed 63K. Plus, the Zhu notebook (Keras/TF) and paper (claims PyTorch) disagree, so some details required interpretation.
- Soft gate assessment: trends matched qualitatively (AWGN < KB2, m=1.2 < m=1.4, Tb=0.5 < Tb=0.3 at high SNR). Absolute BER was off, but the **relative** comparison we care about (our model vs our reproduction) is valid.

### 6. V5 Main Model (Phase 4)
- Architecture: CNN stem (6ch → 128, k=11) + FiLM(SNR-bin) + 4× BiMamba3 block + AvgPool8 + bit head + aux heads.
- Feature pipeline: 5-channel engineered input (I, Q, envelope, phase, Δphase).
- Training: 500K synthetic pretrain (20 epochs) → Zhu finetune (20 epochs). AdamW 3e-4, cosine schedule, EMA 0.9995, grad clip 1.0, bf16 autocast + fp32 SSM params.
- Multi-task loss: BCE on bits + 0.1 × MSE on channel gain + 0.1 × MSE on SNR regression.
- Result: 3-seed ensemble 2.76% BER (-0.36pp vs Zhu).
- Representative code snippet (~30 lines) of the BiMamba3Block.

### 7. Competitor Exploration (Phase 5)
- Motivation: V5 was a single-architecture bet. We trained competitor architectures with otherwise-identical recipe.
- Four architectures tested:
  1. BiTransformer (2-layer pre-norm TransformerEncoder).
  2. BiMamba2 (swap BiMamba3 → BiMamba2, same recipe).
  3. MambaNet-style (MHA → BiMamba2 residual blocks, per Luan et al. 2026 ICASSP).
- Results table: all four vs V5 vs baseline. MambaNet = 2.275% ★.
- Analysis: at L=800, attention is cheap (O(T²) = 10000 ops per head). Attention layer provides global symbol-to-symbol correlation; BiMamba2 refines with linear-time state propagation. Pure-SSM stacks under-perform on this short sequence length. Mamba-3's specific innovations (exp-trapezoidal, complex state) tie with Mamba-2 — suggests Mamba-3's advantages need longer sequences or more retrieval-heavy tasks to show.

### 8. Ablation Study (Phase 6)
- Setup: start from MambaNet, remove one component at a time.
- Three ablations × 3 seeds = 9 runs.
- Results:
  - NoFiLM: 2.284% (+0.009pp) → FiLM ≈ 0 contribution.
  - 2ch (raw I/Q, no feature engineering): 2.274% (-0.001pp) → feature engineering hurts slightly. **This became our final model.**
  - NoPretrain (Zhu-only training): 2.504% (+0.229pp) → pretrain dominates.
- Plain-language conclusion: "The synthetic pretraining stage contributed ~25× more than all other improvements combined."

### 9. Final Evaluation (Phase 7)
- TTA: tested time-reversal and symbol-shift ±1. Both broke the model (23.3% and 4.8% val BER). Likely reason: GMSK's differential-phase encoding is direction-sensitive; frame alignment is critical. Dropped.
- Viterbi post / CRF refinement: both tied or slightly regressed (3.752% / 3.761% vs 3.75% baseline on val). Dropped. Interpretation: BiMamba2 already learns adjacent-bit structure from data.
- LDPC coded BER: not run (future work).
- Final number: **2.275%** on Zhu test set. Absolute improvement -0.845pp. Relative improvement -27%.

### 10. Key Insights (read this for the presentation)
Bulleted list of things we learned:
- Synthetic data scaling + domain coverage >> architectural novelty for physical-layer ML receivers at this scale.
- Attention + SSM hybrids beat pure-SSM stacks on short-to-medium sequence lengths (L=800 here).
- Mamba-3 innovations (March 2026) did not outperform Mamba-2 on this task — suggests the gains are specific to language-modeling use cases that exercise retrieval and longer contexts.
- Feature engineering (explicit envelope, phase, Δphase channels) added ~0 when the architecture has enough capacity and training data.
- Signal-processing priors (Viterbi, TTA) don't help once the model is strong enough — they may even hurt when they disagree with learned priors.
- Our baseline reproduction had a ~1.6% absolute gap vs Zhu's published numbers, likely driven by the Zenodo dataset being smaller than the paper's stated 63K training set.

### 11. Reproducibility
- Environment: PyTorch 2.11.0+cu130, Python 3.12.3, CUDA 12.8+, WSL2 Ubuntu.
- Hardware: RTX 5070 12GB (Blackwell sm_120). mamba-ssm 2.3.1 built from GitHub source (commit 316ed60).
- To re-run the winning model from scratch: step-by-step checklist, script paths, approximate time per phase.
- Checkpoints: where they are, how to load them, how to run `eval` on Zhu test set.

### 12. Code Structure Reference
- Tree of `src/`, with a one-line description per file.
- The 4 most important files get a 20-line code excerpt with annotation.
- Which CSVs contain what.

### 13. Presentation Guide (for the group project)
- Suggested 10-slide outline:
  1. Title + problem
  2. Benchmark (Zhu 2023)
  3. Challenge: K-distribution scintillation + finite data
  4. Approach 1: V5 BiMamba3 compound receiver
  5. Approach 2: competitor exploration → MambaNet
  6. Ablations → synthetic pretrain is the hero
  7. Final result: 2.275% vs 3.12%, -27% relative
  8. Key insight: data scaling > architecture novelty
  9. What didn't work (honest section): TTA, Viterbi, Mamba-3 vs Mamba-2
  10. Q&A / future work
- For each slide: one sentence describing the content + which figure(s) from `figures/` to use.

### 14. Future Work
- LDPC-coded BER pipeline.
- Real-capture validation (DSN logs, Psyche DSOC telemetry).
- Longer-sequence regime where Mamba-3 advantages might manifest.
- Mixed-precision deployment for on-probe inference.

### 15. References
- Zhu et al. 2023 (Radio Science) — DOI 10.1029/2022RS007438.
- Mamba-3: Lahoti et al. 2026, arXiv 2603.15569.
- MambaNet for OFDM CE: Luan et al. 2026, arXiv 2601.17108.
- IQUMamba-1D: Gao et al. 2026, J. King Saud U. CIS.
- CCSDS 401.0-B-32 (GMSK standard).
- Our codebase: `<repo URL placeholder>`.

---

# Sub-phase 8.3 — Architecture Diagrams (`reports/ARCHITECTURE.md`)

Single markdown file. For each diagram below, provide **three** representations in this order:

1. **One-paragraph prose description** — so user can read and understand before drawing.
2. **Mermaid diagram** (```mermaid``` code block) — renders automatically on GitHub/VS Code preview, gives user a rendered starting point.
3. **ASCII block diagram** — fallback for drawio/excalidraw manual recreation. Include exact tensor shapes at every arrow.

**Diagrams needed (in order):**

**D1 — Overall Pipeline.** End-to-end: Zhu dataset + Synthetic generator → feature pipeline → model → evaluation → metrics. Two parallel branches (synthetic pretrain + Zhu finetune). One output (BER on Zhu test set).

**D2 — MambaNet-2ch Architecture (the winner, full detail).** Input (B, 2, 800) → CNN stem (Conv1d 2→128, k=11, p=5, GELU, Conv1d 128→128, GELU) → transpose to (B, 800, 128) → N × MambaNet block → AvgPool over 8 → bit head → output (B, 100). Include tensor shape at every arrow.

**D3 — MambaNet Block Detail.** One residual block = LayerNorm → MHA (8 heads, d=128) → residual → LayerNorm → BiMamba2 (d_state=128, headdim=64) → residual → LayerNorm → FFN (128→512→128) → residual. Show the skip connections explicitly.

**D4 — BiMamba2 Internals.** Input (B, T, D) → split into two parallel paths: (a) forward: Mamba2(x); (b) reverse: flip(Mamba2(flip(x))); → concat → Linear(2D → D). Annotate with fp32-param + bf16-autocast note.

**D5 — Training Pipeline (Two-Stage).** Stage 1: Synthetic pretrain (500K samples, 20 epochs, AdamW, cosine schedule, EMA). Stage 2: Zhu finetune (42K samples, 20 epochs, smaller LR). Arrow from stage 1's final checkpoint → stage 2's initialization.

**D6 — Multi-Task Loss Structure (for the V5 narrative, not the winning MambaNet-2ch).** Bit head → BCE loss. Aux gain head → MSE on |h(t)|. Aux SNR head → MSE on SNR/10. Weighted sum with coefficients. Note: the final MambaNet-2ch dropped aux heads based on ablation — include this evolution in the diagram's caption.

**D7 — Ablation Lineage.** Tree diagram: root = MambaNet. Three branches: NoFiLM, 2ch (= winner), NoPretrain. Each leaf annotated with its BER and delta vs root. Visualizes "what we stripped out and what happened."

**For each diagram, include at the bottom:**
- A "Redraw instructions" section with 3–5 bullets on color/shape conventions the user should use in their final drawio version.
- Source paths to the files the diagram is derived from.

---

# Sub-phase 8.4 — Close-out

1. Update `CLAUDE.md`:
   - Phase 8 → `DONE — figures in ./figures/, bible at reports/BIBLE.md, diagrams at reports/ARCHITECTURE.md`.
2. Append to `RUN_LOG.md`:
   - Final entry with timestamp, figure count, Bible word count, a one-sentence project summary.
3. Final commit: `git add -A && git commit -m "phase 8: figures + bible + architecture diagrams"`.
4. Print a final summary to the user:
   - Paths to all generated files.
   - One-paragraph project summary for copy-paste to classmates/professor.
   - Suggested next user action (review the Bible, then make slides from Section 13).

---

# Failure handling

- If inventory reveals a critical CSV is missing (e.g., no per-seed breakdown for a competitor): note it in `PHASE8_INVENTORY.md`, generate figures with the data that exists, flag the gap clearly in figure captions and in the Bible's caveats section. Do not fabricate.
- If a figure script fails at runtime: capture the error, fix if it's a simple data-loading bug, else skip that figure with a clear `SKIP:` annotation and continue to the next.
- If you realize mid-Bible that a claim can't be supported by the data: weaken the claim, don't remove the section. Honesty over polish.

---

# Start

1. Run inventory (8.0). Write `PHASE8_INVENTORY.md`.
2. Stop. Show the inventory table to the user. Ask: "Proceed to figures?"
3. On confirmation, produce figures P1 → P7 → A1 → A2 in order, committing after each.
4. Produce the Bible.
5. Produce ARCHITECTURE.md.
6. Close out.

Go.
