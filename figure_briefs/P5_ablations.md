# Figure P5 — Ablation Waterfall

## What this chart is

Shows how much each component of the winning model contributes to the final BER. Classic "waterfall" style — start from the worst (baseline or no-pretrain), add each component, end at the winner. Or a horizontal bar showing the cost of removing each component.

## Visual brief

**Option A — Waterfall (cascade)**
- X or Y axis: BER (%) scale
- Start: "No pretrain" (2.504%) — the model with nothing special
- Add pretrain: shows drop to ~2.275% (−0.229 pp jump)
- Add FiLM: shows tiny drop (−0.010 pp)
- Final: MambaNet-2ch winner 2.275%
- Also add a Zhu baseline reference line at 3.122% to show how far we've come from there
- Each band/step = a different color or shade

**Option B — Ablation bar chart (simpler, very clean)**
- Horizontal bars, one per variant
- Show: Winner, A1-NoFiLM, A3-NoPretrain, BiMamba2-only (no attention), Zhu baseline
- Sorted best → worst
- Annotate the gap from the winner to each ablation variant
- Reference line: winner at 2.274% and Zhu at 3.122%

**Suggested labels:**
- "MambaNet-2ch (full)" → 2.274%
- "Remove FiLM" → 2.284% (+0.010 pp)
- "Remove pretrain" → 2.504% (+0.230 pp)
- "Remove attention (BiMamba2 only)" → 2.725% (+0.451 pp)
- "Zhu Bi-LSTM baseline" → 3.122% (+0.848 pp)

## Data — Ensemble BER

| Variant | Ensemble BER% | Delta vs Winner (pp) | What was removed |
|---|---|---|---|
| MambaNet-2ch (winner) | 2.274 | 0.000 | — (reference) |
| A1: No FiLM | 2.284 | +0.010 | SNR-conditional FiLM modulation removed |
| A3: No pretrain | 2.504 | +0.230 | 500K synthetic pretrain removed (finetune on Zhu 42K only) |
| BiMamba2 only (no attention) | 2.725 | +0.451 | MHA attention block removed |
| Zhu Bi-LSTM (baseline) | 3.122 | +0.848 | Our architecture entirely (reference) |

### Per-condition breakdown for ablations (ensemble BER %)

| Condition | Winner (2ch) | A1 NoFiLM | A3 NoPretrain | BiMamba2 only | Baseline |
|---|---|---|---|---|---|
| AWGN BT=0.3 | 1.044 | 1.063 | 1.224 | 1.373 | 1.923 |
| AWGN BT=0.5 | 1.197 | 1.209 | 1.241 | 1.299 | 1.951 |
| KB2 BT=0.3 m=1.2 | 1.864 | 1.899 | 2.161 | 2.426 | 2.667 |
| KB2 BT=0.3 m=1.4 | 3.810 | 3.750 | 4.221 | 4.710 | 4.527 |
| KB2 BT=0.5 m=1.2 | 1.874 | 1.854 | 1.994 | 2.104 | 2.759 |
| KB2 BT=0.5 m=1.4 | 3.860 | 3.931 | 4.181 | 4.441 | 4.903 |
| OVERALL | 2.275 | 2.284 | 2.504 | 2.725 | 3.122 |

Source: `results/mambanet_2ch_ensemble_test.csv`, `results/mambanet_no_film_ensemble_test.csv`, `results/mambanet_no_pretrain_ensemble_test.csv`, `results/bi_mamba2_ensemble_test.csv`, `results/baseline_test_results.csv`
