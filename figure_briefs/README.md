# Figure Briefs — Index

All the data and visual descriptions you need to make each figure yourself.
Every file = one figure. Each contains: what the chart is, visual brief (type, axes, colors, style), and the raw data as tables.

---

## Data-driven charts

| File | Figure | Chart type | Key content |
|---|---|---|---|
| `P1_headline.md` | Headline result | 2 bars | Zhu 3.12% vs winner 2.28% — the one-number slide |
| `P2_model_comparison.md` | All models, overall BER | Sorted bars + seed dots | 6 models ranked, seeds shown as scatter points |
| `P3_per_condition.md` | Baseline vs winner per condition | Grouped bars + delta panel | 6 conditions, how much we win in each |
| `P4_all_models_conditions.md` | All models × all conditions | Grouped bars OR heatmap | Full 6×6 matrix of every model on every condition |
| `P5_ablations.md` | Ablation study | Waterfall OR horizontal bars | What happens when you remove each component |
| `P6_training_curves.md` | Training dynamics | Line chart | MambaNet-2ch 3-seed curves + Zhu baseline 40-epoch curve |
| `A1_seed_variance.md` | Reproducibility | Dot scatter + mean bar | Per-seed BERs for all 6 models |
| `A2_significance.md` | Statistical significance | Heatmap matrix | Pre-computed p-values for every model pair |

## Conceptual diagrams (no data — pure illustration)

| File | Figure | What it shows |
|---|---|---|
| `FIG1_channel_geometry.md` | Space geometry | SEP angle diagram + deep-space signal chain block diagram |
| `P7_architecture.md` | Model architecture | MambaNet-2ch block diagram with tensor shapes at every stage |

---

## Quick numbers cheat sheet

| Thing | Value |
|---|---|
| Zhu baseline overall BER | 3.122% |
| Winner (MambaNet-2ch ensemble) BER | 2.275% |
| Absolute improvement | −0.847 pp |
| Relative improvement | −27.1% |
| p-value (vs baseline, paired t, n=6) | 0.008 |
| Winner seed std | 0.006% |
| Dominant component (ablation) | Synthetic pretrain (+0.230 pp cost if removed) |
| Architecture contribution (MHA) | +0.451 pp cost if removed |
| FiLM contribution | +0.010 pp cost if removed |
| Hardest condition | KB2 BT=0.5 m=1.4 (severe scintillation) |
| Easiest condition | AWGN BT=0.3 |

---

## Notes

- All BER values are in **percent (%)** in this folder (multiply by 100 vs the raw CSVs which use fractions like 0.02275)
- All p-values in A2 are two-sided paired t-test with n=6 (6 per-condition BER pairs), df=5
- The training curve pretrain val BER (~47%) is measured on synthetic val data — NOT comparable to finetune val BER measured on real Zhu data
- "Ensemble" = average of sigmoid(logits) across 3 seeds, then threshold at 0.5
