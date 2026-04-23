# Figure P2 — Model Comparison (Overall BER)

## What this chart is

A bar chart ranking all models from worst to best overall ensemble BER, with individual seed dots overlaid. Shows how our winner compares to all baselines and competitors.

## Visual brief

- **Chart type:** Vertical bar chart, sorted worst → best (left to right)
- One bar per model, height = ensemble BER (%)
- Overlay 3 dots per bar = individual seed BERs (scatter/jitter)
- Add error bar or bracket = std dev across seeds (see std column below)
- Horizontal dashed reference line at Zhu baseline 3.122%
- Label the winner bar clearly (e.g. star icon, different color, bold label)
- Y-axis: BER (%) — range 0 to ~3.5%
- Color suggestion: gradient from warm (bad) to cool (good), or just gray for all except winner which is highlighted
- Annotate winner bar with "2.28%" and baseline with "3.12%" text labels

## Data — Ensemble BER (%, sorted worst→best)

| Model | Ensemble BER% | Seed 0 BER% | Seed 1 BER% | Seed 2 BER% | Std% |
|---|---|---|---|---|---|
| Zhu Bi-LSTM (baseline) | 3.122 | 3.122 | — | — | — |
| V5 (BiMamba3 + FiLM + 5ch) | 2.759 | 3.192 | 2.936 | 2.801 | 0.199 |
| BiTransformer | 2.640 | 2.751 | 3.003 | 2.831 | 0.129 |
| BiMamba2 | 2.725 | 2.790 | 2.824 | 2.821 | 0.019 |
| MambaNet (5ch, full features) | 2.275 | 2.292 | 2.313 | 2.319 | 0.014 |
| **MambaNet-2ch (winner)** | **2.275** | **2.312** | **2.323** | **2.319** | **0.006** |

Note: Baseline has only one seed (single run), plot as a point not a bar, or a flat bar with no std.

Source: `results/*_ensemble_test.csv` (OVERALL row), `results/*_s{0,1,2}_test.csv` (OVERALL row)
