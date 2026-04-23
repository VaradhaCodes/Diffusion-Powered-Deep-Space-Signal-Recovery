# Figure A1 — Seed Variance / Reproducibility

## What this chart is

Shows how reproducible each model is across 3 random seeds. Helps readers trust the results aren't lucky.

## Visual brief

**Chart type:** Scatter plot with bar/mean overlay — one column of dots per model

- X-axis: models (same order as P2, worst → best by ensemble BER)
- Y-axis: BER (%)
- For each model: plot 3 dots (seed 0, seed 1, seed 2)
- Draw a horizontal bar or line at the ensemble BER (avg of 3 seeds)
- Optionally: draw a diamond shape at the ensemble BER
- Baseline: single dot (no seeds), differentiate visually (e.g. hollow dot or X mark)

**Color:** Each model has its own color. Dots for each seed are filled circles, ensemble is a bar or bigger diamond.

**Observation to highlight:** Winner (MambaNet-2ch) has extremely tight seed variance — dots nearly stacked. This is a trust signal.

## Data — Per-seed test BER (%)

| Model | Seed 0 | Seed 1 | Seed 2 | Ensemble | Std (%) |
|---|---|---|---|---|---|
| Zhu Bi-LSTM | 3.122 | — | — | 3.122 | — |
| V5 BiMamba3 | 3.192 | 2.936 | 2.801 | 2.759 | 0.201 |
| BiTransformer | 2.751 | 3.003 | 2.831 | 2.640 | 0.131 |
| BiMamba2 | 2.790 | 2.824 | 2.821 | 2.725 | 0.019 |
| MambaNet (5ch) | 2.292 | 2.313 | 2.319 | 2.275 | 0.014 |
| MambaNet-2ch | 2.312 | 2.323 | 2.319 | 2.275 | 0.006 |

Note: Ensemble BER ≤ best individual seed BER — averaging soft outputs always helps a little.

Source: all `results/*_s{0,1,2}_test.csv` files (OVERALL row for each)
