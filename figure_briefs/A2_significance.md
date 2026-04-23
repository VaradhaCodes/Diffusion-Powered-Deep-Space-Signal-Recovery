# Figure A2 — Statistical Significance Matrix

## What this chart is

Pairwise p-values from paired t-tests between every pair of models. Answers: "Is model A significantly better than model B?" Uses per-condition ensemble BER as paired observations (n=6 conditions, df=5).

## Visual brief

**Chart type:** Square heatmap / matrix

- Rows = models, Columns = models (same set, same order)
- Cell = p-value of paired two-sided t-test comparing row vs column (using 6 per-condition BERs)
- Color: p < 0.05 = green (significant), p ≥ 0.05 = red/warm (not significant)
- Print p-value in each cell (e.g. "0.003" or "0.23")
- Diagonal = always blank/grey (comparing model to itself)
- Upper triangle = mirror of lower triangle (or just show lower triangle)

**Note:** n=6 and df=5 is low power — this is honest, some pairs won't clear 0.05 even if the difference is real.

## Data — Paired t-test p-values (two-sided, n=6 per-condition BER pairs)

The 6 per-condition ensemble BER values used as paired observations:

| Model | AWGN BT=0.3 | AWGN BT=0.5 | KB2 BT=0.3 m=1.2 | KB2 BT=0.3 m=1.4 | KB2 BT=0.5 m=1.2 | KB2 BT=0.5 m=1.4 |
|---|---|---|---|---|---|---|
| Zhu Bi-LSTM | 1.923 | 1.951 | 2.667 | 4.527 | 2.759 | 4.903 |
| V5 BiMamba3 | 1.330 | 1.311 | 2.483 | 4.813 | 2.151 | 4.463 |
| BiTransformer | 1.449 | 1.399 | 2.310 | 4.419 | 2.120 | 4.141 |
| BiMamba2 | 1.373 | 1.299 | 2.426 | 4.710 | 2.104 | 4.441 |
| MambaNet 5ch | 1.053 | 1.194 | 1.906 | 3.751 | 1.893 | 3.853 |
| MambaNet-2ch | 1.044 | 1.197 | 1.864 | 3.810 | 1.874 | 3.860 |

Source: `results/baseline_test_results.csv`, `results/v5_ensemble_test.csv`, `results/bi_transformer_ensemble_test.csv`, `results/bi_mamba2_ensemble_test.csv`, `results/mambanet_ensemble_test.csv`, `results/mambanet_2ch_final_test.csv`

Computed p-values (scipy.stats.ttest_rel, two-sided):

| | Baseline | V5 | BiTransformer | BiMamba2 | MambaNet | MambaNet-2ch |
|---|---|---|---|---|---|---|
| **Baseline** | — | 0.057 | **0.004** | **0.030** | **<0.001** | **<0.001** |
| **V5** | 0.057 | — | 0.230 | 0.160 | **0.018** | **0.014** |
| **BiTransformer** | **0.004** | 0.230 | — | 0.293 | **0.003** | **0.002** |
| **BiMamba2** | **0.030** | 0.160 | 0.293 | — | **0.016** | **0.012** |
| **MambaNet** | **<0.001** | **0.018** | **0.003** | **0.016** | — | 0.991 |
| **MambaNet-2ch** | **<0.001** | **0.014** | **0.002** | **0.012** | 0.991 | — |

**Bold** = p < 0.05 (statistically significant).

Key findings:
- Winner vs Baseline: p < 0.001 — very clearly significant
- Baseline vs V5: p = 0.057 — NOT significant at α=0.05 (V5 is better than baseline but the per-condition variance is too high to reach significance with n=6)
- MambaNet vs MambaNet-2ch: p = 0.991 — essentially identical (expected, 2ch is an ablation of 5ch)
- BiTransformer vs BiMamba2: p = 0.293 — not significantly different from each other

**Threshold line for coloring:** p = 0.05

Note: These p-values use low n=6 (one per condition). The winner vs baseline p<0.001 is robust. The V5 vs baseline result (p=0.057) is borderline — with more conditions or a larger test set the separation would likely reach significance.
