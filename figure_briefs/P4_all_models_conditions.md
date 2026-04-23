# Figure P4 — All Models × All Conditions (Grouped Bar Heatmap)

## What this chart is

The full model comparison broken down by condition — every model × every condition in one chart. More detailed than P2 (which shows overall BER only). Shows which models struggle on specific conditions.

## Visual brief

**Chart type:** Grouped bar chart OR heatmap table

**Option A — Grouped bars:**
- X-axis: 6 test conditions (sorted easy → hard)
- Per condition: one bar per model (6 models = 6 bars per group)
- Y-axis: BER (%)
- Color = model identity (consistent color scheme across all figures)
- This gets crowded but is readable if you use a dark-mode design

**Option B — Heatmap table (cleaner):**
- Rows = models, Columns = conditions + overall
- Cell color = BER value (red=bad, green=good)
- Print the actual % value in each cell
- Sort rows by overall BER (winner at top or bottom)

**Color per model (suggested):**
- Zhu baseline: gray
- V5 BiMamba3: purple
- BiTransformer: orange
- BiMamba2: blue
- MambaNet 5ch: teal
- MambaNet-2ch (winner): bright green / highlighted

## Data — Ensemble BER (%) per model per condition

| Model | AWGN BT=0.3 | AWGN BT=0.5 | KB2 BT=0.3 m=1.2 | KB2 BT=0.3 m=1.4 | KB2 BT=0.5 m=1.2 | KB2 BT=0.5 m=1.4 | OVERALL |
|---|---|---|---|---|---|---|---|
| Zhu Bi-LSTM | 1.923 | 1.951 | 2.667 | 4.527 | 2.759 | 4.903 | 3.122 |
| V5 BiMamba3 | 1.330 | 1.311 | 2.483 | 4.813 | 2.151 | 4.463 | 2.759 |
| BiTransformer | 1.449 | 1.399 | 2.310 | 4.419 | 2.120 | 4.141 | 2.640 |
| BiMamba2 | 1.373 | 1.299 | 2.426 | 4.710 | 2.104 | 4.441 | 2.725 |
| MambaNet 5ch | 1.053 | 1.194 | 1.906 | 3.751 | 1.893 | 3.853 | 2.275 |
| MambaNet-2ch | 1.044 | 1.197 | 1.864 | 3.810 | 1.874 | 3.860 | 2.275 |

Source: all `results/*_ensemble_test.csv` files (one row per condition)
