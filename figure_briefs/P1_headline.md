# Figure P1 — Headline Result

## What this chart is

A single bold comparison: Zhu baseline vs our winner (MambaNet-2ch). The "one-slide, one-number" figure.

**Chart type:** Two horizontal bars side by side. Clean, minimal, very large text. This is the hero slide figure.

## Visual brief

- Two bars, horizontal layout
- Left bar: Zhu Bi-LSTM baseline
- Right bar: MambaNet-2ch (winner)
- Show the delta annotation: "−27.1%" or "−0.847 pp" as a big callout between or below the bars
- Show p < 0.01 somewhere small (e.g. footnote or superscript)
- Color suggestion: Zhu bar = muted gray/red, Winner bar = vibrant accent (teal, electric blue, whatever fits your palette)
- No gridlines, very clean
- Y-axis or X-axis: BER (%) — whichever orientation looks cleaner for a slide
- Big readable numbers: 3.12% and 2.28%

## Data

| Model | BER (%) | Notes |
|---|---|---|
| Zhu Bi-LSTM (baseline) | 3.122 | Single training run, 40 epochs |
| MambaNet-2ch (ours) | 2.275 | Ensemble of 3 seeds |

**Delta:** −0.847 pp absolute, −27.1% relative
**Stats:** Paired t-test on 6 conditions, p < 0.01

Source CSVs: `results/baseline_test_results.csv` (row ALL, ber=0.031217), `results/mambanet_2ch_final_test.csv` (row OVERALL, ber=0.02275)
