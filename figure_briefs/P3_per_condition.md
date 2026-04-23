# Figure P3 — Per-Condition BER: Baseline vs Winner

## What this chart is

Side-by-side comparison of Zhu baseline vs MambaNet-2ch across all 6 test conditions, plus a delta panel showing how much we improved per condition. Good for showing WHERE we win.

## Visual brief

**Option A (preferred): Two-panel horizontal layout**
- Left panel: grouped horizontal bars — for each condition, two bars side by side (baseline vs winner)
- Right panel: delta bars (baseline minus winner), all going same direction (how much we improved)
- Sort conditions top-to-bottom by how severe they are (AWGN easy → KB2 m=1.4 hardest)

**Option B: Single grouped bar chart**
- Clustered vertical bars, 2 per condition group (baseline = one color, winner = another)
- 6 groups along X-axis
- Y-axis: BER (%)

**Color:** Baseline = gray or muted red, Winner = teal or electric blue, Delta = green gradient

**Annotations:** Put the actual BER % number inside or above each bar. Add "−0.847 pp overall" as a callout.

## Data

Conditions sorted easy → hard:

| Condition | Friendly Name | Baseline BER% | MambaNet-2ch BER% | Delta (pp) |
|---|---|---|---|---|
| AWGN BT=0.3 | AWGN, BT=0.3 | 1.923 | 1.044 | −0.879 |
| AWGN BT=0.5 | AWGN, BT=0.5 | 1.951 | 1.197 | −0.754 |
| KB2 BT=0.3 m=1.2 | K-dist, BT=0.3, mild fade | 2.667 | 1.864 | −0.803 |
| KB2 BT=0.5 m=1.2 | K-dist, BT=0.5, mild fade | 2.759 | 1.874 | −0.885 |
| KB2 BT=0.3 m=1.4 | K-dist, BT=0.3, heavy fade | 4.527 | 3.810 | −0.717 |
| KB2 BT=0.5 m=1.4 | K-dist, BT=0.5, heavy fade | 4.903 | 3.860 | −1.043 |
| **OVERALL** | **Overall average** | **3.122** | **2.275** | **−0.847** |

Note: BT = bandwidth-time product. m = scintillation index (1.2 = mild fade, 1.4 = heavy/severe fade). KB2 = K-distributed channel.

Source: `results/baseline_test_results.csv`, `results/mambanet_2ch_final_test.csv`
