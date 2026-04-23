"""
Fig 3 — Per-condition BER breakdown: baseline vs MambaNet-2ch (winner).
Horizontal grouped bars showing absolute BER and delta for each of the 6 conditions.
Output: figures/fig3_per_condition.png  300 DPI
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os

ROOT  = os.path.join(os.path.dirname(__file__), "../../results")
OUT   = os.path.join(os.path.dirname(__file__), "../../figures/fig3_per_condition.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

BG       = "#0D1117"
PANEL_FG = "#E8ECF0"
GRID_C   = "#1F2937"

CONDITIONS = ["Awgn_Tb0d3", "Awgn_Tb0d5",
              "kb2_Tb0d3_m1d2", "kb2_Tb0d3_m1d4",
              "kb2_Tb0d5_m1d2", "kb2_Tb0d5_m1d4", "OVERALL"]
COND_LABELS = ["AWGN Tb=0.3", "AWGN Tb=0.5",
               "KB2 Tb=0.3 m=1.2", "KB2 Tb=0.3 m=1.4",
               "KB2 Tb=0.5 m=1.2", "KB2 Tb=0.5 m=1.4", "OVERALL"]

# baseline
bdf = pd.read_csv(os.path.join(ROOT, "baseline_test_results.csv"))
bdf = bdf[bdf["condition"].isin(CONDITIONS)].set_index("condition")
# Add OVERALL
bdf_overall = bdf.loc[CONDITIONS[:-1], "ber"].mean()
baseline = {c: bdf.loc[c, "ber"]*100 if c != "OVERALL" else bdf_overall*100 for c in CONDITIONS}
baseline_arr = np.array([baseline[c] for c in CONDITIONS])

# Winner
wdf = pd.read_csv(os.path.join(ROOT, "mambanet_2ch_final_test.csv"))
wdf = wdf.set_index("condition")
winner_arr = np.array([wdf.loc[c, "ber"]*100 for c in CONDITIONS])

delta_arr = winner_arr - baseline_arr   # negative = improvement

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG,
                         gridspec_kw={"width_ratios": [2.2, 1.0]})
fig.patch.set_facecolor(BG)

# ── left panel: grouped horizontal bars ───────────────────────────────────────
ax = axes[0]
ax.set_facecolor(BG)
y  = np.arange(len(CONDITIONS))
bh = 0.35

bars_b = ax.barh(y + bh/2, baseline_arr, height=bh, color="#FF6B6B",
                 alpha=0.85, label="Zhu Bi-LSTM (baseline)", edgecolor="white", lw=0.4)
bars_w = ax.barh(y - bh/2, winner_arr,   height=bh, color="#F59E0B",
                 alpha=0.90, label="MambaNet-2ch ★ (winner)", edgecolor="white", lw=0.4)

# Value labels
for i, (bv, wv) in enumerate(zip(baseline_arr, winner_arr)):
    ax.text(bv + 0.04, i + bh/2, f"{bv:.2f}%", va="center", ha="left",
            fontsize=7.5, color=PANEL_FG, fontfamily="monospace")
    ax.text(wv + 0.04, i - bh/2, f"{wv:.2f}%", va="center", ha="left",
            fontsize=7.5, color=PANEL_FG, fontfamily="monospace")

# OVERALL row highlight
rect = mpatches.FancyBboxPatch((-0.1, len(CONDITIONS)-1-bh),
                                8.5, 2*bh+0.05,
                                boxstyle="round,pad=0.05",
                                facecolor="#1A2030", edgecolor="#F59E0B",
                                lw=1.2, alpha=0.6, zorder=0)
ax.add_patch(rect)

ax.set_yticks(y)
ax.set_yticklabels(COND_LABELS, color=PANEL_FG, fontsize=9.5,
                   fontfamily="monospace")
ax.set_xlabel("BER (%)", color=PANEL_FG, fontsize=11, fontfamily="monospace")
ax.tick_params(colors=PANEL_FG, which="both")
for spine in ax.spines.values():
    spine.set_color("#334155")
ax.xaxis.grid(True, color=GRID_C, linestyle="--", lw=0.7, alpha=0.7)
ax.set_axisbelow(True)
ax.invert_yaxis()
ax.legend(loc="lower right", fontsize=9, framealpha=0.2,
          facecolor="#0D1117", edgecolor="#334155", labelcolor=PANEL_FG)
ax.set_title("BER per Operating Condition", color=PANEL_FG,
             fontsize=11, fontweight="bold", fontfamily="monospace", pad=8)

# ── right panel: delta bar chart ──────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(BG)

colors_d = ["#34D399" if d < 0 else "#EF4444" for d in delta_arr]
bars_d = ax2.barh(y, delta_arr, color=colors_d, alpha=0.85,
                  edgecolor="white", lw=0.4)
ax2.axvline(0, color="#AAB8C8", lw=0.8, linestyle="-")

for i, d in enumerate(delta_arr):
    xpos = d - 0.02 if d < 0 else d + 0.02
    ha   = "right" if d < 0 else "left"
    ax2.text(xpos, i, f"{d:+.2f}%", va="center", ha=ha,
             fontsize=7.5, color=PANEL_FG, fontfamily="monospace")

ax2.set_yticks(y)
ax2.set_yticklabels([""] * len(CONDITIONS))
ax2.set_xlabel("ΔBER (winner − baseline)", color=PANEL_FG,
               fontsize=10, fontfamily="monospace")
ax2.tick_params(colors=PANEL_FG)
for spine in ax2.spines.values():
    spine.set_color("#334155")
ax2.xaxis.grid(True, color=GRID_C, linestyle="--", lw=0.7, alpha=0.7)
ax2.set_axisbelow(True)
ax2.invert_yaxis()
ax2.set_title("ΔBER (green = improvement)", color=PANEL_FG,
              fontsize=11, fontweight="bold", fontfamily="monospace", pad=8)

# Overall improvement annotation
overall_delta = delta_arr[-1]
ax2.text(0.5, 0.05, f"Overall\n{overall_delta:+.3f}%\n({abs(overall_delta)/baseline_arr[-1]*100:.1f}% ↓)",
         ha="center", va="bottom", transform=ax2.transAxes,
         fontsize=9, color="#34D399", fontweight="bold",
         fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#0D1117",
                   edgecolor="#34D399", alpha=0.85))

fig.suptitle("Baseline vs MambaNet-2ch: Per-Condition BER Breakdown",
             color=PANEL_FG, fontsize=12, fontweight="bold",
             fontfamily="monospace", y=1.01)

plt.tight_layout(pad=1.5)
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT}")
