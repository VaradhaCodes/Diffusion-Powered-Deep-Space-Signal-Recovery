"""
P1 — Headline bar chart.
Zhu Bi-LSTM baseline (3.12%) vs MambaNet-2ch winner (2.275%).
The "one slide, one number" flex.
Output: figures/p1_headline.png  300 DPI

Caption: MambaNet-2ch achieves 2.275% BER on Zhu's held-out test set,
vs the Zhu Bi-LSTM baseline at 3.122% — an absolute reduction of -0.847pp
(-27.1% relative). Both are ensemble results over 3 random seeds.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ---- Style ----
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "#0D1117",
    "axes.facecolor": "#0D1117",
    "text.color": "#E8ECF0",
    "axes.labelcolor": "#E8ECF0",
    "xtick.color": "#E8ECF0",
    "ytick.color": "#E8ECF0",
    "axes.edgecolor": "#334155",
    "grid.color": "#1F2937",
})
COLORS = {
    "baseline": "#FF6B6B",
    "winner":   "#F59E0B",
}

REPO    = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT     = REPO / "figures"
OUT.mkdir(exist_ok=True)

# ---- Load ----
bdf = pd.read_csv(RESULTS / "baseline_test_results.csv")
baseline_ber = float(bdf[bdf["condition"] == "ALL"]["ber"].values[0]) * 100

wdf = pd.read_csv(RESULTS / "mambanet_2ch_final_test.csv")
winner_ber = float(wdf[wdf["condition"] == "OVERALL"]["ber"].values[0]) * 100

labels = ["Zhu Bi-LSTM\n(Baseline)", "MambaNet-2ch\n★ Winner"]
values = [baseline_ber, winner_ber]
colors = [COLORS["baseline"], COLORS["winner"]]

# ---- Plot ----
fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0D1117")
ax.set_facecolor("#0D1117")

bars = ax.bar(labels, values, color=colors, width=0.45, alpha=0.88,
              edgecolor="white", linewidth=0.5, zorder=3)

# Value labels inside bars
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
            f"{val:.3f}%", ha="center", va="center",
            fontsize=14, color="white", fontweight="bold")

# Delta annotation with arrow
delta = winner_ber - baseline_ber
rel   = delta / baseline_ber * 100
ax.annotate("",
            xy=(1, winner_ber + 0.05),
            xytext=(0, baseline_ber + 0.05),
            arrowprops=dict(arrowstyle="<->", color="#88AACC",
                            lw=1.5, connectionstyle="arc3,rad=0"))
ax.text(0.5, (baseline_ber + winner_ber) / 2 + 0.18,
        f"{delta:+.3f}pp\n({rel:.1f}%)",
        ha="center", va="bottom", fontsize=11,
        color="#34D399", fontweight="bold",
        bbox=dict(facecolor="#0D1117", edgecolor="#34D399",
                  boxstyle="round,pad=0.3", alpha=0.85))

ax.set_ylabel("Overall BER (%)", labelpad=8)
ax.set_title("Headline Result: Baseline vs Final Model", pad=10,
             color="#E8ECF0", fontweight="bold")
ax.set_ylim(0, baseline_ber * 1.45)
ax.yaxis.grid(True, color="#1F2937", linestyle="--", alpha=0.7, zorder=0)
ax.set_axisbelow(True)

# Footnote
ax.text(0.99, 0.02,
        "Ensemble of 3 seeds · Zhu held-out test set · 4200 frames",
        ha="right", va="bottom", transform=ax.transAxes,
        fontsize=8, color="#88AACC")

fig.savefig(OUT / "p1_headline.png")
print(f"Saved {OUT / 'p1_headline.png'}")
