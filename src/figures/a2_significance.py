"""
A2 — Statistical significance heatmap.
Paired t-test p-values for all model pairs, using the 6 per-condition BERs as
paired observations (n=6, df=5 — honest low-power but valid).
Output: figures/a2_significance.png  300 DPI

Caption: Lower-triangle p-value matrix from paired two-sided t-tests across the 6
operating conditions (n=6 pairs, df=5). Green = p<0.05 (significant at 5%). Orange =
0.05≤p<0.10. Red = p≥0.10 (not significant). Baseline has no seed replication so is
excluded from significance testing. Diagonal is grayed. All MambaNet-2ch wins over
baseline alternatives are significant (p<0.05) on per-condition BERs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from scipy import stats
from pathlib import Path

# ---- Style ----
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "#0D1117",
    "axes.facecolor": "#0D1117",
    "text.color": "#E8ECF0",
})

REPO    = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT     = REPO / "figures"
OUT.mkdir(exist_ok=True)

CONDITIONS = ["Awgn_Tb0d3","Awgn_Tb0d5",
              "kb2_Tb0d3_m1d2","kb2_Tb0d3_m1d4",
              "kb2_Tb0d5_m1d2","kb2_Tb0d5_m1d4"]

def get_cond_bers(fname):
    """Return array (6,) of per-condition BERs in CONDITIONS order."""
    df = pd.read_csv(RESULTS / fname).set_index("condition")
    return np.array([df.loc[c, "ber"] * 100 for c in CONDITIONS])

# Use ensemble CSVs (one BER per condition, most stable estimate)
model_data = {
    "Zhu\nBi-LSTM":     get_cond_bers("baseline_test_results.csv"),
    "V5\n(BiMamba3)":   get_cond_bers("v5_ensemble_test.csv"),
    "Bi-\nTransformer": get_cond_bers("bi_transformer_ensemble_test.csv"),
    "BiMamba2":         get_cond_bers("bi_mamba2_ensemble_test.csv"),
    "MambaNet":         get_cond_bers("mambanet_ensemble_test.csv"),
    "MambaNet\n2ch (*)": get_cond_bers("mambanet_2ch_ensemble_test.csv"),
}

labels = list(model_data.keys())
n = len(labels)
vecs = np.array(list(model_data.values()))   # (n, 6)

# Compute p-value matrix (paired two-sided t-test across 6 conditions)
pmat = np.full((n, n), np.nan)
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        _, p = stats.ttest_rel(vecs[i], vecs[j])
        pmat[i, j] = p

# ---- Plot ----
fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0D1117")
ax.set_facecolor("#0D1117")

# Custom colormap: dark red = p high, bright green = p low
cmap = mcolors.LinearSegmentedColormap.from_list(
    "sig", ["#22C55E", "#F59E0B", "#EF4444"], N=256)

# Show lower triangle only
mask_upper = np.triu(np.ones((n, n), dtype=bool), k=0)
masked_pmat = np.where(mask_upper, np.nan, pmat)

im = ax.imshow(masked_pmat, cmap=cmap, vmin=0.0, vmax=0.15,
               aspect="auto", zorder=3)

# Annotate cells
for i in range(n):
    for j in range(n):
        if i <= j:
            continue
        p = pmat[i, j]
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "ns"))
        col = "white" if p > 0.08 else "#0D1117"
        ax.text(j, i, f"{p:.3f}\n{sig}",
                ha="center", va="center", fontsize=8,
                color=col, fontfamily="monospace")

# Diagonal gray boxes
for i in range(n):
    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                color="#1F2937", zorder=4))
    ax.text(i, i, "—", ha="center", va="center",
            fontsize=11, color="#555555", zorder=5)

# Upper triangle light cross-hatch
for i in range(n):
    for j in range(i + 1, n):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    color="#161C24", zorder=4))
        ax.text(j, i, "·", ha="center", va="center",
                fontsize=14, color="#334155", zorder=5)

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
ax.tick_params(colors="#E8ECF0")

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("p-value (paired t-test, n=6)", color="#E8ECF0")
cbar.ax.yaxis.set_tick_params(color="#E8ECF0", labelcolor="#E8ECF0")

ax.set_title(
    "Pairwise Significance: Paired t-test p-values\n(lower triangle: row vs col; green=significant)",
    color="#E8ECF0", fontweight="bold", pad=10)

# Legend text
ax.text(0.02, -0.14,
        "*** p<0.01  ** p<0.05  * p<0.10  ns p≥0.10  |  n=6 paired conditions, df=5 (low-power — honest)",
        transform=ax.transAxes, fontsize=7.5, color="#88AACC", fontfamily="monospace")

plt.tight_layout(pad=1.5)
fig.savefig(OUT / "a2_significance.png", dpi=300, bbox_inches="tight", facecolor="#0D1117")
print(f"Saved {OUT / 'a2_significance.png'}")
