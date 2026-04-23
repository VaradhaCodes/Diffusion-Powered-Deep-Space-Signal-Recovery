"""
Fig 5 — Ablation waterfall: cumulative BER improvement from Baseline → Winner.
Shows how each component (pretrain, FiLM/conditioning, ensemble) contributes.
Uses ensemble BERs from the results CSVs.
Output: figures/fig5_ablations.png  300 DPI
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os

ROOT  = os.path.join(os.path.dirname(__file__), "../../results")
OUT   = os.path.join(os.path.dirname(__file__), "../../figures/fig5_ablations.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

BG       = "#0D1117"
PANEL_FG = "#E8ECF0"
GRID_C   = "#1F2937"

def get_overall(fname):
    df = pd.read_csv(os.path.join(ROOT, fname))
    row = df[df["condition"] == "OVERALL"]
    if len(row):
        return float(row["ber"].values[0]) * 100
    return df["ber"].values.mean() * 100

# ── BER values (overall) ──────────────────────────────────────────────────────
baseline_ber = get_overall("baseline_test_results.csv")

# Ablations (no pretrain, no film) — these show what's LOST by removing a component
no_pretrain_ber  = get_overall("mambanet_no_pretrain_ensemble_test.csv")
no_film_ber      = get_overall("mambanet_no_film_ensemble_test.csv")
mambanet_ber     = get_overall("mambanet_ensemble_test.csv")   # full model (feature eng)
mambanet_2ch_ber = get_overall("mambanet_2ch_ensemble_test.csv")  # winner (raw 2ch)

# Waterfall stages (in improvement order)
stages = [
    ("Zhu Bi-LSTM\n(baseline)",           baseline_ber),
    ("+ Synth pretrain\n(500K frames)",   no_film_ber),        # has pretrain, no FiLM
    ("+ FiLM SNR\nconditioning",          mambanet_ber),       # full MambaNet
    ("− Feature eng.\n(raw 2ch I/Q)",     mambanet_2ch_ber),   # winner, simpler input
]
# NOTE: removing feature engineering gave marginal improvement — winner uses raw I/Q

labels = [s[0] for s in stages]
values = np.array([s[1] for s in stages])
n = len(stages)

fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG,
                         gridspec_kw={"width_ratios": [1.8, 1.0]})
fig.patch.set_facecolor(BG)

# ── Left panel: waterfall ─────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(BG)

# Compute waterfall bars (running total)
bottoms = np.zeros(n)
bottoms[0] = 0
for i in range(1, n):
    bottoms[i] = values[i]   # each bar starts at the floor 0, height = value

# Waterfall: first bar full height, subsequent bars show delta
bar_colors = ["#FF6B6B", "#4ECDC4", "#34D399", "#F59E0B"]
edge_widths = [0.4] * n
edge_widths[-1] = 2.0  # winner has thicker edge

x = np.arange(n)
bars = ax.bar(x, values, color=bar_colors, alpha=0.85,
              edgecolor=["white"]*3 + ["#F59E0B"],
              linewidth=edge_widths, zorder=3)

# Connector lines between bars (waterfall style)
for i in range(n - 1):
    ax.plot([x[i] + 0.4, x[i+1] - 0.4], [values[i], values[i]],
            color="#AAB8C8", lw=1.0, linestyle="--", alpha=0.6, zorder=4)

# Delta annotations
for i in range(1, n):
    delta = values[i] - values[i-1]
    color = "#34D399" if delta < 0 else "#EF4444"
    arrow = "▼" if delta < 0 else "▲"
    ax.text(x[i], values[i] + 0.06, f"{arrow} {delta:+.3f}%",
            ha="center", va="bottom", fontsize=9,
            color=color, fontweight="bold", fontfamily="monospace", zorder=5)

# Value labels on bars
for i, v in enumerate(values):
    ax.text(x[i], v / 2, f"{v:.3f}%",
            ha="center", va="center", fontsize=9.5,
            color="white", fontweight="bold", fontfamily="monospace", zorder=6)

ax.set_xticks(x)
ax.set_xticklabels(labels, color=PANEL_FG, fontsize=9, fontfamily="monospace",
                   multialignment="center")
ax.set_ylabel("Overall BER (%)", color=PANEL_FG, fontsize=11, fontfamily="monospace")
ax.tick_params(colors=PANEL_FG)
for spine in ax.spines.values():
    spine.set_color("#334155")
ax.yaxis.grid(True, color=GRID_C, linestyle="--", lw=0.7, alpha=0.7)
ax.set_axisbelow(True)
ax.set_ylim(0, baseline_ber * 1.25)
ax.set_title("Ablation Waterfall: BER Reduction\n(lower = better)",
             color=PANEL_FG, fontsize=11, fontweight="bold",
             fontfamily="monospace", pad=8)

# Total delta annotation
total_delta = mambanet_2ch_ber - baseline_ber
ax.annotate(f"Total: {total_delta:+.3f}%\n({abs(total_delta)/baseline_ber*100:.1f}% reduction)",
            xy=(n-1, mambanet_2ch_ber), xytext=(n-1 - 1.5, mambanet_2ch_ber + 0.8),
            arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=1.2),
            color="#F59E0B", fontsize=9, fontfamily="monospace", fontweight="bold")

# ── Right panel: per-seed box for ablation models ─────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(BG)

def get_seed_overalls(prefix):
    vals = []
    for s in range(3):
        fname = f"{prefix}_s{s}_test.csv"
        path = os.path.join(ROOT, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        row = df[df["condition"] == "OVERALL"]
        if len(row):
            vals.append(float(row["ber"].values[0]) * 100)
    return vals

abl_data = {
    "Baseline\n(single run)": [baseline_ber],
    "NoPretrain\n(ablation)": get_seed_overalls("mambanet_no_pretrain"),
    "NoFiLM\n(ablation)":     get_seed_overalls("mambanet_no_film"),
    "MambaNet":               get_seed_overalls("mambanet"),
    "MambaNet-2ch\n★":        get_seed_overalls("mambanet_2ch"),
}

abl_colors = ["#FF6B6B", "#FC8181", "#FBBF24", "#34D399", "#F59E0B"]
abl_keys   = list(abl_data.keys())
abl_vals   = [abl_data[k] for k in abl_keys]

# Scatter + mean line
for i, (k, v, col) in enumerate(zip(abl_keys, abl_vals, abl_colors)):
    yv = np.array(v)
    ax2.scatter(np.full(len(yv), i), yv, color=col, s=60, zorder=5,
                edgecolors="white", linewidths=0.5, alpha=0.9)
    mean_v = yv.mean()
    ax2.plot([i-0.28, i+0.28], [mean_v, mean_v], color=col, lw=2.5,
             alpha=0.85, solid_capstyle="round", zorder=4)
    ax2.text(i, mean_v - 0.07, f"{mean_v:.3f}%",
             ha="center", va="top", fontsize=7.5,
             color=col, fontfamily="monospace")

ax2.set_xticks(range(len(abl_keys)))
ax2.set_xticklabels(abl_keys, color=PANEL_FG, fontsize=8.5, fontfamily="monospace",
                    multialignment="center")
ax2.set_ylabel("Overall BER (%)", color=PANEL_FG, fontsize=11, fontfamily="monospace")
ax2.tick_params(colors=PANEL_FG)
for spine in ax2.spines.values():
    spine.set_color("#334155")
ax2.yaxis.grid(True, color=GRID_C, linestyle="--", lw=0.7, alpha=0.7)
ax2.set_axisbelow(True)
ax2.set_title("Per-Seed Scatter\n(bar = mean)", color=PANEL_FG,
              fontsize=11, fontweight="bold", fontfamily="monospace", pad=8)

fig.suptitle("Ablation Study: Contribution of Each V5 Component",
             color=PANEL_FG, fontsize=12, fontweight="bold",
             fontfamily="monospace", y=1.01)

plt.tight_layout(pad=1.5)
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT}")
