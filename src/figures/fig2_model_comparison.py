"""
Fig 2 — Model comparison: BER per operating condition (all models, grouped bars).
Replaces the BER-vs-SNR figure (P4 resolution: no per-SNR data available).
Output: figures/fig2_model_comparison.png  300 DPI
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os

ROOT    = os.path.join(os.path.dirname(__file__), "../../results")
OUT     = os.path.join(os.path.dirname(__file__), "../../figures/fig2_model_comparison.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

BG      = "#0D1117"
PANEL_FG = "#E8ECF0"
GRID_C  = "#1F2937"

# ── load data ─────────────────────────────────────────────────────────────────
CONDITIONS = ["Awgn_Tb0d3", "Awgn_Tb0d5",
              "kb2_Tb0d3_m1d2", "kb2_Tb0d3_m1d4",
              "kb2_Tb0d5_m1d2", "kb2_Tb0d5_m1d4"]
COND_LABELS = ["AWGN\nTb=0.3", "AWGN\nTb=0.5",
               "KB2 Tb=0.3\nm=1.2", "KB2 Tb=0.3\nm=1.4",
               "KB2 Tb=0.5\nm=1.2", "KB2 Tb=0.5\nm=1.4"]

def load_ensemble(name):
    path = os.path.join(ROOT, f"{name}_ensemble_test.csv")
    df = pd.read_csv(path)
    df = df[df["condition"].isin(CONDITIONS)].set_index("condition")
    return df.loc[CONDITIONS, "ber"].values * 100   # → percent

def load_seeds(prefix, seeds=(0,1,2)):
    rows = []
    for s in seeds:
        path = os.path.join(ROOT, f"{prefix}_s{s}_test.csv")
        df = pd.read_csv(path)
        df = df[df["condition"].isin(CONDITIONS)].set_index("condition")
        rows.append(df.loc[CONDITIONS, "ber"].values * 100)
    return np.array(rows)   # (n_seeds, 6)

# Baseline (single run)
bdf = pd.read_csv(os.path.join(ROOT, "baseline_test_results.csv"))
bdf = bdf[bdf["condition"].isin(CONDITIONS)].set_index("condition")
baseline_bers = bdf.loc[CONDITIONS, "ber"].values * 100

models = {
    "Zhu Bi-LSTM\n(baseline)": {"ens": baseline_bers,  "seeds": baseline_bers[None], "color": "#FF6B6B"},
    "V5\n(BiMamba3+FiLM)":     {"ens": load_ensemble("v5"),            "seeds": load_seeds("v5"),           "color": "#4ECDC4"},
    "Bi-Transformer":          {"ens": load_ensemble("bi_transformer"), "seeds": load_seeds("bi_transformer"),"color": "#A78BFA"},
    "BiMamba2":                {"ens": load_ensemble("bi_mamba2"),      "seeds": load_seeds("bi_mamba2"),     "color": "#FCA5A5"},
    "MambaNet":                {"ens": load_ensemble("mambanet"),       "seeds": load_seeds("mambanet"),      "color": "#34D399"},
    "MambaNet-2ch\n★ WINNER":  {"ens": load_ensemble("mambanet_2ch"),  "seeds": load_seeds("mambanet_2ch"), "color": "#F59E0B"},
}

n_models = len(models)
n_cond   = len(CONDITIONS)
x        = np.arange(n_cond)
bw       = 0.13   # bar width
offsets  = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * bw

fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
ax.set_facecolor(BG)

for i, (mname, mdata) in enumerate(models.items()):
    ens  = mdata["ens"]
    seeds = mdata["seeds"]
    col  = mdata["color"]
    xpos = x + offsets[i]
    
    # Bar = ensemble BER
    bars = ax.bar(xpos, ens, width=bw*0.88, color=col, alpha=0.85,
                  zorder=3, label=mname,
                  edgecolor="white", linewidth=0.4)
    
    # Seed scatter dots (only if >1 seed)
    if seeds.shape[0] > 1:
        for j in range(n_cond):
            ax.scatter(np.full(seeds.shape[0], xpos[j]), seeds[:, j],
                       color=col, s=14, zorder=5, alpha=0.6, marker="o",
                       edgecolors="white", linewidths=0.3)
        # std error bars
        stds = seeds.std(axis=0)
        ax.errorbar(xpos, ens, yerr=stds, fmt="none", ecolor="white",
                    elinewidth=0.8, capsize=2, alpha=0.5, zorder=4)

# Winner tick mark
ens_winner = list(models.values())[-1]["ens"]
xw = x + offsets[-1]
for j in range(n_cond):
    ax.text(xw[j], ens_winner[j] + 0.08, "★", ha="center", va="bottom",
            fontsize=7, color="#F59E0B", zorder=6)

ax.set_xticks(x)
ax.set_xticklabels(COND_LABELS, color=PANEL_FG, fontsize=9, fontfamily="monospace")
ax.set_ylabel("BER (%)", color=PANEL_FG, fontsize=11, fontfamily="monospace")
ax.set_xlabel("Test Condition", color=PANEL_FG, fontsize=11, fontfamily="monospace",
              labelpad=6)
ax.tick_params(colors=PANEL_FG, which="both")
ax.spines[:].set_color(GRID_C)
for spine in ax.spines.values():
    spine.set_color("#334155")
ax.yaxis.set_tick_params(labelcolor=PANEL_FG)
ax.xaxis.set_tick_params(labelcolor=PANEL_FG)
ax.set_facecolor(BG)
ax.yaxis.grid(True, color=GRID_C, linestyle="--", linewidth=0.7, alpha=0.7, zorder=0)
ax.set_axisbelow(True)

# Baseline reference dashed line (OVERALL)
ax.axhline(baseline_bers.mean(), color="#FF6B6B", linestyle=":",
           linewidth=1.2, alpha=0.5, zorder=2,
           label=f"Baseline mean ({baseline_bers.mean():.2f}%)")

legend = ax.legend(loc="upper left", fontsize=8, framealpha=0.2,
                   facecolor="#0D1117", edgecolor="#334155",
                   labelcolor=PANEL_FG, ncol=2, columnspacing=0.8,
                   handlelength=1.2)

ax.set_title("BER by Operating Condition — All Models (Ensemble, dots = per-seed)",
             color=PANEL_FG, fontsize=12, fontweight="bold",
             fontfamily="monospace", pad=10)

# Note on dots
ax.text(0.99, 0.97,
        "Bars = ensemble BER  |  Dots = per-seed BER  |  Error bars = 1σ across seeds",
        ha="right", va="top", transform=ax.transAxes,
        fontsize=7.5, color="#88AACC", fontfamily="monospace")

plt.tight_layout(pad=1.5)
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT}")
