"""
A1 — Per-seed variance: scatter + mean bar for each model.
Shows BER distribution across 3 seeds. Baseline = single point.
Output: figures/a1_seed_variance.png  300 DPI

Caption: Per-seed BER (%) for each model on Zhu's held-out test set.
Horizontal bar = mean across seeds. MambaNet-2ch shows the lowest variance
(σ=0.006%), indicating highly stable training. Baseline is a single run with
no variance information.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# ---- Style ----
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
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

REPO    = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT     = REPO / "figures"
OUT.mkdir(exist_ok=True)

# ---- Load ----
def get_seeds(prefix, seeds=(0, 1, 2)):
    vals = []
    for s in seeds:
        try:
            df = pd.read_csv(RESULTS / f"{prefix}_s{s}_test.csv")
            r = df[df["condition"] == "OVERALL"]
            vals.append(float(r["ber"].values[0]) * 100)
        except Exception:
            pass
    return np.array(vals)

def get_overall_from(fname, cond_col="condition", ber_col="ber", cond_val=None):
    df = pd.read_csv(RESULTS / fname)
    for c in (cond_val, "OVERALL", "ALL"):
        if c is None:
            continue
        r = df[df[cond_col] == c]
        if len(r):
            return float(r[ber_col].values[0]) * 100
    return df[ber_col].mean() * 100

bdf = pd.read_csv(RESULTS / "baseline_test_results.csv")
baseline_ber = float(bdf[bdf["condition"] == "ALL"]["ber"].values[0]) * 100

COLORS = {
    "Zhu Bi-LSTM":     "#888888",
    "V5 (BiMamba3)":   "#4C72B0",
    "BiTransformer":   "#8172B2",
    "BiMamba2":        "#55A868",
    "MambaNet":        "#CCB974",
    "MambaNet-2ch (*)": "#C44E52",
}

model_seeds = {
    "Zhu Bi-LSTM":     np.array([baseline_ber]),
    "V5 (BiMamba3)":   get_seeds("v5"),
    "BiTransformer":   get_seeds("bi_transformer"),
    "BiMamba2":        get_seeds("bi_mamba2"),
    "MambaNet":        get_seeds("mambanet"),
    "MambaNet-2ch (*)": get_seeds("mambanet_2ch"),
}

ensemble_bers = {
    "Zhu Bi-LSTM":     baseline_ber,
    "V5 (BiMamba3)":   get_overall_from("v5_ensemble_test.csv"),
    "BiTransformer":   get_overall_from("bi_transformer_ensemble_test.csv"),
    "BiMamba2":        get_overall_from("bi_mamba2_ensemble_test.csv"),
    "MambaNet":        get_overall_from("mambanet_ensemble_test.csv"),
    "MambaNet-2ch (*)": get_overall_from("mambanet_2ch_ensemble_test.csv"),
}

# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0D1117")
ax.set_facecolor("#0D1117")

model_names = list(model_seeds.keys())
x = np.arange(len(model_names))

for i, mname in enumerate(model_names):
    col   = COLORS[mname]
    seeds = model_seeds[mname]
    ens   = ensemble_bers[mname]
    mean  = seeds.mean()
    std   = seeds.std() if len(seeds) > 1 else 0.0

    # Mean bar (thin)
    ax.plot([x[i] - 0.30, x[i] + 0.30], [mean, mean],
            color=col, lw=3.0, solid_capstyle="round", zorder=5, alpha=0.9)
    # Seed dots
    jitter = np.linspace(-0.08, 0.08, len(seeds)) if len(seeds) > 1 else [0]
    ax.scatter(x[i] + np.array(jitter), seeds, color=col, s=55, zorder=6,
               edgecolors="white", linewidths=0.6)
    # Ensemble diamond
    ax.scatter([x[i]], [ens], color=col, marker="D", s=80, zorder=7,
               edgecolors="white", linewidths=0.8, alpha=0.65)

    # Annotation
    ax.text(x[i], max(seeds) + 0.04,
            f"μ={mean:.3f}\nσ={std:.3f}",
            ha="center", va="bottom", fontsize=7.5, color=col,
            fontfamily="monospace")

ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=18, ha="right")
ax.set_ylabel("Overall BER (%)", labelpad=6)
ax.set_title("Per-Seed BER Variance Across Models\n(bar=mean · dot=seed · ◆=ensemble)",
             color="#E8ECF0", fontweight="bold", pad=10)
ax.yaxis.grid(True, color="#1F2937", linestyle="--", alpha=0.7, zorder=0)
ax.set_axisbelow(True)

# legend
from matplotlib.lines import Line2D
l_handles = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#BBBBBB",
           markersize=8, label="Per-seed BER"),
    Line2D([0],[0], lw=3, color="#BBBBBB", label="Seed mean"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="#BBBBBB",
           markersize=8, alpha=0.7, label="Ensemble BER"),
]
ax.legend(handles=l_handles, fontsize=8, framealpha=0.2,
          facecolor="#0D1117", edgecolor="#334155", labelcolor="#E8ECF0",
          loc="upper right")

plt.tight_layout(pad=1.2)
fig.savefig(OUT / "a1_seed_variance.png", dpi=300, bbox_inches="tight", facecolor="#0D1117")
print(f"Saved {OUT / 'a1_seed_variance.png'}")
