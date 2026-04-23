"""
P2 — Full model comparison bar chart.
All 6 trained models' ensemble BER, sorted worst→best.
Error bars = seed std. Winner highlighted.
Output: figures/p2_model_comparison.png  300 DPI

Caption: Ensemble BER (%) for all trained models on Zhu's held-out test set.
Error bars show standard deviation across 3 random seeds. MambaNet-2ch
achieves the best result (2.275%) with the lowest seed variance (σ=0.006%).
Baseline is a single run (no variance reported).
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
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
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
def overall(fname):
    df = pd.read_csv(RESULTS / fname)
    for cond in ("OVERALL", "ALL"):
        r = df[df["condition"] == cond]
        if len(r):
            return float(r["ber"].values[0]) * 100
    return df["ber"].mean() * 100

def seed_overalls(prefix, seeds=(0, 1, 2)):
    vals = []
    for s in seeds:
        try:
            df = pd.read_csv(RESULTS / f"{prefix}_s{s}_test.csv")
            r = df[df["condition"] == "OVERALL"]
            vals.append(float(r["ber"].values[0]) * 100)
        except Exception:
            pass
    return np.array(vals)

bdf = pd.read_csv(RESULTS / "baseline_test_results.csv")
baseline_ber = float(bdf[bdf["condition"] == "ALL"]["ber"].values[0]) * 100

models_raw = {
    "Zhu Bi-LSTM":      {"ens": baseline_ber,                        "seeds": np.array([baseline_ber]), "color": "#888888", "winner": False},
    "V5 (BiMamba3)":    {"ens": overall("v5_ensemble_test.csv"),      "seeds": seed_overalls("v5"),       "color": "#4C72B0", "winner": False},
    "BiTransformer":    {"ens": overall("bi_transformer_ensemble_test.csv"), "seeds": seed_overalls("bi_transformer"), "color": "#8172B2", "winner": False},
    "BiMamba2":         {"ens": overall("bi_mamba2_ensemble_test.csv"),      "seeds": seed_overalls("bi_mamba2"),      "color": "#55A868", "winner": False},
    "MambaNet":         {"ens": overall("mambanet_ensemble_test.csv"),        "seeds": seed_overalls("mambanet"),       "color": "#CCB974", "winner": False},
    "MambaNet-2ch ★":  {"ens": overall("mambanet_2ch_ensemble_test.csv"),    "seeds": seed_overalls("mambanet_2ch"),   "color": "#C44E52", "winner": True},
}

# Sort worst → best (descending BER)
sorted_models = sorted(models_raw.items(), key=lambda x: -x[1]["ens"])

labels  = [m[0] for m in sorted_models]
ens     = np.array([m[1]["ens"] for m in sorted_models])
stds    = np.array([m[1]["seeds"].std() if len(m[1]["seeds"]) > 1 else 0.0
                    for m in sorted_models])
colors  = [m[1]["color"] for m in sorted_models]
winners = [m[1]["winner"] for m in sorted_models]

# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0D1117")
ax.set_facecolor("#0D1117")

x = np.arange(len(labels))
bars = ax.bar(x, ens, color=colors, alpha=0.82, edgecolor="white",
              linewidth=[2.0 if w else 0.5 for w in winners], zorder=3)
ax.errorbar(x, ens, yerr=stds, fmt="none", ecolor="#AAAAAA",
            elinewidth=1.2, capsize=4, zorder=4)

# Value labels
for i, (v, std) in enumerate(zip(ens, stds)):
    ax.text(x[i], v + std + 0.05,
            f"{v:.3f}%" + (f"\n±{std:.3f}" if std > 0 else ""),
            ha="center", va="bottom", fontsize=8.5, color="#E8ECF0")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Overall BER (%)")
ax.set_title("Model Comparison: Ensemble BER (worst → best)", pad=10,
             color="#E8ECF0", fontweight="bold")
ax.set_ylim(0, max(ens) * 1.30)
ax.yaxis.grid(True, color="#1F2937", linestyle="--", alpha=0.7, zorder=0)
ax.set_axisbelow(True)

ax.text(0.99, 0.97,
        "Bars = 3-seed ensemble BER  |  Error bars = seed σ  |  Baseline = single run",
        ha="right", va="top", transform=ax.transAxes,
        fontsize=8, color="#88AACC")

plt.tight_layout(pad=1.5)
fig.savefig(OUT / "p2_model_comparison.png")
print(f"Saved {OUT / 'p2_model_comparison.png'}")
