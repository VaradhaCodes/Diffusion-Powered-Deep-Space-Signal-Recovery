"""
Fig 4 — Training curves: val_ber vs epoch for MambaNet-2ch (3 seeds).
Two phases: pretrain (synth) and finetune (Zhu). Also shows baseline 40-epoch curve.
Output: figures/fig4_training_curves.png  300 DPI
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

ROOT  = os.path.join(os.path.dirname(__file__), "../../results")
OUT   = os.path.join(os.path.dirname(__file__), "../../figures/fig4_training_curves.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

BG       = "#0D1117"
PANEL_FG = "#E8ECF0"
GRID_C   = "#1F2937"

SEED_COLORS = ["#F59E0B", "#34D399", "#60A5FA"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.patch.set_facecolor(BG)

# ── Left panel: MambaNet-2ch training curves ──────────────────────────────────
ax = axes[0]
ax.set_facecolor(BG)

for s in range(3):
    if s == 0:
        # Assemble from two files
        df_partial = pd.read_csv(os.path.join(ROOT, "mambanet_2ch_s0_log_partial_ep1to14.csv"))
        df_main    = pd.read_csv(os.path.join(ROOT, "mambanet_2ch_s0_log.csv"))
        # partial: pretrain ep1-14; main: pretrain ep15-20 + finetune ep1-30
        df = pd.concat([df_partial, df_main], ignore_index=True).reset_index(drop=True)
    else:
        df = pd.read_csv(os.path.join(ROOT, f"mambanet_2ch_s{s}_log.csv"))

    # Create global epoch index
    pre   = df[df["phase"] == "pretrain"].copy()
    fine  = df[df["phase"] == "finetune"].copy()

    pre_epochs  = np.arange(1, len(pre) + 1)
    fine_epochs = np.arange(len(pre) + 1, len(pre) + len(fine) + 1)

    col = SEED_COLORS[s]
    ax.plot(pre_epochs, pre["val_ber"].values * 100, color=col,
            lw=1.6, alpha=0.85,
            label=f"Seed {s}" if s == 0 else f"Seed {s}")
    ax.plot(fine_epochs, fine["val_ber"].values * 100, color=col,
            lw=1.6, alpha=0.85, linestyle="-")

# Phase separator
n_pretrain = 20  # all seeds
ax.axvline(n_pretrain + 0.5, color="#AAB8C8", lw=1.0, linestyle="--", alpha=0.6)
ax.text(n_pretrain + 0.8, 45,
        "finetune →", color="#AAB8C8", fontsize=8,
        fontfamily="monospace", va="top")
ax.text(10, 45,
        "← pretrain", color="#AAB8C8", fontsize=8,
        fontfamily="monospace", va="top", ha="center")

ax.set_xlabel("Epoch", color=PANEL_FG, fontsize=11, fontfamily="monospace")
ax.set_ylabel("Val BER (%)", color=PANEL_FG, fontsize=11, fontfamily="monospace")
ax.tick_params(colors=PANEL_FG)
for spine in ax.spines.values():
    spine.set_color("#334155")
ax.yaxis.grid(True, color=GRID_C, linestyle="--", lw=0.7, alpha=0.7)
ax.set_axisbelow(True)
ax.legend(fontsize=9, framealpha=0.2, facecolor="#0D1117",
          edgecolor="#334155", labelcolor=PANEL_FG)
ax.set_title("MambaNet-2ch Training: Val BER vs Epoch (3 seeds)",
             color=PANEL_FG, fontsize=11, fontweight="bold",
             fontfamily="monospace", pad=8)

# ── Right panel: Baseline training curve ─────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(BG)

bdf = pd.read_csv(os.path.join(ROOT, "baseline_train_log.csv"))
epochs_b = bdf["epoch"].values
ax2.plot(epochs_b, bdf["trn_ber"].values * 100, color="#FF6B6B", lw=1.8,
         alpha=0.9, label="Train BER")
ax2.plot(epochs_b, bdf["val_ber"].values * 100,  color="#FF6B6B", lw=1.8,
         alpha=0.9, label="Val BER", linestyle="--")

ax2.set_xlabel("Epoch", color=PANEL_FG, fontsize=11, fontfamily="monospace")
ax2.set_ylabel("BER (%)", color=PANEL_FG, fontsize=11, fontfamily="monospace")
ax2.tick_params(colors=PANEL_FG)
for spine in ax2.spines.values():
    spine.set_color("#334155")
ax2.yaxis.grid(True, color=GRID_C, linestyle="--", lw=0.7, alpha=0.7)
ax2.set_axisbelow(True)
ax2.legend(fontsize=9, framealpha=0.2, facecolor="#0D1117",
           edgecolor="#334155", labelcolor=PANEL_FG)
ax2.set_title("Zhu Bi-LSTM Baseline: Train & Val BER (40 epochs)",
              color=PANEL_FG, fontsize=11, fontweight="bold",
              fontfamily="monospace", pad=8)

fig.suptitle("Training Dynamics: MambaNet-2ch (left) vs Zhu Baseline (right)",
             color=PANEL_FG, fontsize=12, fontweight="bold",
             fontfamily="monospace", y=1.01)

plt.tight_layout(pad=1.5)
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT}")
