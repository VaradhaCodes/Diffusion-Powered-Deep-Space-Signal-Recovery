"""
P7 — MambaNet-2ch Architecture Block Diagram (matplotlib).
Shows: Input (2×800) → CNN stem → N×[MHA→BiMamba2→FFN] → AvgPool → Bit head.
Output: figures/p7_architecture.png  300 DPI

Caption: MambaNet-2ch architecture. Input is raw 2-channel I/Q (800 samples = 100 bits × 8 samples/bit).
The CNN stem projects to a d=128 feature space. Four MambaNet blocks each apply multi-head self-attention
followed by bidirectional Mamba-2, enabling both global correlation and causal-aware sequence refinement.
Average pooling over 8 samples collapses to 100 symbol positions; the bit head produces 100 log-odds.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import numpy as np

# ---- Style ----
BG       = "#0D1117"
FG       = "#E8ECF0"
GRID_C   = "#1F2937"

REPO = Path(__file__).resolve().parents[2]
OUT  = REPO / "figures"
OUT.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)

def box(x, y, w, h, label, shape_label="", fc="#1E3050", ec="#4A8FD4", lw=1.5, fs=9):
    bb = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec,
                        lw=lw, zorder=4)
    ax.add_patch(bb)
    ax.text(x, y + (0.12 if shape_label else 0), label,
            ha="center", va="center", fontsize=fs, color=FG,
            fontfamily="monospace", fontweight="bold", zorder=5)
    if shape_label:
        ax.text(x, y - 0.28, shape_label, ha="center", va="center",
                fontsize=7, color="#88AACC", fontfamily="monospace", zorder=5)

def arrow(x0, y, x1, label=""):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color="#AAAAAA", lw=1.2),
                zorder=6)
    if label:
        ax.text((x0 + x1) / 2, y + 0.20, label, ha="center", fontsize=7,
                color="#88AACC", fontfamily="monospace", zorder=7)

CY = 2.5   # center y

# Input
box(0.7, CY, 1.1, 1.2, "Input", "B×2×800",
    fc="#142230", ec="#6FE0A0", fs=8)

# Arrow
arrow(1.25, CY, 1.6, "raw I/Q")

# CNN Stem
box(2.3, CY, 1.3, 1.4, "CNN Stem",
    "Conv1d 2→128\nk=11, GELU ×2",
    fc="#1A2E1A", ec="#6FE0A0", fs=8)
arrow(2.95, CY, 3.35, "B×128×800")

# MambaNet Blocks ×4
block_w = 1.35
block_x_start = 4.05
n_blocks = 4

for i in range(n_blocks):
    bx = block_x_start + i * (block_w + 0.15)
    # MambaNet block
    box(bx, CY, block_w, 1.7,
        f"Block {i+1}",
        "LN→MHA(8h)\n→BiMamba2\n→FFN",
        fc="#12183A", ec="#507BD4", lw=1.8, fs=8)
    if i < n_blocks - 1:
        arrow(bx + block_w/2, CY, bx + block_w/2 + 0.15, "B×800×128")

# After blocks
last_x = block_x_start + (n_blocks - 1) * (block_w + 0.15) + block_w/2
arrow(last_x, CY, last_x + 0.25, "B×800×128")

# AvgPool
pool_x = last_x + 0.25 + 0.55
box(pool_x, CY, 1.0, 1.0, "AvgPool", "÷8  →\nB×100×128",
    fc="#1E2030", ec="#A78BFA", fs=8)
arrow(pool_x + 0.5, CY, pool_x + 0.9, "B×100×128")

# Bit head
head_x = pool_x + 0.9 + 0.55
box(head_x, CY, 1.1, 1.0, "Bit Head",
    "Linear 128→1\n→ (B×100)",
    fc="#142230", ec="#F59E0B", lw=2.0, fs=8)
arrow(head_x + 0.55, CY, head_x + 0.9, "logits")

# Output
out_x = head_x + 0.9 + 0.5
box(out_x, CY, 0.85, 0.85, "b̂",
    "100 bits",
    fc="#1A1A0A", ec="#F59E0B", fs=10)

# "Repeat ×4" brace under the blocks
bx_lo = block_x_start - block_w/2
bx_hi = block_x_start + (n_blocks - 1) * (block_w + 0.15) + block_w/2
brace_y = CY - 1.15
ax.plot([bx_lo, bx_lo, bx_hi, bx_hi], [brace_y + 0.08, brace_y, brace_y, brace_y + 0.08],
        color="#507BD4", lw=1.3, alpha=0.7, zorder=5)
ax.text((bx_lo + bx_hi) / 2, brace_y - 0.2, "× 4 MambaNet blocks",
        ha="center", fontsize=8.5, color="#507BD4", fontfamily="monospace")

# Detail box for one MambaNet block
detail_x, detail_y = 5.55, 4.45
ax.text(detail_x, detail_y, "MambaNet Block Detail:",
        ha="center", fontsize=8, color="#88AACC", fontfamily="monospace")
detail_lines = ["LayerNorm → MHA (8 heads, d=128) → residual",
                "LayerNorm → BiMamba2 (d_state=128, headdim=64) → residual",
                "LayerNorm → FFN (128→512→128, SiLU) → residual"]
for i, line in enumerate(detail_lines):
    ax.text(detail_x, detail_y - 0.30 - i * 0.25, line,
            ha="center", fontsize=7, color="#C8D8E8", fontfamily="monospace")

ax.set_title("MambaNet-2ch Architecture  (3-seed ensemble, BER=2.275%)",
             color=FG, fontsize=11, fontweight="bold", fontfamily="monospace",
             pad=4, y=0.98)

fig.savefig(OUT / "p7_architecture.png", dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved {OUT / 'p7_architecture.png'}")
