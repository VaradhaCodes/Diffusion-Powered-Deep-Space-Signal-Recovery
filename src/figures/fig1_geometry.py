"""
Fig 1 — Channel geometry + signal-chain block diagram.
Panel (a): SEP angle geometry (Earth–Sun–Probe).
Panel (b): Signal chain block diagram TX→RX.
Output: figures/fig1_geometry.png  300 DPI
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "../../figures/fig1_geometry.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
SUN_C   = "#FFD700"
EARTH_C = "#4A90E2"
PROBE_C = "#E87040"
MARS_C  = "#C1440E"
JUNO_C  = "#9B59B6"
ARROW_C = "#DDDDDD"
LINK_C  = "#6FE0A0"
BG      = "#0D1117"
PANEL_FG = "#E8ECF0"

fig = plt.figure(figsize=(14, 6), facecolor=BG)
fig.patch.set_facecolor(BG)

# ─────────────────────────────────────────────────────────────────────────────
# Panel (a) — SEP geometry
# ─────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_axes([0.03, 0.06, 0.46, 0.88])
ax1.set_facecolor(BG)
ax1.set_aspect("equal")
ax1.axis("off")

# Positions (arbitrary AU-like units)
sun_pos   = np.array([0.0, 0.0])
earth_pos = np.array([-2.0, 0.0])
voyager_pos = np.array([4.8,  2.0])   # ~1 light-day at 162 AU
mars_pos  = np.array([ 2.8, -1.4])    # Psyche DSOC
juno_pos  = np.array([ 1.5,  3.1])    # Juno at Jupiter-ish

# Faint star field
rng = np.random.default_rng(42)
sx = rng.uniform(-5.5, 5.5, 220)
sy = rng.uniform(-3.2, 3.8, 220)
sz = rng.uniform(0.3, 1.0, 220)
ax1.scatter(sx, sy, s=sz, c="white", alpha=0.4, zorder=0)

# Sun glow
for r, a in [(0.55, 0.08), (0.38, 0.15), (0.22, 0.30)]:
    glow = plt.Circle(sun_pos, r, color=SUN_C, alpha=a, zorder=1)
    ax1.add_patch(glow)
sun_circ = plt.Circle(sun_pos, 0.18, color=SUN_C, zorder=2)
ax1.add_patch(sun_circ)

# Earth
earth_circ = plt.Circle(earth_pos, 0.13, color=EARTH_C, zorder=3)
ax1.add_patch(earth_circ)

# Voyager 1
ax1.plot(*voyager_pos, marker="D", ms=7, color=PROBE_C, zorder=4)

# Mars/Psyche
ax1.plot(*mars_pos, marker="o", ms=9, color=MARS_C, zorder=4)

# Juno
ax1.plot(*juno_pos, marker="^", ms=7, color=JUNO_C, zorder=4)

def draw_link(a, b, col=LINK_C, lw=1.3, ls="--", alpha=0.7, zorder=5):
    ax1.annotate("", xy=b, xytext=a,
                 arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                 linestyle=ls, alpha=alpha),
                 zorder=zorder)

draw_link(earth_pos, voyager_pos, col="#6FE0A0", lw=1.5, ls="-")
draw_link(earth_pos, mars_pos,    col="#A0C4FF", lw=1.2, ls="--")
draw_link(earth_pos, juno_pos,    col="#FFCCAA", lw=1.0, ls=":")

# SEP angle arc
sep_angle_deg = 15.0
theta1 = np.degrees(np.arctan2(voyager_pos[1]-sun_pos[1], voyager_pos[0]-sun_pos[0]))
theta0 = np.degrees(np.arctan2(earth_pos[1]-sun_pos[1], earth_pos[0]-sun_pos[0]))
arc = Arc(sun_pos, 1.2, 1.2, angle=0, theta1=theta0, theta2=theta1,
          color=SUN_C, lw=1.5, alpha=0.7, zorder=6)
ax1.add_patch(arc)
ax1.text(0.35, 0.55, "SEP", color=SUN_C, fontsize=8, ha="center", zorder=7,
         fontfamily="monospace")

# Labels
kw = dict(fontsize=8.5, color=PANEL_FG, zorder=10, fontfamily="monospace",
          path_effects=[pe.withStroke(linewidth=2, foreground=BG)])
ax1.text(*earth_pos + np.array([-0.05, -0.28]), "Earth (DSN)", ha="center", **kw)
ax1.text(*sun_pos   + np.array([0.00, -0.30]),  "Sun",          ha="center", **kw)
ax1.text(*voyager_pos + np.array([0.15, 0.18]), "Voyager 1\n~162 AU (Nov 2026)",
         ha="left", **kw)
ax1.text(*mars_pos  + np.array([0.15, -0.22]), "Mars/Psyche DSOC", ha="left", **kw)
ax1.text(*juno_pos  + np.array([0.15,  0.15]), "Juno (Jupiter)", ha="left", **kw)

ax1.set_xlim(-3.0, 5.8)
ax1.set_ylim(-2.5, 4.2)
ax1.set_title("(a) Deep-Space Geometry + Plasma Scintillation",
              color=PANEL_FG, fontsize=10, pad=4, fontfamily="monospace")

# K-dist annotation box
kdbox = dict(boxstyle="round,pad=0.3", facecolor="#1C2330", edgecolor="#6FE0A0",
             alpha=0.85)
ax1.text(1.0, -1.8,
         r"K-dist fading: $f(h)\propto h^\alpha K_{\alpha-1}(2\sqrt{bh})$"
         "\n" r"$m\in\{1.2,1.4\}$,  $b=2.0$,  BT$\in\{0.3, 0.5\}$",
         color=PANEL_FG, fontsize=7.5, ha="center", va="center",
         bbox=kdbox, zorder=8)

# ─────────────────────────────────────────────────────────────────────────────
# Panel (b) — Signal chain block diagram
# ─────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_axes([0.52, 0.06, 0.46, 0.88])
ax2.set_facecolor(BG)
ax2.axis("off")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

def block(ax, x, y, w, h, label, sublabel="", facecolor="#1E3050", edgecolor="#4A8FD4",
          fontsize=9, lw=1.5):
    bb = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle="round,pad=0.12",
                        facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=4)
    ax.add_patch(bb)
    ax.text(x, y + (0.1 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            color=PANEL_FG, fontweight="bold", zorder=5,
            fontfamily="monospace")
    if sublabel:
        ax.text(x, y - 0.32, sublabel, ha="center", va="center",
                fontsize=7, color="#A8B8C8", zorder=5, fontfamily="monospace")

def arrow(ax, x0, y0, x1, y1, label="", col=ARROW_C, lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw), zorder=6)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.08, my, label, fontsize=7, color="#88AACC",
                va="center", fontfamily="monospace", zorder=7)

# --- TX chain (left column) ---
tx_y = [8.2, 6.9, 5.6]
block(ax2, 2.0, tx_y[0], 2.8, 0.80, "Bit source", "100 bits / frame",
      facecolor="#142230", edgecolor="#6FE0A0", fontsize=8)
block(ax2, 2.0, tx_y[1], 2.8, 0.80, "GMSK Mod",
      "BT∈{0.3,0.5}, 8 samp/sym", facecolor="#1A2E1A", edgecolor="#6FE0A0", fontsize=8)
block(ax2, 2.0, tx_y[2], 2.8, 0.80, "K-dist Channel",
      "h(t): compound-Γ", facecolor="#2E1A1A", edgecolor="#E87040", fontsize=8)

for i in range(len(tx_y)-1):
    arrow(ax2, 2.0, tx_y[i]-0.40, 2.0, tx_y[i+1]+0.40)

# AWGN adds in
block(ax2, 4.6, tx_y[2], 2.0, 0.70, "AWGN  n(t)",
      "σ²=P/SNR", facecolor="#2E1A1A", edgecolor="#E87040", fontsize=8)
arrow(ax2, 3.6, tx_y[2]+0.55, 4.6, tx_y[2]+0.20)
arrow(ax2, 4.6, tx_y[2]-0.35, 4.6, tx_y[2]-0.90, label="r(t)")

# Received signal label
ax2.text(4.6, 4.25, "r(t)=(√p)h(t)x(t)+n(t)",
         ha="center", fontsize=7.5, color="#AAC8DD",
         fontfamily="monospace", zorder=7)

# channel separator arrow
arrow(ax2, 4.6, 3.95, 4.6, 3.30)

# --- RX chain ---
rx_y = [3.0, 1.8]
block(ax2, 4.6, rx_y[0], 3.6, 0.75, "Feature Extract",
      "6-ch: I,Q,|y|²,∂θ/∂t,MF-I,MF-Q", facecolor="#1E1E3A", edgecolor="#A08FD4",
      fontsize=8)

# V5 model block – larger
block(ax2, 4.6, 1.55, 3.6, 1.45, "MambaNet-2ch Receiver",
      "CNN stem → BiMamba2+MHA → FiLM(SNR)\n→ AvgPool(×8) → BCE head",
      facecolor="#12183A", edgecolor="#507BD4", fontsize=8, lw=2.0)

arrow(ax2, 4.6, rx_y[0]-0.38, 4.6, rx_y[0]-0.88)

# output
block(ax2, 4.6, 0.35, 2.6, 0.55, "Hard decisions  b̂ ∈ {0,1}¹⁰⁰",
      facecolor="#142230", edgecolor="#6FE0A0", fontsize=8)
arrow(ax2, 4.6, 0.82, 4.6, 0.62)

ax2.set_title("(b) Signal Chain Block Diagram",
              color=PANEL_FG, fontsize=10, pad=4, fontfamily="monospace",
              x=0.5, y=0.98)

# channel annotation
ax2.text(1.00, 5.2, "TX", color="#6FE0A0", fontsize=11, fontweight="bold",
         fontfamily="monospace", alpha=0.7)
ax2.text(3.50, 2.2, "RX", color="#507BD4", fontsize=11, fontweight="bold",
         fontfamily="monospace", alpha=0.7)

# Overall title
fig.suptitle("Deep-Space Signal Recovery — System Overview",
             color=PANEL_FG, fontsize=12, fontweight="bold",
             fontfamily="monospace", y=0.98)

plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT}")
