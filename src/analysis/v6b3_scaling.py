"""V6 Batch 3 — Scaling curve + power-law fit.

Called by sweep_v6b3.py after the sweep completes.
Can also be run standalone:
  python src/analysis/v6b3_scaling.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

RESULT_DIR  = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SIZE_MAP    = {"500K": 500_000, "1M": 1_000_000, "2M": 2_000_000, "5M": 5_000_000}
SIZE_LABELS = list(SIZE_MAP.keys())


def load_ensemble_ber(sizes_ran: list[str]) -> tuple[list[float], list[float], dict]:
    """Return (x_sizes, y_bers_pct, per_seed_bers_pct).

    per_seed_bers_pct: {size_label: [ber_s0, ber_s1, ber_s2]} in percent.
    """
    x, y, per_seed = [], [], {}
    for sz in sizes_ran:
        path = RESULT_DIR / f"v6b3_{sz}_ensemble_test.csv"
        if not path.exists():
            print(f"  WARNING: {path.name} missing, skipping")
            continue

        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))
        overall = next(r for r in rows if r["condition"] == "OVERALL")
        ber_pct = float(overall["ber"]) * 100
        x.append(SIZE_MAP[sz])
        y.append(ber_pct)

        # Per-seed BERs for error bars
        seed_bers = []
        for seed in (0, 1, 2):
            sp = RESULT_DIR / f"v6b3_{sz}_s{seed}_test.csv"
            if sp.exists():
                with open(sp) as f:
                    srows = list(csv.DictReader(f))
                sovr = next(r for r in srows if r["condition"] == "OVERALL")
                seed_bers.append(float(sovr["ber"]) * 100)
        per_seed[sz] = seed_bers

    return x, y, per_seed


def power_law_fit(x: list[float], y: list[float]):
    """Fit BER = a + b * S^(-c) using Huber loss + L-BFGS.

    Returns (a, b, c) or None if fit fails.
    """
    if len(x) < 3:
        return None

    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    delta = 0.5   # Huber delta

    def huber_loss(params):
        a, log_b, c = params
        b = np.exp(log_b)
        pred = a + b * x_arr ** (-c)
        r = pred - y_arr
        mask = np.abs(r) <= delta
        loss = np.where(mask, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta))
        return loss.sum()

    # Multiple restarts
    best_res = None
    for a0 in [1.0, 2.0]:
        for c0 in [0.3, 0.5, 0.7]:
            try:
                res = minimize(huber_loss, [a0, np.log(1.0), c0],
                               method="L-BFGS-B",
                               bounds=[(-1, 10), (-5, 10), (0.01, 3.0)],
                               options={"maxiter": 2000, "ftol": 1e-12})
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
            except Exception:
                continue

    if best_res is None or not best_res.success:
        return None

    a, log_b, c = best_res.x
    return float(a), float(np.exp(log_b)), float(c)


def bootstrap_ci(x, y, params, n_boot=1000, seed=99):
    """Bootstrap 95% CI on the fitted curve.

    Returns (lo_band, hi_band) arrays evaluated at x_fine.
    """
    if params is None or len(x) < 3:
        return None

    rng = np.random.default_rng(seed)
    a, b, c = params
    x_arr = np.array(x)
    y_arr = np.array(y)
    x_fine = np.geomspace(x_arr.min(), x_arr.max(), 200)
    curves = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        xb, yb = x_arr[idx], y_arr[idx]
        if len(np.unique(xb)) < 2:
            continue
        fit = power_law_fit(list(xb), list(yb))
        if fit is None:
            continue
        a_b, b_b, c_b = fit
        curves.append(a_b + b_b * x_fine ** (-c_b))

    if not curves:
        return None

    curves = np.array(curves)
    return np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)


def extrapolate(params, sizes: list) -> list[float]:
    """Return predicted BER% at given sizes."""
    if params is None:
        return []
    a, b, c = params
    return [float(a + b * s ** (-c)) for s in sizes]


def run_scaling_analysis(sizes_ran: list[str], winner: str, conv_epochs: dict) -> dict:
    """Main entry point. Returns dict with fit params and extrapolations."""
    x, y, per_seed = load_ensemble_ber(sizes_ran)
    if not x:
        print("  No data for scaling curve.")
        return {}

    print(f"\n  Scaling data: {list(zip(sizes_ran[:len(x)], [f'{v:.4f}%' for v in y]))}")

    # --- Fit ---
    params = power_law_fit(x, y)
    if params:
        a, b, c = params
        print(f"  Power-law fit: BER = {a:.4f} + {b:.4f} * S^(-{c:.4f})")
        ci = bootstrap_ci(x, y, params)
        extrap = extrapolate(params, [10_000_000, 50_000_000])
        print(f"  Extrapolated: 10M={extrap[0]:.4f}%  50M={extrap[1]:.4f}%")

        # Save fit
        import csv
        fit_path = RESULT_DIR / "v6b3_scaling_fit.csv"
        with open(fit_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["param", "value"])
            w.writerows([["a", a], ["b", b], ["c", c],
                         ["extrap_10M_pct", extrap[0]], ["extrap_50M_pct", extrap[1]]])
        print(f"  Fit saved → {fit_path.name}")
    else:
        ci = None
        extrap = []
        print("  Power-law fit failed or insufficient data (need ≥3 points).")

    # --- Flat check ---
    flat = (max(y) - min(y)) < 0.15 if len(y) >= 2 else True
    if flat:
        print("  FINDING: Curve is FLAT (range < 0.15 pp). Data size is NOT the bottleneck.")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    x_arr = np.array(x)
    y_arr = np.array(y)

    # Error bars
    y_err = []
    for sz in sizes_ran[:len(x)]:
        seed_bers = per_seed.get(sz, [])
        if len(seed_bers) >= 2:
            y_err.append(np.std(seed_bers))
        else:
            y_err.append(0.0)
    y_err = np.array(y_err)

    ax.errorbar(x_arr, y_arr, yerr=y_err, fmt="o-", color="steelblue",
                capsize=4, linewidth=1.8, markersize=7, label="Ensemble BER")

    # CI band
    if ci is not None and params is not None:
        x_fine = np.geomspace(x_arr.min(), x_arr.max(), 200)
        a, b, c = params
        y_fit = a + b * x_fine ** (-c)
        ax.plot(x_fine, y_fit, "--", color="steelblue", alpha=0.6, label="Power-law fit")
        lo, hi = ci
        ax.fill_between(x_fine, lo, hi, alpha=0.2, color="steelblue", label="95% CI")

    # Mark winner
    if winner in sizes_ran[:len(x)]:
        wi = sizes_ran.index(winner)
        if wi < len(x):
            ax.plot(x[wi], y[wi], "*", markersize=18, color="gold",
                    markeredgecolor="darkorange", zorder=5, label=f"Winner ({winner})")

    # Label points with convergence epoch
    for i, sz in enumerate(sizes_ran[:len(x)]):
        ep = conv_epochs.get(sz, "?")
        ax.annotate(f"ep={ep}", (x[i], y[i]), textcoords="offset points",
                    xytext=(5, 6), fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Pretrain corpus size (samples)", fontsize=12)
    ax.set_ylabel("3-seed ensemble BER (%)", fontsize=12)
    ax.set_title("V6 Batch 3 — mambanet_2ch scaling curve", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    fig_path = FIGURES_DIR / "v6b3_scaling_curve.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved → {fig_path}")

    return {
        "flat": flat,
        "params": params,
        "extrap_10M": extrap[0] if extrap else None,
        "extrap_50M": extrap[1] if len(extrap) > 1 else None,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["500K", "1M", "2M", "5M"])
    ap.add_argument("--winner", default="500K")
    args = ap.parse_args()
    run_scaling_analysis(args.sizes, args.winner, {})
