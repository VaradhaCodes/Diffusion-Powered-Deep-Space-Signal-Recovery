"""V6 Batch 2 Part A — Channel physics validation.

Generates 50K frames per config (4 combos: m in {1.2, 1.4} x BT in {0.3, 0.5}).
Runs A1-A4 checks. Saves results/v6b2_partA_channel_validation.csv and
figures/v6b2_figA_channel_validation.png.

GUARDRAIL: does NOT modify src/synth_gen.py.
"""

import sys, csv, os, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.synth_gen import generate_sample, kdist_fade
from src.physics.kdist import (
    kdist_second_moment, kdist_fourth_moment, kdist_envelope_pdf
)

N_FRAMES   = 50_000
N_ACF      = 10_000
M_CHOICES  = [1.2, 1.4]
BT_CHOICES = [0.3, 0.5]
ALPHA_MAP  = {1.2: 1.2, 1.4: 1.4}   # synth_gen uses m; physics uses alpha (same param)
B          = 2.0
SNR_DB     = 4.0   # mid-range, needed to generate frames

RESULT_DIR = ROOT / "results"
FIG_DIR    = ROOT / "figures"
RESULT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULT_DIR / "v6b2_partA_channel_validation.csv"
FIG_PATH = FIG_DIR / "v6b2_figA_channel_validation.png"


def extract_h_values(m: float, bt: float, n: int, seed: int = 42) -> np.ndarray:
    """Generate n K-dist fading coefficients (one per frame)."""
    rng = np.random.default_rng(seed)
    h = kdist_fade(n, m=m, b=B, rng=rng)
    return h


def run_a1(h: np.ndarray, alpha: float, m: float, bt: float, rows: list) -> bool:
    """A1: Envelope PDF match."""
    envelope = np.abs(h)
    counts, bin_edges = np.histogram(envelope, bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # filter out zero-count bins
    mask = counts > 0
    bc = bin_centers[mask]
    emp = counts[mask]
    theo = kdist_envelope_pdf(bc, alpha=alpha, b=B)

    # percentiles of envelope to find bins at 10th, 50th, 90th pct
    p10, p50, p90 = np.percentile(envelope, [10, 50, 90])
    all_pass = True
    for pct, pval in [(10, p10), (50, p50), (90, p90)]:
        # find nearest bin center
        idx = np.argmin(np.abs(bc - pval))
        if theo[idx] > 1e-12:
            rel_err = abs(emp[idx] - theo[idx]) / theo[idx] * 100
        else:
            rel_err = 0.0
        pf = "PASS" if rel_err <= 10.0 else "WARN"
        if pf == "WARN":
            all_pass = False
        rows.append({
            "check": "A1_pdf",
            "config_m": m, "config_bt": bt,
            "metric": f"rel_err_pct_{pct}th",
            "theoretical": round(float(theo[idx]), 6),
            "empirical": round(float(emp[idx]), 6),
            "rel_error_pct": round(rel_err, 2),
            "pass_fail": pf,
            "reference": "Jao 1984, Eq.(4); DOI:10.1109/TVT.1984.23517",
        })
    return all_pass


def run_a2(h: np.ndarray, alpha: float, m: float, bt: float, rows: list) -> bool:
    """A2: Second-moment check."""
    emp = float(np.mean(np.abs(h)**2))
    theo = kdist_second_moment(alpha=alpha, b=B)
    rel_err = abs(emp - theo) / theo * 100
    pf = "PASS" if rel_err <= 2.0 else "WARN"
    rows.append({
        "check": "A2_second_moment",
        "config_m": m, "config_bt": bt,
        "metric": "E[|h|^2]",
        "theoretical": round(theo, 6),
        "empirical": round(emp, 6),
        "rel_error_pct": round(rel_err, 4),
        "pass_fail": pf,
        "reference": "Jao 1984, Eq.(5); E[|h|^2]=b^2",
    })
    return pf == "PASS"


def run_a3(h: np.ndarray, alpha: float, m: float, bt: float, rows: list) -> bool:
    """A3: Fourth-moment ratio."""
    h2 = np.abs(h)**2
    h4 = h2**2
    emp_ratio = float(np.mean(h4)) / float(np.mean(h2))**2
    theo_4th = kdist_fourth_moment(alpha=alpha, b=B)
    theo_2nd = kdist_second_moment(alpha=alpha, b=B)
    theo_ratio = theo_4th / theo_2nd**2
    rel_err = abs(emp_ratio - theo_ratio) / theo_ratio * 100
    pf = "PASS" if rel_err <= 10.0 else "WARN"
    rows.append({
        "check": "A3_fourth_moment_ratio",
        "config_m": m, "config_bt": bt,
        "metric": "E[|h|^4]/E[|h|^2]^2",
        "theoretical": round(theo_ratio, 6),
        "empirical": round(emp_ratio, 6),
        "rel_error_pct": round(rel_err, 4),
        "pass_fail": pf,
        "reference": "Jao 1984, Eq.(5): E[|h|^4]=b^4*(alpha+1)/alpha",
    })
    return pf == "PASS"


def run_a4(h: np.ndarray, m: float, bt: float, rows: list):
    """A4: Intra-frame autocorrelation (observational, no gate).
    Fading is one scalar per frame → check ACF across frames at lag k.
    """
    env = np.abs(h[:N_ACF])
    env_centered = env - env.mean()
    var = np.var(env_centered)
    for lag in [1, 10, 100, 400]:
        if lag >= len(env_centered):
            continue
        acf = float(np.mean(env_centered[:-lag] * env_centered[lag:])) / (var + 1e-12)
        rows.append({
            "check": f"A4_acf_lag{lag}",
            "config_m": m, "config_bt": bt,
            "metric": f"ACF_lag{lag}",
            "theoretical": 0.0,
            "empirical": round(acf, 6),
            "rel_error_pct": float("nan"),
            "pass_fail": "OBS",
            "reference": "Observational — i.i.d. per frame → should be ~0",
        })


def make_figure(results: list):
    configs = [(m, bt) for m in M_CHOICES for bt in BT_CHOICES]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("V6 Batch 2 Part A — K-dist Channel Validation", fontsize=13)

    for ax, (m, bt) in zip(axes.flat, configs):
        # Generate fresh envelope samples for plotting
        rng = np.random.default_rng(99)
        h = kdist_fade(N_FRAMES, m=m, b=B, rng=rng)
        envelope = np.abs(h)
        r_range = np.linspace(0.01, envelope.max() * 1.05, 500)
        theo_pdf = kdist_envelope_pdf(r_range, alpha=m, b=B)

        ax.hist(envelope, bins=100, density=True, alpha=0.6, label="Empirical", color="steelblue")
        ax.plot(r_range, theo_pdf, "r-", lw=2, label=f"Theoretical K-dist\nα={m}, b={B}")
        ax.set_title(f"m={m}, BT={bt}")
        ax.set_xlabel("|h| (envelope)")
        ax.set_ylabel("PDF")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print(f"Figure saved: {FIG_PATH}")


def main():
    rows = []
    any_warn = False

    for m in M_CHOICES:
        for bt in BT_CHOICES:
            alpha = m   # synth_gen's m == Zhu's alpha (shape param)
            print(f"\n--- Config m={m}, BT={bt} ---")
            t0 = time.time()
            h = extract_h_values(m=m, bt=bt, n=N_FRAMES)
            print(f"  Generated {N_FRAMES} h values in {time.time()-t0:.1f}s")

            ok1 = run_a1(h, alpha, m, bt, rows)
            ok2 = run_a2(h, alpha, m, bt, rows)
            ok3 = run_a3(h, alpha, m, bt, rows)
            run_a4(h, m, bt, rows)

            if not ok1:
                print(f"  WARN A1: PDF mismatch for m={m}, BT={bt}")
                any_warn = True
            if not ok2:
                print(f"  WARN A2: Second-moment mismatch for m={m}, BT={bt}")
                any_warn = True
            if not ok3:
                print(f"  WARN A3: Fourth-moment ratio mismatch for m={m}, BT={bt}")
                any_warn = True

    # Save CSV
    fieldnames = ["check","config_m","config_bt","metric","theoretical","empirical","rel_error_pct","pass_fail","reference"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults saved: {CSV_PATH}")

    # Print summary
    pass_rows = [r for r in rows if r["pass_fail"] == "PASS"]
    warn_rows = [r for r in rows if r["pass_fail"] == "WARN"]
    obs_rows  = [r for r in rows if r["pass_fail"] == "OBS"]
    print(f"\nSummary: PASS={len(pass_rows)}, WARN={len(warn_rows)}, OBS={len(obs_rows)}")

    make_figure(rows)

    status = "WARN" if any_warn else "PASS"
    print(f"\nPartA overall: {status}")
    return status


if __name__ == "__main__":
    main()
