"""V6 Batch 2 Part B4 — SNR estimator gate evaluation.

Gates:
  G1: Overall MAE <= 1.0 dB
  G2: AWGN-only MAE <= 0.5 dB
  G3: KB2 m=1.4 MAE <= 1.5 dB
  G4: No SNR bin signed bias > 2.0 dB
  G5: Decile calibration within ±1 dB

Saves: results/v6b2_snr_estimator_gate_report.csv
       figures/v6b2_snr_estimator_bias.png
"""

import sys, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.snr_estimator import SNREstimator

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = ROOT / "data" / "v6b2_snr_estimator"
CKPT     = ROOT / "checkpoints" / "v6b2_snr_estimator.pt"
RES_DIR  = ROOT / "results"
FIG_DIR  = ROOT / "figures"
RES_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

SNR_LO, SNR_HI = -8.0, 12.0
N_SNR_BINS = 10


def load_model():
    m = SNREstimator().to(DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def predict_val(model):
    va = np.load(DATA_DIR / "val" / "data.npz")
    ds = TensorDataset(torch.from_numpy(va["x"]), torch.from_numpy(va["snr"]))
    dl = DataLoader(ds, batch_size=512, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for x, snr in dl:
            preds.append(model(x.to(DEVICE)).cpu())
            trues.append(snr)
    preds = torch.cat(preds).numpy().astype(np.float32)
    trues = torch.cat(trues).numpy().astype(np.float32)
    return preds, trues


def get_configs():
    va = np.load(DATA_DIR / "val" / "data.npz")
    # Rebuild config labels from deterministic bucket assignment
    # bucket was assigned as i%4 in gen script over 200K; val indices from permutation.
    # We can't recover configs from saved data — the gen script didn't save them.
    # Use SNR range as proxy: all configs mixed, so just return empty config dict.
    # For G2/G3 we need AWGN vs KB2 labels. Add config saving to gen next time.
    # Fallback: generate a small validation set with known configs for G2/G3.
    return None


def eval_gates(preds: np.ndarray, trues: np.ndarray) -> list:
    errors = preds - trues
    abs_errors = np.abs(errors)
    rows = []

    # G1: Overall MAE
    g1_mae = float(abs_errors.mean())
    g1_pass = g1_mae <= 1.0
    rows.append({"gate": "G1_overall_mae", "value": round(g1_mae, 4),
                 "threshold": 1.0, "pass_fail": "PASS" if g1_pass else "FAIL"})
    print(f"G1 Overall MAE: {g1_mae:.4f} dB  {'PASS' if g1_pass else 'FAIL'}")

    # G4: Signed bias per SNR bin
    bins = np.linspace(SNR_LO, SNR_HI, N_SNR_BINS + 1)
    bin_idx = np.digitize(trues, bins) - 1
    bin_idx = np.clip(bin_idx, 0, N_SNR_BINS - 1)
    g4_pass = True
    biases = []
    for b in range(N_SNR_BINS):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        bias = float(errors[mask].mean())
        biases.append(bias)
        if abs(bias) > 2.0:
            g4_pass = False
        rows.append({"gate": f"G4_bias_bin{b}", "value": round(bias, 4),
                     "threshold": 2.0, "pass_fail": "PASS" if abs(bias) <= 2.0 else "FAIL"})
    print(f"G4 Max bin bias: {max(abs(b) for b in biases):.4f} dB  {'PASS' if g4_pass else 'FAIL'}")

    # G5: Decile calibration
    decile_edges = np.percentile(preds, np.linspace(0, 100, 11))
    decile_idx = np.digitize(preds, decile_edges[1:-1])
    g5_pass = True
    for d in range(10):
        mask = decile_idx == d
        if mask.sum() < 5:
            continue
        mean_pred  = float(preds[mask].mean())
        mean_true  = float(trues[mask].mean())
        cal_err = abs(mean_pred - mean_true)
        if cal_err > 1.0:
            g5_pass = False
        rows.append({"gate": f"G5_cal_decile{d}", "value": round(cal_err, 4),
                     "threshold": 1.0, "pass_fail": "PASS" if cal_err <= 1.0 else "FAIL"})
    print(f"G5 Decile calibration  {'PASS' if g5_pass else 'FAIL'}")

    return rows, biases, bins, g4_pass, g5_pass, g1_pass


def eval_config_gates(model):
    """G2, G3: re-generate small known-config val sets."""
    import sys
    sys.path.insert(0, str(ROOT))
    from src.synth_gen import generate_sample

    configs = {
        "awgn":     ("awgn", 1.2),
        "kb2_m1d4": ("kdist", 1.4),
    }
    snr_levels = np.linspace(-4, 8, 13)
    n_per = 100
    results = {}
    rng = np.random.default_rng(999)

    for name, (ch, m_val) in configs.items():
        maes = []
        for snr_db in snr_levels:
            preds_here = []
            for _ in range(n_per):
                x, _, _ = generate_sample(rng, 0.3, float(snr_db), ch, m=m_val)
                xt = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    p = model(xt).item()
                preds_here.append(abs(p - snr_db))
            maes.append(np.mean(preds_here))
        results[name] = float(np.mean(maes))
        print(f"  {name} MAE: {results[name]:.4f} dB")

    return results


def make_bias_figure(biases, bins):
    fig, ax = plt.subplots(figsize=(8, 4))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])[:len(biases)]
    ax.bar(bin_centers, biases, width=(SNR_HI - SNR_LO) / N_SNR_BINS * 0.8,
           color=["red" if abs(b) > 2 else "steelblue" for b in biases])
    ax.axhline(2.0, color="red", linestyle="--", alpha=0.5, label="+2 dB limit")
    ax.axhline(-2.0, color="red", linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("True SNR bin center (dB)")
    ax.set_ylabel("Signed bias (predicted - true, dB)")
    ax.set_title("V6B2 SNR Estimator: Per-bin Signed Bias (G4 Gate)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v6b2_snr_estimator_bias.png", dpi=150)
    plt.close()


def main():
    print(f"Loading model from {CKPT}")
    model = load_model()

    print("\nRunning val-set evaluation...")
    preds, trues = predict_val(model)

    rows, biases, bins, g4_pass, g5_pass, g1_pass = eval_gates(preds, trues)

    print("\nRunning config-specific gates (G2, G3)...")
    config_results = eval_config_gates(model)
    g2_mae = config_results["awgn"]
    g3_mae = config_results["kb2_m1d4"]
    g2_pass = g2_mae <= 0.5
    g3_pass = g3_mae <= 1.5

    rows.insert(1, {"gate": "G2_awgn_mae", "value": round(g2_mae, 4),
                    "threshold": 0.5, "pass_fail": "PASS" if g2_pass else "FAIL"})
    rows.insert(2, {"gate": "G3_kb2_m1d4_mae", "value": round(g3_mae, 4),
                    "threshold": 1.5, "pass_fail": "PASS" if g3_pass else "FAIL"})
    print(f"G2 AWGN MAE: {g2_mae:.4f} dB  {'PASS' if g2_pass else 'FAIL'}")
    print(f"G3 KB2 m=1.4 MAE: {g3_mae:.4f} dB  {'PASS' if g3_pass else 'FAIL'}")

    make_bias_figure(biases, bins)

    csv_path = RES_DIR / "v6b2_snr_estimator_gate_report.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gate","value","threshold","pass_fail"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nGate report: {csv_path}")

    all_pass = g1_pass and g2_pass and g3_pass and g4_pass and g5_pass
    print(f"\nOverall gates: G1={'P' if g1_pass else 'F'} G2={'P' if g2_pass else 'F'} "
          f"G3={'P' if g3_pass else 'F'} G4={'P' if g4_pass else 'F'} G5={'P' if g5_pass else 'F'}")
    print(f"SNR_FIX_STATUS: {'PASS→continue to Part C' if all_pass else 'check failures'}")
    return all_pass


if __name__ == "__main__":
    main()
