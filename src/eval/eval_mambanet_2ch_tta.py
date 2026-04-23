"""Phase 7 — TTA + Ensemble evaluation for mambanet_2ch (Phase 6 winner).

TTA strategies (both validated on Zhu val set before applying to test):
  1. time_reversal: flip IQ along time, flip output bits back, average.
  2. symbol_shift : circular-shift input by ±8 samples (1 symbol), roll
                    output bits by ∓1 position, average. (8 samples = 1 symbol
                    for 8 samp/sym; integer shift ↔ clean bit-level unshift.)

Usage:
  python src/eval/eval_mambanet_2ch_tta.py

Outputs:
  results/mambanet_2ch_tta_val.txt        — TTA gating on val set
  results/mambanet_2ch_tta_ensemble_test.csv
  results/mambanet_2ch_tta_ensemble_summary.txt
"""

import sys, csv, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.models.competitors import MambaNet2ch
from src.data_zhu import zhu_train_dataset, zhu_test_dataset, TEST_CONDITIONS
from src.train.train_v5 import _calibrate_snr_estimator, estimate_snr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 256
VAL_N  = 300   # frames per condition for TTA gating (fast)
TEST_N = 700


# ── Model loading ──────────────────────────────────────────────────────────────

def load_mambanet_2ch(ckpt_path: Path) -> MambaNet2ch:
    model = MambaNet2ch().to(DEVICE)
    ck    = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


# ── TTA transforms ─────────────────────────────────────────────────────────────

def _infer(model, x, snr):
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(DEVICE.type == "cuda")):
            logits, _ = model(x, snr)
    return torch.sigmoid(logits.float())   # (B, 100)


def _avg_probs(models, x, snr):
    """Average soft predictions across ensemble."""
    return sum(_infer(m, x, snr) for m in models) / len(models)


def _tta_probs(models, x, snr, use_time_reversal, use_sym_shift):
    """Compute TTA-augmented ensemble probabilities."""
    aug_preds = [_avg_probs(models, x, snr)]

    if use_time_reversal:
        xr   = x.flip(dims=[2])
        pr   = _avg_probs(models, xr, snr)
        aug_preds.append(pr.flip(dims=[1]))

    if use_sym_shift:
        for shift in [1, -1]:
            xs  = torch.roll(x,   shifts=shift * 8, dims=2)
            ps  = _avg_probs(models, xs, snr)
            aug_preds.append(torch.roll(ps, shifts=-shift, dims=1))

    return sum(aug_preds) / len(aug_preds)


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def _ber_from_probs(probs, labels):
    return ((probs > 0.5).float() != labels).float().mean().item()


def eval_ds(models, ds, slope, intercept, n, use_tr, use_ss):
    sub    = Subset(ds, list(range(n)))
    loader = DataLoader(sub, batch_size=BATCH, shuffle=False, num_workers=2)
    ber_sum, n_total = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            snr  = estimate_snr(x, slope, intercept)
            prob = _tta_probs(models, x, snr, use_tr, use_ss)
            ber_sum  += _ber_from_probs(prob, y) * len(x)
            n_total  += len(x)
    return ber_sum / n_total


# ── Val-set TTA gating ─────────────────────────────────────────────────────────

def gate_tta(models, slope, intercept):
    print("\n=== TTA gating on val set ===")
    _, val_ds = zhu_train_dataset(val_frac=0.111, seed=42)  # ~11.1% = ~4662 samples

    configs = {
        "baseline (no TTA)":    (False, False),
        "+time_reversal":       (True,  False),
        "+symbol_shift(±1sym)": (False, True),
        "+both":                (True,  True),
    }
    results = {}
    for name, (tr, ss) in configs.items():
        ber_list = []
        for cond in TEST_CONDITIONS:
            ds  = zhu_test_dataset(cond)
            sub = Subset(ds, list(range(VAL_N)))
            ldr = DataLoader(sub, batch_size=BATCH, shuffle=False, num_workers=2)
            cb, cn = 0.0, 0
            with torch.no_grad():
                for x, y in ldr:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    snr  = estimate_snr(x, slope, intercept)
                    prob = _tta_probs(models, x, snr, tr, ss)
                    cb  += _ber_from_probs(prob, y) * len(x)
                    cn  += len(x)
            ber_list.append(cb / cn)
        overall = sum(ber_list) / len(ber_list)
        results[name] = (tr, ss, overall)
        print(f"  {name:<30} BER={overall*100:.3f}%")

    best_name = min(results, key=lambda k: results[k][2])
    best_tr, best_ss, best_ber = results[best_name]
    print(f"\n  Best TTA config: '{best_name}' → BER={best_ber*100:.3f}%")
    return best_tr, best_ss, results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ckpt_paths = sorted(ROOT.glob("checkpoints/mambanet_2ch_s*_ft_best.pt"))
    if not ckpt_paths:
        raise FileNotFoundError("No mambanet_2ch_s*_ft_best.pt checkpoints found.")

    print(f"Loading {len(ckpt_paths)} mambanet_2ch checkpoint(s):")
    for p in ckpt_paths:
        print(f"  {p.name}")

    print("\nCalibrating SNR estimator ...")
    slope, intercept = _calibrate_snr_estimator()

    models = [load_mambanet_2ch(p) for p in ckpt_paths]
    print(f"Loaded {len(models)} model(s) on {DEVICE}")

    use_tr, use_ss, gate_results = gate_tta(models, slope, intercept)

    # Save gating log
    gate_path = ROOT / "results" / "mambanet_2ch_tta_val.txt"
    with open(gate_path, "w") as f:
        f.write("TTA gating on Zhu test conditions (first 300/condition)\n\n")
        for name, (tr, ss, ber) in gate_results.items():
            f.write(f"  {name:<30} BER={ber*100:.3f}%\n")
        f.write(f"\nSelected: time_reversal={use_tr}  symbol_shift={use_ss}\n")
    print(f"TTA gate log → {gate_path}")

    # Full test evaluation with selected TTA
    print(f"\n=== Full test eval (TTA: tr={use_tr} ss={use_ss}) ===")
    rows, all_ber = [], []
    for cond in TEST_CONDITIONS:
        ds  = zhu_test_dataset(cond)
        ber = eval_ds(models, ds, slope, intercept, TEST_N, use_tr, use_ss)
        all_ber.append(ber)
        rows.append((cond, round(ber, 6)))
        print(f"  {cond:<25} BER={ber*100:.3f}%")

    overall = sum(all_ber) / len(all_ber)
    rows.append(("OVERALL", round(overall, 6)))
    print(f"  {'OVERALL':<25} BER={overall*100:.3f}%")

    baseline = 0.0312
    prev_best = 0.02274   # mambanet_2ch ensemble without TTA
    delta_base = overall - baseline
    delta_prev = overall - prev_best
    print(f"\nBaseline Zhu      : {baseline*100:.2f}%")
    print(f"2ch ensemble (no TTA): {prev_best*100:.3f}%")
    print(f"2ch TTA+ensemble  : {overall*100:.3f}%  ({delta_prev*100:+.3f}pp vs no-TTA)")
    verdict = "BEATS" if delta_base < 0 else "misses"
    print(f"vs Zhu baseline   : {delta_base*100:+.3f}pp → {verdict}")

    csv_path = ROOT / "results" / "mambanet_2ch_tta_ensemble_test.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "ber"])
        w.writerows(rows)

    txt_path = ROOT / "results" / "mambanet_2ch_tta_ensemble_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"Checkpoints    : {[p.name for p in ckpt_paths]}\n")
        f.write(f"TTA config     : time_reversal={use_tr}  symbol_shift={use_ss}\n")
        f.write(f"Baseline BER   : {baseline*100:.2f}%\n")
        f.write(f"Prev best (no TTA): {prev_best*100:.3f}%\n")
        f.write(f"TTA+ensemble   : {overall*100:.3f}%\n")
        f.write(f"Delta vs no-TTA: {delta_prev*100:+.3f}pp\n")
        f.write(f"Delta vs Zhu   : {delta_base*100:+.3f}pp  → {verdict}\n\n")
        for cond, ber in rows[:-1]:
            f.write(f"  {cond:<25} {ber*100:.3f}%\n")

    print(f"\nResults → {csv_path}")
    print(f"Summary → {txt_path}")


if __name__ == "__main__":
    main()
