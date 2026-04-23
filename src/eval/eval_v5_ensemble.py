"""Ensemble evaluation of V5 checkpoints on Zhu test set.

Averages soft sigmoid outputs across seeds before thresholding.
Works with 1..N checkpoints; best used after all 3 seeds complete.

Usage:
  python src/eval/eval_v5_ensemble.py               # auto-finds v5_s*_ft_best.pt
  python src/eval/eval_v5_ensemble.py --ckpts checkpoints/v5_s0_ft_best.pt checkpoints/v5_s1_ft_best.pt

Outputs:
  results/v5_ensemble_test.csv
  results/v5_ensemble_summary.txt
"""

import sys, csv, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from src.models.v5_model import V5Model, _SNR_MIN, _SNR_RANGE
from src.data_zhu import zhu_test_dataset, TEST_CONDITIONS
from src.train.train_v5 import _calibrate_snr_estimator, estimate_snr

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_N  = 700
BATCH   = 256


def load_model(ckpt_path: Path) -> V5Model:
    model = V5Model().to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def eval_condition(models, ds, slope, intercept, n=TEST_N):
    sub    = Subset(ds, list(range(n)))
    loader = DataLoader(sub, batch_size=BATCH, shuffle=False, num_workers=2)

    ber_sum = n_total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            snr   = estimate_snr(x, slope, intercept)

            # Average soft predictions across models
            logits_sum = torch.zeros(len(x), 100, device=DEVICE)
            for m in models:
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(DEVICE.type == "cuda")):
                    lg, _ = m(x, snr)
                logits_sum += lg.float()
            prob_avg = torch.sigmoid(logits_sum / len(models))

            ber = ((prob_avg > 0.5).float() != y).float().mean().item()
            ber_sum  += ber * len(x)
            n_total  += len(x)

    return ber_sum / n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", type=str, default=None,
                    help="checkpoint paths; auto-detected if omitted")
    args = ap.parse_args()

    # Find checkpoints
    if args.ckpts:
        ckpt_paths = [Path(p) for p in args.ckpts]
    else:
        ckpt_paths = sorted(ROOT.glob("checkpoints/v5_s*_ft_best.pt"))

    if not ckpt_paths:
        raise FileNotFoundError("No v5_s*_ft_best.pt checkpoints found.")

    print(f"Ensemble: {len(ckpt_paths)} model(s)")
    for p in ckpt_paths:
        print(f"  {p.name}")

    # Calibrate SNR estimator
    print("\nCalibrating SNR estimator ...")
    slope, intercept = _calibrate_snr_estimator()

    # Load models
    models = [load_model(p) for p in ckpt_paths]
    print(f"Loaded {len(models)} checkpoint(s) on {DEVICE}")

    # Evaluate per condition
    print(f"\nEvaluating on Zhu test set ({TEST_N}/condition) ...")
    rows = []
    all_ber = []
    for cond in TEST_CONDITIONS:
        ds  = zhu_test_dataset(cond)
        ber = eval_condition(models, ds, slope, intercept)
        all_ber.append(ber)
        print(f"  {cond:<25} BER={ber*100:.3f}%")
        rows.append((cond, round(ber, 6)))

    overall = sum(all_ber) / len(all_ber)
    print(f"  {'OVERALL':<25} BER={overall*100:.3f}%")
    rows.append(("OVERALL", round(overall, 6)))

    # Baseline reference
    baseline_ber = 0.0312
    delta = overall - baseline_ber
    status = "BEATS" if delta < 0 else "misses"
    print(f"\nBaseline BER={baseline_ber*100:.2f}%")
    print(f"Ensemble  BER={overall*100:.3f}%  ({delta*100:+.3f}pp)  → {status} baseline")

    # Save
    csv_path = ROOT / "results" / "v5_ensemble_test.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "ber"])
        w.writerows(rows)

    txt_path = ROOT / "results" / "v5_ensemble_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"Ensemble checkpoints: {[p.name for p in ckpt_paths]}\n")
        f.write(f"Baseline BER : {baseline_ber*100:.2f}%\n")
        f.write(f"Ensemble BER : {overall*100:.3f}%\n")
        f.write(f"Delta        : {delta*100:+.3f}pp\n")
        f.write(f"Verdict      : {status} baseline\n\n")
        for cond, ber in rows[:-1]:
            f.write(f"  {cond:<25} {ber*100:.3f}%\n")

    print(f"\nResults → {csv_path}")
    print(f"Summary → {txt_path}")


if __name__ == "__main__":
    main()
