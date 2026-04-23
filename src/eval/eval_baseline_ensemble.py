"""Ensemble evaluation of 3-seed Zhu baseline — averages soft predictions across seeds.

Usage:
  python src/eval/eval_baseline_ensemble.py [--seeds 0 1 2]

Outputs:
  results/baseline_s{seed}_test.csv    — per-seed (written by train_baseline.py)
  results/baseline_ensemble_test.csv   — ensemble (mean soft prediction)
  results/baseline_ensemble_summary.txt
"""

import sys, os, csv, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

from data_zhu import zhu_test_dataset, TEST_CONDITIONS
from models.zhu_baseline import ZhuBaseline

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_N   = 700
CKPT_DIR = ROOT / "checkpoints"
RES_DIR  = ROOT / "results"


def best_ckpt_for_seed(seed: int) -> Path:
    ckpts = sorted(CKPT_DIR.glob(f"baseline_s{seed}_ep*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints for baseline seed {seed}")
    best, best_loss = None, float("inf")
    for p in ckpts:
        ck = torch.load(p, map_location="cpu")
        if ck.get("val_loss", float("inf")) < best_loss:
            best_loss = ck["val_loss"]
            best = p
    return best


def load_model(seed: int) -> ZhuBaseline:
    ckpt_path = best_ckpt_for_seed(seed)
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model = ZhuBaseline().to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"  seed={seed}  ckpt={ckpt_path.name}  val_loss={ck.get('val_loss', float('nan')):.5f}")
    return model


def ensemble_eval(models, cond):
    ds = Subset(zhu_test_dataset(cond), list(range(TEST_N)))
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    mse_sum = ber_sum = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = torch.stack([m(x) for m in models], dim=0).mean(0)
            mse_sum += ((preds - y) ** 2).mean().item() * len(x)
            ber_sum += ((preds > 0.5).float() != y).float().mean().item() * len(x)
            n += len(x)
    return mse_sum / n, ber_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    print("Loading baseline models...")
    models = [load_model(s) for s in args.seeds]

    rows = []
    print(f"\n{'Condition':<22}  {'MSE':>8}  {'BER':>8}")
    print("-" * 44)
    for cond in TEST_CONDITIONS:
        mse, ber_v = ensemble_eval(models, cond)
        rows.append((cond, mse, ber_v, TEST_N))
        print(f"{cond:<22}  {mse:>8.5f}  {ber_v:>8.5f}")

    overall_n   = sum(r[3] for r in rows)
    overall_mse = sum(r[1] * r[3] for r in rows) / overall_n
    overall_ber = sum(r[2] * r[3] for r in rows) / overall_n
    rows.append(("ALL", overall_mse, overall_ber, overall_n))
    print("-" * 44)
    print(f"{'ALL':<22}  {overall_mse:>8.5f}  {overall_ber:>8.5f}")

    RES_DIR.mkdir(exist_ok=True)
    out_csv = RES_DIR / "baseline_ensemble_test.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "mse", "ber", "n_samples"])
        w.writerows(rows)

    # Per-seed BERs from their individual test CSVs
    seed_bers = {}
    for s in args.seeds:
        p = RES_DIR / f"baseline_s{s}_test.csv"
        if p.exists():
            with open(p) as f:
                for row in csv.DictReader(f):
                    if row["condition"] == "ALL":
                        seed_bers[s] = float(row["ber"])

    lines = [
        f"Seeds              : {args.seeds}",
        *(f"  s{s} BER         : {seed_bers.get(s, float('nan'))*100:.3f}%" for s in args.seeds),
        f"Ensemble BER       : {overall_ber*100:.3f}%",
        f"Zhu paper BER      : 3.12%",
        f"Results saved      : {out_csv}",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(RES_DIR / "baseline_ensemble_summary.txt", "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
