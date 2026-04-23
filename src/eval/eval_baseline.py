"""Evaluate Zhu baseline on held-out test set.

Usage:
  python src/eval/eval_baseline.py [--ckpt checkpoints/baseline_ep40.pt]

Outputs:
  results/baseline_test_results.csv   — per-condition MSE + BER
  results/baseline_test_summary.txt   — soft-gate verdict
"""

import sys, os, csv, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from data_zhu import zhu_test_dataset, TEST_CONDITIONS
from models.zhu_baseline import ZhuBaseline
from torch.utils.data import Subset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Paper Table 2: test = 4200 = 6×700.  Zenodo has 6×1400; use first 700 per condition.
TEST_N_PER_COND = 700

# Paper reports loss→0 / accuracy→1, so reference MSE ≈ 0 (very low).
# We interpret the soft gate as: overall BER < 5% AND trend AWGN<KB2.
SOFT_BER_THRESHOLD = 0.05


def eval_condition(model, ds, batch=512):
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    mse_sum = ber_sum = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            mse_sum += ((pred - y) ** 2).mean().item() * len(x)
            ber_sum += ((pred > 0.5).float() != y).float().mean().item() * len(x)
            n += len(x)
    return mse_sum / n, ber_sum / n


def best_checkpoint(ckpt_dir="checkpoints"):
    ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("baseline_ep"))
    if not ckpts:
        raise FileNotFoundError("No baseline checkpoints found in checkpoints/")
    # pick the one with lowest val_loss
    best, best_loss = None, float("inf")
    for fn in ckpts:
        ck = torch.load(os.path.join(ckpt_dir, fn), map_location="cpu")
        if ck.get("val_loss", float("inf")) < best_loss:
            best_loss = ck["val_loss"]
            best = fn
    return os.path.join(ckpt_dir, best), best_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path, _ = best_checkpoint()
    print(f"Loading: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = ZhuBaseline().to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []
    print(f"\n{'Condition':<22}  {'MSE':>8}  {'BER':>8}  {'n':>5}")
    print("-" * 52)

    for cond in TEST_CONDITIONS:
        ds_full = zhu_test_dataset(cond)
        ds = Subset(ds_full, list(range(TEST_N_PER_COND)))   # first 700 → paper's 4200
        mse, ber_val = eval_condition(model, ds)
        rows.append((cond, mse, ber_val, len(ds)))
        print(f"{cond:<22}  {mse:>8.5f}  {ber_val:>8.5f}  {len(ds):>5}")

    # Overall (4200 pooled)
    from torch.utils.data import ConcatDataset
    ds_all = ConcatDataset([Subset(zhu_test_dataset(c), list(range(TEST_N_PER_COND)))
                            for c in TEST_CONDITIONS])
    mse_all, ber_all = eval_condition(model, ds_all)
    rows.append(("ALL", mse_all, ber_all, len(ds_all)))
    print("-" * 52)
    print(f"{'ALL':<22}  {mse_all:>8.5f}  {ber_all:>8.5f}  {len(ds_all):>5}")

    # Save CSV
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_test_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "mse", "ber", "n_samples"])
        w.writerows(rows)

    # ── Soft gate ─────────────────────────────────────────────────────────────
    # Paper claims loss→0/accuracy→1; soft gate: overall BER<5% AND AWGN<KB2 trend.
    awgn_ber = [ber for cond, _, ber, _ in rows if "Awgn" in cond]
    kb2_ber  = [ber for cond, _, ber, _ in rows if "kb2"  in cond]
    trend_ok = (len(awgn_ber) > 0 and len(kb2_ber) > 0 and
                sum(awgn_ber)/len(awgn_ber) < sum(kb2_ber)/len(kb2_ber))
    ber_ok   = ber_all < SOFT_BER_THRESHOLD
    gate_pass = ber_ok and trend_ok

    lines = [
        f"Checkpoint         : {ckpt_path}",
        f"Test set           : {TEST_N_PER_COND}/condition × 6 = {6*TEST_N_PER_COND} (matches paper)",
        f"Our overall BER    : {ber_all:.4f}",
        f"Our overall MSE    : {mse_all:.4f}",
        f"Soft gate BER<{SOFT_BER_THRESHOLD:.0%} : {'YES' if ber_ok else 'NO'}",
        f"Trend AWGN<KB2     : {'YES' if trend_ok else 'NO'}",
        f"SOFT GATE          : {'PASS' if gate_pass else 'FAIL'}",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open("results/baseline_test_summary.txt", "w") as f:
        f.write(summary + "\n")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
