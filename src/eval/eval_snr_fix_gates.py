"""V6 Batch 2 Part C5 — SNR fix gate evaluation.

Evaluates v6b2_mambanet_snrfix_s1.pt against Zhu test set.
Compares to V5 mambanet_2ch_s1 reference (2.323% per-seed, condition BERs from CSV).

Gates:
  G6: Overall BER improvement >= 0.05 pp  (target <= 2.273%)
  G7: KB2 m=1.4 improvement >= 0.10 pp   (target <= 3.791%)
  G8: No condition regresses > 0.10 pp vs V5 seed-1

If G6+G7+G8 pass: promotes to v6b2_canonical_s1.pt, queues seeds 0 and 2.
"""

import sys, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.competitors import build_model
from src.data_zhu import zhu_test_dataset, TEST_CONDITIONS
from src.infer.snr_helper import estimate_snr_db

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = ROOT / "checkpoints"
RES_DIR  = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)

NEW_CKPT = CKPT_DIR / "mambanet_2ch_s1_ft_best.pt"   # C4 output
NEURAL_CKPT = CKPT_DIR / "v6b2_snr_estimator.pt"
TEST_N   = 700

# V5 seed-1 reference BERs.
# Note: mambanet_2ch_s1_test.csv was overwritten by the C2 plumbing test.
# Spec states overall=2.323%, KB2 m=1.4 avg=3.891% for seed-1.
# Per-condition proxy: use mambanet_2ch_s2_test.csv (closest match to spec's 3.891%).
V5_S1_OVERALL = 0.02323
V5_S1_KB2_M14 = 0.03891   # from spec

V5_S1_PERCON_PROXY = {
    "Awgn_Tb0d3":       0.010829,
    "Awgn_Tb0d5":       0.012186,
    "kb2_Tb0d3_m1d2":   0.019157,
    "kb2_Tb0d3_m1d4":   0.038829,
    "kb2_Tb0d5_m1d2":   0.019143,
    "kb2_Tb0d5_m1d4":   0.039014,
    "OVERALL":          0.02323,   # from spec (not s2 actual)
}


def load_v5_ref():
    # Original s1 CSV was overwritten — use spec values as authoritative
    return dict(V5_S1_PERCON_PROXY)


def eval_model(ckpt_path):
    model = build_model("mambanet_2ch").to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()

    results = {}
    for cond in TEST_CONDITIONS:
        ds  = zhu_test_dataset(cond)
        sub = torch.utils.data.Subset(ds, list(range(TEST_N)))
        tl  = DataLoader(sub, batch_size=256, shuffle=False, num_workers=2)
        ber_sum = n = 0
        with torch.no_grad():
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                snr = estimate_snr_db(x, NEURAL_CKPT, DEVICE).clamp(-4.0, 8.0)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    logits, _ = model(x, snr)
                ber = ((torch.sigmoid(logits.float()) > 0.5).float() != y).float().mean().item()
                ber_sum += ber * len(x); n += len(x)
        results[cond] = ber_sum / n
        print(f"  {cond:<25} BER={results[cond]*100:.4f}%")
    results["OVERALL"] = float(np.mean(list(results.values())))
    print(f"  {'OVERALL':<25} BER={results['OVERALL']*100:.4f}%")
    return results


def main():
    v5_ref = load_v5_ref()

    print(f"\nEvaluating {NEW_CKPT.name} with neural SNR...")
    new_results = eval_model(NEW_CKPT)

    # Save results CSV
    csv_path = RES_DIR / "v6b2_mambanet_snrfix_s1_test.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition","ber"])
        for cond, ber in new_results.items():
            w.writerow([cond, round(ber, 6)])
    print(f"\nResults saved: {csv_path}")

    # Gate evaluation
    overall_new = new_results["OVERALL"]
    overall_v5  = v5_ref.get("OVERALL", V5_S1_OVERALL)

    g6_delta = overall_v5 - overall_new
    g6_pass = g6_delta >= 0.0005   # 0.05 pp
    print(f"\nG6: V5={overall_v5*100:.4f}% → new={overall_new*100:.4f}% delta={g6_delta*100:.4f}pp  {'PASS' if g6_pass else 'FAIL'}")

    # G7: KB2 m=1.4 improvement (average of both BT conditions)
    kb2_m14_conds = [c for c in TEST_CONDITIONS if "m1d4" in c]
    kb2_m14_new = np.mean([new_results[c] for c in kb2_m14_conds])
    kb2_m14_v5  = np.mean([v5_ref.get(c, 0.03891) for c in kb2_m14_conds]) if v5_ref else 0.03891
    g7_delta = kb2_m14_v5 - kb2_m14_new
    g7_pass = g7_delta >= 0.001   # 0.10 pp
    print(f"G7: KB2 m=1.4 V5={kb2_m14_v5*100:.4f}% → new={kb2_m14_new*100:.4f}% delta={g7_delta*100:.4f}pp  {'PASS' if g7_pass else 'FAIL'}")

    # G8: No condition regresses > 0.10 pp
    g8_pass = True
    worst_regression = 0.0
    for cond in TEST_CONDITIONS:
        if cond not in v5_ref:
            continue
        regression = new_results[cond] - v5_ref[cond]
        if regression > worst_regression:
            worst_regression = regression
        if regression > 0.001:   # > 0.10 pp regression
            g8_pass = False
            print(f"  G8 FAIL: {cond} regressed by {regression*100:.4f}pp")
    print(f"G8: Max regression={worst_regression*100:.4f}pp  {'PASS' if g8_pass else 'FAIL'}")

    all_pass = g6_pass and g7_pass and g8_pass
    print(f"\nOverall: G6={'P' if g6_pass else 'F'} G7={'P' if g7_pass else 'F'} G8={'P' if g8_pass else 'F'}")

    if all_pass:
        # Promote to canonical
        import shutil
        canonical = CKPT_DIR / "v6b2_canonical_s1.pt"
        shutil.copy2(NEW_CKPT, canonical)
        print(f"\nSNR_FIX_STATUS=INTEGRATED")
        print(f"Promoted to: {canonical}")
        print(f"\nNext: run seeds 0 and 2 with same recipe.")
    else:
        print(f"\nSNR_FIX_STATUS=gates failed — check debug notes in V6_RUN_LOG.md")

    return all_pass, overall_new


if __name__ == "__main__":
    main()
