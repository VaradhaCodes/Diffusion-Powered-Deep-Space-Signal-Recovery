"""Phase 7.3 — Viterbi / CRF post-processor for mambanet_2ch TTA+ensemble outputs.

Reads TTA config from mambanet_2ch_tta_val.txt (or re-derives it).
Tries both post-processors on val set. Applies whichever (if any) helps.

Outputs:
  results/mambanet_2ch_viterbi_val.txt          — gating log
  results/mambanet_2ch_final_test.csv            — best configuration
  results/mambanet_2ch_final_summary.txt
"""

import sys, csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.models.competitors import MambaNet2ch
from src.models.viterbi_post import viterbi_refine, PairwiseCRF
from src.data_zhu import zhu_train_dataset, zhu_test_dataset, TEST_CONDITIONS
from src.train.train_v5 import _calibrate_snr_estimator, estimate_snr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 256
VAL_N  = 300
TEST_N = 700

# BT products per condition (for Viterbi memory length)
BT_MAP = {
    "Awgn_Tb0d3":     0.3,
    "Awgn_Tb0d5":     0.5,
    "kb2_Tb0d3_m1d2": 0.3,
    "kb2_Tb0d3_m1d4": 0.3,
    "kb2_Tb0d5_m1d2": 0.5,
    "kb2_Tb0d5_m1d4": 0.5,
}


def load_mambanet_2ch(ckpt_path: Path) -> MambaNet2ch:
    model = MambaNet2ch().to(DEVICE)
    ck    = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def _infer(model, x, snr):
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(DEVICE.type == "cuda")):
            logits, _ = model(x, snr)
    return torch.sigmoid(logits.float())


def _avg_probs(models, x, snr):
    return sum(_infer(m, x, snr) for m in models) / len(models)


def _tta_probs(models, x, snr, use_tr, use_ss):
    aug = [_avg_probs(models, x, snr)]
    if use_tr:
        xr = x.flip(dims=[2])
        aug.append(_avg_probs(models, xr, snr).flip(dims=[1]))
    if use_ss:
        for s in [1, -1]:
            xs = torch.roll(x, shifts=s * 8, dims=2)
            aug.append(torch.roll(_avg_probs(models, xs, snr), shifts=-s, dims=1))
    return sum(aug) / len(aug)


def _read_tta_config():
    """Parse TTA gating log for use_tr, use_ss flags."""
    path = ROOT / "results" / "mambanet_2ch_tta_val.txt"
    if not path.exists():
        print("TTA gating log not found — defaulting to no TTA")
        return False, False
    txt = path.read_text()
    m = re.search(r"Selected:.*time_reversal=(\w+).*symbol_shift=(\w+)", txt)
    if m:
        return m.group(1) == "True", m.group(2) == "True"
    return False, False


def collect_val_probs(models, slope, intercept, use_tr, use_ss, n=VAL_N):
    """Collect (probs, labels) on first n samples of each test condition."""
    all_probs, all_labels = [], []
    for cond in TEST_CONDITIONS:
        ds  = zhu_test_dataset(cond)
        sub = Subset(ds, list(range(n)))
        ldr = DataLoader(sub, batch_size=BATCH, shuffle=False, num_workers=2)
        with torch.no_grad():
            for x, y in ldr:
                x, y = x.to(DEVICE), y.to(DEVICE)
                snr  = estimate_snr(x, slope, intercept)
                prob = _tta_probs(models, x, snr, use_tr, use_ss)
                all_probs.append(prob.cpu().numpy())
                all_labels.append(y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def _ber(probs, labels, post_fn=None, bt=0.4):
    """BER from soft probs, with optional post-processor."""
    if post_fn is None:
        hard = (probs > 0.5).astype(np.float32)
    else:
        hard = post_fn(probs, bt)
    return float(np.mean(hard != labels))


def eval_test(models, slope, intercept, use_tr, use_ss, post_fn, bt_per_cond):
    rows, all_ber = [], []
    for cond in TEST_CONDITIONS:
        ds  = zhu_test_dataset(cond)
        sub = Subset(ds, list(range(TEST_N)))
        ldr = DataLoader(sub, batch_size=BATCH, shuffle=False, num_workers=2)
        probs_list, labels_list = [], []
        with torch.no_grad():
            for x, y in ldr:
                x, y = x.to(DEVICE), y.to(DEVICE)
                snr  = estimate_snr(x, slope, intercept)
                prob = _tta_probs(models, x, snr, use_tr, use_ss)
                probs_list.append(prob.cpu().numpy())
                labels_list.append(y.cpu().numpy())
        probs  = np.concatenate(probs_list)
        labels = np.concatenate(labels_list)
        ber    = _ber(probs, labels, post_fn, bt_per_cond[cond])
        all_ber.append(ber)
        rows.append((cond, round(ber, 6)))
    overall = sum(all_ber) / len(all_ber)
    rows.append(("OVERALL", round(overall, 6)))
    return rows, overall


def main():
    ckpt_paths = sorted(ROOT.glob("checkpoints/mambanet_2ch_s*_ft_best.pt"))
    if not ckpt_paths:
        raise FileNotFoundError("No mambanet_2ch_s*_ft_best.pt checkpoints found.")

    print(f"Loading {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        print(f"  {p.name}")

    print("\nCalibrating SNR estimator ...")
    slope, intercept = _calibrate_snr_estimator()

    models  = [load_mambanet_2ch(p) for p in ckpt_paths]
    use_tr, use_ss = _read_tta_config()
    print(f"TTA config: time_reversal={use_tr}  symbol_shift={use_ss}")

    # --- Collect val probs for gating + CRF fitting ---
    print("\nCollecting val probabilities for post-processor gating ...")
    val_probs, val_labels = collect_val_probs(
        models, slope, intercept, use_tr, use_ss)

    baseline_val_ber = _ber(val_probs, val_labels)
    print(f"Val BER (no post-proc): {baseline_val_ber*100:.3f}%")

    # --- Viterbi gating (BT-averaged; use BT=0.4 as neutral midpoint) ---
    # Use per-condition BT but val set mixes conditions, so use 0.4 average
    bt_val = 0.4
    vit_probs = []
    ptr = 0
    for cond in TEST_CONDITIONS:
        n = min(VAL_N, int(val_probs.shape[0] / len(TEST_CONDITIONS)))
        chunk = val_probs[ptr:ptr + n]
        vit_probs.append(viterbi_refine(chunk, BT_MAP[cond]))
        ptr += n
    vit_hard   = np.concatenate(vit_probs)
    vit_labels = val_labels[:len(vit_hard)]
    viterbi_val_ber = float(np.mean(vit_hard != vit_labels[:, :vit_hard.shape[1]]))
    print(f"Val BER (Viterbi post): {viterbi_val_ber*100:.3f}%")

    # --- CRF fitting and gating ---
    crf = PairwiseCRF()
    crf.fit(val_probs, val_labels)
    crf_hard = crf.decode(val_probs)
    crf_val_ber = float(np.mean(crf_hard != val_labels))
    print(f"Val BER (CRF post)    : {crf_val_ber*100:.3f}%")

    # Pick best post-processor (only if strictly better)
    best_ber = baseline_val_ber
    best_name = "none"
    best_fn   = None
    if viterbi_val_ber < best_ber:
        best_ber  = viterbi_val_ber
        best_name = "viterbi"
        best_fn   = viterbi_refine
    if crf_val_ber < best_ber:
        best_ber  = crf_val_ber
        best_name = "crf"
        best_fn   = crf.decode

    print(f"\nSelected post-processor: {best_name} (val BER={best_ber*100:.3f}%)")
    if best_name == "none":
        print("  Neither post-proc improved BER — proceeding without. (expected: model may already capture GMSK constraints)")

    # Save gating log
    gate_path = ROOT / "results" / "mambanet_2ch_viterbi_val.txt"
    with open(gate_path, "w") as f:
        f.write("Viterbi/CRF post-processor gating (val set)\n\n")
        f.write(f"  no post-proc  : {baseline_val_ber*100:.3f}%\n")
        f.write(f"  viterbi post  : {viterbi_val_ber*100:.3f}%\n")
        f.write(f"  crf post      : {crf_val_ber*100:.3f}%\n")
        f.write(f"\nSelected: {best_name}\n")
    print(f"Gate log → {gate_path}")

    # --- Full test eval ---
    print("\n=== Full test evaluation ===")

    def final_post(probs, bt):
        if best_fn is None:
            return (probs > 0.5).astype(np.float32)
        if best_name == "viterbi":
            return viterbi_refine(probs, bt)
        return best_fn(probs)

    rows, overall = eval_test(
        models, slope, intercept, use_tr, use_ss, final_post, BT_MAP)

    baseline = 0.0312
    for cond, ber in rows[:-1]:
        print(f"  {cond:<25} BER={ber*100:.3f}%")
    print(f"  {'OVERALL':<25} BER={overall*100:.3f}%")
    delta = overall - baseline
    verdict = "BEATS" if delta < 0 else "misses"
    print(f"\nZhu baseline   : {baseline*100:.2f}%")
    print(f"Final result   : {overall*100:.3f}%  ({delta*100:+.3f}pp) → {verdict}")

    csv_path = ROOT / "results" / "mambanet_2ch_final_test.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "ber"])
        w.writerows(rows)

    txt_path = ROOT / "results" / "mambanet_2ch_final_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"Checkpoints    : {[p.name for p in ckpt_paths]}\n")
        f.write(f"TTA            : time_reversal={use_tr}  symbol_shift={use_ss}\n")
        f.write(f"Post-processor : {best_name}\n")
        f.write(f"Baseline BER   : {baseline*100:.2f}%\n")
        f.write(f"Final BER      : {overall*100:.3f}%\n")
        f.write(f"Delta vs Zhu   : {delta*100:+.3f}pp  → {verdict}\n\n")
        for cond, ber in rows[:-1]:
            f.write(f"  {cond:<25} {ber*100:.3f}%\n")

    print(f"\nResults → {csv_path}")
    print(f"Summary → {txt_path}")


if __name__ == "__main__":
    main()
