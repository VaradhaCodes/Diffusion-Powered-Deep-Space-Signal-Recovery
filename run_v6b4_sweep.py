"""V6 Batch 4 — full architecture sweep orchestration.

Runs SB1-SB8 sequentially. Handles adaptive sub-batch selection (SB5/6/7 depend
on SB3/4 results). Updates results/v6b4_sweep_summary.csv and V6_RUN_LOG.md after
every sub-batch.

Usage:
  source .venv/bin/activate && python run_v6b4_sweep.py
"""

import sys, csv, json, time, shutil, math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.competitors import MambaNet2ch, MambaNet2chCfg
from src.models.v5_model import _SNR_MIN, _SNR_RANGE
from src.data_zhu import zhu_test_dataset, TEST_CONDITIONS
from src.train.train_v5 import _calibrate_snr_estimator, estimate_snr
from src.train.train_v6b3 import set_seed, ts
from src.train.train_v6b4 import run_finetune_eval, CANONICAL_PRETRAIN, RESULT_DIR, CKPT_DIR, TEST_N, FT_EPOCHS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SWEEP_CSV  = RESULT_DIR / "v6b4_sweep_summary.csv"
RUN_LOG    = ROOT / "V6_RUN_LOG.md"

SWEEP_COLS = [
    "sb_id", "description", "seed", "params",
    "train_wallclock_min", "val_best_ber",
    "test_ber_overall", "test_ber_awgn_bt03", "test_ber_awgn_bt05",
    "test_ber_kb2_bt03_m12", "test_ber_kb2_bt03_m14",
    "test_ber_kb2_bt05_m12", "test_ber_kb2_bt05_m14",
    "batch_size_used", "grad_checkpoint_used", "notes",
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def mean_test_ber(results: list[dict]) -> float:
    vals = [r["test_ber_overall"] for r in results if r["test_ber_overall"] < 1.0]
    return float(np.mean(vals)) if vals else float("inf")


def std_test_ber(results: list[dict]) -> float:
    vals = [r["test_ber_overall"] for r in results if r["test_ber_overall"] < 1.0]
    return float(np.std(vals)) if len(vals) > 1 else float("inf")


def append_sweep_row(result: dict, description: str):
    exists = SWEEP_CSV.exists()
    with open(SWEEP_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_COLS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({**result, "description": description})


def append_runlog(text: str):
    with open(RUN_LOG, "a") as f:
        f.write(text)


# ── Baseline eval ─────────────────────────────────────────────────────────────

def eval_canonical_checkpoints() -> dict:
    """Evaluate v6b3_canonical_s{0,1,2}.pt on test set. Returns per-cond ensemble BER."""
    print(f"\n{'='*70}")
    print(f"[{now()}] BASELINE EVAL — v6b3_canonical_s{{0,1,2}}.pt")
    print(f"{'='*70}")

    slope, intercept = _calibrate_snr_estimator()
    per_seed = {}

    for seed in [0, 1, 2]:
        ckpt_path = CKPT_DIR / f"v6b3_canonical_s{seed}.pt"
        set_seed(seed)
        model = MambaNet2ch().to(DEVICE)
        ck    = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        model.eval()

        seed_results = {}
        for cond in TEST_CONDITIONS:
            ds  = zhu_test_dataset(cond)
            sub = torch.utils.data.Subset(ds, list(range(TEST_N)))
            tl  = DataLoader(sub, batch_size=256, shuffle=False, num_workers=2)
            n_bits = n_errors = 0
            with torch.no_grad():
                for x, y in tl:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    snr = estimate_snr(x, slope, intercept)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        logits, _ = model(x, snr)
                    preds = (torch.sigmoid(logits.float()) > 0.5).float()
                    n_bits   += y.numel()
                    n_errors += (preds != y).sum().item()
            seed_results[cond] = (n_bits, n_errors)
        per_seed[seed] = seed_results
        overall = np.mean([n_errors / n_bits
                           for n_bits, n_errors in seed_results.values()])
        print(f"  seed={seed}  overall_BER={overall*100:.4f}%")
        torch.cuda.empty_cache()

    # Ensemble
    ens_ber = {}
    total_bits = total_errors = 0
    for cond in TEST_CONDITIONS:
        cb = sum(per_seed[s][cond][0] for s in [0, 1, 2])
        ce = sum(per_seed[s][cond][1] for s in [0, 1, 2])
        ens_ber[cond] = ce / cb
        total_bits   += cb
        total_errors += ce

    ensemble_overall = total_errors / total_bits
    ens_ber["OVERALL"] = ensemble_overall
    print(f"\n  Ensemble 3-seed BER = {ensemble_overall*100:.4f}%")

    # Save CSV
    csv_path = RESULT_DIR / "v6b4_pre_canonical_test.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "ber"])
        for cond in TEST_CONDITIONS:
            w.writerow([cond, round(ens_ber[cond], 6)])
        w.writerow(["OVERALL", round(ensemble_overall, 6)])
    print(f"  Saved {csv_path.name}")

    return ens_ber


# ── Sub-batch runner ──────────────────────────────────────────────────────────

def run_sb(sb_id: str, description: str, seeds: list[int],
           d_model: int = 128, n_blocks: int = 1, cnn_k1: int = 7,
           parallel: bool = False, grad_ckpt: bool = False,
           loss_variant: str = "bce") -> list[dict]:
    """Run a sub-batch for given seeds. Returns list of result dicts."""
    results = []
    for seed in seeds:
        try:
            r = run_finetune_eval(
                sb_id=sb_id, seed=seed,
                d_model=d_model, n_blocks=n_blocks, cnn_k1=cnn_k1,
                parallel=parallel, grad_ckpt=grad_ckpt,
                loss_variant=loss_variant, ft_batch=128,
            )
        except Exception as e:
            print(f"  ERROR in {sb_id} seed={seed}: {e}")
            r = {
                "sb_id": sb_id, "seed": seed, "params": 0,
                "train_wallclock_min": 0, "val_best_ber": float("inf"),
                "test_ber_overall": float("inf"),
                "test_ber_awgn_bt03": float("inf"), "test_ber_awgn_bt05": float("inf"),
                "test_ber_kb2_bt03_m12": float("inf"), "test_ber_kb2_bt03_m14": float("inf"),
                "test_ber_kb2_bt05_m12": float("inf"), "test_ber_kb2_bt05_m14": float("inf"),
                "batch_size_used": 128, "grad_checkpoint_used": grad_ckpt,
                "notes": f"EXCEPTION: {str(e)[:100]}",
            }
        results.append(r)
        append_sweep_row(r, description)
        torch.cuda.empty_cache()

    mean_ber = mean_test_ber(results)
    std_ber  = std_test_ber(results)
    print(f"\n  {sb_id} done: mean_BER={mean_ber*100:.4f}%  std={std_ber*100:.4f}%")
    return results


# ── Candidate gate ────────────────────────────────────────────────────────────

def passes_gates(results: list[dict], pre_ber: float, pre_cond_ber: dict) -> tuple[bool, str]:
    """Check C1-C4. Returns (passes, reason_string)."""
    valid = [r for r in results if r["test_ber_overall"] < 1.0]
    if len(valid) < 2:
        return False, "fewer than 2 valid seeds"

    mean_ber = mean_test_ber(valid)
    std_ber  = std_test_ber(valid)
    params   = valid[0]["params"]

    # C1: mean beats pre_ber by >= 0.05pp
    delta_pp = (pre_ber - mean_ber) * 100
    if delta_pp < 0.05:
        return False, f"C1 FAIL: delta={delta_pp:.3f}pp < 0.05pp"

    # C2: std <= 0.08pp
    if std_ber * 100 > 0.08:
        return False, f"C2 FAIL: std={std_ber*100:.3f}pp > 0.08pp"

    # C3: no per-condition regression > 0.15pp
    ens_per_cond = {}
    for cond in TEST_CONDITIONS:
        vals = [r for r in valid if cond in str(RESULT_DIR / f"v6b4_{r['sb_id']}_s{r['seed']}_test.csv")]
    # Read per-condition from CSV files
    for r in valid:
        csv_p = RESULT_DIR / f"v6b4_{r['sb_id']}_s{r['seed']}_test.csv"
        if not csv_p.exists():
            continue
        with open(csv_p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cond = row["condition"]
                if cond != "OVERALL":
                    ens_per_cond.setdefault(cond, []).append(float(row["ber"]))

    for cond in TEST_CONDITIONS:
        if cond not in ens_per_cond:
            continue
        cond_mean = np.mean(ens_per_cond[cond])
        cond_pre  = pre_cond_ber.get(cond, 0.0)
        regression_pp = (cond_mean - cond_pre) * 100
        if regression_pp > 0.15:
            return False, f"C3 FAIL: {cond} regression={regression_pp:.3f}pp > 0.15pp"

    # C4: params <= 2.5M
    if params > 2_500_000:
        return False, f"C4 FAIL: params={params:,} > 2.5M"

    return True, f"PASSES: delta={delta_pp:.3f}pp std={std_ber*100:.3f}pp params={params:,}"


# ── 3-seed promotion ──────────────────────────────────────────────────────────

def run_third_seed(sb_id: str, description: str, d_model: int, n_blocks: int,
                   cnn_k1: int, parallel: bool, grad_ckpt: bool,
                   loss_variant: str = "bce") -> list[dict]:
    """Run seed=2 for the winner config and return all 3 results."""
    r = run_finetune_eval(
        sb_id=sb_id, seed=2,
        d_model=d_model, n_blocks=n_blocks, cnn_k1=cnn_k1,
        parallel=parallel, grad_ckpt=grad_ckpt, loss_variant=loss_variant,
    )
    append_sweep_row(r, description + " [s2]")
    torch.cuda.empty_cache()
    return r


def compute_3seed_ensemble(sb_id: str) -> tuple[dict, float]:
    """Compute 3-seed ensemble BER from saved CSVs. Returns (per_cond_ber, overall)."""
    ens_bits = {c: 0 for c in TEST_CONDITIONS}
    ens_errs = {c: 0 for c in TEST_CONDITIONS}

    for seed in [0, 1, 2]:
        csv_p = RESULT_DIR / f"v6b4_{sb_id}_s{seed}_test.csv"
        if not csv_p.exists():
            print(f"  WARNING: {csv_p.name} not found for ensemble")
            continue
        with open(csv_p) as f:
            for row in csv.DictReader(f):
                cond = row["condition"]
                if cond in TEST_CONDITIONS:
                    ber   = float(row["ber"])
                    n_bits = TEST_N * 100  # 700 frames × 100 bits
                    ens_bits[cond] += n_bits
                    ens_errs[cond] += int(round(ber * n_bits))

    per_cond = {c: ens_errs[c] / ens_bits[c] for c in TEST_CONDITIONS if ens_bits[c] > 0}
    total_bits  = sum(ens_bits[c] for c in TEST_CONDITIONS if ens_bits[c] > 0)
    total_errs  = sum(ens_errs[c] for c in TEST_CONDITIONS if ens_bits[c] > 0)
    overall = total_errs / total_bits if total_bits > 0 else float("inf")
    per_cond["OVERALL"] = overall

    csv_out = RESULT_DIR / f"v6b4_{sb_id}_ensemble_test.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "ber"])
        for cond in TEST_CONDITIONS:
            w.writerow([cond, round(per_cond.get(cond, float("inf")), 6)])
        w.writerow(["OVERALL", round(overall, 6)])
    print(f"  Ensemble {sb_id}: {overall*100:.4f}%  → {csv_out.name}")
    return per_cond, overall


def paired_t_test(pre_cond_ber: dict, new_cond_ber: dict) -> float:
    """Paired t-test across 6 conditions. Returns p-value."""
    from scipy import stats
    diffs = []
    for cond in TEST_CONDITIONS:
        if cond in pre_cond_ber and cond in new_cond_ber:
            diffs.append(pre_cond_ber[cond] - new_cond_ber[cond])
    if len(diffs) < 2:
        return 1.0
    t_stat, p_val = stats.ttest_1samp(diffs, popmean=0)
    return float(p_val)


# ── Main sweep ────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'#'*70}")
    print(f"# V6 Batch 4 sweep — {now()}")
    print(f"# SNR source: linear (SNR_FIX_STATUS=FAILED)")
    print(f"{'#'*70}")

    # ── Step 0: Baseline eval ──────────────────────────────────────────────────
    print(f"\n[STEP 0] Baseline evaluation of v6b3_canonical_s{{0,1,2}}.pt")
    pre_cond_ber = eval_canonical_checkpoints()
    pre_ber      = pre_cond_ber["OVERALL"]
    pre_ber_pct  = pre_ber * 100
    print(f"\n  PRE_B4_CANONICAL_BER = {pre_ber_pct:.4f}%")

    append_runlog(f"""
---

## Batch 4 preflight (auto-written {now()})

### Q1 — MambaNet2ch exact architecture
**CNN stem (2-channel)**:
- Conv1d(2→32, k=7, pad=3) → BN → GELU
- Conv1d(32→64, k=7, pad=3) → BN → GELU
- Conv1d(64→128, k=8, stride=8) → BN → GELU  [800→100 downsampling]

**MHA block**: d_model=128, num_heads=8, dropout=0.0, batch_first=True
- pre-norm residual: `h = norm1(h + attn(h,h,h)[0])`
- NO FFN (attention only)

**BiMamba2 block**: d_state=64, headdim=64, chunk_size=64
- fwd pass + flip(bwd(flip(h))) — sum merge
- pre-norm residual: `h = norm2(h + bi_m2(h))`

**FiLM**: snr_norm = (snr_db - (-4.0)) / 12.0; MLP(1→64→256) → gamma+beta
- Applied after MHA+BiMamba2 blocks (per-token scale+shift)

**Bit head**: Linear(128, 1) per token
**SNR head**: Linear(128, 1) on mean-pooled h

### Q2 — Param count print
train_competitor.py already prints params at line ~163. build_model() now also
prints params (added in competitors.py). No additional changes needed.

### Q3 — CLI flags in train_competitor.py
Added: --depth, --width, --kernel-size, --block-type, --loss
(All map to mambanet_2ch_cfg architecture kwargs.)

### SNR source
SNR_FIX_STATUS=FAILED → --snr-source=linear for all fine-tunes.

### PRE_B4_CANONICAL_BER = {pre_ber_pct:.4f}%
""")

    # ── SB1: LR warmup only ────────────────────────────────────────────────────
    print(f"\n[SB1] LR warmup  d=128 n=1 k=7 serial")
    sb1 = run_sb("sb1", "warmup_only", [0, 1],
                 d_model=128, n_blocks=1, cnn_k1=7, parallel=False, grad_ckpt=False)
    sb1_mean = mean_test_ber(sb1)
    append_runlog(f"""
### SB1 — LR warmup
seed0={sb1[0]['test_ber_overall']*100:.4f}%  seed1={sb1[1]['test_ber_overall']*100:.4f}%
mean={sb1_mean*100:.4f}%  delta_vs_pre={(pre_ber-sb1_mean)*100:.4f}pp
""")

    # ── SB2: wider first CNN kernel ────────────────────────────────────────────
    print(f"\n[SB2] Wider CNN kernel k=31  d=128 n=1 serial")
    sb2 = run_sb("sb2", "wider_cnn_k31", [0, 1],
                 d_model=128, n_blocks=1, cnn_k1=31, parallel=False, grad_ckpt=False)
    sb2_mean = mean_test_ber(sb2)
    append_runlog(f"""
### SB2 — Wider CNN k=31
seed0={sb2[0]['test_ber_overall']*100:.4f}%  seed1={sb2[1]['test_ber_overall']*100:.4f}%
mean={sb2_mean*100:.4f}%  delta_vs_pre={(pre_ber-sb2_mean)*100:.4f}pp
""")

    # ── SB3: depth 2 ──────────────────────────────────────────────────────────
    print(f"\n[SB3] Depth 2x  d=128 n=2 k=31 serial")
    sb3 = run_sb("sb3", "depth2_k31", [0, 1],
                 d_model=128, n_blocks=2, cnn_k1=31, parallel=False, grad_ckpt=False)
    sb3_mean = mean_test_ber(sb3)
    append_runlog(f"""
### SB3 — Depth 2x
seed0={sb3[0]['test_ber_overall']*100:.4f}%  seed1={sb3[1]['test_ber_overall']*100:.4f}%
mean={sb3_mean*100:.4f}%  delta_vs_pre={(pre_ber-sb3_mean)*100:.4f}pp
""")

    # ── SB4: depth 4 ──────────────────────────────────────────────────────────
    print(f"\n[SB4] Depth 4x  d=128 n=4 k=31 serial  [grad_ckpt proactive]")
    sb4 = run_sb("sb4", "depth4_k31", [0, 1],
                 d_model=128, n_blocks=4, cnn_k1=31, parallel=False, grad_ckpt=True)
    sb4_mean = mean_test_ber(sb4)
    append_runlog(f"""
### SB4 — Depth 4x (grad_ckpt)
seed0={sb4[0]['test_ber_overall']*100:.4f}%  seed1={sb4[1]['test_ber_overall']*100:.4f}%
mean={sb4_mean*100:.4f}%  delta_vs_pre={(pre_ber-sb4_mean)*100:.4f}pp
""")

    # ── Depth decision for SB5/6/7 ────────────────────────────────────────────
    best_depth = 2 if sb3_mean <= sb4_mean else 4
    best_depth_mean = min(sb3_mean, sb4_mean)
    print(f"\n  Depth decision: SB3_mean={sb3_mean*100:.4f}% SB4_mean={sb4_mean*100:.4f}% → best_depth={best_depth}")

    # Check if best depth is even valid (not diverged)
    if best_depth_mean > 0.5:
        print(f"  WARNING: best_depth BER={best_depth_mean*100:.4f}% > 50% — both depth configs failed. Using depth=1.")
        best_depth = 1
    append_runlog(f"\n**Depth decision**: SB3={sb3_mean*100:.4f}% SB4={sb4_mean*100:.4f}% → best_depth={best_depth}\n")

    # ── SB5: width 192 ────────────────────────────────────────────────────────
    print(f"\n[SB5] Width 192  d=192 n={best_depth} k=31 serial")
    sb5 = run_sb("sb5", f"width192_depth{best_depth}", [0, 1],
                 d_model=192, n_blocks=best_depth, cnn_k1=31, parallel=False, grad_ckpt=False)
    sb5_mean = mean_test_ber(sb5)
    append_runlog(f"""
### SB5 — Width 192 depth={best_depth}
seed0={sb5[0]['test_ber_overall']*100:.4f}%  seed1={sb5[1]['test_ber_overall']*100:.4f}%
mean={sb5_mean*100:.4f}%  delta_vs_pre={(pre_ber-sb5_mean)*100:.4f}pp
""")

    # ── SB6: width 256 ────────────────────────────────────────────────────────
    print(f"\n[SB6] Width 256  d=256 n={best_depth} k=31 serial  [grad_ckpt proactive]")
    sb6 = run_sb("sb6", f"width256_depth{best_depth}", [0, 1],
                 d_model=256, n_blocks=best_depth, cnn_k1=31, parallel=False, grad_ckpt=True)
    sb6_mean = mean_test_ber(sb6)
    append_runlog(f"""
### SB6 — Width 256 depth={best_depth} (grad_ckpt)
seed0={sb6[0]['test_ber_overall']*100:.4f}%  seed1={sb6[1]['test_ber_overall']*100:.4f}%
mean={sb6_mean*100:.4f}%  delta_vs_pre={(pre_ber-sb6_mean)*100:.4f}pp
""")

    # Width decision for SB7
    sb7_d_model = 192 if sb5_mean <= sb6_mean else 256
    sb7_d_winner_mean = min(sb5_mean, sb6_mean)
    if sb7_d_winner_mean > 0.5:
        sb7_d_model = 128  # fallback
    print(f"\n  Width decision: SB5={sb5_mean*100:.4f}% SB6={sb6_mean*100:.4f}% → sb7_d_model={sb7_d_model}")
    append_runlog(f"\n**Width decision**: SB5={sb5_mean*100:.4f}% SB6={sb6_mean*100:.4f}% → sb7_d_model={sb7_d_model}\n")

    # ── SB7: parallel Mcformer-style ──────────────────────────────────────────
    print(f"\n[SB7] Parallel Mcformer  d={sb7_d_model} n={best_depth} k=31 parallel")
    sb7_gc = (sb7_d_model >= 192 and best_depth >= 2)
    sb7 = run_sb("sb7", f"parallel_d{sb7_d_model}_depth{best_depth}", [0, 1],
                 d_model=sb7_d_model, n_blocks=best_depth, cnn_k1=31,
                 parallel=True, grad_ckpt=sb7_gc)
    sb7_mean = mean_test_ber(sb7)
    append_runlog(f"""
### SB7 — Parallel Mcformer d={sb7_d_model} depth={best_depth}
seed0={sb7[0]['test_ber_overall']*100:.4f}%  seed1={sb7[1]['test_ber_overall']*100:.4f}%
mean={sb7_mean*100:.4f}%  delta_vs_pre={(pre_ber-sb7_mean)*100:.4f}pp
""")

    # ── Winner selection ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"[{now()}] WINNER SELECTION")
    print(f"{'='*70}")

    sb_registry = {
        "sb1": {"results": sb1, "d_model": 128, "n_blocks": 1,   "cnn_k1": 7,  "parallel": False, "grad_ckpt": False},
        "sb2": {"results": sb2, "d_model": 128, "n_blocks": 1,   "cnn_k1": 31, "parallel": False, "grad_ckpt": False},
        "sb3": {"results": sb3, "d_model": 128, "n_blocks": 2,   "cnn_k1": 31, "parallel": False, "grad_ckpt": False},
        "sb4": {"results": sb4, "d_model": 128, "n_blocks": 4,   "cnn_k1": 31, "parallel": False, "grad_ckpt": True},
        "sb5": {"results": sb5, "d_model": 192, "n_blocks": best_depth, "cnn_k1": 31, "parallel": False, "grad_ckpt": False},
        "sb6": {"results": sb6, "d_model": 256, "n_blocks": best_depth, "cnn_k1": 31, "parallel": False, "grad_ckpt": True},
        "sb7": {"results": sb7, "d_model": sb7_d_model, "n_blocks": best_depth, "cnn_k1": 31, "parallel": True, "grad_ckpt": sb7_gc},
    }

    candidates = []
    for sb_id, cfg in sb_registry.items():
        passes, reason = passes_gates(cfg["results"], pre_ber, pre_cond_ber)
        mean_ber = mean_test_ber(cfg["results"])
        params   = cfg["results"][0]["params"] if cfg["results"] else 0
        print(f"  {sb_id}: mean={mean_ber*100:.4f}% | {reason}")
        if passes:
            candidates.append((sb_id, mean_ber, params, cfg))

    winner_id = None
    winner_cfg = None
    winner_mean = None

    if not candidates:
        print("\n  NO sub-batch passed gates C1-C4.")
        print("  V6 Batch4: no architectural improvement found. V6 canonical stays as v6b3.")
        append_runlog(f"""
### Winner selection
No sub-batch passed all gates (C1-C4). Architecture ceiling finding.
V6 canonical remains v6b3 (BER={pre_ber_pct:.4f}%).
""")
    else:
        # Sort by mean BER, tiebreak by params, secondary tiebreak by kb2 m=1.4 mean
        def sort_key(item):
            sb_id, mean_ber, params, cfg = item
            kb2_m14 = np.mean([r.get("test_ber_kb2_bt03_m14", 1.0) + r.get("test_ber_kb2_bt05_m14", 1.0)
                                for r in cfg["results"] if r["test_ber_overall"] < 1.0])
            return (round(mean_ber * 10000), params, kb2_m14)

        candidates.sort(key=sort_key)
        winner_id, winner_mean, winner_params, winner_cfg = candidates[0]

        # Check tiebreak: within 0.03pp → prefer smaller params
        top_ber = candidates[0][1]
        within_tie = [c for c in candidates if abs(c[1] - top_ber) * 100 <= 0.03]
        if len(within_tie) > 1:
            within_tie.sort(key=lambda c: c[2])  # smallest params wins
            winner_id, winner_mean, winner_params, winner_cfg = within_tie[0]

        print(f"\n  WINNER: {winner_id}  mean={winner_mean*100:.4f}%  params={winner_params:,}")
        append_runlog(f"""
### Winner selection
Winner: {winner_id}  mean_BER={winner_mean*100:.4f}%  delta_vs_pre={(pre_ber-winner_mean)*100:.4f}pp
Params: {winner_params:,}
""")

    # ── Promotion to 3-seed ────────────────────────────────────────────────────
    if winner_id is None:
        append_runlog("\n### 3-seed promotion\nSKIPPED (no winner).\n")
        # Still run SB8 at baseline arch
        winner_id_for_sb8 = "sb1"  # warmup baseline as SB8 base
        winner_cfg_for_sb8 = sb_registry["sb1"]
    else:
        # Check if we already have seed=2 result from sweep (we don't — sweep ran seeds 0,1)
        print(f"\n[PROMOTE] Running seed=2 for winner={winner_id}")
        r2 = run_third_seed(
            winner_id, f"{winner_id}_3seed",
            d_model=winner_cfg["d_model"], n_blocks=winner_cfg["n_blocks"],
            cnn_k1=winner_cfg["cnn_k1"], parallel=winner_cfg["parallel"],
            grad_ckpt=winner_cfg["grad_ckpt"],
        )

        # Compute 3-seed ensemble
        mid_ens_cond, mid_ens_overall = compute_3seed_ensemble(winner_id)
        print(f"  Mid-canonical ensemble BER = {mid_ens_overall*100:.4f}%")

        # Save as mid-canonical checkpoints
        for seed in [0, 1, 2]:
            src = CKPT_DIR / f"v6b4_{winner_id}_s{seed}_ft.pt"
            dst = CKPT_DIR / f"v6b4_mid_canonical_s{seed}.pt"
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Copied {src.name} → {dst.name}")

        # Paired t-test vs pre-Batch-4 canonical
        p_val = paired_t_test(pre_cond_ber, mid_ens_cond)
        improvement_pp = (pre_ber - mid_ens_overall) * 100
        print(f"  Improvement vs pre-B4: {improvement_pp:.4f}pp  p={p_val:.4f}")

        append_runlog(f"""
### 3-seed promotion
Winner seed=2 BER={r2['test_ber_overall']*100:.4f}%
3-seed ensemble BER = {mid_ens_overall*100:.4f}%
Improvement vs pre-B4 canonical: {improvement_pp:.4f}pp
Paired t-test (n=6 conditions) p={p_val:.4f}
""")

        if improvement_pp < 0.05 or p_val > 0.10:
            print(f"  Improvement {improvement_pp:.4f}pp < 0.05pp OR p={p_val:.4f} > 0.10 → NO-OP")
            print(f"  v6b3_canonical stays as V6 final. Running SB8 at mid-canonical arch anyway.")
            append_runlog("Decision: NO-OP (insufficient improvement). v6b3 stays final.\n")
            winner_id_for_sb8 = winner_id
            winner_cfg_for_sb8 = winner_cfg
        else:
            print(f"  Promotion ACCEPTED. v6b4_mid_canonical is new base.")
            append_runlog("Decision: PROMOTED. v6b4_mid_canonical is new base.\n")
            winner_id_for_sb8 = winner_id
            winner_cfg_for_sb8 = winner_cfg

    # ── SB8: loss function ablation ────────────────────────────────────────────
    print(f"\n[SB8] Loss ablation at winner arch (seed=0 only)")
    sb8_cfg = winner_cfg_for_sb8

    # Determine base checkpoint for SB8
    # If we promoted to 3-seed, use mid_canonical s0; otherwise use winner s0
    sb8_base_ckpt = CKPT_DIR / f"v6b4_{winner_id_for_sb8}_s0_ft.pt"
    # But we retrain from canonical_pretrain always
    sb8_results = {}
    for variant in ["bce", "bce_ls", "focal", "focal_ls"]:
        print(f"\n  SB8 variant={variant}")
        sb8_tag = f"sb8_{variant}"
        try:
            r = run_finetune_eval(
                sb_id=sb8_tag, seed=0,
                d_model=sb8_cfg["d_model"], n_blocks=sb8_cfg["n_blocks"],
                cnn_k1=sb8_cfg["cnn_k1"], parallel=sb8_cfg["parallel"],
                grad_ckpt=sb8_cfg["grad_ckpt"], loss_variant=variant,
            )
        except Exception as e:
            print(f"  SB8 {variant} FAILED: {e}")
            r = {"sb_id": sb8_tag, "seed": 0, "test_ber_overall": float("inf"),
                 "params": 0, "train_wallclock_min": 0, "val_best_ber": float("inf"),
                 "test_ber_awgn_bt03": float("inf"), "test_ber_awgn_bt05": float("inf"),
                 "test_ber_kb2_bt03_m12": float("inf"), "test_ber_kb2_bt03_m14": float("inf"),
                 "test_ber_kb2_bt05_m12": float("inf"), "test_ber_kb2_bt05_m14": float("inf"),
                 "batch_size_used": 128, "grad_checkpoint_used": False, "notes": f"EXCEPTION: {e}"}
        sb8_results[variant] = r
        append_sweep_row(r, f"sb8_{variant}")
        torch.cuda.empty_cache()

    sb8_best_variant = min(sb8_results, key=lambda v: sb8_results[v]["test_ber_overall"])
    sb8_best_ber     = sb8_results[sb8_best_variant]["test_ber_overall"]
    sb8_bce_ber      = sb8_results["bce"]["test_ber_overall"]

    # If all within 0.03pp of each other: pick (a) bce
    all_valid = [v for v in sb8_results if sb8_results[v]["test_ber_overall"] < 1.0]
    if all_valid:
        sb8_min = min(sb8_results[v]["test_ber_overall"] for v in all_valid)
        sb8_max = max(sb8_results[v]["test_ber_overall"] for v in all_valid)
        if (sb8_max - sb8_min) * 100 <= 0.03:
            sb8_best_variant = "bce"
            sb8_best_ber     = sb8_results["bce"]["test_ber_overall"]
            print(f"  SB8: all variants within 0.03pp → using plain BCE")

    print(f"\n  SB8 winner: {sb8_best_variant}  BER={sb8_best_ber*100:.4f}%")
    append_runlog(f"""
### SB8 — Loss ablation
| variant | seed | BER |
|---------|------|-----|
""")
    for variant in ["bce", "bce_ls", "focal", "focal_ls"]:
        r = sb8_results[variant]
        append_runlog(f"| {variant} | 0 | {r['test_ber_overall']*100:.4f}% |\n")
    append_runlog(f"\n**SB8 winner**: {sb8_best_variant} (BER={sb8_best_ber*100:.4f}%)\n")

    # If SB8 winner is not BCE: run seeds 1 and 2
    final_sb_id = winner_id_for_sb8 if winner_id else "sb1"
    if sb8_best_variant != "bce":
        print(f"\n  SB8 winner is not BCE — running seeds 1, 2 with {sb8_best_variant}")
        for seed in [1, 2]:
            try:
                r = run_finetune_eval(
                    sb_id=f"sb8_{sb8_best_variant}", seed=seed,
                    d_model=sb8_cfg["d_model"], n_blocks=sb8_cfg["n_blocks"],
                    cnn_k1=sb8_cfg["cnn_k1"], parallel=sb8_cfg["parallel"],
                    grad_ckpt=sb8_cfg["grad_ckpt"], loss_variant=sb8_best_variant,
                )
            except Exception as e:
                print(f"  SB8 {sb8_best_variant} s{seed} FAILED: {e}")
                r = None
            if r:
                append_sweep_row(r, f"sb8_{sb8_best_variant}_full")
                torch.cuda.empty_cache()
        final_sb_id = f"sb8_{sb8_best_variant}"
        # Copy to v6b4_final_s{0,1,2}.pt
        for seed in [0, 1, 2]:
            src = CKPT_DIR / f"v6b4_sb8_{sb8_best_variant}_s{seed}_ft.pt"
            dst = CKPT_DIR / f"v6b4_final_s{seed}.pt"
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  {src.name} → {dst.name}")
        _, final_ensemble_ber = compute_3seed_ensemble(f"sb8_{sb8_best_variant}")
    else:
        # Use mid_canonical (winner arch, bce loss)
        if winner_id is not None:
            final_sb_id = winner_id
            for seed in [0, 1, 2]:
                src = CKPT_DIR / f"v6b4_mid_canonical_s{seed}.pt"
                dst = CKPT_DIR / f"v6b4_final_s{seed}.pt"
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f"  {src.name} → {dst.name}")
            _, final_ensemble_ber = compute_3seed_ensemble(winner_id)
        else:
            # No improvement found anywhere: copy v6b3_canonical as v6b4_final
            final_ensemble_ber = pre_ber
            for seed in [0, 1, 2]:
                src = CKPT_DIR / f"v6b3_canonical_s{seed}.pt"
                dst = CKPT_DIR / f"v6b4_final_s{seed}.pt"
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f"  No-op copy: {src.name} → {dst.name}")

    # ── Copy final to v6_final_s{0,1,2}.pt ─────────────────────────────────────
    for seed in [0, 1, 2]:
        src = CKPT_DIR / f"v6b4_final_s{seed}.pt"
        dst = CKPT_DIR / f"v6_final_s{seed}.pt"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  {src.name} → {dst.name}")

    final_ber_pct = final_ensemble_ber * 100

    # ── Final log entry ────────────────────────────────────────────────────────
    summary_lines = "\n".join([
        f"| {sb_id} | {mean_test_ber(cfg['results'])*100:.4f}% |"
        for sb_id, cfg in sb_registry.items()
    ])

    append_runlog(f"""
### Final promotion
V6_FINAL_BER = {final_ber_pct:.4f}%  (config: {winner_id or 'v6b3_canonical_no_change'})
PRE_B4_CANONICAL_BER = {pre_ber_pct:.4f}%
Delta = {(pre_ber - final_ensemble_ber)*100:.4f}pp
Checkpoints: v6_final_s{{0,1,2}}.pt

### Sweep summary (2-seed mean BER)
| sb | mean_BER |
|----|---------|
{summary_lines}
""")

    print(f"\n{'#'*70}")
    print(f"# V6 Batch 4 COMPLETE — {now()}")
    print(f"# V6_FINAL_BER = {final_ber_pct:.4f}%")
    print(f"# PRE_B4_CANONICAL_BER = {pre_ber_pct:.4f}%")
    print(f"# Delta = {(pre_ber - final_ensemble_ber)*100:.4f}pp")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
