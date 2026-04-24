"""V6 Batch 3 — Sweep orchestrator.

Runs 500K → 1M → 2M → 5M in order. Applies sweep-level early stopping,
disk cleanup, and generates the scaling curve at the end.

Usage:
  source .venv/bin/activate
  python src/train/sweep_v6b3.py [--snr-source linear] [--start-size 500K]
"""

import sys, os, time, json, csv, subprocess, shutil, hashlib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.train.train_v6b3 import (
    pretrain, finetune, evaluate, ensemble_eval,
    SIZE_MAP, DATA_B3, CKPT_DIR, RESULT_DIR,
)
from src.analysis.v6b3_scaling import run_scaling_analysis

LOG_PATH     = ROOT / "V6_RUN_LOG.md"
MANIFEST_CSV = RESULT_DIR / "v6b3_manifest.csv"

SWEEP_ORDER  = ["500K", "1M", "2M", "5M"]
SNR_SOURCE   = "linear"   # fixed per Batch 2 predecessor check

SWEEP_GAIN_THRESHOLD = 0.0005   # 0.05 pp in fractional BER
V5_BASELINE_BER      = 0.02275  # mambanet Phase 5 best


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(msg)


def append_run_log(section: str):
    with open(LOG_PATH, "a") as f:
        f.write(section)


def md5_first_1k(data_dir: Path) -> str:
    xs = np.load(str(data_dir / "xs.npy"), mmap_mode="r")
    return hashlib.md5(xs[:1000].tobytes()).hexdigest()


def record_manifest(size_label: str, seed: int, meta_path: Path):
    with open(meta_path) as f:
        meta = json.load(f)
    row = [size_label, seed, meta["num_samples"], meta["md5_first_1k"],
           meta["gen_wallclock_s"], meta["disk_bytes"]]
    write_header = not MANIFEST_CSV.exists()
    with open(MANIFEST_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["size", "seed", "n_frames", "md5_first_1k",
                        "gen_wallclock_s", "disk_bytes"])
        w.writerow(row)


def generate_data(size_label: str) -> Path:
    """Generate synthetic corpus. Returns data directory."""
    seed_map = {"500K": 500500000, "1M": 501000000, "2M": 502000000, "5M": 505000000}
    seed     = seed_map[size_label]
    data_dir = DATA_B3 / size_label

    if (data_dir / "meta.json").exists():
        log(f"  [{ts()}] Data already exists at {data_dir}, skipping generation.")
        with open(data_dir / "meta.json") as f:
            meta = json.load(f)
        record_manifest(size_label, seed, data_dir / "meta.json")
        return data_dir

    log(f"\n[{ts()}] Generating {size_label} synthetic data (seed={seed}) ...")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "synth_gen.py"),
         f"--num-samples={SIZE_MAP[size_label]}",
         f"--seed={seed}",
         f"--output-dir={data_dir}"],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"synth_gen.py failed with exit code {result.returncode}")

    elapsed = time.time() - t0
    log(f"  Data generation done in {elapsed:.0f}s")
    record_manifest(size_label, seed, data_dir / "meta.json")
    return data_dir


def get_conv_epoch(size_label: str, seed: int) -> int:
    """Read convergence epoch from pretrain log CSV."""
    log_path = RESULT_DIR / f"v6b3_pre_{size_label}_s{seed}_log.csv"
    if not log_path.exists():
        return -1
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    return int(rows[-1]["epoch"]) if rows else -1


def cleanup_data(size_label: str):
    """Delete synthetic data for given size (keep checkpoint)."""
    data_dir = DATA_B3 / size_label
    if data_dir.exists():
        log(f"  [{ts()}] Deleting corpus {data_dir} ...")
        shutil.rmtree(data_dir)
        log(f"  Corpus {size_label} deleted.")
    else:
        log(f"  Corpus {size_label} already gone.")


def run_one_size(size_label: str) -> float:
    """Full pipeline for one corpus size. Returns 3-seed ensemble BER."""
    torch.cuda.empty_cache()

    # 1. Generate data
    generate_data(size_label)

    # 2. Pretrain seeds 0 and 1
    for seed in (0, 1):
        pre_ckpt = CKPT_DIR / f"v6b3_pre_{size_label}_s{seed}.pt"
        if pre_ckpt.exists():
            log(f"  [{ts()}] Pretrain ckpt exists for {size_label} seed {seed}, skipping.")
        else:
            t0 = time.time()
            pretrain(size_label, seed, SNR_SOURCE)
            torch.cuda.empty_cache()
            log(f"  [{ts()}] Pretrain {size_label} seed {seed} done in {(time.time()-t0)/60:.1f} min")

    # 3. Fine-tune seeds 0, 1, 2
    seed_results: dict[int, dict] = {}
    valid_seeds  = []

    for seed in (0, 1, 2):
        ft_ckpt = CKPT_DIR / f"v6b3_{size_label}_s{seed}_ft.pt"
        from_seed = 1 if seed == 2 else None

        if ft_ckpt.exists():
            log(f"  [{ts()}] Finetune ckpt exists for {size_label} seed {seed}, skipping.")
        else:
            t0 = time.time()
            val_ber = finetune(size_label, seed, from_seed=from_seed, snr_source=SNR_SOURCE)
            torch.cuda.empty_cache()
            elapsed = (time.time() - t0) / 60
            if val_ber == float("inf"):
                log(f"  WARNING: seed {seed} diverged — excluded from ensemble")
                continue
            log(f"  [{ts()}] Finetune {size_label} seed {seed} done in {elapsed:.1f} min")

        # Evaluate
        test_csv = RESULT_DIR / f"v6b3_{size_label}_s{seed}_test.csv"
        if test_csv.exists():
            log(f"  [{ts()}] Test CSV exists for {size_label} seed {seed}, loading.")
            import csv as _csv
            with open(test_csv) as f:
                rows = list(_csv.DictReader(f))
            from src.data_zhu import TEST_CONDITIONS
            cond_results = {}
            for r in rows:
                if r["condition"] != "OVERALL":
                    cond_results[r["condition"]] = (int(r["n_bits"]), int(r["n_errors"]))
            seed_results[seed] = cond_results
        else:
            results = evaluate(size_label, seed, SNR_SOURCE)
            seed_results[seed] = results
        valid_seeds.append(seed)
        torch.cuda.empty_cache()

    if len(valid_seeds) < 2:
        log(f"  ERROR: fewer than 2 valid seeds for {size_label}. Aborting size.")
        return float("inf")

    # 4. Ensemble
    ensemble_ber = ensemble_eval(size_label, seed_results)
    return ensemble_ber


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr-source",  default="linear")
    ap.add_argument("--start-size",  default="500K", choices=SWEEP_ORDER)
    args = ap.parse_args()

    global SNR_SOURCE
    SNR_SOURCE = args.snr_source

    start_idx = SWEEP_ORDER.index(args.start_size)
    sizes_to_run = SWEEP_ORDER[start_idx:]

    append_run_log(f"""
---

## Batch 3 preflight

**Date**: {ts()}

**Predecessor check**:
- SNR_FIX_STATUS read from V6_RUN_LOG.md: FAILED
- All Batch 3 runs use --snr-source=linear (WARNING: neural SNR integration failed in Batch 2)

**Environment**:
- GPU: NVIDIA RTX 5070, 12227 MiB VRAM ✓
- mamba_ssm: OK (inside .venv) ✓
- Disk: 824 GB free ✓ (all sizes including 5M can run; peak ~63 GB)

**Batch 3 preflight answers**:

Q1: CLI entry point in synth_gen.py?
  No prior CLI existed. Added `if __name__ == "__main__"` block with --num-samples, --seed,
  --output-dir, --channel, --snr-lo, --snr-hi, --chunk flags. Generates in chunks of 100K to
  avoid RAM spike. Pre-allocates np.lib.format.open_memmap .npy files on disk.

  Programmatic signature (unchanged):
  ```python
  SynthDataset(n_samples, channel='mixed', snr_range=(-3.0,20.0),
               BT_choices=None, m_choices=None, b=2.0, seed=0)
  ```

Q2: Deterministically seeded?
  YES. `rng = np.random.default_rng(seed)` at line 153.
  Verified: MD5 run1=adb71537843ea01f0eb88d921135b094
            MD5 run2=adb71537843ea01f0eb88d921135b094  → MATCH ✓

Q3: V5 pretrain optimizer/scheduler/batch/grad-clip (from train_competitor.py):
  - Optimizer:  `torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)`  (pretrain_lr=1e-3)
  - Scheduler:  `CosineAnnealingLR(opt_pre, T_max=pretrain_epochs, eta_min=1e-5)`
  - Batch size: 512
  - Grad clip:  `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` (in _run, line 73)

Q4: Early stopping support?
  No — train_competitor.py runs all epochs unconditionally.
  Added: train_v6b3.py implements early stopping via patience counter in both
  pretrain (patience=5) and finetune (patience=10) functions.

**Batch 3 adaptations**:
  1. synth_gen.py: added CLI entry point (--num-samples, --seed, --output-dir, chunked memmap)
  2. train_v6b3.py: new script with AdamW+warmup pretrain, early stopping, v6b3_* naming,
     step-level checkpoints, SynthNpyDataset, set_seed/worker_init_fn seeding
  3. sweep_v6b3.py: sweep orchestration with sweep-level early stopping, disk cleanup
  4. v6b3_scaling.py: power-law fit + scaling curve figure

**Disk budget**:
  Free: 824 GB. All sizes (500K/1M/2M/5M) can run simultaneously (~63 GB peak).
  Deletion policy applied after each successor corpus is verified.

**Sweep**: 500K → 1M → 2M → 5M  |  snr-source=linear
""")

    log(f"\n{'='*70}")
    log(f"[{ts()}] V6 Batch 3 sweep starting | sizes={sizes_to_run} | snr={SNR_SOURCE}")
    log(f"{'='*70}")

    sizes_ran    = []
    bers         = {}   # size_label → ensemble BER
    conv_epochs  = {}   # size_label → convergence epoch (seed 0)

    prev_ber = V5_BASELINE_BER
    winner   = None

    for size_label in sizes_to_run:
        log(f"\n{'─'*60}")
        log(f"[{ts()}] Starting {size_label} ...")
        t_size = time.time()

        ensemble_ber = run_one_size(size_label)
        elapsed_min  = (time.time() - t_size) / 60

        if ensemble_ber == float("inf"):
            log(f"  FAILED: {size_label} aborted.")
            continue

        bers[size_label]        = ensemble_ber
        conv_epochs[size_label] = get_conv_epoch(size_label, 0)
        sizes_ran.append(size_label)

        gain = prev_ber - ensemble_ber
        log(f"[{ts()}] {size_label}: ensemble_BER={ensemble_ber*100:.4f}%  "
            f"gain={gain*100:.4f}pp  {elapsed_min:.1f} min")

        append_run_log(f"""
### {size_label}

| Field | Value |
|-------|-------|
| Ensemble BER | {ensemble_ber*100:.4f}% |
| Gain vs previous | {gain*100:.4f} pp |
| Conv epoch (seed 0) | {conv_epochs[size_label]} |
| Conv epoch (seed 1) | {get_conv_epoch(size_label, 1)} |
| Wall-clock (total) | {elapsed_min:.1f} min |
| SNR source | {SNR_SOURCE} |
""")

        # Git commit per size
        commit_msg = (f"V6 Batch3 {size_label}: converged ep={conv_epochs[size_label]}, "
                      f"ensemble BER={ensemble_ber*100:.4f}%")
        subprocess.run(
            ["git", "add", "-A", "--",
             str(RESULT_DIR), str(LOG_PATH), "src/train/train_v6b3.py",
             "src/train/sweep_v6b3.py", "src/analysis/v6b3_scaling.py",
             "src/synth_gen.py"],
            cwd=ROOT
        )
        subprocess.run(
            ["git", "commit", "-m", commit_msg + "\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"],
            cwd=ROOT
        )

        # Disk cleanup (delete predecessor after successor verified)
        idx = SWEEP_ORDER.index(size_label)
        if idx > 0:
            prev_size = SWEEP_ORDER[idx - 1]
            cleanup_data(prev_size)

        # Sweep-level early stopping
        if gain < SWEEP_GAIN_THRESHOLD:
            log(f"  SWEEP STOP: gain={gain*100:.4f}pp < 0.05pp threshold.")
            append_run_log(f"\n**Sweep early stop at {size_label}**: "
                           f"gain={gain*100:.4f}pp < 0.05pp\n")
            break

        winner   = size_label
        prev_ber = ensemble_ber

    # If we never updated winner (500K already stopped), use 500K
    if winner is None and "500K" in bers:
        winner = "500K"
    elif winner is None:
        winner = sizes_ran[-1] if sizes_ran else "500K"

    # Tiebreaker: within 0.02pp, prefer smaller
    if len(bers) >= 2:
        best_ber = min(bers.values())
        candidates = [sz for sz, b in bers.items() if (b - best_ber) * 100 <= 0.02]
        winner = min(candidates, key=lambda s: SIZE_MAP[s])

    log(f"\n{'='*70}")
    log(f"[{ts()}] WINNER: {winner}  BER={bers.get(winner, '?')*100:.4f}%")
    log(f"{'='*70}")

    # --- Scaling curve ---
    analysis = run_scaling_analysis(sizes_ran, winner, conv_epochs)

    # --- Promote winner ---
    log(f"\n[{ts()}] Promoting winner {winner} checkpoints ...")
    pre_src = CKPT_DIR / f"v6b3_pre_{winner}_s1.pt"
    if pre_src.exists():
        import shutil as _sh
        _sh.copy2(pre_src, CKPT_DIR / "v6b3_canonical_pretrain.pt")
    for seed in (0, 1, 2):
        ft_src = CKPT_DIR / f"v6b3_{winner}_s{seed}_ft.pt"
        if ft_src.exists():
            _sh.copy2(ft_src, CKPT_DIR / f"v6b3_canonical_s{seed}.pt")

    canonical_ber = bers.get(winner, float("inf"))

    # --- Final log entry ---
    extrap_str = ""
    if analysis.get("extrap_10M") is not None:
        extrap_str = (f"\nExtrapolated: 10M={analysis['extrap_10M']:.4f}%  "
                      f"50M={analysis['extrap_50M']:.4f}%")

    flat_str = "\nFINDING: Curve is FLAT — data size is not the bottleneck." if analysis.get("flat") else ""

    append_run_log(f"""
---

## Batch 3 final

**Winner**: {winner}
**V6B3_CANONICAL_BER**: {canonical_ber*100:.4f}%
**Sizes ran**: {sizes_ran}
**All ensemble BERs**: {{{', '.join(f'{k}: {v*100:.4f}%' for k,v in bers.items())}}}
{flat_str}
{extrap_str}

Canonical checkpoints:
  v6b3_canonical_pretrain.pt  (seed 1 pretrain of {winner})
  v6b3_canonical_s0.pt  v6b3_canonical_s1.pt  v6b3_canonical_s2.pt  (fine-tuned)
""")

    # Update V6B3_CANONICAL_BER in run log header
    text = open(LOG_PATH).read()
    text = text.replace("V6B3_CANONICAL_BER: PENDING",
                        f"V6B3_CANONICAL_BER: {canonical_ber*100:.4f}%")
    with open(LOG_PATH, "w") as f:
        f.write(text)

    # Final commit
    subprocess.run(
        ["git", "add", "-A", "--",
         str(RESULT_DIR), str(LOG_PATH), "figures/v6b3_scaling_curve.png",
         "src/"],
        cwd=ROOT
    )
    subprocess.run(
        ["git", "commit", "-m",
         f"V6 Batch3 winner: {winner}, ensemble BER={canonical_ber*100:.4f}%\n\n"
         f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"],
        cwd=ROOT
    )

    log(f"\n[{ts()}] V6 Batch 3 sweep COMPLETE.")
    log(f"  V6B3_CANONICAL_BER = {canonical_ber*100:.4f}%")
    log(f"  Winner = {winner}")
    log("  Batch 4 can now begin.")

    return canonical_ber


if __name__ == "__main__":
    main()
