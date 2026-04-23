"""V6 Batch 2 Part B1 — Generate 200K labelled frames for SNR estimator training.

Label per frame: instantaneous received Eb/N0 in dB.
In the generator, snr_db IS the instantaneous received Eb/N0 (noise computed
from post-fading signal power — see src/synth_gen.py _awgn_sigma).

Config split (B1 spec):
  25% AWGN (BT 50/50)
  25% KB2 m=1.2
  25% KB2 m=1.4
  25% KB2 m uniform in [1.0, 1.8]
  50% BT=0.3, 50% BT=0.5
  TX-side SNR uniform in [-8, +12] dB

Saves: data/v6b2_snr_estimator/train/data.npz
       data/v6b2_snr_estimator/val/data.npz
"""

import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.synth_gen import generate_sample

SEED      = 42
N_TOTAL   = 200_000
TRAIN_VAL_SPLIT = 0.9   # 90/10
SNR_LO    = -8.0
SNR_HI    = 12.0
BT_CHOICES = [0.3, 0.5]
OUT_DIR   = ROOT / "data" / "v6b2_snr_estimator"
(OUT_DIR / "train").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "val").mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_seed(SEED)
    rng = np.random.default_rng(SEED)

    xs, ys, snrs = [], [], []
    t0 = time.time()
    for i in range(N_TOTAL):
        if i % 20000 == 0:
            print(f"  [{i}/{N_TOTAL}] {time.time()-t0:.0f}s")

        # Config selection
        bucket = i % 4  # deterministic 25% split
        BT = BT_CHOICES[rng.integers(2)]
        snr_db = float(rng.uniform(SNR_LO, SNR_HI))

        if bucket == 0:
            # AWGN
            ch, m = "awgn", 1.2
        elif bucket == 1:
            ch, m = "kdist", 1.2
        elif bucket == 2:
            ch, m = "kdist", 1.4
        else:
            ch, m = "kdist", float(rng.uniform(1.0, 1.8))

        x, y, snr_out = generate_sample(rng, BT, snr_db, ch, m=m, b=2.0)
        # snr_out == snr_db == instantaneous received Eb/N0
        xs.append(x)
        ys.append(y)
        snrs.append(snr_out)

    xs   = np.stack(xs)           # (200K, 2, 800)
    ys   = np.stack(ys)           # (200K, 100)
    snrs = np.array(snrs, dtype=np.float32)

    print(f"Generated {N_TOTAL} frames in {time.time()-t0:.1f}s")
    print(f"SNR range: [{snrs.min():.2f}, {snrs.max():.2f}] dB, mean={snrs.mean():.2f}")

    # Deterministic 90/10 split
    n_train = int(N_TOTAL * TRAIN_VAL_SPLIT)
    perm = np.random.default_rng(SEED + 1).permutation(N_TOTAL)
    tr_idx  = perm[:n_train]
    val_idx = perm[n_train:]

    np.savez_compressed(OUT_DIR / "train" / "data.npz",
                        x=xs[tr_idx], y=ys[tr_idx], snr=snrs[tr_idx])
    np.savez_compressed(OUT_DIR / "val" / "data.npz",
                        x=xs[val_idx], y=ys[val_idx], snr=snrs[val_idx])

    print(f"Train: {len(tr_idx)}, Val: {len(val_idx)}")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
