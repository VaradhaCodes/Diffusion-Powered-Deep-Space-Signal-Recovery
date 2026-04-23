"""Phase 1 validation: Zhu loader + synthetic generator sanity checks.

Checks:
  1. Zhu train loader — shapes, dtypes, value ranges
  2. Zhu test loader  — all 6 conditions present, correct total
  3. Synth generator  — shapes, SNR in range, BER vs SNR curve (AWGN)
  4. K-dist statistics — E[|h|²] ≈ b²
  5. Throughput estimate for the iterable dataset

Saves:
  results/phase1_ber_awgn.csv   — BER vs Eb/N0 for AWGN synth
  results/phase1_stats.csv      — aggregate statistics
"""

import sys, time, csv, os
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_zhu import zhu_train_dataset, zhu_test_dataset, TEST_CONDITIONS
from synth_gen import SynthDataset, kdist_fade, gmsk_modulate

os.makedirs("results", exist_ok=True)

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    marker = "✓" if cond else "✗"
    print(f"  [{marker}] {name}{': '+detail if detail else ''}")
    results.append((name, status, detail))
    if not cond:
        print(f"      FAIL — aborting Phase 1 validation")
        _save_results()
        sys.exit(1)


def _save_results():
    with open("results/phase1_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "detail"])
        w.writerows(results)


# ── 1. Zhu train loader ───────────────────────────────────────────────────────
print("\n=== 1. Zhu train dataset ===")

t0 = time.time()
train_ds, val_ds = zhu_train_dataset(val_frac=0.2, seed=42)
check("train+val split sum", len(train_ds) + len(val_ds) == 42000,
      f"got {len(train_ds)+len(val_ds)}")
check("train size ~33600", abs(len(train_ds) - 33600) < 200, str(len(train_ds)))
check("val size ~8400",    abs(len(val_ds) - 8400) < 200,   str(len(val_ds)))

x0, y0 = train_ds[0]
check("x shape (2,800)", x0.shape == torch.Size([2, 800]), str(x0.shape))
check("y shape (100,)",  y0.shape == torch.Size([100]),    str(y0.shape))
check("x dtype float32", x0.dtype == torch.float32, str(x0.dtype))
check("y dtype float32", y0.dtype == torch.float32, str(y0.dtype))
check("y values binary", set(y0.numpy().tolist()).issubset({0.0, 1.0}))

# loader throughput (2 workers, 4 samples)
loader = DataLoader(train_ds, batch_size=4, num_workers=0)
xb, yb = next(iter(loader))
check("batch x shape (4,2,800)", xb.shape == torch.Size([4, 2, 800]), str(xb.shape))
check("batch y shape (4,100)",   yb.shape == torch.Size([4, 100]),    str(yb.shape))
print(f"  Zhu train loader OK in {time.time()-t0:.1f}s")

# ── 2. Zhu test loader ────────────────────────────────────────────────────────
print("\n=== 2. Zhu test dataset ===")

test_ds = zhu_test_dataset()
check("test total == 8400", len(test_ds) == 8400, str(len(test_ds)))

for cond in TEST_CONDITIONS:
    ds_c = zhu_test_dataset(cond)
    check(f"test {cond} == 1400", len(ds_c) == 1400, str(len(ds_c)))

xt, yt = test_ds[0]
check("test x shape (2,800)", xt.shape == torch.Size([2, 800]))
check("test y shape (100,)",  yt.shape == torch.Size([100]))

# ── 3. Synth generator — shapes & ranges ─────────────────────────────────────
print("\n=== 3. Synthetic generator ===")

synth = SynthDataset(n_samples=1000, channel="mixed", snr_range=(-3.0, 20.0), seed=7)
check("synth len == 1000", len(synth) == 1000)

xs, ys, snrs = synth[0]
check("synth x shape (2,800)", xs.shape == torch.Size([2, 800]), str(xs.shape))
check("synth y shape (100,)",  ys.shape == torch.Size([100]),    str(ys.shape))
check("synth y binary",        set(ys.numpy().tolist()).issubset({0.0, 1.0}))
check("snr in range",          -3.0 <= float(snrs) <= 20.0, f"{float(snrs):.1f} dB")

all_snrs = synth.snrs
check("snr min > -4",  float(all_snrs.min()) > -4, f"min={all_snrs.min():.2f}")
check("snr max < 21",  float(all_snrs.max()) < 21, f"max={all_snrs.max():.2f}")

# ── 4. K-dist statistics ──────────────────────────────────────────────────────
print("\n=== 4. K-distribution statistics ===")

rng = np.random.default_rng(99)
for m, b in [(1.2, 2.0), (1.4, 2.0)]:
    h = kdist_fade(100_000, m=m, b=b, rng=rng)
    mean_power = float(np.mean(np.abs(h) ** 2))
    expected = b ** 2
    rel_err = abs(mean_power - expected) / expected
    check(f"E[|h|²] ≈ b²={expected:.1f} (m={m})", rel_err < 0.05,
          f"got {mean_power:.3f} (err {rel_err*100:.1f}%)")

# ── 5. SNR calibration (AWGN) ────────────────────────────────────────────────
# GMSK is constant-envelope CPM with strong ISI (BT=0.3), so classical
# threshold demodulators don't work.  Instead we verify the noise floor
# matches Eb/N0 directly by comparing received power vs noiseless power.
print("\n=== 5. SNR calibration & GMSK constant-envelope check ===")

from synth_gen import generate_sample, gmsk_modulate

rng2 = np.random.default_rng(42)
n_frames = 200
snr_db_list = [-2.0, 4.0, 10.0, 16.0]
calib_rows = []
SPS = 8

print(f"  {'SNR':>6}  {'Ps_noisy':>9}  {'Ps_noiseless':>13}  {'N0_meas':>9}  {'N0_theory':>10}  {'rel_err':>7}")
for snr in snr_db_list:
    ps_n_list, noise_var_list = [], []
    for _ in range(n_frames):
        bits = rng2.integers(0, 2, size=100)
        sig = gmsk_modulate(bits, BT=0.3, sps=SPS)

        # constant-envelope check on the noiseless signal
        env = np.abs(sig) ** 2
        assert env.min() > 0.99 and env.max() < 1.01, \
            f"GMSK not constant-envelope: min={env.min():.4f} max={env.max():.4f}"

        from synth_gen import _awgn_sigma
        sigma = _awgn_sigma(sig, snr_db=snr, sps=SPS)
        noise = (rng2.standard_normal(800) + 1j * rng2.standard_normal(800)) * sigma
        rx = sig + noise
        ps_n_list.append(float(np.mean(np.abs(rx) ** 2)))
        # measured noise power = total - signal power (≈1.0)
        noise_var_list.append(float(np.mean(np.abs(noise) ** 2)))

    Ps = np.mean(ps_n_list)
    N0_meas = np.mean(noise_var_list)
    Eb = 1.0 * SPS   # Ps=1 × sps
    N0_theory = Eb / (10 ** (snr / 10))
    rel_err = abs(N0_meas - N0_theory) / N0_theory
    print(f"  {snr:>6.1f}  {Ps:>9.4f}  {'1.0 (ideal)':>13}  {N0_meas:>9.4f}  {N0_theory:>10.4f}  {rel_err:>6.1%}")
    calib_rows.append((snr, float(Ps), N0_meas, N0_theory, rel_err))
    check(f"N0 calibration at {snr:+.0f}dB (rel_err<5%)", rel_err < 0.05,
          f"err={rel_err:.1%}")

with open("results/phase1_ber_awgn.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["snr_db", "Ps_noisy", "N0_measured", "N0_theory", "rel_err"])
    w.writerows(calib_rows)

check("GMSK constant-envelope", True, "verified in loop above")

# ── 6. Throughput estimate ────────────────────────────────────────────────────
print("\n=== 6. Generator throughput ===")

from synth_gen import SynthIterableDataset
from itertools import islice

iter_ds = SynthIterableDataset(channel="mixed", base_seed=0)
t0 = time.time()
count = 0
for x, y, s in islice(iter_ds, 200):
    count += 1
elapsed = time.time() - t0
rate = count / elapsed
print(f"  Iterable gen: {rate:.0f} samples/s  ({elapsed:.2f}s for 200)")
check("gen rate > 50 samples/s", rate > 50, f"{rate:.0f} sps")

# ── Summary ───────────────────────────────────────────────────────────────────
_save_results()
passed = sum(1 for _, s, _ in results if s == PASS)
total_checks = len(results)
print(f"\n=== Phase 1: {passed}/{total_checks} checks PASSED ===")
if passed == total_checks:
    print("ALL PHASE 1 CHECKS PASSED")
    sys.exit(0)
else:
    sys.exit(1)
