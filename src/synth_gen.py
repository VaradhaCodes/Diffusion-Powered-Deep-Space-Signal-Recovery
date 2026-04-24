"""Synthetic GMSK + K-distribution channel generator.

Produces samples matching Zhu dataset format:
  x: (2, 800) float32   I/Q, 100 symbols × 8 sps
  y: (100,)   float32   bit labels {0,1}
  snr_db: scalar float  known Eb/N0

Channel modes
  'awgn'  : AWGN only
  'kdist' : K-distribution scintillation + AWGN

K-distribution compound model (Zhu 2023 Eq. 3–4):
  α = 2/(m²−1)          [shape from scintillation index; inverts m=sqrt(1+2/α)]
  G ~ Gamma(α, b²/α)    [random power, E[G]=b²]
  h = sqrt(G/2) * (N_r + jN_i) with N_r,N_i~N(0,1)  → E[|h|²]=b²
  With b=2: E[|h|²]=4, matching Zhu's channel params.
  Noise is computed from the UNFADED signal (transmitted Eb/N0, Eq. 2).

Usage:
  ds = SynthDataset(n_samples=500_000, channel='kdist')
  x, y, snr = ds[0]
"""

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

# ── GMSK pulse ────────────────────────────────────────────────────────────────

def _gmsk_pulse(BT: float, sps: int, n_taps: int = 5) -> np.ndarray:
    """Gaussian frequency pulse of length n_taps*sps samples, normalised to ∫=0.5."""
    t = np.arange(-n_taps * sps // 2, n_taps * sps // 2 + 1) / sps
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * BT)
    h = (np.exp(-t**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)) / sps
    h /= h.sum()        # area = 1
    h *= 0.5            # GMSK freq pulse area = 0.5
    return h.astype(np.float64)


_PULSE_CACHE: dict = {}

def _get_pulse(BT: float, sps: int = 8) -> np.ndarray:
    key = (BT, sps)
    if key not in _PULSE_CACHE:
        _PULSE_CACHE[key] = _gmsk_pulse(BT, sps)
    return _PULSE_CACHE[key]


def gmsk_modulate(bits: np.ndarray, BT: float, sps: int = 8) -> np.ndarray:
    """Modulate bits→GMSK complex baseband. Returns (N*sps,) complex128."""
    n_bits = len(bits)
    nrz = 2 * bits.astype(np.float64) - 1   # {0,1} → {-1,+1}
    pulse = _get_pulse(BT, sps)
    # upsample NRZ
    nrz_up = np.zeros(n_bits * sps + len(pulse) - 1)
    nrz_up[sps // 2::sps][:n_bits] = nrz
    freq = np.convolve(nrz_up, pulse, mode="full")[: n_bits * sps]
    phase = np.pi * np.cumsum(freq)           # integrate frequency (pulse sum=0.5 → Δφ=π/2 per bit)
    return np.exp(1j * phase)


# ── K-distribution channel ─────────────────────────────────────────────────

def kdist_fade(n_frames: int, m: float, b: float, rng: np.random.Generator) -> np.ndarray:
    """Per-frame complex K-dist fading coefficients, shape (n_frames,).

    m is the scintillation index (paper Eq. 1).
    Compound model: I=G*W, W~Exp(1) → m_sci=sqrt(1+2/α) → α=2/(m²-1).
    E[G] = b², so E[|h|²] = b².
    """
    # For the compound Gamma*Exp model: m_sci = sqrt(1 + 2/α)
    # Inverting to hit the paper's stated scintillation index: α = 2/(m²-1)
    # E[G] = b² regardless of α via scale = b²/α.
    alpha = 2.0 / (m ** 2 - 1)
    scale = b**2 / alpha
    G = rng.gamma(shape=alpha, scale=scale, size=n_frames)
    # complex Rayleigh conditioned on G
    real = rng.standard_normal(n_frames) * np.sqrt(G / 2)
    imag = rng.standard_normal(n_frames) * np.sqrt(G / 2)
    return real + 1j * imag


# ── SNR → noise power ──────────────────────────────────────────────────────

def _awgn_sigma(signal: np.ndarray, snr_db: float, sps: int = 8) -> float:
    """Noise std for Eb/N0 = snr_db dB, given complex baseband signal."""
    Ps = np.mean(np.abs(signal) ** 2)
    Eb = Ps * sps          # energy per bit (sps samples per symbol)
    N0 = Eb / (10 ** (snr_db / 10))
    return np.sqrt(N0 / 2)  # per I or Q component


# ── Single sample generator ────────────────────────────────────────────────

def generate_sample(
    rng: np.random.Generator,
    BT: float,
    snr_db: float,
    channel: str,
    m: float = 1.2,
    b: float = 2.0,
    n_bits: int = 100,
    sps: int = 8,
):
    """Returns (x: (2,800) float32, y: (100,) float32, snr_db: float)."""
    bits = rng.integers(0, 2, size=n_bits)
    sig = gmsk_modulate(bits, BT, sps)          # complex (800,)

    # Noise set from UNFADED signal → transmitted Eb/N0 (paper Eq. 2 / Table 2)
    sigma = _awgn_sigma(sig, snr_db, sps)

    if channel == "kdist":
        h = kdist_fade(1, m, b, rng)[0]
        sig = sig * h
    noise = (rng.standard_normal(len(sig)) + 1j * rng.standard_normal(len(sig))) * sigma
    rx = sig + noise                            # complex (800,)

    x = np.stack([rx.real, rx.imag], axis=0).astype(np.float32)  # (2,800)
    y = bits.astype(np.float32)
    return x, y, float(snr_db)


# ── Dataset ────────────────────────────────────────────────────────────────

class SynthDataset(Dataset):
    """Pre-generated fixed synthetic dataset.

    Args:
        n_samples: total samples to pre-generate.
        channel:   'awgn' or 'kdist' or 'mixed' (50/50 at gen time).
        snr_range: (lo, hi) Eb/N0 in dB; drawn uniformly per sample.
        BT_choices: list of BT products to sample uniformly.
        m_choices:  scintillation indices (kdist only).
        b:          K-dist scale.
        seed:       RNG seed.
    """

    def __init__(
        self,
        n_samples: int = 100_000,
        channel: str = "mixed",
        snr_range: tuple[float, float] = (-3.0, 20.0),
        BT_choices: list[float] | None = None,
        m_choices: list[float] | None = None,
        b: float = 2.0,
        seed: int = 0,
    ):
        if BT_choices is None:
            BT_choices = [0.3, 0.5]
        if m_choices is None:
            m_choices = [1.2, 1.4]

        rng = np.random.default_rng(seed)
        xs, ys, snrs = [], [], []

        channels = ["awgn", "kdist"] if channel == "mixed" else [channel]

        for i in range(n_samples):
            ch = channels[rng.integers(len(channels))]
            BT = BT_choices[rng.integers(len(BT_choices))]
            snr = float(rng.uniform(*snr_range))
            m = m_choices[rng.integers(len(m_choices))]
            x, y, s = generate_sample(rng, BT, snr, ch, m=m, b=b)
            xs.append(x)
            ys.append(y)
            snrs.append(s)

        self.xs = np.stack(xs)           # (N, 2, 800)
        self.ys = np.stack(ys)           # (N, 100)
        self.snrs = np.array(snrs, dtype=np.float32)  # (N,)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.xs[idx]),
            torch.from_numpy(self.ys[idx]),
            torch.tensor(self.snrs[idx]),
        )


class SynthIterableDataset(IterableDataset):
    """Infinite on-the-fly generator — avoids pre-allocating large arrays.

    Yields (x, y, snr) tuples indefinitely; wrap with itertools.islice.
    Each worker gets its own RNG seed to avoid duplicates.
    """

    def __init__(
        self,
        channel: str = "mixed",
        snr_range: tuple[float, float] = (-3.0, 20.0),
        BT_choices: list[float] | None = None,
        m_choices: list[float] | None = None,
        b: float = 2.0,
        base_seed: int = 0,
    ):
        self.channel = channel
        self.snr_range = snr_range
        self.BT_choices = BT_choices or [0.3, 0.5]
        self.m_choices = m_choices or [1.2, 1.4]
        self.b = b
        self.base_seed = base_seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.base_seed + (worker_info.id if worker_info else 0)
        rng = np.random.default_rng(seed)
        channels = ["awgn", "kdist"] if self.channel == "mixed" else [self.channel]
        while True:
            ch = channels[rng.integers(len(channels))]
            BT = self.BT_choices[rng.integers(len(self.BT_choices))]
            snr = float(rng.uniform(*self.snr_range))
            m = self.m_choices[rng.integers(len(self.m_choices))]
            x, y, s = generate_sample(rng, BT, snr, ch, m=m, b=self.b)
            yield (
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.tensor(s, dtype=torch.float32),
            )


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, gc, hashlib, json, time
    from pathlib import Path as _Path

    ap = argparse.ArgumentParser(description="Generate synthetic GMSK dataset to disk.")
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--seed",        type=int, required=True)
    ap.add_argument("--output-dir",  type=str, required=True)
    ap.add_argument("--channel",     type=str, default="mixed")
    ap.add_argument("--snr-lo",      type=float, default=-4.0)
    ap.add_argument("--snr-hi",      type=float, default=8.0)
    ap.add_argument("--chunk",       type=int, default=100_000)
    args = ap.parse_args()

    out = _Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    N = args.num_samples
    CHUNK = args.chunk
    BT_choices = [0.3, 0.5]
    m_choices  = [1.2, 1.4]
    channels   = ["awgn", "kdist"] if args.channel == "mixed" else [args.channel]
    snr_range  = (args.snr_lo, args.snr_hi)

    # Pre-allocate memory-mapped .npy files (numpy format, loadable with mmap_mode='r')
    xs_mm   = np.lib.format.open_memmap(str(out / "xs.npy"),   mode="w+",
                                        dtype=np.float32, shape=(N, 2, 800))
    ys_mm   = np.lib.format.open_memmap(str(out / "ys.npy"),   mode="w+",
                                        dtype=np.float32, shape=(N, 100))
    snrs_mm = np.lib.format.open_memmap(str(out / "snrs.npy"), mode="w+",
                                        dtype=np.float32, shape=(N,))

    rng = np.random.default_rng(args.seed)
    t0  = time.time()

    for start in range(0, N, CHUNK):
        end  = min(start + CHUNK, N)
        size = end - start

        xs_c   = np.empty((size, 2, 800), dtype=np.float32)
        ys_c   = np.empty((size, 100),    dtype=np.float32)
        snrs_c = np.empty(size,           dtype=np.float32)

        for j in range(size):
            ch  = channels[rng.integers(len(channels))]
            BT  = BT_choices[rng.integers(len(BT_choices))]
            snr = float(rng.uniform(*snr_range))
            m   = m_choices[rng.integers(len(m_choices))]
            x, y, s = generate_sample(rng, BT, snr, ch, m=m, b=2.0)
            xs_c[j]   = x
            ys_c[j]   = y
            snrs_c[j] = s

        xs_mm[start:end]   = xs_c
        ys_mm[start:end]   = ys_c
        snrs_mm[start:end] = snrs_c
        xs_mm.flush(); ys_mm.flush(); snrs_mm.flush()
        del xs_c, ys_c, snrs_c
        gc.collect()
        print(f"  {end}/{N} samples written  ({time.time()-t0:.0f}s)")

    # MD5 of first 1000 frames
    md5 = hashlib.md5(xs_mm[:1000].tobytes()).hexdigest()
    wallclock = time.time() - t0
    disk_bytes = sum(f.stat().st_size for f in out.iterdir())

    meta = {
        "num_samples": N, "seed": args.seed, "channel": args.channel,
        "snr_lo": args.snr_lo, "snr_hi": args.snr_hi,
        "md5_first_1k": md5, "gen_wallclock_s": round(wallclock, 1),
        "disk_bytes": disk_bytes,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. N={N}  seed={args.seed}  md5={md5}  {wallclock:.0f}s  {disk_bytes/1e9:.2f} GB")
    print(f"Saved to {out}/")
