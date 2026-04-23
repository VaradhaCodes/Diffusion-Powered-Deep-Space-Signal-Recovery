"""Synthetic GMSK + K-distribution channel generator.

Produces samples matching Zhu dataset format:
  x: (2, 800) float32   I/Q, 100 symbols × 8 sps
  y: (100,)   float32   bit labels {0,1}
  snr_db: scalar float  known Eb/N0

Channel modes
  'awgn'  : AWGN only
  'kdist' : K-distribution scintillation + AWGN

K-distribution compound model:
  G ~ Gamma(m, b²/m)     [random power, E[G]=b²]
  h = sqrt(G/2) * (N_r + jN_i) with N_r,N_i~N(0,1)  → E[|h|²]=b²
  With b=2: E[|h|²]=4, matching Zhu's channel params.

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

    E[|h|²] = b² (unnormalised); caller normalises signal power separately.
    """
    # G ~ Gamma(m, b²/m)  → E[G]=b², so E[|h|²]=b²
    scale = b**2 / m
    G = rng.gamma(shape=m, scale=scale, size=n_frames)
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

    if channel == "kdist":
        h = kdist_fade(1, m, b, rng)[0]
        sig = sig * h

    sigma = _awgn_sigma(sig, snr_db, sps)
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
