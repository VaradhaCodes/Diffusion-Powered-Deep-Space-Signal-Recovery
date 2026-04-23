"""Phase 5 competitor baseline models.

All three share the V5 CNN stem, FiLM(SNR) conditioning, and multi-task heads
so that the only variable is the sequence backbone.

Models
------
BiTransformer  — CNN stem + 2-layer Transformer encoder (non-causal → bidirectional)
BiMamba2       — CNN stem + Bidirectional Mamba-2 (fwd+bwd flip, sum-merge)
MambaNet       — CNN stem + MultiheadAttn → BiMamba2 (attention-assisted Mamba,
                 following MambaNet 2026 architecture: Luan et al., ICASSP)

Shared hypers
  d_model = 128, CNN 5→32→64→128 with 8× stride, FiLM from SNR, BCE+SNR loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2

from src.features.feature_extract import FeatureExtractor
from src.models.v5_model import _FiLM, v5_loss, _SNR_MIN, _SNR_RANGE

_D = 128   # shared d_model


# ── CNN stem (identical to V5) ────────────────────────────────────────────────

def _cnn_stem() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(5, 32, kernel_size=7, padding=3),
        nn.BatchNorm1d(32),
        nn.GELU(),
        nn.Conv1d(32, 64, kernel_size=7, padding=3),
        nn.BatchNorm1d(64),
        nn.GELU(),
        nn.Conv1d(64, _D, kernel_size=8, stride=8),   # 800→100
        nn.BatchNorm1d(_D),
        nn.GELU(),
    )


# ── Mamba-2 bidirectional block ───────────────────────────────────────────────

def _make_mamba2() -> Mamba2:
    return Mamba2(d_model=_D, d_state=64, headdim=64, chunk_size=64)


class _BiMamba2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd = _make_mamba2()
        self.bwd = _make_mamba2()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_f = self.fwd(x)
        h_b = torch.flip(self.bwd(torch.flip(x, [1])), [1])
        return h_f + h_b


# ══════════════════════════════════════════════════════════════════════════════
# 1. BiTransformer
# ══════════════════════════════════════════════════════════════════════════════

class BiTransformer(nn.Module):
    """CNN stem + 2-layer Transformer encoder (bidirectional by default — no causal mask)."""

    def __init__(self):
        super().__init__()
        self.feat = FeatureExtractor()
        self.cnn  = _cnn_stem()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=_D, nhead=8, dim_feedforward=256,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.film     = _FiLM(_D)
        self.bit_head = nn.Linear(_D, 1)
        self.snr_head = nn.Linear(_D, 1)

    def forward(self, iq, snr_db):
        snr_norm = ((snr_db - _SNR_MIN) / _SNR_RANGE).unsqueeze(-1)
        h = self.cnn(self.feat(iq)).permute(0, 2, 1)   # (B,100,128)
        h = self.encoder(h)
        h = self.film(h, snr_norm)
        return self.bit_head(h).squeeze(-1), self.snr_head(h.mean(1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. BiMamba2
# ══════════════════════════════════════════════════════════════════════════════

class BiMamba2(nn.Module):
    """CNN stem + Bidirectional Mamba-2 (direct swap of Mamba-3 in V5)."""

    def __init__(self):
        super().__init__()
        self.feat      = FeatureExtractor()
        self.cnn       = _cnn_stem()
        self.bi_mamba2 = _BiMamba2Block()
        self.film      = _FiLM(_D)
        self.bit_head  = nn.Linear(_D, 1)
        self.snr_head  = nn.Linear(_D, 1)

    def forward(self, iq, snr_db):
        snr_norm = ((snr_db - _SNR_MIN) / _SNR_RANGE).unsqueeze(-1)
        h = self.cnn(self.feat(iq)).permute(0, 2, 1)
        h = self.bi_mamba2(h)
        h = self.film(h, snr_norm)
        return self.bit_head(h).squeeze(-1), self.snr_head(h.mean(1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MambaNet-style
# ══════════════════════════════════════════════════════════════════════════════

class MambaNet(nn.Module):
    """CNN stem + [MHA + LN residual] + [BiMamba2 + LN residual].

    Follows MambaNet (Luan et al. 2026): multi-head attention captures local
    inter-symbol correlations first, then bidirectional Mamba propagates refined
    features across the full sequence with linear complexity.
    """

    def __init__(self):
        super().__init__()
        self.feat = FeatureExtractor()
        self.cnn  = _cnn_stem()

        # Attention-assisted block
        self.attn    = nn.MultiheadAttention(_D, num_heads=8, dropout=0.0, batch_first=True)
        self.norm1   = nn.LayerNorm(_D)

        # Mamba propagation block
        self.bi_m2   = _BiMamba2Block()
        self.norm2   = nn.LayerNorm(_D)

        self.film     = _FiLM(_D)
        self.bit_head = nn.Linear(_D, 1)
        self.snr_head = nn.Linear(_D, 1)

    def forward(self, iq, snr_db):
        snr_norm = ((snr_db - _SNR_MIN) / _SNR_RANGE).unsqueeze(-1)
        h = self.cnn(self.feat(iq)).permute(0, 2, 1)   # (B,100,128)

        # Attention block (pre-norm, residual)
        h = self.norm1(h + self.attn(h, h, h, need_weights=False)[0])

        # BiMamba-2 block (pre-norm, residual)
        h = self.norm2(h + self.bi_m2(h))

        h = self.film(h, snr_norm)
        return self.bit_head(h).squeeze(-1), self.snr_head(h.mean(1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6 Ablations (all derived from MambaNet)
# ══════════════════════════════════════════════════════════════════════════════

def _cnn_stem_2ch() -> nn.Sequential:
    """CNN stem for 2-channel raw IQ input (no feature engineering)."""
    return nn.Sequential(
        nn.Conv1d(2, 32, kernel_size=7, padding=3),
        nn.BatchNorm1d(32),
        nn.GELU(),
        nn.Conv1d(32, 64, kernel_size=7, padding=3),
        nn.BatchNorm1d(64),
        nn.GELU(),
        nn.Conv1d(64, _D, kernel_size=8, stride=8),
        nn.BatchNorm1d(_D),
        nn.GELU(),
    )


class MambaNetNoFiLM(nn.Module):
    """Ablation A1 — MambaNet with FiLM replaced by identity (no SNR conditioning)."""

    def __init__(self):
        super().__init__()
        self.feat     = FeatureExtractor()
        self.cnn      = _cnn_stem()
        self.attn     = nn.MultiheadAttention(_D, num_heads=8, dropout=0.0, batch_first=True)
        self.norm1    = nn.LayerNorm(_D)
        self.bi_m2    = _BiMamba2Block()
        self.norm2    = nn.LayerNorm(_D)
        self.bit_head = nn.Linear(_D, 1)
        self.snr_head = nn.Linear(_D, 1)

    def forward(self, iq, snr_db):
        h = self.cnn(self.feat(iq)).permute(0, 2, 1)
        h = self.norm1(h + self.attn(h, h, h, need_weights=False)[0])
        h = self.norm2(h + self.bi_m2(h))
        # No FiLM — SNR input silently ignored for fair loss API compatibility
        return self.bit_head(h).squeeze(-1), self.snr_head(h.mean(1)).squeeze(-1)


class MambaNet2ch(nn.Module):
    """Ablation A2 — MambaNet with raw 2-channel IQ (no feature engineering)."""

    def __init__(self):
        super().__init__()
        # No FeatureExtractor — raw IQ passed directly to 2-ch CNN stem
        self.cnn      = _cnn_stem_2ch()
        self.attn     = nn.MultiheadAttention(_D, num_heads=8, dropout=0.0, batch_first=True)
        self.norm1    = nn.LayerNorm(_D)
        self.bi_m2    = _BiMamba2Block()
        self.norm2    = nn.LayerNorm(_D)
        self.film     = _FiLM(_D)
        self.bit_head = nn.Linear(_D, 1)
        self.snr_head = nn.Linear(_D, 1)

    def forward(self, iq, snr_db):
        snr_norm = ((snr_db - _SNR_MIN) / _SNR_RANGE).unsqueeze(-1)
        h = self.cnn(iq).permute(0, 2, 1)              # raw IQ, no feature extraction
        h = self.norm1(h + self.attn(h, h, h, need_weights=False)[0])
        h = self.norm2(h + self.bi_m2(h))
        h = self.film(h, snr_norm)
        return self.bit_head(h).squeeze(-1), self.snr_head(h.mean(1)).squeeze(-1)


# ── Registry ──────────────────────────────────────────────────────────────────

MODELS = {
    "bi_transformer":       BiTransformer,
    "bi_mamba2":            BiMamba2,
    "mambanet":             MambaNet,
    # ablations (separate names → separate output files, no clobbering)
    "mambanet_no_film":     MambaNetNoFiLM,
    "mambanet_2ch":         MambaNet2ch,
    "mambanet_no_pretrain": MambaNet,   # same arch, trained with --skip-pretrain
}


def build_model(name: str) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    return MODELS[name]()
