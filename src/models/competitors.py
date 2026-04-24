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


# ══════════════════════════════════════════════════════════════════════════════
# V6 Batch 4 — Configurable MambaNet2ch architecture sweep
# ══════════════════════════════════════════════════════════════════════════════

def _make_mamba2_d(d_model: int) -> Mamba2:
    return Mamba2(d_model=d_model, d_state=64, headdim=64, chunk_size=64)


class _BiMamba2BlockCfg(nn.Module):
    """Bidirectional Mamba-2 with configurable d_model."""
    def __init__(self, d_model: int):
        super().__init__()
        self.fwd = _make_mamba2_d(d_model)
        self.bwd = _make_mamba2_d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_f = self.fwd(x)
        h_b = torch.flip(self.bwd(torch.flip(x, [1])), [1])
        return h_f + h_b


class _SerialBlock(nn.Module):
    """MHA → BiMamba2 serial block, pre-norm residuals, optional grad checkpoint."""
    def __init__(self, d_model: int, grad_ckpt: bool = False):
        super().__init__()
        n_heads = max(1, d_model // 16)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.bi_m2 = _BiMamba2BlockCfg(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self._grad_ckpt = grad_ckpt

    def _body(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm1(h + self.attn(h, h, h, need_weights=False)[0])
        h = self.norm2(h + self.bi_m2(h))
        return h

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self._grad_ckpt and self.training:
            import torch.utils.checkpoint as cp
            return cp.checkpoint(self._body, h, use_reentrant=False)
        return self._body(h)


class _ParallelBlock(nn.Module):
    """Mcformer-style: LN(h + MHA(h) + BiMamba2_fwd(h) + BiMamba2_bwd(flip(h)))."""
    def __init__(self, d_model: int, grad_ckpt: bool = False):
        super().__init__()
        n_heads = max(1, d_model // 16)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.fwd_m = _make_mamba2_d(d_model)
        self.bwd_m = _make_mamba2_d(d_model)
        self.norm  = nn.LayerNorm(d_model)
        self._grad_ckpt = grad_ckpt

    def _body(self, h: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(h, h, h, need_weights=False)[0]
        fwd_out  = self.fwd_m(h)
        bwd_out  = torch.flip(self.bwd_m(torch.flip(h, [1])), [1])
        return self.norm(h + attn_out + fwd_out + bwd_out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self._grad_ckpt and self.training:
            import torch.utils.checkpoint as cp
            return cp.checkpoint(self._body, h, use_reentrant=False)
        return self._body(h)


class MambaNet2chCfg(nn.Module):
    """Configurable MambaNet2ch for V6 Batch 4 architecture sweep.

    d_model   : embedding dim (128 / 192 / 256)
    n_blocks  : stacked (MHA + BiMamba2) blocks
    cnn_k1    : first CNN kernel size (7 = baseline, 31 = SB2+)
    parallel  : serial vs Mcformer-style parallel block
    grad_ckpt : per-block gradient checkpointing
    """
    def __init__(self, d_model: int = 128, n_blocks: int = 1,
                 cnn_k1: int = 7, parallel: bool = False, grad_ckpt: bool = False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=cnn_k1, padding=cnn_k1 // 2),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=8, stride=8),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )
        cls = _ParallelBlock if parallel else _SerialBlock
        self.blocks   = nn.ModuleList([cls(d_model, grad_ckpt) for _ in range(n_blocks)])
        self.film      = _FiLM(d_model)
        self.bit_head  = nn.Linear(d_model, 1)
        self.snr_head  = nn.Linear(d_model, 1)

    def forward(self, iq: torch.Tensor, snr_db: torch.Tensor):
        snr_norm = ((snr_db - _SNR_MIN) / _SNR_RANGE).unsqueeze(-1)
        h = self.cnn(iq).permute(0, 2, 1)
        for block in self.blocks:
            h = block(h)
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
    # V6 Batch 4 configurable sweep model
    "mambanet_2ch_cfg":     MambaNet2chCfg,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    model = MODELS[name](**kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[build_model] {name}  params={n_params:,}  kwargs={kwargs}")
    return model
