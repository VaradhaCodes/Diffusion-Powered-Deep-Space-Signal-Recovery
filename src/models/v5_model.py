"""V5 model: CNN stem + Bidirectional Mamba-3 + FiLM(SNR) + multi-task heads.

Architecture:
  FeatureExtractor   (B,2,800) → (B,5,800)
  CNN stem           (B,5,800) → (B,128,100)   8× downsample = 1 step/symbol
  Bi-Mamba-3         (B,100,128) → (B,100,128) fwd+bwd sum-merge
  FiLM(SNR)          elementwise modulate with learned γ,β from SNR scalar
  Bit head           Linear(128→1) per step → (B,100)  [primary]
  SNR head           global mean-pool → Linear(128→1)  [auxiliary]

Non-negotiables (CLAUDE.md):
  - No silent Mamba-3 fallback.
  - fp32 SSM parameters. Mamba3 constructed with dtype=bfloat16 for Blackwell
    kernel compatibility; A_log / dt_bias pinned back to fp32 after __init__.
  - bf16 autocast in training loop (external).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba3

from src.features.feature_extract import FeatureExtractor

# SNR normalisation constants matching Zhu dataset range
_SNR_MIN = -4.0
_SNR_RANGE = 12.0   # 8 − (−4)

# Mamba-3 hyperparameters
_D_MODEL  = 128
_D_STATE  = 64
_HEADDIM  = 64   # nheads = d_model / headdim = 2
_CHUNK    = 64


def _make_mamba3() -> Mamba3:
    m = Mamba3(
        d_model=_D_MODEL,
        d_state=_D_STATE,
        headdim=_HEADDIM,
        is_mimo=False,
        chunk_size=_CHUNK,
        dtype=torch.bfloat16,   # required for Blackwell (sm_120) kernel
    )
    # Pin SSM parameters to fp32 so they accumulate gradients accurately.
    # Projection / conv weights stay bf16 as set by dtype=bfloat16 above.
    for name, p in m.named_parameters():
        if any(k in name for k in ("A_log", "dt_bias", "D")):
            p.data = p.data.float()
    return m


class _BiMamba3(nn.Module):
    """Bidirectional Mamba-3 with sum-merge."""
    def __init__(self):
        super().__init__()
        self.fwd = _make_mamba3()
        self.bwd = _make_mamba3()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_f = self.fwd(x)
        h_b = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        return h_f + h_b


class _FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on scalar SNR."""
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, d_model * 2),
        )

    def forward(self, h: torch.Tensor, snr_norm: torch.Tensor) -> torch.Tensor:
        """
        h         : (B, T, D)
        snr_norm  : (B, 1)  values in [0, 1]
        returns   : (B, T, D)
        """
        params  = self.mlp(snr_norm)          # (B, 2D)
        gamma   = params[:, :h.shape[-1]]     # (B, D)
        beta    = params[:, h.shape[-1]:]
        return (1 + gamma).unsqueeze(1) * h + beta.unsqueeze(1)


class V5Model(nn.Module):
    """Full V5 model. Forward returns (bit_pred, snr_pred)."""

    def __init__(self):
        super().__init__()

        self.feat = FeatureExtractor()

        # CNN stem: (B,5,800) → (B,128,100)
        self.cnn = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, _D_MODEL, kernel_size=8, stride=8),  # 800→100
            nn.BatchNorm1d(_D_MODEL),
            nn.GELU(),
        )

        self.bi_mamba = _BiMamba3()
        self.film     = _FiLM(_D_MODEL)

        # Bit detection head: per-symbol logit → sigmoid
        self.bit_head = nn.Linear(_D_MODEL, 1)

        # Auxiliary SNR regression head
        self.snr_head = nn.Linear(_D_MODEL, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        iq: torch.Tensor,       # (B, 2, 800)
        snr_db: torch.Tensor,   # (B,) known or estimated Eb/N0 in dB
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            bit_pred : (B, 100)  probabilities in (0,1)
            snr_pred : (B,)      predicted SNR (normalised, for loss only)
        """
        snr_norm = ((snr_db - _SNR_MIN) / _SNR_RANGE).unsqueeze(-1)  # (B,1)

        feat = self.feat(iq)            # (B, 5, 800)
        h    = self.cnn(feat)           # (B, 128, 100)
        h    = h.permute(0, 2, 1)       # (B, 100, 128)

        h = self.bi_mamba(h)            # (B, 100, 128)
        h = self.film(h, snr_norm)      # (B, 100, 128)

        # Return logits (no sigmoid); use binary_cross_entropy_with_logits in loss.
        # Call torch.sigmoid(bit_logits) at inference to get probabilities.
        bit_logits = self.bit_head(h).squeeze(-1)               # (B, 100)
        snr_pred   = self.snr_head(h.mean(dim=1)).squeeze(-1)   # (B,)

        return bit_logits, snr_pred


def v5_loss(
    bit_logits: torch.Tensor,   # (B, 100)  raw logits
    bit_target: torch.Tensor,   # (B, 100)  float {0,1}
    snr_pred:   torch.Tensor,   # (B,)      normalised prediction
    snr_target: torch.Tensor,   # (B,)      true normalised SNR
    lambda_snr: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    bce      = F.binary_cross_entropy_with_logits(bit_logits, bit_target)
    snr_loss = F.mse_loss(snr_pred, snr_target)
    loss     = bce + lambda_snr * snr_loss
    return loss, {"bce": bce.item(), "snr_mse": snr_loss.item()}
