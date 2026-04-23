"""Phase 3: IQ feature extraction for V5 model.

Converts raw (B, 2, 800) IQ → (B, 5, 800):
  ch0: I       raw in-phase
  ch1: Q       raw quadrature
  ch2: A       instantaneous amplitude  sqrt(I²+Q²)
  ch3: phi     instantaneous phase      atan2(Q, I)
  ch4: dphi    differential phase       phase-diff wrapped to [-π,π], zero-padded at t=0
"""

import torch
import torch.nn as nn


def extract_features(iq: torch.Tensor) -> torch.Tensor:
    """
    Args:
        iq: (B, 2, T) or (2, T) float tensor, ch0=I ch1=Q
    Returns:
        (B, 5, T) or (5, T) float tensor
    """
    unbatched = iq.dim() == 2
    if unbatched:
        iq = iq.unsqueeze(0)

    I   = iq[:, 0, :]
    Q   = iq[:, 1, :]
    A   = torch.sqrt(I.pow(2) + Q.pow(2) + 1e-8)
    phi = torch.atan2(Q, I)

    dphi = torch.diff(phi, dim=-1)
    dphi = (dphi + torch.pi) % (2 * torch.pi) - torch.pi
    dphi = torch.cat([torch.zeros_like(dphi[:, :1]), dphi], dim=-1)

    out = torch.stack([I, Q, A, phi, dphi], dim=1)
    return out.squeeze(0) if unbatched else out


class FeatureExtractor(nn.Module):
    """Differentiable, stateless feature extractor — embeds inside model."""
    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return extract_features(iq)
