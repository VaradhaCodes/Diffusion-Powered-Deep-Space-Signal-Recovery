"""V6 Batch 2 Part B2 — Neural SNR estimator.

Architecture (B2 spec):
  Input: (B, 2, 800) raw IQ
  Stem:  Conv1d(2->32, k=7, pad=3) -> GELU
         Conv1d(32->64, k=7, pad=3, stride=2) -> GELU
         Conv1d(64->128, k=7, pad=3, stride=2) -> GELU
  Pool:  AdaptiveAvgPool1d(1) -> flatten -> 128
  Head:  Linear(128->64) -> GELU -> Dropout(0.1) -> Linear(64->1)
  Output: scalar SNR estimate in dB.

REF architecture family: Xie et al. 2019, "An SNR Estimation Technique Based on Deep
  Learning", MDPI Electronics 8(10):1139. https://www.mdpi.com/2079-9292/8/10/1139
"""

import torch
import torch.nn as nn


class SNREstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3, stride=2),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3, stride=2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        """iq: (B, 2, 800) → (B,) SNR estimates in dB."""
        h = self.stem(iq)
        h = self.pool(h).squeeze(-1)   # (B, 128)
        return self.head(h).squeeze(-1)  # (B,)
