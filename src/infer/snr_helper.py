"""V6 Batch 2 Part C1 — SNR estimator inference helper.

Exposes: estimate_snr_db(iq_tensor, ckpt_path, device) -> Tensor (B,)

Estimator is frozen, eval-mode, no gradients. Checkpoint cached by path.
"""

import torch
from pathlib import Path

_MODEL_CACHE: dict = {}


def estimate_snr_db(
    iq: torch.Tensor,
    ckpt_path: str | Path,
    device: torch.device,
) -> torch.Tensor:
    """Estimate instantaneous received Eb/N0 in dB for each frame.

    Args:
        iq       : (B, 2, 800) IQ tensor (any device/dtype)
        ckpt_path: path to v6b2_snr_estimator.pt
        device   : inference device

    Returns:
        (B,) SNR estimates in dB, on `device`, float32
    """
    ckpt_path = str(ckpt_path)
    if ckpt_path not in _MODEL_CACHE:
        from src.models.snr_estimator import SNREstimator
        model = SNREstimator().to(device)
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _MODEL_CACHE[ckpt_path] = model

    model = _MODEL_CACHE[ckpt_path]
    with torch.no_grad():
        return model(iq.to(device).float())
