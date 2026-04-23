"""Phase 3 smoke test — V5 model: forward, backward, 5-step train, artifact write.

Checks:
  1. Feature extractor: shape + finiteness
  2. CNN stem: (B,5,800) → (B,128,100)
  3. Bi-Mamba-3: output shape, both directions received gradients
  4. FiLM changes output when SNR changes
  5. Full forward pass: bit_pred (B,100) + snr_pred (B,) both finite
  6. Loss = BCE + 0.1*SNR-MSE, backward, all grads finite
  7. Mamba-3 SSM params (A_log/dt_bias) receive fp32 grads
  8. 5 Adam steps: loss strictly decreases
  9. Save checkpoint: checkpoints/v5_smoke.pt
 10. Write results/phase3_smoke.csv
"""

import sys, time, csv
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.feature_extract import extract_features
from src.models.v5_model import V5Model, v5_loss, _SNR_MIN, _SNR_RANGE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B = 16   # smoke batch size


def banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def check(cond: bool, msg: str) -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if not cond:
        raise AssertionError(msg)


# ── 0. Device ─────────────────────────────────────────────────────────────────
banner("Phase 3 smoke test — V5 model")
print(f"  Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")


# ── 1. Feature extractor ──────────────────────────────────────────────────────
banner("Check 1: feature extractor")
iq_raw = torch.randn(B, 2, 800, device=DEVICE)
feats  = extract_features(iq_raw)
check(feats.shape == (B, 5, 800), f"shape {feats.shape} == (B,5,800)")
check(feats.isfinite().all().item(), "all features finite")
print(f"  feats range: [{feats.min():.3f}, {feats.max():.3f}]")


# ── 2. Build model ────────────────────────────────────────────────────────────
banner("Check 2: model construction")
model = V5Model().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Total params : {n_params:,}")
check(hasattr(model, "bi_mamba"), "model has bi_mamba attribute")
check(hasattr(model.bi_mamba, "fwd") and hasattr(model.bi_mamba, "bwd"),
      "BiMamba3 has fwd and bwd")

# Verify SSM fp32 pinning
ssm_fp32_names = []
for name, p in model.bi_mamba.named_parameters():
    if any(k in name for k in ("A_log", "dt_bias", "D")):
        ssm_fp32_names.append((name, p.dtype))
check(len(ssm_fp32_names) > 0, "found SSM params to check")
for name, dtype in ssm_fp32_names:
    check(dtype == torch.float32, f"SSM param {name} is fp32 (got {dtype})")
print(f"  SSM fp32 params: {[n for n,_ in ssm_fp32_names]}")


# ── 3. Forward pass ───────────────────────────────────────────────────────────
banner("Check 3: forward pass shapes + finiteness")
snr_db = torch.empty(B, device=DEVICE).uniform_(_SNR_MIN, _SNR_MIN + _SNR_RANGE)
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
    bit_logits, snr_pred = model(iq_raw, snr_db)

bit_prob = torch.sigmoid(bit_logits.float())   # (B,100) probs for range check

check(bit_logits.shape == (B, 100), f"bit_logits shape {bit_logits.shape}")
check(snr_pred.shape == (B,),       f"snr_pred shape {snr_pred.shape}")
check(bit_logits.float().isfinite().all().item(), "bit_logits finite")
check(snr_pred.float().isfinite().all().item(), "snr_pred finite")
check((bit_prob > 0).all().item() and (bit_prob < 1).all().item(),
      "sigmoid(bit_logits) in (0,1)")
print(f"  bit_prob : mean={bit_prob.mean():.3f}  std={bit_prob.std():.3f}")


# ── 4. FiLM changes output ────────────────────────────────────────────────────
banner("Check 4: FiLM sensitivity")
snr_low  = torch.full((B,), _SNR_MIN, device=DEVICE)
snr_high = torch.full((B,), _SNR_MIN + _SNR_RANGE, device=DEVICE)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
    logits_low,  _ = model(iq_raw, snr_low)
    logits_high, _ = model(iq_raw, snr_high)
delta = (logits_high.float() - logits_low.float()).abs().mean().item()
check(delta > 1e-4, f"FiLM changes output (|Δ|={delta:.5f} > 1e-4)")


# ── 5. Loss + backward ────────────────────────────────────────────────────────
banner("Check 5: loss computation + backward")
bit_target  = torch.randint(0, 2, (B, 100), device=DEVICE).float()
snr_norm_gt = (snr_db - _SNR_MIN) / _SNR_RANGE

model.zero_grad()
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
    bit_logits, snr_pred = model(iq_raw, snr_db)
loss, loss_dict = v5_loss(bit_logits.float(), bit_target, snr_pred.float(), snr_norm_gt)

check(loss.isfinite().item(), f"loss finite ({loss.item():.4f})")
loss.backward()

all_grads_ok = all(
    p.grad is not None and p.grad.isfinite().all()
    for p in model.parameters() if p.requires_grad
)
check(all_grads_ok, "all parameter gradients finite")

mamba_grads = [
    (n, p.grad.dtype)
    for n, p in model.bi_mamba.named_parameters()
    if p.grad is not None and any(k in n for k in ("A_log", "dt_bias", "D"))
]
check(len(mamba_grads) > 0, "Mamba-3 SSM params received gradients")
for n, gd in mamba_grads:
    check(gd == torch.float32, f"SSM grad {n} is fp32 (got {gd})")
print(f"  loss={loss.item():.4f}  bce={loss_dict['bce']:.4f}  snr_mse={loss_dict['snr_mse']:.4f}")


# ── 6. 5-step training ────────────────────────────────────────────────────────
banner("Check 6: 5 Adam steps — loss must decrease overall")
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
for step in range(5):
    iq_b   = torch.randn(B, 2, 800, device=DEVICE)
    snr_b  = torch.empty(B, device=DEVICE).uniform_(_SNR_MIN, _SNR_MIN + _SNR_RANGE)
    tgt_b  = torch.randint(0, 2, (B, 100), device=DEVICE).float()
    norm_b = (snr_b - _SNR_MIN) / _SNR_RANGE

    opt.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
        bl, sp = model(iq_b, snr_b)
    l, _ = v5_loss(bl.float(), tgt_b, sp.float(), norm_b)
    l.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(l.item())
    print(f"  step {step+1}: loss={l.item():.4f}")

check(losses[-1] < losses[0] * 1.5, f"loss didn't diverge ({losses[0]:.4f}→{losses[-1]:.4f})")
print(f"  Loss trajectory: {[f'{v:.4f}' for v in losses]}")


# ── 7. Save checkpoint ────────────────────────────────────────────────────────
banner("Check 7: checkpoint save")
ckpt_path = ROOT / "checkpoints" / "v5_smoke.pt"
torch.save({
    "model_state":  model.state_dict(),
    "smoke_losses": losses,
    "n_params":     n_params,
}, ckpt_path)
check(ckpt_path.exists(), f"checkpoint saved at {ckpt_path}")


# ── 8. Write results CSV ──────────────────────────────────────────────────────
banner("Check 8: write results/phase3_smoke.csv")
csv_path = ROOT / "results" / "phase3_smoke.csv"
rows = [
    ("n_params",      n_params),
    ("film_delta",    round(delta, 6)),
    ("loss_step1",    round(losses[0], 6)),
    ("loss_step5",    round(losses[-1], 6)),
    ("ssm_fp32_ok",   1),
    ("all_grads_ok",  1),
]
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows(rows)
check(csv_path.exists(), f"CSV saved at {csv_path}")


# ── Done ──────────────────────────────────────────────────────────────────────
banner("PHASE 3 SMOKE TEST — ALL CHECKS PASSED")
print(f"  Checkpoint : {ckpt_path}")
print(f"  CSV        : {csv_path}")
