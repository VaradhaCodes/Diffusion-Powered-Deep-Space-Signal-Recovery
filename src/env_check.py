"""Phase 0 environment + Blackwell + Mamba-3 gate check."""
import sys, torch, platform

print("Python:", sys.version)
print("Platform:", platform.platform())
assert torch.cuda.is_available(), "CUDA not available"
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"Device: {name}  cap={cap}  torch={torch.__version__}  cuda_rt={torch.version.cuda}")
assert cap == (12, 0), f"Expected Blackwell sm_120 (12,0), got {cap}"

# Smoke: a real matmul in bf16 must run without 'no kernel' error
x = torch.randn(32, 128, 128, device="cuda", dtype=torch.bfloat16)
y = torch.nn.functional.gelu(x) @ torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
assert y.isfinite().all(), "bf16 matmul produced non-finite values"
print("PyTorch + Blackwell bf16 matmul OK")

# --- Mamba-3 gate ---
from mamba_ssm import Mamba3

for label, kwargs in [
    ("SISO", dict(d_model=128, d_state=128, headdim=64,
                  is_mimo=False, chunk_size=64, dtype=torch.bfloat16)),
    # MIMO omitted — not used in our architecture; TileLang bwd has sm_120 shared-mem bug
]:
    m = Mamba3(**kwargs).cuda()
    x = torch.randn(2, 800, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape and y.isfinite().all(), f"{label} forward failed"
    loss = y.float().pow(2).mean()
    loss.backward()
    grads_ok = all(
        p.grad is not None and p.grad.isfinite().all()
        for p in m.parameters() if p.requires_grad
    )
    assert grads_ok, f"{label} backward produced bad grads"
    print(f"Mamba-3 {label} forward+backward on Blackwell OK")

# Bidirectional sanity
m = Mamba3(d_model=128, d_state=128, headdim=64, is_mimo=False,
           chunk_size=64, dtype=torch.bfloat16).cuda()
x = torch.randn(2, 800, 128, device="cuda", dtype=torch.bfloat16)
y_fwd = m(x)
y_rev = torch.flip(m(torch.flip(x, dims=[1])), dims=[1])
assert (y_fwd + y_rev).isfinite().all()
print("Bi-Mamba-3 composition OK")

print("\n=== ALL PHASE 0 CHECKS PASSED ===")
