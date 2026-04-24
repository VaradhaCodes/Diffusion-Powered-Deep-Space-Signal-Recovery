"""V6 Batch 4 — single-run fine-tune + eval for architecture sweep.

Loads v6b3_canonical_pretrain.pt, fine-tunes on Zhu 42K, evaluates on test set.
Used by run_v6b4_sweep.py to run each sub-batch.

CLI usage (for standalone testing):
  python src/train/train_v6b4.py --sb-id sb1 --seed 0
  python src/train/train_v6b4.py --sb-id sb2 --seed 0 --kernel-size 31
  python src/train/train_v6b4.py --sb-id sb3 --seed 0 --kernel-size 31 --depth 2
"""

import sys, csv, time, math, argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.competitors import MambaNet2chCfg
from src.models.v5_model import v5_loss, _SNR_MIN, _SNR_RANGE
from src.data_zhu import zhu_train_dataset, zhu_test_dataset, TEST_CONDITIONS
from src.train.train_v5 import _calibrate_snr_estimator, estimate_snr, _preload_zhu, _norm_snr
from src.train.train_v6b3 import make_warmup_cosine, set_seed, worker_init_fn, ts

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR   = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results"
CKPT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

CANONICAL_PRETRAIN = CKPT_DIR / "v6b3_canonical_pretrain.pt"

FT_EPOCHS   = 30
FT_LR       = 3e-4
FT_LR_MIN   = 1e-6
WARMUP_STEPS = 500
FT_PATIENCE  = 10
TEST_N       = 700

# Condition → CSV column name mapping
COND_TO_COL = {
    "Awgn_Tb0d3":    "test_ber_awgn_bt03",
    "Awgn_Tb0d5":    "test_ber_awgn_bt05",
    "kb2_Tb0d3_m1d2": "test_ber_kb2_bt03_m12",
    "kb2_Tb0d3_m1d4": "test_ber_kb2_bt03_m14",
    "kb2_Tb0d5_m1d2": "test_ber_kb2_bt05_m12",
    "kb2_Tb0d5_m1d4": "test_ber_kb2_bt05_m14",
}


# ── Pretrain key remapping ─────────────────────────────────────────────────

def _remap_pretrain_keys(state_dict: dict, parallel: bool) -> dict:
    """Remap MambaNet2ch checkpoint keys (attn.*, norm1.*, bi_m2.*, norm2.*)
    to MambaNet2chCfg format (blocks.0.*). Handles serial and parallel block layouts.
    """
    new_sd = {}
    for k, v in state_dict.items():
        # Keys that are identical in both architectures
        if (k.startswith("cnn.") or k.startswith("film.") or
                k in ("bit_head.weight", "bit_head.bias",
                       "snr_head.weight", "snr_head.bias")):
            new_sd[k] = v
        elif k.startswith("attn."):
            new_sd["blocks.0." + k] = v
        elif k.startswith("norm1."):
            if not parallel:
                new_sd["blocks.0." + k] = v
            # parallel block has no separate norm1 — skip
        elif k.startswith("bi_m2.fwd."):
            suffix = k[len("bi_m2.fwd."):]
            if parallel:
                new_sd[f"blocks.0.fwd_m.{suffix}"] = v
            else:
                new_sd[f"blocks.0.bi_m2.fwd.{suffix}"] = v
        elif k.startswith("bi_m2.bwd."):
            suffix = k[len("bi_m2.bwd."):]
            if parallel:
                new_sd[f"blocks.0.bwd_m.{suffix}"] = v
            else:
                new_sd[f"blocks.0.bi_m2.bwd.{suffix}"] = v
        elif k.startswith("norm2."):
            if parallel:
                new_sd[k.replace("norm2.", "blocks.0.norm.")] = v
            else:
                new_sd["blocks.0." + k] = v
    return new_sd


def _load_pretrain(model: nn.Module, ckpt_path: Path, parallel: bool,
                   d_model: int, cnn_k1: int) -> int:
    """Load pretrain checkpoint with shape-safe key remapping. Returns n loaded keys."""
    ck  = torch.load(ckpt_path, map_location=DEVICE)
    raw = ck["model"] if "model" in ck else ck
    remapped = _remap_pretrain_keys(raw, parallel)

    model_sd = model.state_dict()
    compatible = {}
    skipped_shape = []
    skipped_missing = []
    for k, v in remapped.items():
        if k not in model_sd:
            skipped_missing.append(k)
        elif model_sd[k].shape != v.shape:
            skipped_shape.append(f"{k}: ckpt={tuple(v.shape)} model={tuple(model_sd[k].shape)}")
        else:
            compatible[k] = v

    model.load_state_dict(compatible, strict=False)
    n = len(compatible)
    total = len(model_sd)
    print(f"  Loaded {n}/{total} param tensors from pretrain ckpt.")
    if skipped_shape:
        print(f"  Shape mismatch (→ random init): {skipped_shape}")
    if skipped_missing:
        print(f"  Missing in model (skipped): {len(skipped_missing)} keys")
    return n


# ── Loss variants (SB8) ───────────────────────────────────────────────────

def _focal_bce(logits: torch.Tensor, targets: torch.Tensor,
               gamma: float = 2.0) -> torch.Tensor:
    bce_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce_elem)
    return ((1 - p_t) ** gamma * bce_elem).mean()


def compute_loss(logits, targets, snr_pred, snr_target, loss_variant: str):
    if loss_variant == "bce":
        return v5_loss(logits, targets, snr_pred, snr_target)
    elif loss_variant == "bce_ls":
        smooth = targets * 0.95 + 0.025
        bce = F.binary_cross_entropy_with_logits(logits, smooth)
        snr_loss = F.mse_loss(snr_pred, snr_target)
        loss = bce + 0.1 * snr_loss
        return loss, {"bce": bce.item(), "snr_mse": snr_loss.item()}
    elif loss_variant == "focal":
        bce = _focal_bce(logits, targets, gamma=2.0)
        snr_loss = F.mse_loss(snr_pred, snr_target)
        loss = bce + 0.1 * snr_loss
        return loss, {"bce": bce.item(), "snr_mse": snr_loss.item()}
    elif loss_variant == "focal_ls":
        smooth = targets * 0.95 + 0.025
        bce = _focal_bce(logits, smooth, gamma=2.0)
        snr_loss = F.mse_loss(snr_pred, snr_target)
        loss = bce + 0.1 * snr_loss
        return loss, {"bce": bce.item(), "snr_mse": snr_loss.item()}
    else:
        raise ValueError(f"Unknown loss_variant: {loss_variant}")


# ── Main fine-tune + eval function ────────────────────────────────────────

def run_finetune_eval(
    sb_id: str,
    seed: int,
    d_model: int       = 128,
    n_blocks: int      = 1,
    cnn_k1: int        = 7,
    parallel: bool     = False,
    grad_ckpt: bool    = False,
    loss_variant: str  = "bce",
    warmup_steps: int  = WARMUP_STEPS,
    ft_batch: int      = 128,
    ft_epochs: int     = FT_EPOCHS,
    pretrain_ckpt: Path = CANONICAL_PRETRAIN,
    description: str   = "",
) -> dict:
    """Fine-tune MambaNet2chCfg from pretrain checkpoint. Returns results dict.

    Results dict keys: val_best_ber, test_ber_overall, test_ber_awgn_bt03, ...,
                       params, train_wallclock_min, batch_size_used, grad_checkpoint_used,
                       notes, converged_epoch
    """
    set_seed(seed)
    torch.cuda.empty_cache()

    tag = f"v6b4_{sb_id}_s{seed}"
    print(f"\n{'='*70}")
    print(f"[{ts()}] {tag}  d_model={d_model}  n_blocks={n_blocks}  "
          f"cnn_k1={cnn_k1}  parallel={parallel}  grad_ckpt={grad_ckpt}")
    print(f"  loss={loss_variant}  warmup={warmup_steps}  batch={ft_batch}  {DEVICE}")
    print(f"{'='*70}")

    slope, intercept = _calibrate_snr_estimator()

    # Zhu data
    zhu_tr_raw, zhu_val_raw = zhu_train_dataset(val_frac=0.111, seed=seed)
    zhu_tr  = _preload_zhu(zhu_tr_raw,  "Zhu-train")
    zhu_val = _preload_zhu(zhu_val_raw, "Zhu-val  ")

    g = torch.Generator(); g.manual_seed(seed)
    ft_loader  = DataLoader(zhu_tr, batch_size=ft_batch, shuffle=True,
                            pin_memory=True, generator=g, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(zhu_val, batch_size=ft_batch, shuffle=False, pin_memory=True)

    # Model
    model = MambaNet2chCfg(d_model=d_model, n_blocks=n_blocks, cnn_k1=cnn_k1,
                           parallel=parallel, grad_ckpt=grad_ckpt).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}")

    # Load pretrain
    n_loaded = _load_pretrain(model, pretrain_ckpt, parallel, d_model, cnn_k1)

    # Optimizer + warmup cosine scheduler
    steps_per_epoch = math.ceil(len(zhu_tr) / ft_batch)
    total_steps     = ft_epochs * steps_per_epoch
    opt   = torch.optim.Adam(model.parameters(), lr=FT_LR)
    sched = make_warmup_cosine(opt, warmup_steps, total_steps, FT_LR_MIN / FT_LR)

    log_path = RESULT_DIR / f"{tag}_log.csv"
    with open(log_path, "w", newline="") as lf:
        lw = csv.writer(lf)
        lw.writerow(["epoch", "train_loss", "train_ber", "val_ber", "lr", "elapsed_s"])

    best_ber     = float("inf")
    patience_ctr = 0
    best_ckpt_path = CKPT_DIR / f"{tag}_ft.pt"
    notes         = ""
    wall_start    = time.time()

    def _run_one_epoch(is_train: bool, loader):
        model.train(is_train)
        tot_loss = tot_ber = n = 0
        with torch.set_grad_enabled(is_train):
            accum_step = 0
            if is_train:
                opt.zero_grad()
            for batch in loader:
                x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                snr = estimate_snr(x, slope, intercept)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                    logits, snr_pred = model(x, snr)
                snr_norm = _norm_snr(snr)
                loss, _ = compute_loss(logits.float(), y, snr_pred.float(), snr_norm, loss_variant)

                if is_train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    sched.step()

                B = len(x)
                tot_loss += loss.item() * B
                with torch.no_grad():
                    ber = ((torch.sigmoid(logits.float()) > 0.5).float() != y).float().mean().item()
                tot_ber += ber * B
                n += B
        return tot_loss / n, tot_ber / n

    actual_batch = ft_batch
    grad_ckpt_used = grad_ckpt
    conv_epoch = ft_epochs

    for ep in range(1, ft_epochs + 1):
        t0 = time.time()
        try:
            train_loss, train_ber = _run_one_epoch(True,  ft_loader)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if not grad_ckpt_used:
                # First OOM: enable gradient checkpointing
                grad_ckpt_used = True
                notes += f"[ep{ep} OOM→grad_ckpt] "
                print(f"  [ep{ep}] OOM — enabling gradient checkpointing and retrying")
                for block in model.blocks:
                    block._grad_ckpt = True
                try:
                    train_loss, train_ber = _run_one_epoch(True, ft_loader)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    # Second OOM: reduce batch
                    actual_batch = 64
                    notes += f"[ep{ep} OOM→batch64] "
                    print(f"  [ep{ep}] OOM again — reducing batch to 64 (eff=128 with accum=2)")
                    g2 = torch.Generator(); g2.manual_seed(seed + ep)
                    ft_loader_new = DataLoader(zhu_tr, batch_size=actual_batch, shuffle=True,
                                               pin_memory=True, generator=g2,
                                               worker_init_fn=worker_init_fn)
                    # grad accum = 2 to keep effective batch ~128
                    model.train()
                    tot_loss2 = tot_ber2 = n2 = 0
                    opt.zero_grad()
                    for i, batch in enumerate(ft_loader_new):
                        x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                        snr = estimate_snr(x, slope, intercept)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                            logits, snr_pred = model(x, snr)
                        snr_norm = _norm_snr(snr)
                        loss, _ = compute_loss(logits.float(), y, snr_pred.float(),
                                               snr_norm, loss_variant)
                        (loss / 2).backward()
                        if (i + 1) % 2 == 0:
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            opt.step()
                            opt.zero_grad()
                            sched.step()
                        B = len(x)
                        tot_loss2 += loss.item() * B
                        with torch.no_grad():
                            ber2 = ((torch.sigmoid(logits.float()) > 0.5).float() != y).float().mean().item()
                        tot_ber2 += ber2 * B; n2 += B
                        # Update ft_loader for future epochs
                    ft_loader = ft_loader_new
                    train_loss, train_ber = tot_loss2 / n2, tot_ber2 / n2

        _, val_ber = _run_one_epoch(False, val_loader)
        lr      = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"[{ts()}] epoch={ep:02d} tr_loss={train_loss:.4f} tr_ber={train_ber:.4f} "
              f"val_ber={val_ber:.4f} lr={lr:.2e} {elapsed:.0f}s")

        with open(log_path, "a", newline="") as lf:
            csv.writer(lf).writerow([ep, round(train_loss, 6), round(train_ber, 6),
                                     round(val_ber, 6), f"{lr:.2e}", round(elapsed, 1)])

        # Divergence check at ep 5
        if ep == 5 and val_ber > 0.30:
            print(f"  WARNING: val_ber={val_ber:.3f} > 30% at ep5 — DIVERGED")
            notes += "[DIVERGED_EP5] "
            conv_epoch = ep
            break

        if val_ber < best_ber:
            best_ber     = val_ber
            patience_ctr = 0
            conv_epoch   = ep
            torch.save({"model": model.state_dict(), "val_ber": best_ber, "epoch": ep,
                        "d_model": d_model, "n_blocks": n_blocks, "cnn_k1": cnn_k1,
                        "parallel": parallel}, best_ckpt_path)
            print(f"    *** new best val_ber={best_ber:.4f} → {best_ckpt_path.name}")
        else:
            patience_ctr += 1
            if patience_ctr >= FT_PATIENCE:
                print(f"  Early stop at ep {ep}.")
                conv_epoch = ep
                break

    wall_min = (time.time() - wall_start) / 60.0

    if best_ber > 0.30:
        print(f"  DIVERGED: best_val_ber={best_ber:.4f}. Skipping test eval.")
        return {
            "sb_id": sb_id, "seed": seed, "params": n_params,
            "train_wallclock_min": round(wall_min, 1),
            "val_best_ber": best_ber,
            "test_ber_overall": float("inf"),
            "test_ber_awgn_bt03": float("inf"), "test_ber_awgn_bt05": float("inf"),
            "test_ber_kb2_bt03_m12": float("inf"), "test_ber_kb2_bt03_m14": float("inf"),
            "test_ber_kb2_bt05_m12": float("inf"), "test_ber_kb2_bt05_m14": float("inf"),
            "batch_size_used": actual_batch, "grad_checkpoint_used": grad_ckpt_used,
            "notes": notes + "DIVERGED",
            "n_loaded_pretrain_keys": n_loaded, "converged_epoch": conv_epoch,
        }

    # ── Test evaluation ──────────────────────────────────────────────────────
    print(f"\n--- Test eval: {tag} ---")
    ck = torch.load(best_ckpt_path, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()

    per_cond_ber = {}
    for cond in TEST_CONDITIONS:
        ds  = zhu_test_dataset(cond)
        sub = torch.utils.data.Subset(ds, list(range(TEST_N)))
        tl  = DataLoader(sub, batch_size=256, shuffle=False, num_workers=2)
        n_bits = n_errors = 0
        with torch.no_grad():
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                snr = estimate_snr(x, slope, intercept)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                    logits, _ = model(x, snr)
                preds = (torch.sigmoid(logits.float()) > 0.5).float()
                n_bits   += y.numel()
                n_errors += (preds != y).sum().item()
        ber = n_errors / n_bits
        per_cond_ber[cond] = ber
        print(f"  {cond:<26} BER={ber:.4f}")

    overall = float(np.mean(list(per_cond_ber.values())))
    print(f"  {'OVERALL':<26} BER={overall:.4f}")

    # Write per-condition CSV
    csv_path = RESULT_DIR / f"{tag}_test.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "ber"])
        for cond in TEST_CONDITIONS:
            w.writerow([cond, round(per_cond_ber[cond], 6)])
        w.writerow(["OVERALL", round(overall, 6)])

    result = {
        "sb_id": sb_id, "seed": seed, "params": n_params,
        "train_wallclock_min": round(wall_min, 1),
        "val_best_ber": round(best_ber, 6),
        "test_ber_overall": round(overall, 6),
        "test_ber_awgn_bt03":    round(per_cond_ber.get("Awgn_Tb0d3",    float("inf")), 6),
        "test_ber_awgn_bt05":    round(per_cond_ber.get("Awgn_Tb0d5",    float("inf")), 6),
        "test_ber_kb2_bt03_m12": round(per_cond_ber.get("kb2_Tb0d3_m1d2", float("inf")), 6),
        "test_ber_kb2_bt03_m14": round(per_cond_ber.get("kb2_Tb0d3_m1d4", float("inf")), 6),
        "test_ber_kb2_bt05_m12": round(per_cond_ber.get("kb2_Tb0d5_m1d2", float("inf")), 6),
        "test_ber_kb2_bt05_m14": round(per_cond_ber.get("kb2_Tb0d5_m1d4", float("inf")), 6),
        "batch_size_used": actual_batch, "grad_checkpoint_used": grad_ckpt_used,
        "notes": notes.strip() if notes.strip() else "OK",
        "n_loaded_pretrain_keys": n_loaded, "converged_epoch": conv_epoch,
    }
    return result


# ── CLI (for standalone testing) ──────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb-id",        type=str,   default="test")
    ap.add_argument("--seed",          type=int,   default=0)
    ap.add_argument("--depth",         type=int,   default=1)
    ap.add_argument("--d-model",       type=int,   default=128)
    ap.add_argument("--kernel-size",   type=int,   default=7)
    ap.add_argument("--parallel",      action="store_true")
    ap.add_argument("--grad-ckpt",     action="store_true")
    ap.add_argument("--loss-variant",  type=str,   default="bce",
                    choices=["bce", "bce_ls", "focal", "focal_ls"])
    ap.add_argument("--warmup-steps",  type=int,   default=WARMUP_STEPS)
    ap.add_argument("--batch",         type=int,   default=128)
    ap.add_argument("--epochs",        type=int,   default=FT_EPOCHS)
    ap.add_argument("--pretrain-ckpt", type=str,   default=str(CANONICAL_PRETRAIN))
    args = ap.parse_args()

    result = run_finetune_eval(
        sb_id=args.sb_id, seed=args.seed,
        d_model=args.d_model, n_blocks=args.depth,
        cnn_k1=args.kernel_size, parallel=args.parallel,
        grad_ckpt=args.grad_ckpt, loss_variant=args.loss_variant,
        warmup_steps=args.warmup_steps, ft_batch=args.batch,
        ft_epochs=args.epochs, pretrain_ckpt=Path(args.pretrain_ckpt),
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
