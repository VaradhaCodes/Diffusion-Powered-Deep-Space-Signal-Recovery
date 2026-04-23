# Figure — Architecture Diagram: MambaNet-2ch

## What this diagram is

A block diagram showing how data flows through the MambaNet-2ch model from raw I/Q waveform to bit decisions. Intended as a visual explanation for slides or a paper figure — NOT a network graph, more like a clean pipeline diagram.

Source: `src/models/competitors.py` — class `MambaNet2ch`

---

## Diagram structure (draw top-to-bottom or left-to-right)

### Inputs

```
[Received Frame]
   Raw I/Q signal — NO feature engineering
   Shape: (2 channels × 800 samples)
   = 100 symbols × 8 samples/symbol
   I channel (real part)    ch0
   Q channel (imaginary part) ch1
```

Note: Unlike V5 (which uses a FeatureExtractor to produce 5 channels: I, Q, amplitude, phase, differential phase), MambaNet-2ch takes raw 2-channel IQ directly. This was the Phase 6 ablation A2 that turned out to match the full model's accuracy.

---

### Stage 1 — CNN Stem

```
↓
[CNN Stem]
   Conv1d(2 → 32, kernel_size=7, padding=3) → BatchNorm1d(32) → GELU
   Conv1d(32 → 64, kernel_size=7, padding=3) → BatchNorm1d(64) → GELU
   Conv1d(64 → 128, kernel_size=8, stride=8) → BatchNorm1d(128) → GELU
   Channel sequence: 2 → 32 → 64 → 128
   The third Conv1d does BOTH: project to 128 channels AND 8× downsample (stride=8)
   Input: (2, 800) → Output: (128, 100)
```

---

### Stage 2 — Transpose

```
↓
[Transpose]
   (128, 100) → (100, 128)
   Now (sequence_length=100, d_model=128)
   One feature vector per symbol
```

---

### Stage 3 — Multi-Head Attention Block (post-norm residual)

```
↓
[Multi-Head Self-Attention]
   8 heads, d_model=128
   All 100 symbols attend to ALL other symbols
   Non-causal (no mask) → bidirectional by default
   attn_out = MHA(h, h, h)
   ↓
[Residual Add + Layer Norm]  ← POST-NORM (norm AFTER residual add)
   h = LayerNorm(h + attn_out)
   Output shape: (100, 128)
```

Note: There is NO feed-forward (FFN) layer in this block. Just MHA → residual → LayerNorm.

---

### Stage 4 — Bidirectional Mamba-2 Block (post-norm residual)

```
↓
[Bidirectional Mamba-2]
   Two separate Mamba2 instances (d_model=128, d_state=64, headdim=64, chunk_size=64)
   Forward pass:   h_fwd = Mamba2_fwd(h)
   Backward pass:  h_bwd = flip(Mamba2_bwd(flip(h, dim=1)), dim=1)
   Sum-merge:      bi_out = h_fwd + h_bwd
   O(T) complexity (linear in sequence length)
   ↓
[Residual Add + Layer Norm]  ← POST-NORM
   h = LayerNorm(h + bi_out)
   Output shape: (100, 128)
```

---

### Stage 5 — FiLM(SNR) Conditioning (applied ONCE, after both blocks)

```
↓
[FiLM Conditioning]
   SNR input: continuous float (snr_db), normalized:
     snr_norm = (snr_db − (−4.0)) / 12.0  → scalar in [0, 1]
   MLP: Linear(1 → 64) → GELU → Linear(64 → 256)
   Split 256-dim output: γ (128 dims), β (128 dims)
   Modulate: h' = (1 + γ) ⊙ h + β
     (γ and β broadcast across all 100 symbols)
   Output shape: (100, 128)
```

Note: SNR input is a single continuous scalar, NOT a binned lookup table. The MLP learns to map any SNR value in the range to a feature modulation.

---

### Stage 6 — Output Heads

```
↓
[Bit Head]  ← USED AT INFERENCE
   Linear(128 → 1) applied per symbol → squeeze
   logits: (100,) raw logits (pre-sigmoid)
   Apply sigmoid → (100,) probabilities in (0, 1)
   Threshold at 0.5 → (100,) hard bit decisions {0, 1}

[SNR Head]  ← TRAINING ONLY, dropped at inference
   Linear(128 → 1) applied after global mean over symbols
   snr_pred: scalar — predicted normalized SNR
```

---

### Training loss (not used at inference)

```
L_total = BCE_with_logits(bit_logits, bit_target)
         + 0.1 × MSE(snr_pred, snr_norm)
```

Only two terms: bit loss (primary) + SNR regression (auxiliary). No channel gain head or gain loss term.

At inference: only the Bit Head output is used.

---

## Full data flow summary

| Stage | Operation | Input shape | Output shape |
|---|---|---|---|
| Input | Raw I/Q frame (no feature extraction) | (2, 800) | (2, 800) |
| CNN Stem | Conv1d(2→32, k=7) + Conv1d(32→64, k=7) + Conv1d(64→128, k=8, stride=8) | (2, 800) | (128, 100) |
| Transpose | Reshape for sequence processing | (128, 100) | (100, 128) |
| MHA Block | Self-attention (8 heads) + residual + LayerNorm (post-norm) | (100, 128) | (100, 128) |
| BiMamba2 Block | Fwd Mamba2 + Bwd Mamba2 (sum) + residual + LayerNorm (post-norm) | (100, 128) | (100, 128) |
| FiLM(SNR) | MLP(scalar snr_norm) → γ, β; h' = (1+γ)⊙h + β | (100, 128) + scalar | (100, 128) |
| Bit Head | Linear(128→1) per symbol → sigmoid → threshold | (100, 128) | (100,) |

Total parameters: ~400K

---

## Key conceptual points to call out in the diagram

1. **Raw I/Q input** — no manual feature engineering. The model learns its own features from raw I and Q channels. This matched the full 5-channel model's accuracy (Phase 6 ablation A2).
2. **FiLM is applied ONCE** — after both MHA and BiMamba2 blocks, not interleaved with them.
3. **Post-norm residuals** — both MHA and BiMamba2 blocks use post-norm (LayerNorm after the residual add), not pre-norm.
4. **No FFN in attention block** — unlike a standard Transformer encoder, there is no feed-forward sublayer. Just MHA → add → norm.
5. **BiMamba2 = two separate instances** — explicit forward and backward Mamba2 modules, sum-merged. Not a single bidirectional SSM.
6. **SNR is a continuous scalar** — normalized to [0,1], passed directly to the FiLM MLP. No discrete bin embedding.

---

## Diagram style suggestions

- Clean rectangular boxes connected by arrows
- Group "MHA Block" and "BiMamba2 Block" in separate dashed-border "Block" containers
- Show the FiLM path as a branch coming in AFTER BiMamba2 (not before)
- Use tensor shape annotations (e.g. "(100 × 128)") on the arrows between stages
- Color the components: CNN stem = one color, MHA = second, BiMamba2 = third, FiLM = accent
- Keep the diagram tall and vertical (flows naturally top to bottom)
- Annotate the 3-layer CNN with channel counts: 2 → 32 → 64 → 128
