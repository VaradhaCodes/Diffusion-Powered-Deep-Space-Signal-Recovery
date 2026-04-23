# V6 Post-Mortem: Deep Space Signal Recovery

**Date:** 2026-04-22  
**Project:** deepspace_v5 — GMSK demodulation over K-distribution fading  
**Final result:** MambaNet-2ch 2.275% ensemble BER vs Zhu et al. 2023 baseline 3.122% (−0.847 pp, −27.1% relative)  
**Purpose of this document:** Honest analysis of every decision, every flaw, every missed opportunity, and a grounded v6 roadmap. No celebration. Source of truth is RUN_LOG.md and the result CSVs.

---

## 1. What We Actually Achieved

The headline is real. On Zhu's Zenodo test set (4,200 frames across 6 channel conditions), our MambaNet-2ch model ensemble achieved 2.275% BER vs the Bi-LSTM baseline at 3.122%. The gap is −0.847 pp (−27.1% relative). Per-condition:

| Condition | Winner | Baseline | Delta |
|---|---|---|---|
| AWGN BT=0.3 | 1.044% | 1.923% | −0.879 pp |
| AWGN BT=0.5 | 1.197% | 1.951% | −0.754 pp |
| KB2 BT=0.3 m=1.2 | 1.864% | 2.667% | −0.803 pp |
| KB2 BT=0.3 m=1.4 | 3.810% | 4.527% | −0.717 pp |
| KB2 BT=0.5 m=1.2 | 1.874% | 2.759% | −0.885 pp |
| KB2 BT=0.5 m=1.4 | 3.860% | 4.903% | −1.043 pp |

Statistically: paired t-test (n=6 conditions), p < 0.001 vs baseline. The win is real.

But the headline obscures three things that matter enormously:

1. **The win comes almost entirely from synthetic pretraining, not the architecture.** Without 500K synthetic pretrain, the same model scores 2.504% — the architecture itself beats baseline by only 0.618 pp instead of 0.847 pp. The attention+Mamba hybrid helps, but pretrain helps more.

2. **The training dataset is not the paper's dataset.** Zhu et al. trained on 63,000 samples. Zenodo has 42,000. We trained on 42,000. Our baseline reproduction at 3.122% matches the paper, but we are not competing on a level field — the baseline we reproduced is a weakened version of the original (15K fewer training samples, no access to the paper's exact split or augmentations). We beat a weakened baseline.

3. **KB2 m=1.4 heavy scintillation is still the weak point.** At 3.810–3.860%, we're at ~79% BER overhead relative to AWGN. This condition is the hardest, most physically realistic, and we did least well on it proportionally. The model does not understand the channel physics — it pattern-matches.

---

## 2. Dataset Situation

### 2.1 The 63K vs 42K Gap

CLAUDE.md initially said 63,000/7,875/4,200 train/val/test (from the Zhu paper Table 2). The actual Zenodo upload has 42,000 train + 8,400 test. The paper was trained on a private dataset version with 21K more training samples. We never had access to those.

This matters because:
- Our baseline reproduction (3.122%) matches the paper despite fewer training samples, which means either the Bi-LSTM saturates early OR the extra 21K samples in the paper's training set were not particularly helpful for the baseline model.
- For our model, we compensated with 500K synthetic samples for pretraining. This substitution is not equivalent — the synthetic data quality and distribution differ from real measured data.
- Any comparison to Zhu et al. numbers should note this gap. We beat a reproduced baseline, not a definitively stronger version of the paper's model.

### 2.2 Synthetic Data Quality

The synthetic generator (`src/synth_gen.py`) produces GMSK signals with K-distribution fading (Gamma envelope × Rayleigh phase). The Phase 1 validation showed Eb/N0 calibration error < 0.3% and correct K-distribution second moment (E[|h|²] = b² = 4, α=10, b=2).

What was never validated:
- **Higher-order statistics of the channel.** K-distribution is parameterized by (m, α, b). We matched the Phase 1 mean power check, but not the kurtosis, LCR (level crossing rate), or autocorrelation structure of the channel impulse response. Real ionospheric K-distribution channels have time correlation. Our simulator generates i.i.d. per-frame fading, which is correct for the Zhu dataset's structure but not for realistic link budgets.
- **Modulation accuracy at BT=0.3 vs BT=0.5.** The GMSK pulse shape is generated from a Gaussian filter applied to an NRZ sequence. We didn't verify that the BT products in our simulator produce bit error curves matching theory to within 0.5 dB at all SNRs — only the AWGN curves were spot-checked.
- **SNR range in synthetic pretrain.** We pretraining on SNR −4 to +8 dB, matching Zhu's test conditions. But KB2 channels with heavy scintillation effectively push instantaneous SNR well below −4 dB in fade events. The pretrain distribution doesn't cover the tail.

### 2.3 SNR Estimation and the FiLM Calibration Problem

FiLM conditioning relies on an estimated per-frame SNR. The SNR estimator (`src/train/train_v5.py`) uses a power-based linear model calibrated on synthetic AWGN data: `snr_db = slope * log10(rx_power) + intercept`, where slope and intercept are fitted by least-squares on synthetic samples.

**This estimator is fundamentally miscalibrated for KB2 channels.** In K-distribution fading:
- The received power is the product of the signal power, the channel gain (K-distributed), and noise. The channel gain is NOT unity.
- A power measurement on a KB2 frame gives a compressed, biased estimate of SNR — the fade events push apparent SNR lower than the true system SNR.
- RUN_LOG.md Phase 4 NOTE explicitly says: "FiLM conditioning limited for K-dist (SNR estimate compressed to -4 to 0 dB range for KB2)." The estimator clips everything into a narrow band.

The consequence: FiLM is effectively a fixed bias for KB2 inputs. The ablation result (NoFiLM = +0.009 pp loss) reflects this — when the SNR estimate is uninformative, removing FiLM costs almost nothing. FiLM adds ~0.009 pp on average, with the marginal gains concentrated entirely on AWGN conditions where the power estimate is accurate. On KB2 m=1.4, NoFiLM actually beat the full model by 0.018 pp (3.750% vs 3.810%), which means FiLM is actively hurting on that condition.

**We shipped a broken SNR conditioning component and called it an ablation result.** The correct comparison is: FiLM with accurate SNR vs FiLM with wrong SNR vs no FiLM. We only measured the last two.

---

## 3. Architecture Decisions

### 3.1 What Actually Worked: The MHA→BiMamba2 Hybrid

MambaNet-2ch is: CNN stem → MHA (8 heads, post-norm, no FFN) → BiMamba2 (fwd+bwd sum-merge, post-norm) → FiLM → bit head.

The Phase 5 competition results tell a clear story:

| Model | Ensemble BER | Architecture |
|---|---|---|
| MambaNet-2ch | 2.275% | MHA + BiMamba2 |
| BiTransformer | 2.640% | MHA only (2 layers) |
| BiMamba2 | 2.725% | BiMamba2 only |
| V5 (BiMamba3) | 2.759% | BiMamba3 only |
| Zhu Bi-LSTM | 3.122% | CNN + Bi-LSTM |

The gap between BiMamba2 (2.725%) and MambaNet (2.275%) is 0.450 pp — larger than the entire gap between BiMamba2 and the Zhu baseline (0.397 pp). Adding attention on top of BiMamba2 delivers more improvement than switching from Bi-LSTM to BiMamba2 in the first place. The attention component is the dominant architectural win.

Why does MHA→BiMamba2 outperform each alone? The most plausible explanation: at T=100 symbols, full self-attention (O(T²) = 10,000 pairs) can learn global ISI patterns — which bits interact across the entire 100-symbol frame due to GMSK pulse spreading and channel memory. Then BiMamba2 propagates these attended representations forward and backward, refining local structure. The two mechanisms are genuinely complementary here.

What we don't know: **which heads are firing on what.** We never ran attention weight visualization. We don't know whether the 8 MHA heads specialize (e.g., some heads track the differential encoding structure, others track ISI across BT products). This is a critical open question for v6.

### 3.2 What Got Lucky: BiMamba2 vs BiMamba3 Equivalence

BiMamba2 (2.725%) and V5/BiMamba3 (2.759%) are effectively equivalent on this task. The gap is 0.034 pp (< 1.2% relative). This was the main "Phase 5 key finding" — Mamba-2 and Mamba-3 are interchangeable here.

This is a lucky result, not a design choice. We chose Mamba-3 in Phase 0 for principled reasons (the original project goal), then discovered that Mamba-3 is unavailable in PyPI and requires building from source. We kept it. Had we known Mamba-2 was equivalent on this dataset, we could have saved considerable setup time in Phase 0.

The implication: the SSM order (2 vs 3) doesn't matter for 100-step GMSK sequences. The sequence is too short and too structured for higher-order SSM dynamics to express additional representational capacity.

### 3.3 What Was Suboptimal: The CNN Stem

The CNN stem is Conv1d(2→32, k=7) → BN → GELU → Conv1d(32→64, k=7) → BN → GELU → Conv1d(64→128, k=8, stride=8) → BN → GELU. It takes 800 samples and produces 100 symbol embeddings (exactly 8 samples/symbol).

The stride-8 final layer is doing two things simultaneously: projecting channels (64→128) and downsampling (stride 8). This is a non-obvious design choice with two potential problems:

1. **Downsampling before any inter-symbol context.** The stride-8 conv locks in the 8-sample alignment. There's no shift-invariance — a 1-sample timing offset in the frame would produce a different set of 100 symbol embeddings. The model is never tested with timing offsets.

2. **The 7-sample kernel is mismatched to the GMSK pulse.** GMSK at BT=0.3 has a pulse that spans approximately 5–6 symbol periods = 40–48 samples. The 7-tap kernel in the first two layers covers only 7 samples (< 1 symbol period). It's too narrow to capture ISI structure directly — the CNN stem is essentially doing local feature extraction within a symbol, and the attention layer has to pick up all the inter-symbol ISI. This works, but a wider CNN (k=31 or k=47) might allow the CNN stem to absorb more ISI before attention, potentially reducing the load on the attention layer and allowing a shallower transformer to match performance.

### 3.4 What Was Never Tried: Depth vs Width

The model is shallow: one MHA block, one BiMamba2 block, both at d_model=128 with 8 heads. The total parameter count is ~400K — far below what modern signal processing models use.

We never swept depth (e.g., 2×MHA + 2×BiMamba2, or alternating MHA-BiMamba2-MHA-BiMamba2). We never swept width (d_model=256). We chose these values from MambaNet's original paper (Luan et al., OFDM context) and applied them directly to GMSK without tuning. This is a significant missed opportunity.

The 400K parameter count is almost certainly in the under-parameterization regime for this task. For reference: the Zhu Bi-LSTM baseline has fewer parameters and achieves 3.122%. A 2× or 4× wider model might push BER below 2.0%.

---

## 4. Training Setup Critique

### 4.1 The Seed 0 Pretrain Bug

Phase 4 RUN_LOG.md: "seed=0 used ep01 pretrain checkpoint for fine-tune (bug, since fixed); seeds 1/2 used ep20."

Seed 0 V5 final BER = 3.192%. Seeds 1 and 2 = 2.936%, 2.801%. The seed 0 number makes the V5 ensemble BER look worse than it is: with ep20 pretrain, seed 0 might have reached ~2.7–2.8%, making the V5 ensemble potentially around 2.6–2.7% instead of 2.759%. 

This bug propagates into the Phase 5 ablation comparison (A3-NoPretrain=2.504%) — the MambaNet training used the correct pretrain checkpoint, so the ablation comparison is clean. But the V5 numbers in the paper draft and documentation are contaminated by seed 0's defective initialization.

The honest conclusion: V5 (BiMamba3) with correct training for all 3 seeds might close some of the gap with MambaNet. We can't know without rerunning seed 0.

### 4.2 Synthetic Pretrain: 500K Is an Arbitrary Choice

We pretrained on 500K synthetic samples. The ablation (A3-NoPretrain) shows the pretrain contributes +0.229 pp to BER when removed. This is the largest single contributor — larger than the entire architectural advantage of MambaNet over BiMamba2.

But we never measured:
- **Pretrain scaling laws.** Is 500K optimal? What does BER look like at 50K, 200K, 1M, 5M synthetic samples? Given the A3 result (+0.229 pp from removing all pretrain), there's likely significant performance left on the table if the model hasn't saturated pretrain data.
- **Pretrain distribution diversity.** We pretrained on the same 6 condition types as the test set (AWGN BT∈{0.3,0.5}, KB2 with m∈{1.2,1.4}, SNR −4 to 8 dB). This is a favorable setup. In a real deployment, the pretrain distribution should be broader (wider SNR range, additional m values, multipath, Doppler).
- **Pretrain epochs vs data.** We ran 20 pretrain epochs on 500K samples = 10M gradient updates. Was this over-pretrained? Under-pretrained? We have no pretrain val BER convergence analysis.

CSRD2025 (200TB synthetic radio dataset, from web search results) would give access to a vastly larger and more diverse pretraining corpus. If synthetic data scaling is the primary lever (which our ablation suggests), access to 100× more diverse synthetic data could be transformative for v6.

### 4.3 Loss Function: The 0.1× SNR Auxiliary Weight

The training loss is:
```
L = BCE(bit_logits, bit_targets) + 0.1 × MSE(snr_pred, snr_norm)
```

The 0.1 weight was chosen by analogy with other multi-task learning setups and was never ablated. The SNR head is discarded at inference. Its role is to provide a gradient signal that forces the model's mean-pooled representation to encode SNR — which in turn helps FiLM conditioning by making the pre-FiLM features SNR-aligned.

But if FiLM conditioning is broken for KB2 (as argued in §2.3), then the SNR auxiliary loss is providing a distorted gradient signal. The model is being regularized toward an SNR representation that will be used with a miscalibrated estimator at inference. It's possible the 0.1× weight is the right call in spite of this, but it's untested.

### 4.4 Learning Rate Schedule: No Warmup

Both pretrain and finetune phases use cosine LR decay (1e-3 → 1e-5 for pretrain, 3e-4 → 1e-6 for finetune). There is no LR warmup.

For the Mamba-2 SSM parameters (which are fp32-pinned), the initial gradient magnitude from the SSM path can be large before the model has learned coherent representations. A brief warmup (e.g., 500 steps, linear 0 → peak LR) is standard practice for hybrid architectures. We skipped it and got acceptable results, but it's possible warmup would reduce seed variance and produce better checkpoints at intermediate training steps.

### 4.5 Finetune Initialization: Best Val vs Last Pretrain

The finetune stage loads the **last pretrain epoch** (not the best pretrain val BER checkpoint). This choice was made explicitly after the seed 0 bug — seed 0 accidentally loaded ep01 (best early Zhu val checkpoint), which was nearly untrained. The fix was to always use the last epoch.

This is actually the correct choice for a two-phase curriculum: the pretrain phase's objective is not to minimize Zhu val BER, it's to build general feature representations. The last-epoch pretrain checkpoint has seen the most synthetic data. But "last epoch" doesn't necessarily mean "best feature representations" — if the pretrain phase is over-fit to synthetic data, the last checkpoint might have worse generalization to Zhu data than an early-stopped checkpoint. We never measured this.

---

## 5. Failure Analysis

### 5.1 TTA: Time-Reversal Is Physically Invalid for GMSK

Time-reversal TTA flipped the input along the sample dimension (dim 2) and averaged the model's probabilities with the un-flipped output. Result: 23.249% BER (vs 3.752% baseline) — a catastrophic +19.5 pp regression.

The failure reason is clear in retrospect: GMSK is a **differentially encoded continuous-phase modulation**. The differential encoding means each bit is encoded as a phase transition relative to the previous symbol. Reversing the time axis reverses the direction of the phase transitions — the model is now decoding a signal with the wrong differential relationship. The bit probability at position t in the reversed signal doesn't correspond to bit t in the original message. This is not a learnable transform — it's a physical incompatibility.

The correct TTA augmentations for GMSK would have been:
- **SNR jitter:** Add ±0.5 dB of simulated noise to the input before inference, average the resulting probabilities. This works because adding a small noise perturbation is like sampling from the SNR uncertainty distribution.
- **Phase rotation:** Rotate the I/Q phasor by θ ∈ {0°, 90°, 180°, 270°} (only valid for BPSK/QPSK symmetry; GMSK is not phase-symmetric, so this would also fail).
- **Nothing.** For GMSK, the correct TTA is no TTA. The signal has a definite physical orientation and encoding direction. We spent Phase 7 time discovering this from data rather than reasoning about it from physics first.

### 5.2 Viterbi: Correct Implementation, Wrong Integration

The Viterbi post-processor is correctly implemented as a Viterbi decoder on the GMSK trellis with 2 states (phase 0, phase π/2). But the branch metrics use **state-independent log-probabilities** from the neural network:
```python
log_p[t, 0] = log(1 - p_bit[t])   # bit = 0
log_p[t, 1] = log(p_bit[t])        # bit = 1
```

This is mathematically equivalent to thresholding with Viterbi enforcement of bit-sequence validity, but it ignores the differential encoding and GMSK trellis transition structure. A proper GMSK Viterbi uses branch metrics based on Euclidean distance in the I/Q constellation (or log-likelihood from a matched filter), not posterior bit probabilities.

The result — 3.752% (identical to baseline) — makes sense: the neural network already produces probabilities that implicitly capture the trellis constraints via bidirectional training. Viterbi adds nothing because the model hasn't made errors that violate trellis validity; its errors are in the probability values, not in trellis-invalid sequences.

The missed opportunity: **BCJR-informed training**, not Viterbi post-processing. If the training loss incorporated explicit GMSK sequence-level likelihoods (via a BCJR forward-backward pass as a differentiable layer), the model could learn bit posteriors that are explicitly calibrated to the GMSK trellis, rather than learning trellis structure implicitly. This was never attempted.

### 5.3 The Power Outage Interruption (A2 Seed 0)

Phase 6 RUN_LOG.md: "A2 seed 0 resumed from ep14 after power outage." The A2 (MambaNet-2ch) training for seed 0 was interrupted mid-finetune and resumed from ep14. The partial log is saved as `results/mambanet_2ch_s0_log_partial_ep1to14.csv`.

The final seed 0 BER for A2 is 2.312%. Across seeds: 2.312%, 2.323%, 2.319%. These are extremely tight (std=0.006%), which suggests the resume was clean. But there's a subtle risk: if the LR schedule was not correctly restored from the checkpoint state (only the model weights were restored, not the optimizer + scheduler state), the final 16 epochs of seed 0 trained with the wrong LR profile. If the cosine schedule was restarted from ep0 of a 30-ep schedule at ep14, the effective LR for eps 14-30 was too high relative to seeds 1 and 2. The result matches seeds 1 and 2 closely enough that this probably didn't matter, but it's untested.

---

## 6. Evaluation Validity

### 6.1 Statistical Power: 4,200 Samples Across 6 Conditions

The test set has 700 frames per condition × 6 conditions = 4,200 total. At 100 bits/frame, this is 420,000 bit errors assessed. The overall BER is measured as total errors / total bits.

For the winner at 2.275% BER: expected errors ≈ 9,555 bits. The standard error on this estimate is approximately sqrt(p(1-p)/N) = sqrt(0.02275 × 0.97725 / 420000) ≈ 0.00023 = 0.023 pp. So the 2.275% figure is precise to ±0.05 pp (2-sigma). The gap to baseline (0.847 pp) is ~36 standard errors. The headline result is statistically unambiguous.

The concern is at the per-condition level: 700 frames × 100 bits = 70,000 bits per condition. For KB2 m=1.4 at 3.810%, expected errors ≈ 2,667. Standard error ≈ sqrt(0.0381 × 0.9619 / 70000) ≈ 0.00073 = 0.073 pp. The KB2 m=1.4 winner vs A1-NoFiLM gap is only 0.060 pp (3.810% vs 3.750%) — this is within 1 standard error. **The per-condition FiLM ablation comparison for KB2 m=1.4 is not statistically significant at the individual condition level.**

The paired t-test (n=6) across conditions is significant, but this is because the test detects the global pattern across all 6 conditions, not because KB2 m=1.4 individually is resolved.

### 6.2 Single Run Baseline: A Reproducibility Problem

The Zhu baseline was run once (no multiple seeds — the baseline training script exits after one run). We don't know the seed variance of the Bi-LSTM baseline. If the baseline has high seed variance (like V5 BiMamba3, which showed std=0.199%), the 3.122% could be a lucky or unlucky seed, and the true baseline mean might differ by ±0.2 pp. The MambaNet-2ch result uses a 3-seed ensemble, which is more robust. We are comparing a robust ensemble against a single run.

The paired t-test reported (p<0.001) treats 3.122% as the true baseline BER, which it isn't — it's a point estimate with unknown variance. A proper comparison would require the Zhu baseline to also be run with multiple seeds.

### 6.3 Distribution Shift: Zenodo Test vs Paper Test

The Zhu paper reports results on a 4,200-sample test set, same as Zenodo. But we don't know if the Zenodo test set is the same 4,200 samples used in the paper's reported results, or a different draw from the same distribution. If the paper chose their test set after observing model performance, there could be selection bias. More likely, the paper's test set was fixed before training. We cannot verify this.

---

## 7. What Was Never Tried

### 7.1 Architecture: Alternatives Not Explored

**Deeper networks.** One MHA + one BiMamba2 is very shallow. Standard recommendation for sequence models on 100-step problems: 4–6 layers. We never tested 2×(MHA+BiMamba2) or 4× depth.

**Cross-attention between forward/backward streams.** Instead of summing BiMamba2 forward and backward outputs, use cross-attention: forward attends to backward and vice versa. This could capture asymmetric ISI structure (GMSK ISI is causal-heavy at BT=0.3 because the Gaussian filter is causal in practice). We used sum-merge because it's simpler and comes from the MambaNet reference.

**Conformer (Conv+Attention).** Conformer blocks (used in speech recognition: attention + convolution in parallel, with relative positional encoding) have shown excellent performance on fixed-length sequences of the same length scale (100 tokens is common in speech). GMSK demodulation has much in common with phoneme recognition — both involve sequence classification with local and global dependencies.

**Physics-informed input representation.** Instead of raw I/Q or the 5-channel feature set (I, Q, A, φ, Δφ), use features derived from GMSK differential decoding theory:
- The instantaneous phase differences between adjacent samples
- The in-phase and quadrature components relative to the expected GMSK phasor trajectory
- The log-likelihood ratio from a matched filter for each bit position
Any of these would inject domain knowledge that the CNN stem currently has to learn from scratch.

**SNR-conditioned architecture switching.** At high SNR (>6 dB), a shallow model suffices — errors are rare and the channel is nearly AWGN. At low SNR (<−2 dB), deeper processing is needed. A mixture-of-experts or conditional computation approach could allocate more capacity to hard cases. We used a single fixed architecture for all SNR conditions.

### 7.2 Training: Improvements Not Attempted

**Pretrain scaling.** From 500K to 5M or 50M synthetic samples. The A3 ablation shows pretrain is the dominant lever. Scaling the pretrain data is the most directly motivated next step.

**Curriculum learning in finetune.** After synthetic pretrain, fine-tune starting from easy conditions (AWGN, high SNR) and progressively adding harder ones (KB2 m=1.4, low SNR). This is standard curriculum learning and could improve convergence on the hard conditions.

**Data augmentation during finetune.** The Zhu training set has 42K samples. With augmentation (SNR jitter ±0.5 dB during training, random circular time-shift within a symbol period, random phase rotation by ±small angle), the effective training set size increases without synthetic data bias. We never applied any augmentation during finetune.

**Label smoothing.** BCE loss with hard 0/1 targets can be overconfident. Label smoothing (soft targets of 0.05/0.95 instead of 0/1) often improves calibration and generalization, especially when training labels are noise-free ground truth (as they are here). Not tried.

**Focal loss.** To emphasize rare errors (hard examples in heavy fading conditions). Standard BCE weights all frames equally. A focal loss would upweight KB2 m=1.4 frames where the model still struggles.

**Better SNR estimation.** Replace the linear power estimator with a neural SNR estimator trained end-to-end. Or use the model's own confidence (mean entropy of bit predictions) as a proxy for SNR. The current estimator is explicitly calibrated on AWGN and produces wrong estimates for KB2.

### 7.3 Post-Processing: BCJR Integration

The missed opportunity from §5.2: incorporate a differentiable BCJR (forward-backward) layer as the final stage. The BCJR layer would take per-bit log-probabilities from the neural network and compute exact MAP bit decisions on the GMSK trellis. This is differentiable (the forward-backward algorithm can be written in log-space with smooth max approximations) and would enforce GMSK sequence validity in the gradients, not just at inference. The expected gain is modest on clean channels but could be 0.1–0.3 pp on KB2 m=1.4 where error propagation from one wrong bit can corrupt multiple subsequent bits under the differential encoding.

### 7.4 Evaluation: Missing Analyses

**BER vs SNR curves.** The test set has per-frame estimated SNR. We could sort frames by estimated SNR and compute BER in 1-dB bins to produce a waterfall curve. This would show where the model breaks down (likely below −2 dB for KB2 m=1.4) and whether it approaches theory. We have all the data needed for this — it was noted in Phase 8 as "not produced because test CSVs have per-condition means only." This is wrong — the individual frame CSVs exist in `results/*_s{0,1,2}_test.csv` and could be re-parsed to extract per-SNR data.

**Calibration curves.** Does sigmoid(logit) = 0.4 actually mean the model is wrong 40% of the time on that bit? Probability calibration matters if these outputs are used in downstream decoders (e.g., as channel LLRs for LDPC). We never measured calibration.

**LDPC coded BER.** This was designated "optional, future work" in Phase 7. For a real communications application, the FEC-decoded BER at a target coded BER (e.g., 10^-5) is the relevant metric, not raw uncoded BER. Our 2.275% uncoded BER maps to a coding gain that depends on the LDPC code rate, which we never measured.

---

## 8. Current SOTA Context (April 2026)

Based on web searches conducted for this post-mortem:

**IQUMamba-1D (Springer, January 2026):** Applies Mamba-based SSMs to signal separation in the I/Q domain. Direct relevance — shows Mamba is being actively adopted for raw I/Q processing beyond our project timeline.

**MambaNet (Luan et al., arXiv 2601.17108, January 2026):** The paper we derived our architecture from. They apply MHA→BiMamba in an OFDM channel estimation context. Our work extends this to GMSK demodulation with GMSK-specific challenges (differential encoding, GMSK ISI). No public follow-up from Luan et al. was found.

**Mamba-3 (ICLR 2026):** The version we built from source. The academic community is using it; our source-build approach was correct given PyPI availability issues.

**M1 — Test-Time Compute for Mamba (arXiv 2504.10449, April 2026):** Explores test-time computation scaling for SSMs. Conceptually relevant to TTA: instead of data augmentation TTA, run multiple inference passes with different sequence orderings and aggregate. For GMSK, this would require understanding which orderings are physically valid (not time-reversal). Not directly applicable but points to an active research direction.

**CSRD2025 (200TB synthetic radio dataset):** Announced at CSRD2025 symposium. This would provide access to orders-of-magnitude more diverse synthetic data for pretraining. Given that our A3 ablation shows pretrain is the dominant performance lever, access to CSRD2025 data for v6 pretraining could be significant.

**RWKV-TS and xLSTM 7B (2025-2026):** Competing linear-time sequence model architectures. No GMSK or wireless communications papers found using these as of April 2026. Likely not the right direction for a 100-step fixed-length sequence task where attention is already tractable.

**FiLM + wireless communications:** No papers found combining FiLM conditioning with channel-adaptive demodulation (other than our work). This appears to be a genuine unexplored area.

**PINN + GMSK:** No papers combining physics-informed neural networks with GMSK demodulation. Open territory.

---

## 9. Where We Got Lucky

1. **The task length (T=100) makes attention tractable.** O(T²) = 10,000 — small enough that a single MHA block with 8 heads runs in milliseconds. If the task were T=1000 symbols, attention would be impractical and the architectural choice would need to change entirely. We didn't design for this — we got fortunate that the sequence length is GMSK's natural 100-symbol frame.

2. **The synthetic channel model was good enough.** Our GMSK+K-dist simulator produced samples that transferred well to real Zenodo data. We could have gotten this wrong — wrong BT product pulse shape, wrong K-dist parameterization, wrong noise model. The Phase 1 validation caught the obvious calibration errors, but we never did a rigorous distribution matching test. The model trained on synthetic data and transferred to Zhu real data. That's not guaranteed.

3. **The seed variance of MambaNet-2ch was extremely low (std=0.006%).** Across 3 seeds, the model varied by only 0.011 pp (2.312–2.323%). This made the ensemble essentially equal to the best individual seed. We didn't engineer this — it emerged from the architecture. A high-variance model would have needed more seeds to produce a reliable ensemble estimate.

4. **BiMamba2 ≈ BiMamba3 on this task.** We spent Phase 0 fighting to install Mamba-3 (building from source, fixing sm_120 issues). Then discovered the architecture we actually needed (MambaNet) performs equally well with Mamba-2, which is in PyPI and stable. The Phase 0 investment was unnecessary for the final result.

5. **The MHA block without FFN worked.** Standard Transformer encoders have an FFN sublayer after attention. We removed it (following MambaNet's design). This could have hurt badly if the MHA output needed a feedforward transform to be useful as BiMamba2 input. It didn't. The BiMamba2 block implicitly provides the nonlinearity that FFN normally handles.

---

## 10. V6 Roadmap

### Priority 1: Fix the SNR Estimator for KB2

Before any architecture changes, fix the fundamental calibration problem. A neural SNR estimator (small MLP operating on frame statistics) trained on a mix of AWGN and K-dist frames would replace the broken linear power model. If FiLM conditioning actually received accurate SNR estimates, the A1-NoFiLM gap would increase substantially. Current A1 gap = 0.009 pp; with accurate SNR, plausible gap = 0.1–0.3 pp. This single fix, with no other changes, might push BER from 2.275% to ~2.0%.

Implementation: add a small neural SNR estimator as a frozen auxiliary network, trained offline on synthetic data with known SNR labels. Input: frame I/Q features (power, kurtosis, zero-crossing rate). Output: SNR estimate. Calibrate separately for AWGN vs KB2 using the per-condition BT value as a context variable (which IS known at test time from the frame structure).

### Priority 2: Pretrain Scaling

Run pretrain at 50K, 200K, 500K, 2M, 5M samples and plot BER vs pretrain data size. If the curve hasn't saturated at 500K (likely — we have no saturation evidence), increase to 2M–5M. Access CSRD2025 data for diverse out-of-distribution pretraining to improve robustness.

Expected gain: 0.1–0.3 pp BER improvement from 500K → 5M synthetic. Basis: pretrain accounts for 0.229 pp total contribution; scaling to 10× more data plausibly captures a portion of remaining headroom.

### Priority 3: Depth Sweep

Test 1×, 2×, 4× layers of MHA+BiMamba2. Current: 1×. Expected optimum: 2×–3× at 800K–1.2M parameters. This is the highest-reward architectural experiment that was completely skipped in v5.

### Priority 4: BCJR Integration

Implement a differentiable BCJR layer as the final stage, replacing the per-symbol sigmoid threshold. Train end-to-end: BCE loss computed on BCJR MAP decisions, not per-symbol probabilities. This enforces GMSK sequence validity in the gradient and is particularly likely to help on KB2 m=1.4 where error propagation is significant.

### Priority 5: BER vs SNR Analysis and Calibration

Using the existing result CSVs (individual-frame data), produce BER-vs-SNR waterfall curves. Identify where the model breaks down. Use calibration curves (reliability diagrams) to assess whether the model's output probabilities are well-calibrated. If miscalibrated, apply temperature scaling.

### Priority 6: LDPC Coded BER

Implement a simple rate-1/2 LDPC code (e.g., DVB-S2 LDPC, publicly available). Pass the model's bit log-likelihoods (not hard decisions) as channel LLRs into a standard belief propagation decoder. Measure coded BER at target 10^-5. This is the metric that matters for real communications links.

### What v6 Should NOT Do

- Change the MHA→BiMamba2 core — it's the right architecture for this task and the gain is established.
- Rerun seeds of the current model — the seed variance is already tight. More seeds give diminishing returns.
- Add more ablations for their own sake — we have the necessary ablation evidence. New experiments should target specific improvements.
- Use time-reversal TTA — this breaks GMSK. Any TTA must respect differential encoding direction.

---

## 11. Summary of Issues by Severity

| Issue | Severity | Impact | Fixable in v6? |
|---|---|---|---|
| SNR estimator miscalibrated for KB2 | High | FiLM conditioning provides wrong signal for hardest condition | Yes — neural estimator |
| Baseline not multi-seed | Medium | Cannot compare ensemble vs single-run fairly | Yes — rerun baseline 3× |
| Pretrain scaling not measured | Medium | Could leave 0.1–0.3 pp on table | Yes — data scaling sweep |
| Seed 0 V5 bug contaminates V5 comparison | Medium | V5 ensemble BER is too pessimistic | Yes — rerun V5 seed 0 |
| Model depth not swept | Medium | Almost certainly under-parameterized | Yes — depth sweep |
| No attention weight analysis | Low | Can't explain what the model learned | Yes — attention viz |
| No BER vs SNR curves | Low | Missing analysis, not wrong results | Yes — from existing CSVs |
| No calibration curves | Low | Matters for coded BER / LLR output | Yes — temperature scaling |
| KB2 m=1.4 stat. power | Low | Per-condition FiLM ablation insignificant | Partially — more test frames needed |
| No LDPC coded BER | Low | Not a fair comparison to operational systems | Yes — DVB-S2 LDPC |
| No curriculum or augmentation | Low | May leave finetune performance on table | Yes — augmentation pass |

---

*Source of truth: RUN_LOG.md and results/ CSVs. All numbers in this document trace to either a results CSV file or an explicit RUN_LOG.md entry.*
