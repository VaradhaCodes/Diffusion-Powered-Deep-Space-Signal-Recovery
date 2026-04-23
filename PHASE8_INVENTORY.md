# Phase 8 Inventory

Generated: 2026-04-22. Ground truth is local disk; this document corrects any
discrepancies between PROMPT_PHASE8.md assumptions and reality.

---

## 1. Master Results Table

All BERs in percent (%). Ensemble = average of 3-seed soft outputs. Seed std is
std([s0, s1, s2]) on overall BER.

| Model | s0 BER% | s1 BER% | s2 BER% | Mean% | Std% | Ensemble% |
|---|---|---|---|---|---|---|
| Zhu Bi-LSTM (baseline) | — | — | — | — | — | **3.122** |
| V5 (BiMamba3+FiLM+5ch) | 3.192 | 2.936 | 2.801 | 2.976 | 0.201 | 2.759 |
| BiTransformer | 2.751 | 3.003 | 2.831 | 2.862 | 0.131 | 2.640 |
| BiMamba2 | 2.790 | 2.824 | 2.821 | 2.812 | 0.018 | 2.726 |
| MambaNet | 2.292 | 2.313 | 2.319 | 2.308 | 0.014 | 2.275 |
| **MambaNet-2ch (WINNER)** | **2.312** | **2.323** | **2.319** | **2.318** | **0.006** | **2.275** |
| Ablation: NoFiLM | 2.323 | 2.476 | 2.326 | 2.375 | 0.087 | 2.284 |
| Ablation: NoPretrain | 2.771 | 2.550 | 2.579 | 2.633 | 0.119 | 2.504 |

Notes:
- Baseline has no seed splits (single training run). No variance reported.
- MambaNet and MambaNet-2ch ensemble BERs are identical to 3 significant figures
  (2.275% each). MambaNet-2ch was selected as the final model because it is the
  ablation that removes feature engineering entirely (raw 2-ch I/Q) and never
  regressed; the 0.001pp margin is noise-level.
- V5 ensemble BER in CLAUDE.md is listed as 2.76%; CSV value is 2.759% — consistent.
- Phase 7 final eval confirmed mambanet_2ch at 2.275% (mambanet_2ch_final_test.csv)
  using no TTA and no Viterbi post (both gated out).

---

## 2. Per-Condition BER Breakdown (ensemble, %)

| Condition | Baseline | MambaNet-2ch | Delta |
|---|---|---|---|
| AWGN Tb=0.3 | 1.923 | 1.044 | -0.879 |
| AWGN Tb=0.5 | 1.951 | 1.197 | -0.754 |
| KB2 Tb=0.3 m=1.2 | 2.667 | 1.867 | -0.800 |
| KB2 Tb=0.3 m=1.4 | 4.527 | 3.806 | -0.721 |
| KB2 Tb=0.5 m=1.2 | 2.759 | 1.874 | -0.885 |
| KB2 Tb=0.5 m=1.4 | 4.903 | 3.859 | -1.044 |
| **OVERALL** | **3.122** | **2.275** | **-0.847** |

(Using mambanet_2ch_final_test.csv for per-condition; OVERALL computed as
simple average of 6 conditions.)

---

## 3. Artifact Availability per Figure

| Figure | Required Data | Available? | Source CSV(s) |
|---|---|---|---|
| P1 Headline bar | baseline + winner BER | ✅ | baseline_test_results.csv, mambanet_2ch_final_test.csv |
| P2 Model comparison | all 6 ensembles + seed stds | ✅ | *_ensemble_test.csv, *_s{0,1,2}_test.csv |
| P3 Per-condition bars | per-condition BER for baseline + winner | ✅ | baseline_test_results.csv, mambanet_2ch_final_test.csv |
| P4 BER vs SNR | per-SNR BER curves | ❌ **MISSING** | Test CSVs have 6 operating-condition means only; no SNR axis |
| P5 Ablation waterfall | ablation ensemble BERs | ✅ | mambanet_*_ensemble_test.csv |
| P6 Training curves | epoch-level val_ber per seed | ✅ (partial) | mambanet_2ch_s{0,1,2}_log.csv (s0 needs two-file assembly) |
| P7 Architecture | code inspection | ✅ | src/models/competitors.py |
| A1 Seed box plot | per-seed BER per model | ✅ (baseline=1 pt) | *_s{0,1,2}_test.csv |
| A2 Significance heatmap | per-condition BERs per model per seed | ✅ (n=6, df=5) | *_s{0,1,2}_test.csv |

**P4 RESOLUTION:** Figure P4 as described (BER vs Eb/N0 curves) is not
producible — test set conditions are labeled by (channel, Tb, m) combos, not
by SNR. The 6 conditions map to different SNR regimes but no per-SNR
measurements were taken. Figure P4 will be replaced with a re-styled version
of P3 (BER per operating condition, all models, not just baseline+winner).
This is noted in the Bible and figure caption.

**P6 NOTE:** mambanet_2ch seed-0 has a split log:
- pretrain ep1-14: results/mambanet_2ch_s0_log_partial_ep1to14.csv
- pretrain ep15-20 + finetune ep1-30: results/mambanet_2ch_s0_log.csv
P6 script assembles these. Seeds 1 and 2 have complete logs (51 rows each).

---

## 4. Available Artifacts

### CSVs (`results/`)
- `baseline_test_results.csv` — baseline per-condition: condition, mse, ber, n_samples
- `baseline_train_log.csv` — baseline training: epoch, trn_loss, trn_ber, val_loss, val_ber, seconds (40 epochs)
- `v5_s{0,1,2}_log.csv` — V5 training: phase, epoch, ..., val_ber, lr (50 rows: 20 pretrain + 30 finetune)
- `v5_s{0,1,2}_test.csv` — V5 per-seed test: condition, ber
- `v5_ensemble_test.csv` — V5 ensemble test
- `bi_transformer_s{0,1,2}_{log,test}.csv` — same structure
- `bi_transformer_ensemble_test.csv`
- `bi_mamba2_s{0,1,2}_{log,test}.csv`
- `bi_mamba2_ensemble_test.csv`
- `mambanet_s{0,1,2}_{log,test}.csv`
- `mambanet_ensemble_test.csv`
- `mambanet_2ch_s{0,1,2}_{log,test}.csv` (s0 has partial companion)
- `mambanet_2ch_ensemble_test.csv`
- `mambanet_2ch_tta_ensemble_test.csv` — Phase 7 TTA result (TTA gated out)
- `mambanet_2ch_final_test.csv` — Phase 7 canonical final result
- `mambanet_no_film_s{0,1,2}_{log,test}.csv` + ensemble
- `mambanet_no_pretrain_s{0,1,2}_{log,test}.csv` + ensemble
- `phase1_ber_awgn.csv` — synth generator vs theory: snr_db, Ps_noisy, N0_measured, N0_theory, rel_err
- `phase1_stats.csv` — generator validation checks
- `phase3_smoke.csv` — V5 smoke test

### Checkpoints (`checkpoints/`)
- `baseline_ep{01..40}.pt` — Zhu Bi-LSTM, one run, 40 epochs
- `mambanet_2ch_s{0,1,2}_ft_best.pt` — winner, 3 seeds, best finetune epoch
- `mambanet_2ch_s{0,1,2}_pre_best.pt` — winner pretrain best
- `v5_s{0,1,2}_mambanet_2ch_ft_ep{01..30}.pt` — per-epoch finetune checkpoints (s0)
- (similar structures for other models)

### Scripts (`src/`)
- `data_zhu.py` — Zhu dataset loader, splits, TEST_CONDITIONS
- `synth_gen.py` — GMSK + K-dist synthetic generator
- `features/feature_extract.py` — 5-channel feature engineering (unused by winner)
- `models/v5_model.py` — V5 BiMamba3+FiLM architecture
- `models/competitors.py` — BiTransformer, BiMamba2, MambaNet, MambaNet2ch (winner)
- `models/viterbi_post.py` — Viterbi + CRF post-processors (gated out in Phase 7)
- `models/zhu_baseline.py` — Zhu Bi-LSTM reproduction
- `train/train_v5.py` — V5 training script (SNR calibrator, pretrain+finetune)
- `train/train_competitor.py` — competitor training script
- `train/train_baseline.py` — baseline training script
- `eval/eval_baseline.py` — baseline evaluator
- `eval/eval_v5_ensemble.py` — V5 ensemble evaluator
- `eval/eval_mambanet_2ch_tta.py` — Phase 7 TTA evaluator
- `eval/eval_mambanet_2ch_viterbi.py` — Phase 7 Viterbi/CRF evaluator
- `phase1_validate.py` — synth generator validator
- `phase3_smoke.py` — V5 smoke test

---

## 5. Discrepancies vs PROMPT_PHASE8.md

| Item | Prompt assumption | Actual reality |
|---|---|---|
| MambaNet-2ch description | "MHA → BiMamba2 hybrid, no FiLM, no feature engineering" | Correct — but MambaNet-2ch DOES have FiLM. NoFiLM is a separate ablation. MambaNet-2ch ablation is "no feature engineering (raw 2ch I/Q)". |
| Winner BER | "2.275%" | Confirmed. Phase 7 canonical: 2.275%. |
| Per-SNR BER curves | Assumed available | NOT available. Test CSVs are per operating condition only. P4 replaced. |
| V5 BiMamba3 architecture | "4× BiMamba3 block" | Need to verify from v5_model.py — inventory notes to check before writing §6 of Bible. |
| Training log structure | Assumed present | Confirmed. `phase,epoch,val_ber` columns exist. S0 of 2ch model has split log. |
| Baseline "ensemble" | Expected ensemble CSV | No baseline ensemble CSV exists — baseline was a single run. |
| Paired t-test | "paired across 3 seeds" | With n=3 seeds and df=2, p-values will be low-power. We use 6 per-condition BERs as paired obs instead (n=6, df=5) for each model pair. |

---

## 6. Figures Decision Log

| # | Decision |
|---|---|
| P4 | REPLACED: BER vs SNR → BER by operating condition (all models, grouped). Retains comparative value. |
| P6 | S0 log assembled from two files. Seeds 1&2 complete. |
| A2 | Paired t-test uses 6 per-condition BERs as paired observations (n=6). Low power but honest. |
| Baseline box | Single run, plotted as a point, not a box. |
