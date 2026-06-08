# TabPFN-3 six-position benchmark — findings (2026-06-08)

Snapshot of the opt-in [TabPFN-3 variant](../docs/adr/0003-three-way-model-comparison-no-ensemble.md)
benchmarked against the four incumbent models and **RotoWire** across all six positions.
Single seed, 2025 test split. Reproduce with the committed CLI:

```
python -m src.analysis.analysis_tabpfn_benchmark --cache-dir /tmp/tabpfn_bench --out report.md
# memory-constrained / WSL: run one at a time sharing the cache, e.g. --positions WR
```

(TabPFN is non-commercial / benchmark-only — never served. Needs `tabpfn` installed + a Prior Labs
`TABPFN_TOKEN`. See ADR-0003.)

## TL;DR

TabPFN-3 wins the **headline metric** (overall MAE on 5 of 6 positions) but is the **weakest model on the
elite/startable tier** (top-30, Q4 "boom" band). LightGBM owns that tier among the models; **RotoWire still
owns the very top of the volatile skill positions (RB/WR)**. The "best model" flips with the slice you optimize.

## Section A — full 2025 test set

**Overall MAE** (bold = best of all five):

| Pos | n | Ridge | NN | AttnNN | LightGBM | TabPFN |
|---|--|--|--|--|--|--|
| QB | 676 | 6.169 | 6.096 | 5.990 | 5.907 | **5.789** |
| RB | 1643 | 4.180 | 4.099 | 4.004 | 4.193 | **3.952** |
| WR | 2498 | 4.131 | 4.044 | 3.983 | 4.118 | **3.941** |
| TE | 1281 | 3.413 | 3.376 | 3.409 | 3.462 | **3.324** |
| K | 535 | **4.008** | 4.079 | 4.193 | 4.063 | 4.115 |
| DST | 544 | 5.089 | 5.166 | 5.105 | 5.089 | **5.001** |

**R² — full vs top-30** (TabPFN leads full R² only on QB; **LightGBM wins top-30 R² on all 6**):

| Pos | best full-R² | TabPFN full | best top-30 R² | TabPFN top-30 |
|---|--|--|--|--|
| QB | **TabPFN 0.383** | 0.383 | LightGBM 0.142 | 0.138 |
| RB | LightGBM 0.474 | 0.472 | LightGBM 0.123 | 0.087 |
| WR | LightGBM 0.392 | 0.387 | LightGBM 0.040 | 0.019 |
| TE | LightGBM 0.361 | 0.343 | LightGBM 0.060 | 0.006 |
| K | Ridge 0.018 | −0.018 | Ridge 0.006 | −0.027 |
| DST | LightGBM 0.108 | 0.101 | LightGBM 0.107 | 0.092 |

**Q1–Q4 bands (RMSE by actual-FP quartile)** — TabPFN owns the floor (Q1–Q2), LightGBM/Ridge own the ceiling
(Q4). QB illustrates the pattern (full per-position bands via the CLI):

| Model | Q1 | Q2 | Q3 | Q4 (boom) |
|---|--|--|--|--|
| Ridge | 8.47 | 6.77 | 3.86 | **9.89** |
| TabPFN | **7.23** | **6.02** | 4.10 | 10.41 |
| LightGBM | 7.99 | 6.40 | **3.83** | 9.93 |

## Section B — vs RotoWire (matched 2025; models re-scored on the same rows; no K)

| Pos | matched n | best MAE | RotoWire MAE | best RMSE | RotoWire RMSE | best top-30 R² | RotoWire top-30 R² |
|---|--|--|--|--|--|--|--|
| QB | 545 | LGBM 6.167 | 6.243 | LGBM 7.726 | 7.754 | LGBM 0.063 | 0.034 |
| RB | 1447 | TabPFN 4.295 | 4.328 | **RotoWire 6.043** | 6.043 | **RotoWire 0.156** | 0.156 |
| WR | 2295 | **TabPFN 4.171** | 4.228 | **RotoWire 5.837** | 5.837 | **RotoWire 0.074** | 0.074 |
| TE | 1165 | **TabPFN 3.502** | 3.611 | LGBM 5.071 | 5.125 | LGBM 0.078 | 0.074 |
| DST | 544 | **TabPFN 5.001** | 5.117 | LGBM 6.526 | 6.556 | LGBM 0.107 | 0.096 |

The best model **beats RotoWire's MAE at all five covered positions** (the project's "beat the expert"
milestone), but **RotoWire wins RMSE + top-30 on RB and WR** — the volatile skill-position ceiling the models
(TabPFN most of all) under-predict.

## Interpretation

- **TabPFN-3 = best *average* accuracy, worst *ceiling*.** Its calibrated predictive-distribution mean nails
  the low/mid majority (best Q1–Q2) but regresses to the mean on boom games (worst-or-near Q4 + top-30). Classic
  L1-vs-L2: it minimizes average error while shaving the peaks.
- **For the served product, LightGBM is the most decision-relevant** of the four served models — it wins
  top-30 R² across all six positions, i.e. the startable players users actually choose among. (No served model
  dominates *overall* MAE: AttnNN wins RB/WR, NN wins TE, Ridge wins K/DST, LightGBM wins QB.)
- **The Attention NN** (the custom-architecture deliverable) leads skill-position *average* MAE (best served on
  RB/WR) but is mid-pack on the elite tier.
- **The real frontier is the RB/WR ceiling** — where every model under-predicts and RotoWire still wins. Closing
  it (boom-signal features; TabPFN tuning levers from #1086 like a lower `softmax_temperature`) is the
  highest-value work.
- **K is hard for everyone** (all R² ≈ 0; Ridge marginally best; RotoWire doesn't cover it).
- **Doc nit:** ADR-0003's "LightGBM falls apart on K/DST" is now outdated for DST — LightGBM is the **best**
  model on DST R² (0.108). On K, *everyone* fails, so it's not LightGBM-specific.

## Caveats

- **Single seed.** The TabPFN MAE edge and the RotoWire top-30 advantage are within plausible seed noise on the
  thinner positions; a 3-seed rerun would firm them up.
- **Section B R² look tiny (QB 0.07 vs 0.34 in A) — that's range restriction, not a regression.** The
  RotoWire-matched subset is startable-players-only → compressed FP variance → low R². MAE/RMSE are the fair read.
- Matched-subset MAE is *higher* than full-test MAE (RB 4.30 vs 3.95): the matched rows are the higher-scoring,
  higher-variance startable set.
- **top-30 n is players × weeks** (e.g. QB 437), not 30 rows — top-30 players selected by season-total FP, then
  all their weekly rows scored.
- Metrics computed directly from per-row `pred_*_total` vs `fantasy_points` (≈0.01 off the pipeline's
  per-target-aggregated comparison print); WR Ridge carries ~0.02 known PCA-SVD context noise.
