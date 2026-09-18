# WR PR1564 causal investigation

Retrospective 2025 comparison: 2,761 identical player-weeks, corrected shared-component actuals, three seeds, four full-production model families. Twelve fitted cells produce 24 scored intervention cells.

## Complete correction

| Model | Baseline MAE | Fixed MAE | Delta MAE | Delta RMSE | Delta bias |
|---|---:|---:|---:|---:|---:|
| ridge | 3.8657 | 4.1196 | +0.2539 ± 0.0000 | +0.0550 ± 0.0000 | +0.6078 |
| nn | 3.7848 | 3.9452 | +0.1604 ± 0.0227 | -0.0001 ± 0.0137 | +0.8044 |
| attn_nn | 3.7925 | 3.9007 | +0.1082 ± 0.0810 | -0.0800 ± 0.0442 | +0.7140 |
| lgbm | 3.9102 | 4.0233 | +0.1131 ± 0.0146 | +0.0270 ± 0.0073 | +0.3755 |

## Attribution of MAE change

Exact Shapley allocation averages all six factor orders, allocating interactions rather than ignoring them. Values sum to each model's complete MAE change.

| Model | Depth normalization | Availability semantics | Other source/identity changes |
|---|---:|---:|---:|
| ridge | +0.2547 | +0.0004 | -0.0012 |
| nn | +0.1395 | +0.0096 | +0.0113 |
| attn_nn | +0.1468 | -0.0291 | -0.0095 |
| lgbm | +0.1147 | -0.0016 | -0.0001 |

## Fixed-model mechanism

Depth changes below rescore the fully corrected fitted models; weights and fitted preprocessing stay fixed. Actual-output bins describe outcomes retrospectively and are not model-selected evaluation pools.

### ridge

| Actual points | n | Forecast shift | Delta MAE | Delta MSE |
|---|---:|---:|---:|---:|
| [-inf,2) | 1105 | +0.8462 | +0.7861 | +4.1112 |
| [2,5) | 430 | +0.7274 | +0.1660 | +1.5594 |
| [5,10) | 395 | +0.6359 | -0.1929 | -1.3302 |
| [10,20) | 277 | +0.5744 | -0.4800 | -7.8224 |
| [20,inf) | 65 | +0.4477 | -0.4477 | -13.3783 |

### nn

| Actual points | n | Forecast shift | Delta MAE | Delta MSE |
|---|---:|---:|---:|---:|
| [-inf,2) | 1105 | +0.5787 | +0.5433 | +3.7476 |
| [2,5) | 430 | +0.8831 | +0.4049 | +3.5821 |
| [5,10) | 395 | +1.0430 | -0.1345 | -0.2544 |
| [10,20) | 277 | +1.0936 | -0.8772 | -11.9346 |
| [20,inf) | 65 | +1.2507 | -1.2507 | -37.6657 |

### attn_nn

| Actual points | n | Forecast shift | Delta MAE | Delta MSE |
|---|---:|---:|---:|---:|
| [-inf,2) | 1105 | +0.6763 | +0.6278 | +4.2891 |
| [2,5) | 430 | +1.0604 | +0.4166 | +4.0341 |
| [5,10) | 395 | +1.2685 | -0.1378 | -0.6444 |
| [10,20) | 277 | +1.3395 | -1.1257 | -15.3970 |
| [20,inf) | 65 | +1.6556 | -1.6556 | -50.8456 |

### lgbm

| Actual points | n | Forecast shift | Delta MAE | Delta MSE |
|---|---:|---:|---:|---:|
| [-inf,2) | 1105 | +0.3360 | +0.3301 | +2.6562 |
| [2,5) | 430 | +0.5532 | +0.3973 | +2.7212 |
| [5,10) | 395 | +0.6256 | -0.0491 | +0.1702 |
| [10,20) | 277 | +0.6449 | -0.5630 | -7.7265 |
| [20,inf) | 65 | +0.6830 | -0.6830 | -20.9589 |

## Seven added player-weeks

These rows are absent from the baseline and excluded from every paired delta. The table describes the fully corrected arm only; the JSON retains all four source-corrected arms.

| Model | MAE | RMSE | Bias |
|---|---:|---:|---:|
| ridge | 3.1406 | 3.3543 | +3.1406 |
| nn | 2.4113 | 2.6678 | +2.4113 |
| attn_nn | 1.5684 | 1.6888 | +1.5684 |
| lgbm | 2.8540 | 3.0309 | +2.8540 |

## Provenance and limits

Worker source: `9f4dae30ca55d79985176ab7e7f767d2c592cc43`. Input archive SHA-256: `4a914ff282b47472661abe063d8fd35720650cdd4b99fe4d2279c989eddaa2aa`. Maximum saved-inference discrepancy: `0.0`.

The inputs reconstruct archived production generations. They are not claimed to be the deleted local CPU experiment artifacts. All new comparisons use one pinned CUDA FP32 eager regime; seed standard deviations do not measure uncertainty across NFL seasons. No model changes, promotion, deployment, or retuning are implied.
