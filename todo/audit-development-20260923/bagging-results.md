# Isolated development evidence

Candidate: `bagging`. Purpose: **development**.

Verified grid: 108/108 cells. Verification: PASS. Development qualification: False.

These results do not authorize a merge or establish confirmation.

| Position | Season | Model | MAE delta | RMSE delta | Elite MAE delta | Elite RMSE delta | Screen |
|---|---:|---|---:|---:|---:|---:|---|
| QB | 2022 | lgbm | -0.008601 | -0.023043 | +0.008265 | -0.000888 | hold |
| QB | 2023 | lgbm | +0.037549 | +0.039289 | +0.025659 | +0.023613 | hold |
| RB | 2022 | lgbm | +0.006918 | +0.015732 | +0.024542 | +0.029832 | hold |
| RB | 2023 | lgbm | -0.011272 | -0.000228 | -0.029995 | +0.002060 | hold |
| WR | 2022 | lgbm | +0.002796 | +0.008522 | +0.009571 | +0.032648 | hold |
| WR | 2023 | lgbm | +0.001277 | +0.008023 | +0.013030 | +0.025005 | hold |
| TE | 2022 | lgbm | -0.008870 | -0.008050 | -0.007371 | -0.017069 | pass |
| TE | 2023 | lgbm | -0.018394 | -0.002072 | -0.003712 | -0.004966 | pass |
| K | 2022 | lgbm | -0.001840 | -0.007587 | +0.002236 | -0.004320 | hold |
| K | 2023 | lgbm | +0.005674 | -0.005515 | +0.004070 | -0.008195 | hold |
| DST | 2022 | lgbm | -0.004391 | -0.012084 | -0.005253 | -0.021534 | pass |
| DST | 2023 | lgbm | -0.009496 | -0.015212 | +0.004089 | -0.013453 | hold |

Unavailable before 2024; required confirmation remains separate.

Paired sample standard deviations and Student-t intervals describe three training seeds within each retrospective season, not independent season uncertainty.
