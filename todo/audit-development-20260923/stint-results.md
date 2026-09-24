# Isolated development evidence

Candidate: `stint_reset`. Purpose: **development**.

Verified grid: 36/36 cells. Verification: PASS. Development qualification: False.

These results do not authorize a merge or establish confirmation.

| Position | Season | Model | MAE delta | RMSE delta | Elite MAE delta | Elite RMSE delta | Screen |
|---|---:|---|---:|---:|---:|---:|---|
| WR | 2022 | ridge | -0.000312 | -0.000135 | -0.000302 | -0.000298 | pass |
| WR | 2022 | nn | +0.008629 | +0.001314 | +0.002149 | +0.013506 | hold |
| WR | 2022 | lgbm | -0.004243 | +0.002536 | +0.004756 | +0.011852 | hold |
| WR | 2023 | ridge | -0.000116 | -0.000063 | -0.000054 | -0.000106 | pass |
| WR | 2023 | nn | -0.046514 | +0.050376 | -0.035880 | +0.132825 | hold |
| WR | 2023 | lgbm | -0.000360 | +0.002956 | +0.006243 | +0.003381 | hold |
| TE | 2022 | ridge | +0.000120 | +0.000074 | +0.000744 | +0.001076 | hold |
| TE | 2022 | nn | -0.023407 | +0.030166 | -0.012459 | -0.026716 | hold |
| TE | 2022 | lgbm | +0.002263 | +0.001975 | +0.017461 | +0.014274 | hold |
| TE | 2023 | ridge | -0.000309 | -0.000349 | -0.000306 | -0.000629 | pass |
| TE | 2023 | nn | +0.023062 | -0.000882 | +0.009918 | +0.040323 | hold |
| TE | 2023 | lgbm | -0.006697 | -0.005336 | -0.010033 | -0.009985 | pass |

Unavailable before 2024; required confirmation remains separate.

Paired sample standard deviations and Student-t intervals describe three training seeds within each retrospective season, not independent season uncertainty.
