# Isolated development evidence

Candidate: `stint_reset`. Purpose: **smoke**.

Verified grid: 3/3 cells. Verification: PASS. Development qualification: False.

These results do not authorize a merge or establish confirmation.

| Position | Season | Model | MAE delta | RMSE delta | Elite MAE delta | Elite RMSE delta | Screen |
|---|---:|---|---:|---:|---:|---:|---|
| TE | 2022 | ridge | +0.000120 | +0.000074 | +0.000744 | +0.001076 | hold |
| TE | 2022 | nn | +0.010519 | -0.010094 | -0.003956 | -0.020508 | hold |
| TE | 2022 | lgbm | -0.005455 | +0.010454 | +0.004069 | +0.045695 | hold |

Unavailable before 2024; required confirmation remains separate.

Paired sample standard deviations and Student-t intervals describe three training seeds within each retrospective season, not independent season uncertainty.
