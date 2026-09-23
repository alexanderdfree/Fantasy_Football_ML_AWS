# Isolated development evidence

Candidate: `stint_reset`. Purpose: **smoke**.

Verified grid: 3/3 cells. Verification: PASS. Development qualification: False.

These results do not authorize a merge or establish confirmation.

| Position | Season | Model | MAE delta | RMSE delta | Elite MAE delta | Elite RMSE delta | Screen |
|---|---:|---|---:|---:|---:|---:|---|
| WR | 2022 | ridge | -0.000312 | -0.000135 | -0.000302 | -0.000298 | pass |
| WR | 2022 | nn | +0.005779 | +0.000769 | +0.020233 | +0.010531 | hold |
| WR | 2022 | lgbm | +0.005316 | +0.014485 | +0.054648 | +0.039734 | hold |

Unavailable before 2024; required confirmation remains separate.

Paired sample standard deviations and Student-t intervals describe three training seeds within each retrospective season, not independent season uncertainty.
