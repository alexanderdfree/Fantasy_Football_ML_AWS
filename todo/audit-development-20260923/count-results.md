# Isolated development evidence

Candidate: `count_precision`. Purpose: **development**.

Verified grid: 54/54 cells. Verification: PASS. Development qualification: False.

These results do not authorize a merge or establish confirmation.

| Position | Season | Model | MAE delta | RMSE delta | Elite MAE delta | Elite RMSE delta | Screen |
|---|---:|---|---:|---:|---:|---:|---|
| RB | 2022 | attn_nn | +0.010491 | -0.010245 | +0.018964 | -0.005051 | hold |
| RB | 2023 | attn_nn | -0.004978 | -0.004877 | +0.006279 | +0.007554 | hold |
| WR | 2022 | attn_nn | -0.017466 | +0.008361 | -0.015810 | +0.033721 | hold |
| WR | 2023 | attn_nn | -0.000350 | -0.000254 | -0.022398 | -0.054796 | pass |
| TE | 2022 | attn_nn | +0.000971 | +0.020793 | +0.035131 | +0.068174 | hold |
| TE | 2023 | attn_nn | +0.006651 | +0.022755 | +0.032129 | +0.052917 | hold |

Unavailable before 2024; required confirmation remains separate.

Paired sample standard deviations and Student-t intervals describe three training seeds within each retrospective season, not independent season uncertainty.
