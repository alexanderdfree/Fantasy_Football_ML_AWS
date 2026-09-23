# Isolated development evidence

Candidate: `count_precision`. Purpose: **development**.

Verified grid: 36/54 cells. Verification: FAIL. Development qualification: False.

These results do not authorize a merge or establish confirmation.

| Position | Season | Model | MAE delta | RMSE delta | Elite MAE delta | Elite RMSE delta | Screen |
|---|---:|---|---:|---:|---:|---:|---|
| RB | 2022 | attn_nn | +0.010491 | -0.010245 | +0.018964 | -0.005051 | hold |
| RB | 2023 | attn_nn | -0.004978 | -0.004877 | +0.006279 | +0.007554 | hold |
| WR | 2022 | attn_nn | -0.017466 | +0.008361 | -0.015810 | +0.033721 | hold |
| WR | 2023 | attn_nn | -0.000350 | -0.000254 | -0.022398 | -0.054796 | pass |
| TE | 2022 | attn_nn | unavailable | unavailable | unavailable | unavailable | hold |
| TE | 2023 | attn_nn | unavailable | unavailable | unavailable | unavailable | hold |

Unavailable before 2024; required confirmation remains separate.

Paired sample standard deviations and Student-t intervals describe three training seeds within each retrospective season, not independent season uncertainty.

Verification errors:

- Incomplete grid: missing=[('TE', 2022, 7, 'baseline'), ('TE', 2022, 7, 'baseline_rep'), ('TE', 2022, 7, 'count_precision'), ('TE', 2022, 42, 'baseline'), ('TE', 2022, 42, 'baseline_rep'), ('TE', 2022, 42, 'count_precision'), ('TE', 2022, 123, 'baseline'), ('TE', 2022, 123, 'baseline_rep'), ('TE', 2022, 123, 'count_precision'), ('TE', 2023, 7, 'baseline'), ('TE', 2023, 7, 'baseline_rep'), ('TE', 2023, 7, 'count_precision'), ('TE', 2023, 42, 'baseline'), ('TE', 2023, 42, 'baseline_rep'), ('TE', 2023, 42, 'count_precision'), ('TE', 2023, 123, 'baseline'), ('TE', 2023, 123, 'baseline_rep'), ('TE', 2023, 123, 'count_precision')], extra=[]
- ab_runs/audit-isolated-development-20260923/full-count-2022-90265546/: Missing completed cells: [('TE', 7, 'baseline'), ('TE', 7, 'baseline_rep'), ('TE', 7, 'count_precision'), ('TE', 42, 'baseline'), ('TE', 42, 'baseline_rep'), ('TE', 42, 'count_precision'), ('TE', 123, 'baseline'), ('TE', 123, 'baseline_rep'), ('TE', 123, 'count_precision')]
- ab_runs/audit-isolated-development-20260923/full-count-2023-90265546/: Missing completed cells: [('TE', 7, 'baseline'), ('TE', 7, 'baseline_rep'), ('TE', 7, 'count_precision'), ('TE', 42, 'baseline'), ('TE', 42, 'baseline_rep'), ('TE', 42, 'count_precision'), ('TE', 123, 'baseline'), ('TE', 123, 'baseline_rep'), ('TE', 123, 'count_precision')]
