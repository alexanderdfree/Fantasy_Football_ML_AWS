# Feature-selection report — DST

- spec: `src.tuning.ab_feature_screen_dst`
- run-id: `ab_feature_screen_dst-20260620T013402Z-2e7455c`
- seeds: [42, 123, 7]
- noise floor: 0.02 FP (AGENTS.md); effect = MAE/RMSE delta when the group is DROPPED.
- **Sign:** `+` = dropping RAISES error = group carries signal (keep). `-` = dropping LOWERS error = drop candidate.
- **Ridge:** PCA is disabled for the screen (raw features) for robustness + clean attribution. Production RB/WR/DST ship PCA-Ridge — confirm a final cut on the production config (Stage 3 / the benchmark gate), not this screen's Ridge column.

## Per-group effect by model (MAE | RMSE)

| group | verdict | Ridge MAE | Ridge RMSE | LightGBM MAE | LightGBM RMSE | NN MAE | NN RMSE | Attention NN MAE | Attention NN RMSE |
|---|---|---|---|---|---|---|---|---|---|
| `weather` | DROP-CAND | -0.084±0.000 | -0.098±0.000 | -0.056±0.013 | -0.072±0.017 | -0.096±0.059 | -0.101±0.042 | -0.057±0.052 | -0.091±0.039 |
| `pts_yds_allowed` | DROP-CAND | -0.070±0.000 | -0.088±0.000 | -0.079±0.013 | -0.086±0.014 | +0.003±0.083 | -0.029±0.051 | -0.080±0.009 | -0.098±0.006 |
| `prior_season` | DROP-CAND | -0.009±0.000 | -0.036±0.000 | -0.004±0.012 | -0.026±0.013 | +0.019±0.016 | -0.001±0.017 | -0.065±0.002 | -0.096±0.016 |
| `dst_points` | DROP-CAND | -0.045±0.000 | -0.035±0.000 | -0.034±0.001 | -0.040±0.003 | -0.008±0.054 | -0.014±0.043 | -0.007±0.013 | -0.009±0.010 |
| `opp_qb` | DROP-CAND | +0.007±0.000 | +0.019±0.000 | -0.004±0.004 | +0.007±0.005 | -0.031±0.011 | +0.003±0.003 | +0.000±0.026 | -0.010±0.020 |
| `dst_trend` | DROP-CAND | -0.001±0.000 | -0.013±0.000 | -0.021±0.006 | -0.019±0.004 | +0.020±0.040 | -0.007±0.018 | -0.000±0.026 | +0.010±0.020 |
| `dst_production` | MIXED | +0.026±0.000 | +0.027±0.000 | +0.042±0.007 | +0.036±0.008 | +0.006±0.050 | +0.010±0.043 | +0.007±0.013 | +0.009±0.010 |
| `opp_offense` | MIXED | +0.031±0.000 | +0.031±0.000 | +0.034±0.008 | +0.032±0.009 | +0.021±0.029 | +0.024±0.030 | +0.012±0.024 | +0.011±0.018 |
| `game_context` | KEEP | +0.229±0.000 | +0.296±0.000 | +0.194±0.007 | +0.270±0.011 | +0.176±0.030 | +0.271±0.021 | +0.282±0.038 | +0.384±0.031 |

## Suggested conservative cut (review — not auto-applied)

Groups neutral-or-helpful (MAE effect ≤ noise) for **every** model present. Dropping them should not regress any model beyond the noise floor — but confirm with a combined-drop run at high seed count (Stage 3) before applying:

- `dst_points` — columns: `dst_pts_L3`, `dst_pts_L5`, `dst_pts_L8`, `dst_pts_ewma`
- `pts_yds_allowed` — columns: `pts_allowed_L3`, `pts_allowed_L5`, `pts_allowed_ewma`, `yards_allowed_L3`, `yards_allowed_L5`, `yards_allowed_ewma`
- `dst_trend` — columns: `dst_scoring_std_L3`, `pts_allowed_std_L3`, `pts_allowed_trend`, `sack_trend`, `turnover_trend`
- `opp_qb` — columns: `opp_qb_epa_L5`, `opp_qb_int_rate_L5`, `opp_qb_rush_yds_L5`, `opp_qb_sack_rate_L5`
- `weather` — columns: `is_dome`
- `prior_season` — columns: `prior_season_dst_pts_avg`, `prior_season_pts_allowed_avg`

Apply (after review):

```
python -m src.tuning.feature_selection apply --position DST --drop dst_pts_L3 dst_pts_L5 dst_pts_L8 dst_pts_ewma dst_scoring_std_L3 is_dome opp_qb_epa_L5 opp_qb_int_rate_L5 opp_qb_rush_yds_L5 opp_qb_sack_rate_L5 prior_season_dst_pts_avg prior_season_pts_allowed_avg pts_allowed_L3 pts_allowed_L5 pts_allowed_ewma pts_allowed_std_L3 pts_allowed_trend sack_trend turnover_trend yards_allowed_L3 yards_allowed_L5 yards_allowed_ewma --pr
```

## Caveats

- **Neutral overall ≠ useless.** A group flat on overall MAE/RMSE may still carry subgroup signal (rookies/RB, returners). Judge subgroup value by *bias*, not overall MAE (draft-capital lesson). Check `result['test_df']` cohorts before dropping a borderline group.
- **Stacked vs eager:** skill positions screen stacked (vmap, FP32/LN/fixed-epochs); K/DST screen eager. Compare stacked arms only against stacked arms — never seed-by-seed against an eager run.
- **Confirm before applying.** PB main effects assume additivity; re-run the chosen drop-set together (Stage 3) at high seed count to catch interactions.
