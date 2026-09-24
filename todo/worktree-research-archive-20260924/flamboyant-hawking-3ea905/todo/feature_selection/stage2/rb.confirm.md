# Feature-selection Stage-3 confirm — RB

- spec: `src.tuning.ab_feature_confirm`  run-id: `confirm-rb-20260620T163815Z-7edcbd2`  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
- columns dropped together (12): `air_yards_share`, `opp_def_pass_td_allowed_L5`, `opp_def_pts_allowed_L5`, `opp_def_rush_yds_allowed_L5`, `opp_def_sacks_L5`, `opp_rush_pts_allowed_to_pos`, `snap_pct`, `target_share_L3`, `trend_carries`, `trend_fantasy_points`, `trend_snap_pct`, `trend_targets`
- **Production config — PCA-Ridge ON.** This is the faithful gate the skip-PCA screen feeds; judge the combined drop here, not on the screen's raw-Ridge column.
- **Sign:** `+` = dropping the set RAISES error = the set carries signal (KEEP it). `-`/flat (≤ 0.02 FP) = the combined drop is safe.

## Combined drop-set effect by model (MAE | RMSE)

| model | verdict | MAE | RMSE |
|---|---|---|---|
| Ridge | KEEP | +0.029±0.000 | +0.047±0.000 |
| LightGBM | DROP-CAND | -0.023±0.000 | -0.000±0.000 |
| NN | DROP-CAND | -0.044±0.000 | -0.060±0.000 |
| Attention NN | DROP-CAND | -0.007±0.026 | +0.033±0.089 |

**Mixed:** helps some models, hurts others on the production config — operator's call; consider a smaller subset and re-confirm.
