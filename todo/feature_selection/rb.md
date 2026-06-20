# Feature-selection report — RB

- spec: `src.tuning.ab_feature_screen_extended`
- run-id: `ab_feature_screen_extended-20260620T013357Z-2e7455c`
- seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
- noise floor: 0.02 FP (AGENTS.md); effect = MAE/RMSE delta when the group is DROPPED.
- **Sign:** `+` = dropping RAISES error = group carries signal (keep). `-` = dropping LOWERS error = drop candidate.
- **Ridge:** PCA is disabled for the screen (raw features) for robustness + clean attribution. Production RB/WR/DST ship PCA-Ridge — confirm a final cut on the production config (Stage 3 / the benchmark gate), not this screen's Ridge column.

## Per-group effect by model (MAE | RMSE)

| group | verdict | Ridge MAE | Ridge RMSE | LightGBM MAE | LightGBM RMSE | NN MAE | NN RMSE | Attention NN MAE | Attention NN RMSE |
|---|---|---|---|---|---|---|---|---|---|
| `trend` | DROP-CAND | -0.074±0.000 | -0.080±0.000 | -0.113±0.000 | -0.097±0.000 | -0.108±0.000 | -0.110±0.000 | -0.064±0.022 | -0.074±0.050 |
| `matchup` | DROP-CAND | -0.036±0.000 | -0.039±0.000 | -0.068±0.000 | -0.061±0.000 | -0.070±0.000 | -0.035±0.000 | -0.038±0.024 | -0.023±0.060 |
| `specific` | MIXED | +0.041±0.000 | +0.040±0.000 | -0.005±0.000 | -0.013±0.000 | +0.015±0.000 | +0.014±0.000 | -0.050±0.026 | -0.051±0.065 |
| `defense` | DROP-CAND | -0.035±0.000 | -0.029±0.000 | -0.036±0.000 | -0.034±0.000 | -0.012±0.000 | -0.063±0.000 | -0.009±0.026 | -0.016±0.059 |
| `prior_season` | MIXED | -0.024±0.000 | -0.048±0.000 | +0.016±0.000 | -0.015±0.000 | -0.020±0.000 | -0.004±0.000 | +0.036±0.023 | +0.021±0.061 |
| `rolling` | MIXED | +0.063±0.000 | +0.064±0.000 | +0.095±0.000 | +0.098±0.000 | +0.070±0.000 | +0.113±0.000 | -0.020±0.023 | -0.027±0.054 |
| `share` | DROP-CAND | +0.012±0.000 | +0.033±0.000 | +0.013±0.000 | +0.029±0.000 | +0.009±0.000 | +0.050±0.000 | -0.003±0.027 | -0.005±0.056 |
| `weather_vegas` | KEEP | +0.032±0.000 | +0.043±0.000 | +0.046±0.000 | +0.044±0.000 | +0.034±0.000 | +0.037±0.000 | +0.043±0.024 | +0.031±0.056 |
| `contextual` | KEEP | +0.096±0.000 | +0.075±0.000 | +0.120±0.000 | +0.093±0.000 | +0.096±0.000 | +0.117±0.000 | +0.130±0.022 | +0.158±0.053 |

## Suggested conservative cut (review — not auto-applied)

Groups neutral-or-helpful (MAE effect ≤ noise) for **every** model present. Dropping them should not regress any model beyond the noise floor — but confirm with a combined-drop run at high seed count (Stage 3) before applying:

- `trend` — columns: `trend_carries`, `trend_fantasy_points`, `trend_snap_pct`, `trend_targets`
- `share` — columns: `air_yards_share`, `carry_share_L3`, `carry_share_L5`, `snap_pct`, `target_share_L3`, `target_share_L5`
- `matchup` — columns: `opp_def_rank_vs_pos`, `opp_fantasy_pts_allowed_to_pos`, `opp_recv_pts_allowed_to_pos`, `opp_rush_pts_allowed_to_pos`
- `defense` — columns: `opp_def_ints_L5`, `opp_def_pass_td_allowed_L5`, `opp_def_pass_yds_allowed_L5`, `opp_def_pts_allowed_L5`, `opp_def_rush_yds_allowed_L5`, `opp_def_sacks_L5`

Apply (after review):

```
python -m src.tuning.feature_selection apply --position RB --drop air_yards_share carry_share_L3 carry_share_L5 opp_def_ints_L5 opp_def_pass_td_allowed_L5 opp_def_pass_yds_allowed_L5 opp_def_pts_allowed_L5 opp_def_rank_vs_pos opp_def_rush_yds_allowed_L5 opp_def_sacks_L5 opp_fantasy_pts_allowed_to_pos opp_recv_pts_allowed_to_pos opp_rush_pts_allowed_to_pos snap_pct target_share_L3 target_share_L5 trend_carries trend_fantasy_points trend_snap_pct trend_targets --pr
```

## Caveats

- **Neutral overall ≠ useless.** A group flat on overall MAE/RMSE may still carry subgroup signal (rookies/RB, returners). Judge subgroup value by *bias*, not overall MAE (draft-capital lesson). Check `result['test_df']` cohorts before dropping a borderline group.
- **Stacked vs eager:** skill positions screen stacked (vmap, FP32/LN/fixed-epochs); K/DST screen eager. Compare stacked arms only against stacked arms — never seed-by-seed against an eager run.
- **Confirm before applying.** PB main effects assume additivity; re-run the chosen drop-set together (Stage 3) at high seed count to catch interactions.
