# Feature-selection report — QB

- spec: `src.tuning.ab_feature_screen_extended`
- run-id: `ab_feature_screen_extended-20260619T234253Z-ff086c1`
- seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
- noise floor: 0.02 FP (AGENTS.md); effect = MAE/RMSE delta when the group is DROPPED.
- **Sign:** `+` = dropping RAISES error = group carries signal (keep). `-` = dropping LOWERS error = drop candidate.

## Per-group effect by model (MAE | RMSE)

| group | verdict | Ridge MAE | Ridge RMSE | LightGBM MAE | LightGBM RMSE | NN MAE | NN RMSE | Attention NN MAE | Attention NN RMSE |
|---|---|---|---|---|---|---|---|---|---|
| `prior_season` | DROP-CAND | -0.167±0.000 | -0.141±0.000 | -0.123±0.000 | -0.126±0.000 | -0.446±0.000 | -0.628±0.000 | -0.032±0.139 | -0.034±0.203 |
| `trend` | DROP-CAND | -0.152±0.000 | -0.152±0.000 | -0.061±0.000 | -0.083±0.000 | -0.165±0.000 | -0.155±0.000 | -0.116±0.126 | -0.118±0.187 |
| `share` | DROP-CAND | -0.056±0.000 | -0.045±0.000 | -0.086±0.000 | -0.097±0.000 | -0.013±0.000 | -0.045±0.000 | -0.137±0.104 | -0.166±0.159 |
| `specific` | DROP-CAND | -0.083±0.000 | -0.083±0.000 | +0.009±0.000 | -0.011±0.000 | -0.050±0.000 | -0.011±0.000 | -0.120±0.096 | -0.125±0.138 |
| `rolling` | MIXED | +0.095±0.000 | +0.089±0.000 | -0.058±0.000 | -0.042±0.000 | +0.070±0.000 | +0.060±0.000 | -0.095±0.111 | -0.104±0.169 |
| `defense` | DROP-CAND | -0.036±0.000 | -0.047±0.000 | -0.039±0.000 | -0.048±0.000 | -0.028±0.000 | -0.027±0.000 | -0.069±0.078 | -0.097±0.121 |
| `matchup` | MIXED | -0.013±0.000 | -0.008±0.000 | -0.067±0.000 | -0.048±0.000 | +0.077±0.000 | +0.102±0.000 | -0.049±0.084 | -0.042±0.138 |
| `weather_vegas` | MIXED | +0.044±0.000 | +0.039±0.000 | +0.071±0.000 | +0.065±0.000 | -0.025±0.000 | -0.008±0.000 | +0.096±0.102 | +0.137±0.159 |
| `contextual` | KEEP | +0.583±0.000 | +0.567±0.000 | +0.519±0.000 | +0.544±0.000 | +0.842±0.000 | +0.915±0.000 | +0.676±0.092 | +0.700±0.132 |

## Suggested conservative cut (review — not auto-applied)

Groups neutral-or-helpful (MAE effect ≤ noise) for **every** model present. Dropping them should not regress any model beyond the noise floor — but confirm with a combined-drop run at high seed count (Stage 3) before applying:

- `prior_season` — columns: `prior_season_games_played`, `prior_season_max_attempts`, `prior_season_max_carries`, `prior_season_max_passing_yards`, `prior_season_max_receiving_yards`, `prior_season_max_receptions`, `prior_season_max_rushing_yards`, `prior_season_max_snap_pct`, `prior_season_max_targets`, `prior_season_mean_attempts`, `prior_season_mean_carries`, `prior_season_mean_catch_rate`, `prior_season_mean_fumbles_lost`, `prior_season_mean_pass_touchdown_exp`, `prior_season_mean_pass_yards_gained_exp`, `prior_season_mean_passing_yards`, `prior_season_mean_pts_added`, `prior_season_mean_qbr_total`, `prior_season_mean_rec_yards_gained_exp`, `prior_season_mean_receiving_yards`, `prior_season_mean_receptions`, `prior_season_mean_receptions_exp`, `prior_season_mean_redzone_touches_per_game`, `prior_season_mean_rush_yards_gained_exp`, `prior_season_mean_rushing_yards`, `prior_season_mean_snap_pct`, `prior_season_mean_targets`, `prior_season_mean_total_fantasy_points_exp`, `prior_season_mean_yards_per_carry`, `prior_season_std_attempts`, `prior_season_std_carries`, `prior_season_std_passing_yards`, `prior_season_std_receiving_yards`, `prior_season_std_receptions`, `prior_season_std_rushing_yards`, `prior_season_std_snap_pct`, `prior_season_std_targets`, `prior_season_total_redzone_touches`, `prior_season_total_touchdowns`, `prior_season_total_yards`
- `trend` — columns: `trend_carries`, `trend_fantasy_points`, `trend_snap_pct`, `trend_targets`
- `share` — columns: `air_yards_share`, `carry_share_L3`, `carry_share_L5`, `snap_pct`, `target_share_L3`, `target_share_L5`
- `defense` — columns: `opp_def_ints_L5`, `opp_def_pass_td_allowed_L5`, `opp_def_pass_yds_allowed_L5`, `opp_def_pts_allowed_L5`, `opp_def_rush_yds_allowed_L5`, `opp_def_sacks_L5`
- `specific` — columns: `air_yards_per_target_L3`, `career_carries`, `completion_pct_L3`, `deep_ball_rate_L3`, `int_rate_L3`, `opportunity_index_L3`, `pass_first_down_rate_L3`, `passing_epa_per_dropback_L3`, `qb_rushing_share_L3`, `receiving_epa_per_target_L3`, `receiving_first_down_rate_L3`, `reception_rate_L3`, `redzone_target_share_L3`, `redzone_targets_L3`, `rush_first_down_rate_L3`, `rushing_epa_per_attempt_L3`, `rushing_epa_per_carry_L3`, `rushing_first_down_rate_L3`, `sack_damage_per_dropback_L3`, `sack_rate_L3`, `td_rate_L3`, `td_rate_per_target_L3`, `team_rb_carry_hhi_L3`, `team_rb_carry_share_L3`, `team_rb_target_hhi_L3`, `team_rb_target_share_L3`, `team_te_target_share_L3`, `team_wr_target_share_L3`, `yac_per_reception_L3`, `yac_rate_L3`, `yards_per_attempt_L3`, `yards_per_carry_L3`, `yards_per_reception_L3`, `yards_per_target_L3`

Apply (after review):

```
python -m src.tuning.feature_selection apply --position QB --drop air_yards_per_target_L3 air_yards_share career_carries carry_share_L3 carry_share_L5 completion_pct_L3 deep_ball_rate_L3 int_rate_L3 opp_def_ints_L5 opp_def_pass_td_allowed_L5 opp_def_pass_yds_allowed_L5 opp_def_pts_allowed_L5 opp_def_rush_yds_allowed_L5 opp_def_sacks_L5 opportunity_index_L3 pass_first_down_rate_L3 passing_epa_per_dropback_L3 prior_season_games_played prior_season_max_attempts prior_season_max_carries prior_season_max_passing_yards prior_season_max_receiving_yards prior_season_max_receptions prior_season_max_rushing_yards prior_season_max_snap_pct prior_season_max_targets prior_season_mean_attempts prior_season_mean_carries prior_season_mean_catch_rate prior_season_mean_fumbles_lost prior_season_mean_pass_touchdown_exp prior_season_mean_pass_yards_gained_exp prior_season_mean_passing_yards prior_season_mean_pts_added prior_season_mean_qbr_total prior_season_mean_rec_yards_gained_exp prior_season_mean_receiving_yards prior_season_mean_receptions prior_season_mean_receptions_exp prior_season_mean_redzone_touches_per_game prior_season_mean_rush_yards_gained_exp prior_season_mean_rushing_yards prior_season_mean_snap_pct prior_season_mean_targets prior_season_mean_total_fantasy_points_exp prior_season_mean_yards_per_carry prior_season_std_attempts prior_season_std_carries prior_season_std_passing_yards prior_season_std_receiving_yards prior_season_std_receptions prior_season_std_rushing_yards prior_season_std_snap_pct prior_season_std_targets prior_season_total_redzone_touches prior_season_total_touchdowns prior_season_total_yards qb_rushing_share_L3 receiving_epa_per_target_L3 receiving_first_down_rate_L3 reception_rate_L3 redzone_target_share_L3 redzone_targets_L3 rush_first_down_rate_L3 rushing_epa_per_attempt_L3 rushing_epa_per_carry_L3 rushing_first_down_rate_L3 sack_damage_per_dropback_L3 sack_rate_L3 snap_pct target_share_L3 target_share_L5 td_rate_L3 td_rate_per_target_L3 team_rb_carry_hhi_L3 team_rb_carry_share_L3 team_rb_target_hhi_L3 team_rb_target_share_L3 team_te_target_share_L3 team_wr_target_share_L3 trend_carries trend_fantasy_points trend_snap_pct trend_targets yac_per_reception_L3 yac_rate_L3 yards_per_attempt_L3 yards_per_carry_L3 yards_per_reception_L3 yards_per_target_L3 --pr
```

## Caveats

- **Neutral overall ≠ useless.** A group flat on overall MAE/RMSE may still carry subgroup signal (rookies/RB, returners). Judge subgroup value by *bias*, not overall MAE (draft-capital lesson). Check `result['test_df']` cohorts before dropping a borderline group.
- **Stacked vs eager:** skill positions screen stacked (vmap, FP32/LN/fixed-epochs); K/DST screen eager. Compare stacked arms only against stacked arms — never seed-by-seed against an eager run.
- **Confirm before applying.** PB main effects assume additivity; re-run the chosen drop-set together (Stage 3) at high seed count to catch interactions.
