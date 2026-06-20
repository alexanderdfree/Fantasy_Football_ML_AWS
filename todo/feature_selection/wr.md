# Feature-selection report — WR

- spec: `src.tuning.ab_feature_screen_extended`
- run-id: `ab_feature_screen_extended-20260620T013357Z-2e7455c`
- seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
- noise floor: 0.02 FP (AGENTS.md); effect = MAE/RMSE delta when the group is DROPPED.
- **Sign:** `+` = dropping RAISES error = group carries signal (keep). `-` = dropping LOWERS error = drop candidate.
- **Ridge:** PCA is disabled for the screen (raw features) for robustness + clean attribution. Production RB/WR/DST ship PCA-Ridge — confirm a final cut on the production config (Stage 3 / the benchmark gate), not this screen's Ridge column.

## Per-group effect by model (MAE | RMSE)

| group | verdict | Ridge MAE | Ridge RMSE | LightGBM MAE | LightGBM RMSE | NN MAE | NN RMSE | Attention NN MAE | Attention NN RMSE |
|---|---|---|---|---|---|---|---|---|---|
| `matchup` | DROP-CAND | -0.095±0.000 | -0.083±0.000 | -0.049±0.000 | -0.071±0.000 | -0.034±0.000 | -0.069±0.000 | -0.043±0.014 | -0.028±0.074 |
| `trend` | DROP-CAND | -0.093±0.000 | -0.121±0.000 | -0.072±0.000 | -0.066±0.000 | -0.032±0.000 | -0.074±0.000 | -0.054±0.006 | -0.023±0.009 |
| `defense` | DROP-CAND | -0.093±0.000 | -0.105±0.000 | -0.035±0.000 | -0.048±0.000 | -0.048±0.000 | -0.068±0.000 | -0.008±0.007 | -0.004±0.010 |
| `specific` | DROP-CAND | -0.029±0.000 | +0.057±0.000 | -0.038±0.000 | -0.020±0.000 | -0.052±0.000 | +0.009±0.000 | -0.057±0.008 | -0.031±0.008 |
| `weather_vegas` | DROP-CAND | -0.017±0.000 | +0.098±0.000 | +0.007±0.000 | +0.047±0.000 | +0.008±0.000 | +0.058±0.000 | +0.011±0.008 | +0.018±0.060 |
| `rolling` | MIXED | +0.074±0.000 | +0.137±0.000 | +0.064±0.000 | +0.068±0.000 | +0.081±0.000 | +0.098±0.000 | -0.007±0.008 | +0.004±0.009 |
| `share` | DROP-CAND | -0.004±0.000 | +0.095±0.000 | +0.011±0.000 | +0.031±0.000 | +0.006±0.000 | +0.053±0.000 | -0.003±0.006 | -0.005±0.009 |
| `prior_season` | KEEP | +0.036±0.000 | +0.057±0.000 | +0.049±0.000 | +0.089±0.000 | +0.104±0.000 | +0.088±0.000 | +0.048±0.015 | +0.071±0.063 |
| `contextual` | KEEP | +0.324±0.000 | -0.101±0.000 | +0.127±0.000 | +0.021±0.000 | +0.063±0.000 | -0.035±0.000 | +0.142±0.014 | +0.019±0.071 |

## Suggested conservative cut (review — not auto-applied)

Groups neutral-or-helpful (MAE effect ≤ noise) for **every** model present. Dropping them should not regress any model beyond the noise floor — but confirm with a combined-drop run at high seed count (Stage 3) before applying:

- `trend` — columns: `trend_carries`, `trend_fantasy_points`, `trend_snap_pct`, `trend_targets`
- `share` — columns: `air_yards_share`, `carry_share_L3`, `carry_share_L5`, `snap_pct`, `target_share_L3`, `target_share_L5`
- `matchup` — columns: `opp_def_rank_vs_pos`, `opp_fantasy_pts_allowed_to_pos`, `opp_recv_pts_allowed_to_pos`, `opp_rush_pts_allowed_to_pos`
- `defense` — columns: `opp_def_ints_L5`, `opp_def_pass_td_allowed_L5`, `opp_def_pass_yds_allowed_L5`, `opp_def_pts_allowed_L5`, `opp_def_rush_yds_allowed_L5`, `opp_def_sacks_L5`
- `weather_vegas` — columns: `implied_opp_total`, `implied_team_total`, `is_divisional`, `is_dome`, `rest_advantage`, `temp_adjusted`, `wind_adjusted`
- `specific` — columns: `air_yards_per_target_L3`, `career_carries`, `completion_pct_L3`, `deep_ball_rate_L3`, `int_rate_L3`, `opportunity_index_L3`, `pass_first_down_rate_L3`, `passing_epa_per_dropback_L3`, `qb_rushing_share_L3`, `receiving_epa_per_target_L3`, `receiving_first_down_rate_L3`, `reception_rate_L3`, `redzone_target_share_L3`, `redzone_targets_L3`, `rush_first_down_rate_L3`, `rushing_epa_per_attempt_L3`, `rushing_epa_per_carry_L3`, `rushing_first_down_rate_L3`, `sack_damage_per_dropback_L3`, `sack_rate_L3`, `td_rate_L3`, `td_rate_per_target_L3`, `team_rb_carry_hhi_L3`, `team_rb_carry_share_L3`, `team_rb_target_hhi_L3`, `team_rb_target_share_L3`, `team_te_target_share_L3`, `team_wr_target_share_L3`, `yac_per_reception_L3`, `yac_rate_L3`, `yards_per_attempt_L3`, `yards_per_carry_L3`, `yards_per_reception_L3`, `yards_per_target_L3`

Apply (after review):

```
python -m src.tuning.feature_selection apply --position WR --drop air_yards_per_target_L3 air_yards_share career_carries carry_share_L3 carry_share_L5 completion_pct_L3 deep_ball_rate_L3 implied_opp_total implied_team_total int_rate_L3 is_divisional is_dome opp_def_ints_L5 opp_def_pass_td_allowed_L5 opp_def_pass_yds_allowed_L5 opp_def_pts_allowed_L5 opp_def_rank_vs_pos opp_def_rush_yds_allowed_L5 opp_def_sacks_L5 opp_fantasy_pts_allowed_to_pos opp_recv_pts_allowed_to_pos opp_rush_pts_allowed_to_pos opportunity_index_L3 pass_first_down_rate_L3 passing_epa_per_dropback_L3 qb_rushing_share_L3 receiving_epa_per_target_L3 receiving_first_down_rate_L3 reception_rate_L3 redzone_target_share_L3 redzone_targets_L3 rest_advantage rush_first_down_rate_L3 rushing_epa_per_attempt_L3 rushing_epa_per_carry_L3 rushing_first_down_rate_L3 sack_damage_per_dropback_L3 sack_rate_L3 snap_pct target_share_L3 target_share_L5 td_rate_L3 td_rate_per_target_L3 team_rb_carry_hhi_L3 team_rb_carry_share_L3 team_rb_target_hhi_L3 team_rb_target_share_L3 team_te_target_share_L3 team_wr_target_share_L3 temp_adjusted trend_carries trend_fantasy_points trend_snap_pct trend_targets wind_adjusted yac_per_reception_L3 yac_rate_L3 yards_per_attempt_L3 yards_per_carry_L3 yards_per_reception_L3 yards_per_target_L3 --pr
```

## Caveats

- **Neutral overall ≠ useless.** A group flat on overall MAE/RMSE may still carry subgroup signal (rookies/RB, returners). Judge subgroup value by *bias*, not overall MAE (draft-capital lesson). Check `result['test_df']` cohorts before dropping a borderline group.
- **Stacked vs eager:** skill positions screen stacked (vmap, FP32/LN/fixed-epochs); K/DST screen eager. Compare stacked arms only against stacked arms — never seed-by-seed against an eager run.
- **Confirm before applying.** PB main effects assume additivity; re-run the chosen drop-set together (Stage 3) at high seed count to catch interactions.
