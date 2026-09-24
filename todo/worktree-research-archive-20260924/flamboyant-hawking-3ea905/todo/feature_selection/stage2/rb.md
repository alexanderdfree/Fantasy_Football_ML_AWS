# Feature-selection Stage-2 (sub-family zoom) — RB

- families zoomed: `defense`, `trend`, `share`, `matchup`, `prior_season`, `specific`, `rolling`
- noise floor: 0.02 FP (AGENTS.md)

## Combined suggested drop columns (review — not auto-applied)

Union of every zoomed family's suggested sub-cut (neutral-or-helpful for **every** model). CONFIRM them together on the production PCA-Ridge config (Stage 3) before applying — the screen is skip-PCA and PB assumes additivity:

- `air_yards_per_target_L3`
- `air_yards_share`
- `career_carries`
- `opp_def_pass_td_allowed_L5`
- `opp_def_pts_allowed_L5`
- `opp_def_rush_yds_allowed_L5`
- `opp_def_sacks_L5`
- `opp_rush_pts_allowed_to_pos`
- `opportunity_index_L3`
- `prior_season_games_played`
- `prior_season_max_carries`
- `prior_season_max_snap_pct`
- `prior_season_max_targets`
- `prior_season_mean_carries`
- `prior_season_mean_catch_rate`
- `prior_season_mean_rush_yards_gained_exp`
- `prior_season_mean_snap_pct`
- `prior_season_mean_targets`
- `prior_season_mean_yards_per_carry`
- `prior_season_std_carries`
- `prior_season_std_snap_pct`
- `prior_season_std_targets`
- `prior_season_total_redzone_touches`
- `prior_season_total_touchdowns`
- `prior_season_total_yards`
- `receiving_epa_per_target_L3`
- `rolling_max_carries_L3`
- `rolling_max_carries_L8`
- `rolling_max_fantasy_points_L3`
- `rolling_max_fantasy_points_L8`
- `rolling_max_receiving_yards_L3`
- `rolling_max_receiving_yards_L8`
- `rolling_max_rushing_yards_L3`
- `rolling_max_rushing_yards_L8`
- `rolling_max_targets_L3`
- `rolling_max_targets_L8`
- `rolling_mean_carries_L3`
- `rolling_mean_carries_L8`
- `rolling_mean_fantasy_points_L3`
- `rolling_mean_fantasy_points_L8`
- `rolling_mean_receiving_yards_L3`
- `rolling_mean_receiving_yards_L8`
- `rolling_mean_rushing_yards_L3`
- `rolling_mean_rushing_yards_L8`
- `rolling_mean_targets_L3`
- `rolling_mean_targets_L8`
- `rolling_min_fantasy_points_L3`
- `rolling_min_fantasy_points_L5`
- `rolling_min_fantasy_points_L8`
- `rolling_std_carries_L3`
- `rolling_std_carries_L8`
- `rolling_std_fantasy_points_L3`
- `rolling_std_fantasy_points_L8`
- `rolling_std_receiving_yards_L3`
- `rolling_std_receiving_yards_L8`
- `rolling_std_rushing_yards_L3`
- `rolling_std_rushing_yards_L8`
- `rolling_std_targets_L3`
- `rolling_std_targets_L8`
- `rushing_epa_per_attempt_L3`
- `rushing_first_down_rate_L3`
- `snap_pct`
- `target_share_L3`
- `team_rb_carry_share_L3`
- `team_rb_target_hhi_L3`
- `team_rb_target_share_L3`
- `trend_carries`
- `trend_fantasy_points`
- `trend_snap_pct`
- `trend_targets`
- `yards_per_carry_L3`

Confirm (Stage 3 — production PCA-Ridge, high seed count):

```
python -m src.tuning.feature_selection confirm --position RB --from-stage2
```

Then apply the cut YOU choose after a clean confirm (draft PR):

```
python -m src.tuning.feature_selection apply --position RB --drop air_yards_per_target_L3 air_yards_share career_carries opp_def_pass_td_allowed_L5 opp_def_pts_allowed_L5 opp_def_rush_yds_allowed_L5 opp_def_sacks_L5 opp_rush_pts_allowed_to_pos opportunity_index_L3 prior_season_games_played prior_season_max_carries prior_season_max_snap_pct prior_season_max_targets prior_season_mean_carries prior_season_mean_catch_rate prior_season_mean_rush_yards_gained_exp prior_season_mean_snap_pct prior_season_mean_targets prior_season_mean_yards_per_carry prior_season_std_carries prior_season_std_snap_pct prior_season_std_targets prior_season_total_redzone_touches prior_season_total_touchdowns prior_season_total_yards receiving_epa_per_target_L3 rolling_max_carries_L3 rolling_max_carries_L8 rolling_max_fantasy_points_L3 rolling_max_fantasy_points_L8 rolling_max_receiving_yards_L3 rolling_max_receiving_yards_L8 rolling_max_rushing_yards_L3 rolling_max_rushing_yards_L8 rolling_max_targets_L3 rolling_max_targets_L8 rolling_mean_carries_L3 rolling_mean_carries_L8 rolling_mean_fantasy_points_L3 rolling_mean_fantasy_points_L8 rolling_mean_receiving_yards_L3 rolling_mean_receiving_yards_L8 rolling_mean_rushing_yards_L3 rolling_mean_rushing_yards_L8 rolling_mean_targets_L3 rolling_mean_targets_L8 rolling_min_fantasy_points_L3 rolling_min_fantasy_points_L5 rolling_min_fantasy_points_L8 rolling_std_carries_L3 rolling_std_carries_L8 rolling_std_fantasy_points_L3 rolling_std_fantasy_points_L8 rolling_std_receiving_yards_L3 rolling_std_receiving_yards_L8 rolling_std_rushing_yards_L3 rolling_std_rushing_yards_L8 rolling_std_targets_L3 rolling_std_targets_L8 rushing_epa_per_attempt_L3 rushing_first_down_rate_L3 snap_pct target_share_L3 team_rb_carry_share_L3 team_rb_target_hhi_L3 team_rb_target_share_L3 trend_carries trend_fantasy_points trend_snap_pct trend_targets yards_per_carry_L3 --pr
```

## `defense` sub-groups

- run-id: `subscreen-rb-defense-20260620T080604Z-503d31f`  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
- suggested sub-cut: `opp_def_pass_td_allowed_L5`, `opp_def_pts_allowed_L5`, `opp_def_rush_yds_allowed_L5`, `opp_def_sacks_L5`  -> columns: `opp_def_pass_td_allowed_L5`, `opp_def_pts_allowed_L5`, `opp_def_rush_yds_allowed_L5`, `opp_def_sacks_L5`

| sub-group | verdict | Ridge MAE | Ridge RMSE | LightGBM MAE | LightGBM RMSE | NN MAE | NN RMSE | Attention NN MAE | Attention NN RMSE |
|---|---|---|---|---|---|---|---|---|---|
| `opp_def_rush_yds_allowed_L5` | DROP-CAND | +0.008±0.000 | +0.008±0.000 | +0.009±0.000 | +0.021±0.000 | -0.028±0.000 | -0.014±0.000 | +0.000±0.000 | +0.000±0.000 |
| `opp_def_pts_allowed_L5` | DROP-CAND | -0.005±0.000 | -0.011±0.000 | +0.004±0.000 | -0.001±0.000 | -0.024±0.000 | -0.036±0.000 | +0.000±0.000 | +0.000±0.000 |
| `opp_def_pass_td_allowed_L5` | DROP-CAND | +0.010±0.000 | +0.016±0.000 | -0.002±0.000 | -0.000±0.000 | -0.016±0.000 | -0.015±0.000 | +0.000±0.000 | +0.000±0.000 |
| `opp_def_pass_yds_allowed_L5` | MIXED | -0.012±0.000 | +0.005±0.000 | +0.000±0.000 | +0.012±0.000 | +0.044±0.000 | -0.031±0.000 | +0.000±0.000 | +0.000±0.000 |
| `opp_def_ints_L5` | MIXED | +0.012±0.000 | +0.023±0.000 | -0.006±0.000 | -0.007±0.000 | +0.049±0.000 | +0.009±0.000 | +0.000±0.000 | +0.000±0.000 |
| `opp_def_sacks_L5` | DROP-CAND | +0.004±0.000 | +0.009±0.000 | +0.007±0.000 | +0.008±0.000 | +0.002±0.000 | +0.020±0.000 | +0.000±0.000 | +0.000±0.000 |

## `trend` sub-groups

- run-id: `subscreen-rb-trend-20260620T080604Z-503d31f`  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
- suggested sub-cut: `trend_carries`, `trend_fantasy_points`, `trend_snap_pct`, `trend_targets`  -> columns: `trend_carries`, `trend_fantasy_points`, `trend_snap_pct`, `trend_targets`

| sub-group | verdict | Ridge MAE | Ridge RMSE | LightGBM MAE | LightGBM RMSE | NN MAE | NN RMSE | Attention NN MAE | Attention NN RMSE |
|---|---|---|---|---|---|---|---|---|---|
| `trend_fantasy_points` | DROP-CAND | +0.002±0.000 | +0.008±0.000 | -0.005±0.000 | -0.007±0.000 | -0.026±0.000 | -0.048±0.000 | +0.000±0.000 | +0.000±0.000 |
| `trend_targets` | DROP-CAND | +0.002±0.000 | +0.003±0.000 | +0.002±0.000 | +0.004±0.000 | -0.023±0.000 | -0.109±0.000 | +0.000±0.000 | +0.000±0.000 |
| `trend_snap_pct` | DROP-CAND | +0.002±0.000 | +0.008±0.000 | +0.005±0.000 | +0.007±0.000 | -0.022±0.000 | +0.039±0.000 | +0.000±0.000 | +0.000±0.000 |
| `trend_carries` | DROP-CAND | +0.004±0.000 | +0.006±0.000 | -0.002±0.000 | -0.005±0.000 | +0.007±0.000 | -0.004±0.000 | +0.000±0.000 | +0.000±0.000 |

## `share` sub-groups

- run-id: `subscreen-rb-share-20260620T080604Z-503d31f`  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
- suggested sub-cut: `air_yards_share`, `snap_pct`, `target_share_L3`  -> columns: `air_yards_share`, `snap_pct`, `target_share_L3`

| sub-group | verdict | Ridge MAE | Ridge RMSE | LightGBM MAE | LightGBM RMSE | NN MAE | NN RMSE | Attention NN MAE | Attention NN RMSE |
|---|---|---|---|---|---|---|---|---|---|
| `air_yards_share` | DROP-CAND | +0.003±0.000 | -0.000±0.000 | -0.017±0.000 | -0.023±0.000 | +0.012±0.000 | +0.036±0.000 | +0.000±0.000 | +0.000±0.000 |
| `target_share_L3` | DROP-CAND | +0.006±0.000 | +0.006±0.000 | +0.011±0.000 | +0.000±0.000 | -0.000±0.000 | -0.061±0.000 | +0.000±0.000 | +0.000±0.000 |
| `snap_pct` | DROP-CAND | +0.006±0.000 | +0.010±0.000 | +0.007±0.000 | +0.019±0.000 | +0.006±0.000 | -0.020±0.000 | +0.000±0.000 | +0.000±0.000 |

## `matchup` sub-groups

- run-id: `subscreen-rb-matchup-20260620T080604Z-503d31f`  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
- suggested sub-cut: `opp_rush_pts_allowed_to_pos`  -> columns: `opp_rush_pts_allowed_to_pos`

| sub-group | verdict | Ridge MAE | Ridge RMSE | LightGBM MAE | LightGBM RMSE | NN MAE | NN RMSE | Attention NN MAE | Attention NN RMSE |
|---|---|---|---|---|---|---|---|---|---|
| `opp_rush_pts_allowed_to_pos` | DROP-CAND | +0.002±0.000 | +0.002±0.000 | -0.010±0.000 | -0.004±0.000 | -0.029±0.000 | -0.062±0.000 | -0.015±0.020 | +0.018±0.068 |
| `opp_recv_pts_allowed_to_pos` | MIXED | -0.002±0.000 | -0.002±0.000 | +0.010±0.000 | +0.004±0.000 | +0.029±0.000 | +0.062±0.000 | +0.015±0.020 | -0.018±0.068 |

## `prior_season` sub-groups

- run-id: `subscreen-rb-prior_season-20260620T080604Z-503d31f`  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
- suggested sub-cut: `carries`, `catch_rate`, `games_played`, `redzone_touches`, `rush_yards_gained_exp`, `snap_pct`, `targets`, `touchdowns`, `yards`, `yards_per_carry`  -> columns: `prior_season_games_played`, `prior_season_max_carries`, `prior_season_max_snap_pct`, `prior_season_max_targets`, `prior_season_mean_carries`, `prior_season_mean_catch_rate`, `prior_season_mean_rush_yards_gained_exp`, `prior_season_mean_snap_pct`, `prior_season_mean_targets`, `prior_season_mean_yards_per_carry`, `prior_season_std_carries`, `prior_season_std_snap_pct`, `prior_season_std_targets`, `prior_season_total_redzone_touches`, `prior_season_total_touchdowns`, `prior_season_total_yards`

| sub-group | verdict | Ridge MAE | Ridge RMSE | LightGBM MAE | LightGBM RMSE | NN MAE | NN RMSE | Attention NN MAE | Attention NN RMSE |
|---|---|---|---|---|---|---|---|---|---|
| `targets` | DROP-CAND | +0.008±0.000 | +0.017±0.000 | -0.011±0.000 | +0.000±0.000 | -0.046±0.000 | -0.026±0.000 | -0.002±0.026 | -0.000±0.052 |
| `snap_pct` | DROP-CAND | -0.006±0.000 | -0.012±0.000 | +0.004±0.000 | +0.013±0.000 | -0.037±0.000 | -0.007±0.000 | -0.017±0.024 | -0.009±0.044 |
| `yards` | DROP-CAND | -0.023±0.000 | -0.028±0.000 | -0.014±0.000 | -0.007±0.000 | -0.002±0.000 | -0.066±0.000 | +0.016±0.117 | +0.072±0.303 |
| `catch_rate` | DROP-CAND | +0.011±0.000 | +0.012±0.000 | +0.001±0.000 | -0.003±0.000 | -0.019±0.000 | -0.056±0.000 | +0.001±0.016 | -0.021±0.034 |
| `fumbles_lost` | MIXED | +0.024±0.000 | +0.025±0.000 | -0.006±0.000 | -0.004±0.000 | -0.012±0.000 | -0.056±0.000 | -0.018±0.016 | -0.018±0.059 |
| `touchdowns` | DROP-CAND | +0.012±0.000 | +0.025±0.000 | +0.013±0.000 | +0.013±0.000 | -0.015±0.000 | -0.031±0.000 | -0.005±0.024 | -0.013±0.046 |
| `total_fantasy_points_exp` | MIXED | -0.011±0.000 | -0.010±0.000 | -0.013±0.000 | -0.008±0.000 | +0.022±0.000 | -0.024±0.000 | +0.001±0.016 | -0.012±0.034 |
| `yards_per_carry` | DROP-CAND | -0.009±0.000 | -0.008±0.000 | -0.010±0.000 | -0.021±0.000 | +0.002±0.000 | -0.116±0.000 | +0.002±0.019 | -0.015±0.034 |
| `redzone_touches_per_game` | MIXED | -0.009±0.000 | -0.012±0.000 | +0.004±0.000 | -0.006±0.000 | +0.026±0.000 | +0.257±0.000 | -0.001±0.017 | -0.004±0.049 |
| `games_played` | DROP-CAND | +0.011±0.000 | +0.007±0.000 | +0.002±0.000 | +0.009±0.000 | -0.009±0.000 | -0.051±0.000 | +0.005±0.016 | -0.005±0.061 |
| `receptions_exp` | MIXED | -0.007±0.000 | -0.008±0.000 | -0.004±0.000 | +0.001±0.000 | +0.047±0.000 | +0.324±0.000 | -0.004±0.018 | -0.001±0.058 |
| `carries` | DROP-CAND | -0.006±0.000 | -0.009±0.000 | +0.007±0.000 | -0.003±0.000 | +0.001±0.000 | -0.049±0.000 | -0.000±0.026 | -0.001±0.054 |
| `redzone_touches` | DROP-CAND | -0.000±0.000 | -0.002±0.000 | +0.004±0.000 | +0.007±0.000 | +0.014±0.000 | -0.102±0.000 | -0.005±0.019 | -0.017±0.030 |
| `rush_yards_gained_exp` | DROP-CAND | -0.004±0.000 | -0.004±0.000 | +0.019±0.000 | +0.011±0.000 | -0.001±0.000 | -0.009±0.000 | -0.001±0.016 | -0.018±0.035 |
| `receiving_yards` | MIXED | +0.012±0.000 | +0.007±0.000 | +0.001±0.000 | -0.001±0.000 | +0.031±0.000 | +0.012±0.000 | +0.029±0.101 | +0.063±0.281 |

## `specific` sub-groups

- run-id: `subscreen-rb-specific-20260620T080604Z-503d31f`  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
- suggested sub-cut: `air_yards_per_target_L3`, `career_carries`, `opportunity_index_L3`, `receiving_epa_per_target_L3`, `rushing_epa_per_attempt_L3`, `rushing_first_down_rate_L3`, `team_rb_carry_share_L3`, `team_rb_target_hhi_L3`, `team_rb_target_share_L3`, `yards_per_carry_L3`  -> columns: `air_yards_per_target_L3`, `career_carries`, `opportunity_index_L3`, `receiving_epa_per_target_L3`, `rushing_epa_per_attempt_L3`, `rushing_first_down_rate_L3`, `team_rb_carry_share_L3`, `team_rb_target_hhi_L3`, `team_rb_target_share_L3`, `yards_per_carry_L3`

| sub-group | verdict | Ridge MAE | Ridge RMSE | LightGBM MAE | LightGBM RMSE | NN MAE | NN RMSE | Attention NN MAE | Attention NN RMSE |
|---|---|---|---|---|---|---|---|---|---|
| `yards_per_carry_L3` | DROP-CAND | -0.003±0.000 | -0.006±0.000 | +0.010±0.000 | +0.008±0.000 | -0.033±0.000 | +0.032±0.000 | +0.000±0.000 | +0.000±0.000 |
| `rushing_epa_per_attempt_L3` | DROP-CAND | +0.007±0.000 | +0.006±0.000 | -0.008±0.000 | -0.002±0.000 | -0.025±0.000 | -0.034±0.000 | +0.000±0.000 | +0.000±0.000 |
| `team_rb_carry_share_L3` | DROP-CAND | +0.002±0.000 | +0.008±0.000 | -0.022±0.000 | -0.008±0.000 | -0.019±0.000 | -0.033±0.000 | +0.000±0.000 | +0.000±0.000 |
| `receiving_epa_per_target_L3` | DROP-CAND | -0.006±0.000 | -0.007±0.000 | -0.019±0.000 | -0.016±0.000 | -0.009±0.000 | -0.031±0.000 | +0.000±0.000 | +0.000±0.000 |
| `rushing_first_down_rate_L3` | DROP-CAND | -0.005±0.000 | -0.011±0.000 | +0.009±0.000 | +0.009±0.000 | -0.018±0.000 | -0.038±0.000 | +0.000±0.000 | +0.000±0.000 |
| `air_yards_per_target_L3` | DROP-CAND | -0.004±0.000 | -0.001±0.000 | +0.008±0.000 | +0.004±0.000 | -0.017±0.000 | -0.007±0.000 | +0.000±0.000 | +0.000±0.000 |
| `yac_per_reception_L3` | MIXED | -0.009±0.000 | -0.010±0.000 | -0.014±0.000 | -0.010±0.000 | +0.047±0.000 | +0.074±0.000 | +0.000±0.000 | +0.000±0.000 |
| `career_carries` | DROP-CAND | +0.002±0.000 | +0.029±0.000 | -0.014±0.000 | -0.023±0.000 | +0.002±0.000 | -0.041±0.000 | +0.000±0.000 | +0.000±0.000 |
| `opportunity_index_L3` | DROP-CAND | +0.009±0.000 | -0.004±0.000 | +0.002±0.000 | +0.002±0.000 | -0.009±0.000 | -0.035±0.000 | +0.000±0.000 | +0.000±0.000 |
| `team_rb_target_hhi_L3` | DROP-CAND | +0.008±0.000 | +0.005±0.000 | -0.006±0.000 | -0.000±0.000 | -0.009±0.000 | -0.064±0.000 | +0.000±0.000 | +0.000±0.000 |
| `receiving_first_down_rate_L3` | MIXED | -0.003±0.000 | -0.005±0.000 | -0.005±0.000 | -0.005±0.000 | +0.029±0.000 | +0.032±0.000 | +0.000±0.000 | +0.000±0.000 |
| `reception_rate_L3` | MIXED | -0.004±0.000 | -0.008±0.000 | +0.026±0.000 | +0.003±0.000 | -0.004±0.000 | +0.008±0.000 | +0.000±0.000 | +0.000±0.000 |
| `team_rb_target_share_L3` | DROP-CAND | -0.004±0.000 | -0.008±0.000 | +0.011±0.000 | +0.015±0.000 | +0.008±0.000 | +0.055±0.000 | +0.000±0.000 | +0.000±0.000 |
| `team_rb_carry_hhi_L3` | MIXED | +0.009±0.000 | +0.012±0.000 | +0.021±0.000 | +0.023±0.000 | +0.058±0.000 | +0.083±0.000 | +0.000±0.000 | +0.000±0.000 |

## `rolling` sub-groups

- run-id: `subscreen-rb-rolling-20260620T080604Z-503d31f`  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
- suggested sub-cut: `carries`, `fantasy_points`, `receiving_yards`, `rushing_yards`, `targets`  -> columns: `rolling_max_carries_L3`, `rolling_max_carries_L8`, `rolling_max_fantasy_points_L3`, `rolling_max_fantasy_points_L8`, `rolling_max_receiving_yards_L3`, `rolling_max_receiving_yards_L8`, `rolling_max_rushing_yards_L3`, `rolling_max_rushing_yards_L8`, `rolling_max_targets_L3`, `rolling_max_targets_L8`, `rolling_mean_carries_L3`, `rolling_mean_carries_L8`, `rolling_mean_fantasy_points_L3`, `rolling_mean_fantasy_points_L8`, `rolling_mean_receiving_yards_L3`, `rolling_mean_receiving_yards_L8`, `rolling_mean_rushing_yards_L3`, `rolling_mean_rushing_yards_L8`, `rolling_mean_targets_L3`, `rolling_mean_targets_L8`, `rolling_min_fantasy_points_L3`, `rolling_min_fantasy_points_L5`, `rolling_min_fantasy_points_L8`, `rolling_std_carries_L3`, `rolling_std_carries_L8`, `rolling_std_fantasy_points_L3`, `rolling_std_fantasy_points_L8`, `rolling_std_receiving_yards_L3`, `rolling_std_receiving_yards_L8`, `rolling_std_rushing_yards_L3`, `rolling_std_rushing_yards_L8`, `rolling_std_targets_L3`, `rolling_std_targets_L8`

| sub-group | verdict | Ridge MAE | Ridge RMSE | LightGBM MAE | LightGBM RMSE | NN MAE | NN RMSE | Attention NN MAE | Attention NN RMSE |
|---|---|---|---|---|---|---|---|---|---|
| `targets` | DROP-CAND | +0.007±0.000 | +0.003±0.000 | +0.006±0.000 | +0.009±0.000 | -0.027±0.000 | -0.065±0.000 | +0.000±0.000 | +0.000±0.000 |
| `snap_pct` | MIXED | +0.011±0.000 | +0.001±0.000 | -0.006±0.000 | +0.014±0.000 | +0.025±0.000 | -0.014±0.000 | +0.000±0.000 | +0.000±0.000 |
| `receiving_yards` | DROP-CAND | -0.002±0.000 | +0.003±0.000 | +0.016±0.000 | +0.009±0.000 | +0.007±0.000 | -0.086±0.000 | +0.000±0.000 | +0.000±0.000 |
| `rushing_yards` | DROP-CAND | +0.004±0.000 | +0.015±0.000 | -0.001±0.000 | -0.010±0.000 | +0.007±0.000 | +0.063±0.000 | +0.000±0.000 | +0.000±0.000 |
| `carries` | DROP-CAND | +0.005±0.000 | +0.004±0.000 | +0.017±0.000 | +0.015±0.000 | +0.009±0.000 | -0.049±0.000 | +0.000±0.000 | +0.000±0.000 |
| `fantasy_points` | DROP-CAND | +0.000±0.000 | +0.000±0.000 | +0.008±0.000 | +0.014±0.000 | +0.018±0.000 | +0.088±0.000 | +0.000±0.000 | +0.000±0.000 |
| `receptions` | MIXED | +0.007±0.000 | +0.011±0.000 | +0.025±0.000 | +0.026±0.000 | +0.003±0.000 | -0.089±0.000 | +0.000±0.000 | +0.000±0.000 |

## Caveats

- **Sign:** `+` = dropping the sub-group RAISES error = it carries signal (keep). `-` = drop candidate.
- **Skip-PCA screen.** Ridge here runs on raw features (PCA off) for clean attribution; production RB/WR/DST ship PCA-Ridge. The combined drop-set MUST be confirmed on the production config (Stage 3) before `apply`.
- **Stacked vs eager / subgroup bias.** Skill stacks (vmap, FP32/LN/fixed-epochs); K/DST eager. Judge borderline sub-groups by subgroup *bias*, not overall MAE, and confirm the combined drop at high seed count — PB main effects assume additivity.
