# Feature-selection report — K

- spec: `src.tuning.ab_feature_screen_k`
- run-id: `ab_feature_screen_k-20260619T234433Z-ff086c1`
- seeds: [42, 123, 7]
- noise floor: 0.02 FP (AGENTS.md); effect = MAE/RMSE delta when the group is DROPPED.
- **Sign:** `+` = dropping RAISES error = group carries signal (keep). `-` = dropping LOWERS error = drop candidate.

## Per-group effect by model (MAE | RMSE)

| group | verdict | Ridge MAE | Ridge RMSE | LightGBM MAE | LightGBM RMSE | NN MAE | NN RMSE | Attention NN MAE | Attention NN RMSE |
|---|---|---|---|---|---|---|---|---|---|
| `fg_accuracy` | MIXED | +0.010±0.000 | +0.014±0.000 | +0.024±0.004 | +0.034±0.002 | -0.259±0.461 | -0.309±0.498 | -0.005±0.009 | -0.000±0.012 |
| `fg_volume` | DROP-CAND | +0.003±0.000 | -0.006±0.000 | -0.009±0.007 | -0.014±0.004 | -0.231±0.364 | -0.266±0.413 | +0.005±0.009 | +0.000±0.012 |
| `k_trend` | DROP-CAND | -0.004±0.000 | -0.013±0.000 | -0.010±0.002 | -0.018±0.006 | -0.204±0.377 | -0.238±0.423 | +0.005±0.009 | +0.000±0.012 |
| `game_context` | DROP-CAND | +0.012±0.000 | +0.016±0.000 | +0.019±0.003 | +0.018±0.002 | -0.184±0.289 | -0.221±0.306 | -0.056±0.008 | -0.053±0.013 |
| `weather` | MIXED | +0.018±0.000 | +0.033±0.000 | +0.025±0.001 | +0.045±0.003 | -0.158±0.312 | -0.149±0.344 | +0.038±0.024 | +0.051±0.027 |
| `fg_distance` | MIXED | +0.006±0.000 | +0.016±0.000 | +0.011±0.002 | +0.019±0.005 | +0.216±0.364 | +0.273±0.395 | -0.005±0.009 | -0.000±0.012 |

## Suggested conservative cut (review — not auto-applied)

Groups neutral-or-helpful (MAE effect ≤ noise) for **every** model present. Dropping them should not regress any model beyond the noise floor — but confirm with a combined-drop run at high seed count (Stage 3) before applying:

- `fg_volume` — columns: `fg_attempts_L3`, `pat_volume_L3`, `total_k_pts_L3`
- `k_trend` — columns: `k_pts_std_L3`, `k_pts_trend`
- `game_context` — columns: `implied_team_total`, `is_home`, `total_line`, `week`

Apply (after review):

```
python -m src.tuning.feature_selection apply --position K --drop fg_attempts_L3 implied_team_total is_home k_pts_std_L3 k_pts_trend pat_volume_L3 total_k_pts_L3 total_line week --pr
```

## Caveats

- **Neutral overall ≠ useless.** A group flat on overall MAE/RMSE may still carry subgroup signal (rookies/RB, returners). Judge subgroup value by *bias*, not overall MAE (draft-capital lesson). Check `result['test_df']` cohorts before dropping a borderline group.
- **Stacked vs eager:** skill positions screen stacked (vmap, FP32/LN/fixed-epochs); K/DST screen eager. Compare stacked arms only against stacked arms — never seed-by-seed against an eager run.
- **Confirm before applying.** PB main effects assume additivity; re-run the chosen drop-set together (Stage 3) at high seed count to catch interactions.
