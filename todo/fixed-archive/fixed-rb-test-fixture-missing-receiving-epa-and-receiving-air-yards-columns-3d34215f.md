> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] RB test fixture missing `receiving_epa` and `receiving_air_yards` columns
- **File:** `tests/rb/test_features.py:10-52`
- **What:** `_make_player_games()` fixture didn't include `receiving_epa` or `receiving_air_yards` columns added with features 10-11. The feature list `RB_FEATURE_COLS` also had stale names (`first_down_rate_L3` instead of split `rushing_first_down_rate_L3` / `receiving_first_down_rate_L3`). 11 tests failed with `KeyError: 'receiving_epa'`.
- **Fix:** Added missing columns to the fixture and updated `RB_FEATURE_COLS` to match the current 11 features in `rb_config.py`.
- **Lesson:** When adding new features to `*_features.py`, update the corresponding test fixture and expected feature list.
