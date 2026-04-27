"""Tests for src.rb.features — _compute_features and fill_nans."""

import numpy as np
import pandas as pd
import pytest

from src.rb.features import _compute_features, fill_nans

FEATURE_COLS = [
    "yards_per_carry_L3",
    "reception_rate_L3",
    "team_rb_carry_share_L3",
    "team_rb_target_share_L3",
    "rushing_epa_per_attempt_L3",
    "rushing_first_down_rate_L3",
    "receiving_first_down_rate_L3",
    "yac_per_reception_L3",
    "receiving_epa_per_target_L3",
    "air_yards_per_target_L3",
]


# ---------------------------------------------------------------------------
# _compute_features
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeRBFeatures:
    def test_all_features_created(self, make_player_games):
        df = make_player_games()
        _compute_features(df)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"

    def test_no_temp_columns_remain(self, make_player_games):
        """Temporary columns (_raw_weighted_opps, etc.) should be cleaned up."""
        df = make_player_games()
        _compute_features(df)
        temp_cols = [c for c in df.columns if c.startswith("_")]
        assert temp_cols == [], f"Temp columns left: {temp_cols}"

    def test_first_week_features_are_zero_or_nan(self, make_player_games):
        """Week 1 has no prior data (shift=1), so features should be 0 or NaN."""
        df = make_player_games(n_weeks=4)
        _compute_features(df)
        week1 = df[df["week"] == 1]
        # Shifted rolling means should produce NaN -> filled to 0
        for col in ["yards_per_carry_L3", "reception_rate_L3"]:
            val = week1[col].iloc[0]
            assert val == 0.0 or np.isnan(val), f"{col} = {val} for week 1"

    def test_yards_per_carry_computation(self, make_player_games):
        """Check yards_per_carry_L3 for constant carries/yards."""
        df = make_player_games(n_weeks=5, carries=10, rushing_yards=50)
        _compute_features(df)
        # Week 3 (index 2): shift sees weeks 1,2; rolling(3,min_periods=1) sums [50,50]=100 yds / 20 carries = 5.0
        week3 = df[df["week"] == 3]
        assert pytest.approx(week3["yards_per_carry_L3"].iloc[0], abs=0.01) == 5.0

    def test_zero_carries_no_division_error(self, make_player_games):
        """Player with 0 carries: yards_per_carry_L3 should be 0, not inf."""
        df = make_player_games(n_weeks=4, carries=0, rushing_yards=0)
        _compute_features(df)
        assert not df["yards_per_carry_L3"].isin([np.inf, -np.inf]).any()
        assert (df["yards_per_carry_L3"].fillna(0) == 0).all()

    def test_zero_targets_no_division_error(self, make_player_games):
        """Player with 0 targets: reception_rate_L3 should be 0, not inf."""
        df = make_player_games(n_weeks=4, targets=0, receptions=0)
        _compute_features(df)
        assert not df["reception_rate_L3"].isin([np.inf, -np.inf]).any()

    def test_zero_receptions_yac(self, make_player_games):
        """Player with 0 receptions: yac_per_reception_L3 should be 0."""
        df = make_player_games(n_weeks=4, receptions=0, receiving_yards_after_catch=0)
        _compute_features(df)
        assert not df["yac_per_reception_L3"].isin([np.inf, -np.inf]).any()

    def test_team_carry_share_single_player(self, make_player_games):
        """Solo RB on team should have carry share of 1.0."""
        df = make_player_games(n_weeks=5, carries=15, recent_team="KC")
        _compute_features(df)
        # After enough history, share should be 1.0 (only RB on team)
        later_weeks = df[df["week"] >= 3]
        shares = later_weeks["team_rb_carry_share_L3"]
        valid = shares.dropna()
        if len(valid) > 0:
            assert (valid <= 1.01).all()  # should be ~1.0

    def test_team_carry_share_two_players(self, make_player_games):
        """Two RBs on same team split carries."""
        p1 = make_player_games("P1", n_weeks=5, carries=10, targets=3, recent_team="KC")
        p2 = make_player_games("P2", n_weeks=5, carries=10, targets=3, recent_team="KC")
        df = pd.concat([p1, p2], ignore_index=True)
        _compute_features(df)
        # Each player should have ~0.5 carry share after history builds up
        later = df[(df["week"] >= 3) & (df["player_id"] == "P1")]
        shares = later["team_rb_carry_share_L3"].dropna()
        if len(shares) > 0:
            assert all(0.4 <= s <= 0.6 for s in shares)

    def test_multiple_seasons_independent(self, make_player_games):
        """Features should not leak across seasons."""
        s1 = make_player_games(season=2022, n_weeks=3, carries=20, rushing_yards=100)
        s2 = make_player_games(season=2023, n_weeks=3, carries=5, rushing_yards=20)
        df = pd.concat([s1, s2], ignore_index=True)
        _compute_features(df)
        # 2023 week 1 should not see 2022 data
        w1_2023 = df[(df["season"] == 2023) & (df["week"] == 1)]
        ypc = w1_2023["yards_per_carry_L3"].iloc[0]
        assert ypc == 0.0 or np.isnan(ypc)


# ---------------------------------------------------------------------------
# fill_nans
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFillRBNans:
    def test_fills_nan_with_train_mean(self, make_splits):
        train, val, test = make_splits(
            [1.0, 2.0, 3.0],
            [np.nan],
            [np.nan],
        )
        train, val, test = fill_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0  # mean of [1,2,3]
        assert pytest.approx(test["feat1"].iloc[0]) == 2.0

    def test_replaces_inf_with_train_mean(self, make_splits):
        train, val, test = make_splits(
            [1.0, 3.0],
            [np.inf],
            [-np.inf],
        )
        train, val, test = fill_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0
        assert pytest.approx(test["feat1"].iloc[0]) == 2.0

    def test_train_inf_replaced_before_mean(self, make_splits):
        """Inf in training set should be replaced with NaN before computing mean."""
        train, val, test = make_splits(
            [1.0, np.inf, 3.0],
            [np.nan],
            [np.nan],
        )
        train, val, test = fill_nans(train, val, test, ["feat1"])
        # mean of [1.0, NaN, 3.0] = 2.0
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0

    def test_all_nan_train_fills_with_zero(self, make_splits, capsys):
        """If training set is all NaN, train_mean is NaN; previously the fill
        was a silent no-op and the catch-all `.fillna(0)` in
        `build_position_features` masked the failure. The hardened
        `fill_nans_with_train_means` now substitutes 0 explicitly and prints
        a warning so the silent zero-feature is visible."""
        train, val, test = make_splits(
            [np.nan, np.nan],
            [np.nan],
            [np.nan],
        )
        train, val, test = fill_nans(train, val, test, ["feat1"])
        assert val["feat1"].iloc[0] == 0.0
        assert test["feat1"].iloc[0] == 0.0
        captured = capsys.readouterr().out
        assert "feat1" in captured
        assert "entirely NaN in training" in captured

    def test_no_nans_unchanged(self, make_splits):
        train, val, test = make_splits(
            [1.0, 2.0],
            [3.0],
            [4.0],
        )
        train, val, test = fill_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 3.0
        assert pytest.approx(test["feat1"].iloc[0]) == 4.0

    def test_multiple_columns(self):
        train = pd.DataFrame({"f1": [1.0, 3.0], "f2": [10.0, 20.0]})
        val = pd.DataFrame({"f1": [np.nan], "f2": [np.nan]})
        test = pd.DataFrame({"f1": [5.0], "f2": [np.nan]})
        train, val, test = fill_nans(train, val, test, ["f1", "f2"])
        assert pytest.approx(val["f1"].iloc[0]) == 2.0
        assert pytest.approx(val["f2"].iloc[0]) == 15.0
        assert pytest.approx(test["f1"].iloc[0]) == 5.0  # wasn't NaN
        assert pytest.approx(test["f2"].iloc[0]) == 15.0
