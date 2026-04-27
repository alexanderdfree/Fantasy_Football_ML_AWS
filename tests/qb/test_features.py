"""Tests for QB.qb_features — _compute_features and fill_nans."""

import numpy as np
import pandas as pd
import pytest

from src.qb.features import _compute_features, fill_nans


def _make_player_games(
    player_id="QB1",
    season=2023,
    n_weeks=5,
    completions=20,
    attempts=30,
    passing_yards=250,
    passing_tds=2,
    interceptions=1,
    sacks=2,
    rushing_yards=15,
    passing_epa=5.0,
    passing_air_yards=200,
    carries=3,
    passing_first_downs=10,
    rushing_first_downs=1,
    rushing_epa=1.0,
    passing_yards_after_catch=120,
    sack_yards=14,
):
    """Create a multi-week DataFrame for one QB."""
    return pd.DataFrame(
        {
            "player_id": [player_id] * n_weeks,
            "season": [season] * n_weeks,
            "week": list(range(1, n_weeks + 1)),
            "completions": [completions] * n_weeks,
            "attempts": [attempts] * n_weeks,
            "passing_yards": [passing_yards] * n_weeks,
            "passing_tds": [passing_tds] * n_weeks,
            "interceptions": [interceptions] * n_weeks,
            "sacks": [sacks] * n_weeks,
            "rushing_yards": [rushing_yards] * n_weeks,
            "passing_epa": [passing_epa] * n_weeks,
            "passing_air_yards": [passing_air_yards] * n_weeks,
            "carries": [carries] * n_weeks,
            "passing_first_downs": [passing_first_downs] * n_weeks,
            "rushing_first_downs": [rushing_first_downs] * n_weeks,
            "rushing_epa": [rushing_epa] * n_weeks,
            "passing_yards_after_catch": [passing_yards_after_catch] * n_weeks,
            "sack_yards": [sack_yards] * n_weeks,
        }
    )


FEATURE_COLS = [
    "completion_pct_L3",
    "yards_per_attempt_L3",
    "td_rate_L3",
    "int_rate_L3",
    "sack_rate_L3",
    "qb_rushing_share_L3",
    "passing_epa_per_dropback_L3",
    "deep_ball_rate_L3",
    "pass_first_down_rate_L3",
    "rushing_epa_per_carry_L3",
    "rush_first_down_rate_L3",
    "yac_rate_L3",
    "sack_damage_per_dropback_L3",
]


# ---------------------------------------------------------------------------
# _compute_features
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeQBFeatures:
    def test_all_features_created(self):
        df = _make_player_games()
        _compute_features(df)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"

    def test_first_week_features_are_zero(self):
        """Week 1 has no prior data (shift=1), so features should be 0 (filled)."""
        df = _make_player_games(n_weeks=4)
        _compute_features(df)
        week1 = df[df["week"] == 1]
        for col in ["completion_pct_L3", "yards_per_attempt_L3", "td_rate_L3"]:
            val = week1[col].iloc[0]
            assert val == 0.0 or np.isnan(val), f"{col} = {val} for week 1"

    def test_completion_pct_computation(self):
        """completion_pct_L3 = rolling completions / rolling attempts."""
        df = _make_player_games(n_weeks=5, completions=20, attempts=30)
        _compute_features(df)
        # Week 3 (index 2): shift sees weeks 1,2; rolling(3, sum) = [40]/[60] = 0.667
        week3 = df[df["week"] == 3]
        assert pytest.approx(week3["completion_pct_L3"].iloc[0], abs=0.01) == 20 / 30

    def test_yards_per_attempt_computation(self):
        df = _make_player_games(n_weeks=5, attempts=30, passing_yards=240)
        _compute_features(df)
        week3 = df[df["week"] == 3]
        # 480 / 60 = 8.0
        assert pytest.approx(week3["yards_per_attempt_L3"].iloc[0], abs=0.01) == 8.0

    def test_td_rate_computation(self):
        df = _make_player_games(n_weeks=5, attempts=25, passing_tds=2)
        _compute_features(df)
        week3 = df[df["week"] == 3]
        # 4 / 50 = 0.08
        assert pytest.approx(week3["td_rate_L3"].iloc[0], abs=0.01) == 2 / 25

    def test_int_rate_computation(self):
        df = _make_player_games(n_weeks=5, attempts=25, interceptions=1)
        _compute_features(df)
        week3 = df[df["week"] == 3]
        assert pytest.approx(week3["int_rate_L3"].iloc[0], abs=0.01) == 1 / 25

    def test_sack_rate_computation(self):
        """sack_rate = sacks / (attempts + sacks)."""
        df = _make_player_games(n_weeks=5, attempts=30, sacks=2)
        _compute_features(df)
        week3 = df[df["week"] == 3]
        # sacks=4, dropbacks=60+4=64, rate = 4/64 = 0.0625
        assert pytest.approx(week3["sack_rate_L3"].iloc[0], abs=0.01) == 4 / 64

    def test_rushing_share_computation(self):
        """qb_rushing_share = rush_yds / (pass_yds + rush_yds)."""
        df = _make_player_games(n_weeks=5, passing_yards=300, rushing_yards=30)
        _compute_features(df)
        week3 = df[df["week"] == 3]
        # 60 / 660 = 0.0909
        assert pytest.approx(week3["qb_rushing_share_L3"].iloc[0], abs=0.01) == 60 / 660

    def test_zero_attempts_no_division_error(self):
        """QB with 0 attempts should have 0 rates (not inf)."""
        df = _make_player_games(
            n_weeks=4,
            completions=0,
            attempts=0,
            passing_yards=0,
            passing_tds=0,
            interceptions=0,
            sacks=0,
            passing_air_yards=0,
            passing_epa=0,
            passing_first_downs=0,
            passing_yards_after_catch=0,
            sack_yards=0,
        )
        _compute_features(df)
        for col in [
            "completion_pct_L3",
            "yards_per_attempt_L3",
            "td_rate_L3",
            "int_rate_L3",
            "deep_ball_rate_L3",
            "pass_first_down_rate_L3",
        ]:
            assert not df[col].isin([np.inf, -np.inf]).any(), f"{col} has inf"
            assert (df[col].fillna(0) == 0).all(), f"{col} should be 0"

    def test_zero_carries_no_division_error(self):
        """QB with 0 carries should have 0 rushing rates (not inf)."""
        df = _make_player_games(
            n_weeks=4, carries=0, rushing_yards=0, rushing_first_downs=0, rushing_epa=0
        )
        _compute_features(df)
        assert not df["rushing_epa_per_carry_L3"].isin([np.inf, -np.inf]).any()
        assert not df["rush_first_down_rate_L3"].isin([np.inf, -np.inf]).any()

    def test_zero_passing_yards_yac_rate(self):
        """QB with 0 passing yards should have 0 yac_rate (not inf)."""
        df = _make_player_games(n_weeks=4, passing_yards=0, passing_yards_after_catch=0)
        _compute_features(df)
        assert not df["yac_rate_L3"].isin([np.inf, -np.inf]).any()

    def test_multiple_seasons_independent(self):
        """Features should not leak across seasons."""
        s1 = _make_player_games(season=2022, n_weeks=3, attempts=30, passing_yards=300)
        s2 = _make_player_games(season=2023, n_weeks=3, attempts=20, passing_yards=150)
        df = pd.concat([s1, s2], ignore_index=True)
        _compute_features(df)
        w1_2023 = df[(df["season"] == 2023) & (df["week"] == 1)]
        ypa = w1_2023["yards_per_attempt_L3"].iloc[0]
        assert ypa == 0.0 or np.isnan(ypa)


# ---------------------------------------------------------------------------
# fill_nans
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFillQBNans:
    def test_fills_nan_with_train_mean(self, make_splits):
        train, val, test = make_splits([1.0, 2.0, 3.0], [np.nan], [np.nan])
        train, val, test = fill_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0
        assert pytest.approx(test["feat1"].iloc[0]) == 2.0

    def test_replaces_inf_with_train_mean(self, make_splits):
        train, val, test = make_splits([1.0, 3.0], [np.inf], [-np.inf])
        train, val, test = fill_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0
        assert pytest.approx(test["feat1"].iloc[0]) == 2.0

    def test_train_inf_replaced_before_mean(self, make_splits):
        """Inf in training set should be replaced with NaN before computing mean."""
        train, val, test = make_splits([1.0, np.inf, 3.0], [np.nan], [np.nan])
        train, val, test = fill_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0

    def test_no_nans_unchanged(self, make_splits):
        train, val, test = make_splits([1.0, 2.0], [3.0], [4.0])
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
        assert pytest.approx(test["f1"].iloc[0]) == 5.0
        assert pytest.approx(test["f2"].iloc[0]) == 15.0
