"""Tests for WR.wr_features — _compute_wr_features and fill_wr_nans."""

import numpy as np
import pandas as pd
import pytest

from src.WR.wr_config import WR_SPECIFIC_FEATURES
from src.WR.wr_features import _compute_wr_features, fill_wr_nans

# ---------------------------------------------------------------------------
# _compute_wr_features
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeWRFeatures:
    def test_all_features_created(self, wr_player_games_factory):
        df = wr_player_games_factory()
        _compute_wr_features(df)
        for col in WR_SPECIFIC_FEATURES:
            assert col in df.columns, f"Missing feature: {col}"

    def test_first_week_features_are_zero(self, wr_player_games_factory):
        df = wr_player_games_factory(n_weeks=4)
        _compute_wr_features(df)
        week1 = df[df["week"] == 1]
        for col in ["yards_per_reception_L3", "reception_rate_L3"]:
            val = week1[col].iloc[0]
            assert val == 0.0 or np.isnan(val), f"{col} = {val} for week 1"

    def test_yards_per_reception_computation(self, wr_player_games_factory):
        df = wr_player_games_factory(n_weeks=5, receptions=5, receiving_yards=70)
        _compute_wr_features(df)
        # Week 3: rolling sum of [70, 70] = 140 yds / 10 receptions = 14.0
        week3 = df[df["week"] == 3]
        assert pytest.approx(week3["yards_per_reception_L3"].iloc[0], abs=0.01) == 14.0

    def test_reception_rate_computation(self, wr_player_games_factory):
        df = wr_player_games_factory(n_weeks=5, receptions=5, targets=8)
        _compute_wr_features(df)
        week3 = df[df["week"] == 3]
        # 10 / 16 = 0.625
        assert pytest.approx(week3["reception_rate_L3"].iloc[0], abs=0.01) == 5 / 8

    def test_yards_per_target_computation(self, wr_player_games_factory):
        df = wr_player_games_factory(n_weeks=5, targets=8, receiving_yards=80)
        _compute_wr_features(df)
        week3 = df[df["week"] == 3]
        # 160 / 16 = 10.0
        assert pytest.approx(week3["yards_per_target_L3"].iloc[0], abs=0.01) == 10.0

    def test_air_yards_per_target_computation(self, wr_player_games_factory):
        df = wr_player_games_factory(n_weeks=5, targets=8, receiving_air_yards=100)
        _compute_wr_features(df)
        week3 = df[df["week"] == 3]
        # 200 / 16 = 12.5
        assert pytest.approx(week3["air_yards_per_target_L3"].iloc[0], abs=0.01) == 12.5

    def test_zero_targets_no_division_error(self, wr_player_games_factory):
        df = wr_player_games_factory(
            n_weeks=4,
            targets=0,
            receptions=0,
            receiving_yards=0,
            receiving_air_yards=0,
            receiving_yards_after_catch=0,
            receiving_epa=0,
            receiving_first_downs=0,
        )
        _compute_wr_features(df)
        for col in [
            "yards_per_target_L3",
            "reception_rate_L3",
            "air_yards_per_target_L3",
            "receiving_epa_per_target_L3",
        ]:
            assert not df[col].isin([np.inf, -np.inf]).any(), f"{col} has inf"

    def test_zero_receptions_yac(self, wr_player_games_factory):
        df = wr_player_games_factory(
            n_weeks=4,
            receptions=0,
            receiving_yards_after_catch=0,
            receiving_first_downs=0,
            receiving_yards=0,
        )
        _compute_wr_features(df)
        assert not df["yac_per_reception_L3"].isin([np.inf, -np.inf]).any()
        assert not df["receiving_first_down_rate_L3"].isin([np.inf, -np.inf]).any()

    def test_team_target_share_single_player(self, wr_player_games_factory):
        """Solo WR on team should have target share close to 1.0."""
        df = wr_player_games_factory(n_weeks=5, targets=8, recent_team="KC")
        _compute_wr_features(df)
        later = df[df["week"] >= 3]
        shares = later["team_wr_target_share_L3"].dropna()
        if len(shares) > 0:
            assert (shares <= 1.01).all()

    def test_team_target_share_two_players(self, wr_player_games_factory):
        """Two WRs should roughly split target share."""
        p1 = wr_player_games_factory("W1", n_weeks=5, targets=8, recent_team="KC")
        p2 = wr_player_games_factory("W2", n_weeks=5, targets=8, recent_team="KC")
        df = pd.concat([p1, p2], ignore_index=True)
        _compute_wr_features(df)
        later = df[(df["week"] >= 3) & (df["player_id"] == "W1")]
        shares = later["team_wr_target_share_L3"].dropna()
        if len(shares) > 0:
            assert all(0.4 <= s <= 0.6 for s in shares)

    def test_multiple_seasons_independent(self, wr_player_games_factory):
        s1 = wr_player_games_factory(season=2022, n_weeks=3, targets=12, receiving_yards=120)
        s2 = wr_player_games_factory(season=2023, n_weeks=3, targets=4, receiving_yards=30)
        df = pd.concat([s1, s2], ignore_index=True)
        _compute_wr_features(df)
        w1_2023 = df[(df["season"] == 2023) & (df["week"] == 1)]
        ypt = w1_2023["yards_per_target_L3"].iloc[0]
        assert ypt == 0.0 or np.isnan(ypt)


# ---------------------------------------------------------------------------
# fill_wr_nans
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFillWRNans:
    def test_fills_nan_with_train_mean(self, wr_nan_splits_factory):
        train, val, test = wr_nan_splits_factory([1.0, 2.0, 3.0], [np.nan], [np.nan])
        train, val, test = fill_wr_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0
        assert pytest.approx(test["feat1"].iloc[0]) == 2.0

    def test_replaces_inf_with_train_mean(self, wr_nan_splits_factory):
        train, val, test = wr_nan_splits_factory([1.0, 3.0], [np.inf], [-np.inf])
        train, val, test = fill_wr_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0
        assert pytest.approx(test["feat1"].iloc[0]) == 2.0

    def test_train_inf_replaced_before_mean(self, wr_nan_splits_factory):
        train, val, test = wr_nan_splits_factory([1.0, np.inf, 3.0], [np.nan], [np.nan])
        train, val, test = fill_wr_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0

    def test_no_nans_unchanged(self, wr_nan_splits_factory):
        train, val, test = wr_nan_splits_factory([1.0, 2.0], [3.0], [4.0])
        train, val, test = fill_wr_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 3.0
        assert pytest.approx(test["feat1"].iloc[0]) == 4.0

    def test_multiple_columns(self):
        train = pd.DataFrame({"f1": [1.0, 3.0], "f2": [10.0, 20.0]})
        val = pd.DataFrame({"f1": [np.nan], "f2": [np.nan]})
        test = pd.DataFrame({"f1": [5.0], "f2": [np.nan]})
        train, val, test = fill_wr_nans(train, val, test, ["f1", "f2"])
        assert pytest.approx(val["f1"].iloc[0]) == 2.0
        assert pytest.approx(val["f2"].iloc[0]) == 15.0
        assert pytest.approx(test["f1"].iloc[0]) == 5.0
        assert pytest.approx(test["f2"].iloc[0]) == 15.0
