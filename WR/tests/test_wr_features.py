"""Tests for WR.wr_features — _compute_wr_features and fill_wr_nans."""

import numpy as np
import pandas as pd
import pytest

from WR.wr_features import _compute_wr_features, fill_wr_nans


def _make_player_games(
    player_id="W1",
    season=2023,
    n_weeks=5,
    receptions=5,
    targets=8,
    receiving_yards=70,
    receiving_air_yards=100,
    receiving_yards_after_catch=30,
    receiving_epa=2.0,
    receiving_first_downs=3,
    recent_team="KC",
):
    """Create a multi-week DataFrame for one WR."""
    return pd.DataFrame({
        "player_id": [player_id] * n_weeks,
        "season": [season] * n_weeks,
        "week": list(range(1, n_weeks + 1)),
        "receptions": [receptions] * n_weeks,
        "targets": [targets] * n_weeks,
        "receiving_yards": [receiving_yards] * n_weeks,
        "receiving_air_yards": [receiving_air_yards] * n_weeks,
        "receiving_yards_after_catch": [receiving_yards_after_catch] * n_weeks,
        "receiving_epa": [receiving_epa] * n_weeks,
        "receiving_first_downs": [receiving_first_downs] * n_weeks,
        "recent_team": [recent_team] * n_weeks,
    })


WR_FEATURE_COLS = [
    "yards_per_reception_L3",
    "yards_per_target_L3",
    "reception_rate_L3",
    "air_yards_per_target_L3",
    "yac_per_reception_L3",
    "team_wr_target_share_L3",
    "receiving_epa_per_target_L3",
    "receiving_first_down_rate_L3",
]


# ---------------------------------------------------------------------------
# _compute_wr_features
# ---------------------------------------------------------------------------

class TestComputeWRFeatures:
    def test_all_features_created(self):
        df = _make_player_games()
        _compute_wr_features(df)
        for col in WR_FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"

    def test_first_week_features_are_zero(self):
        df = _make_player_games(n_weeks=4)
        _compute_wr_features(df)
        week1 = df[df["week"] == 1]
        for col in ["yards_per_reception_L3", "reception_rate_L3"]:
            val = week1[col].iloc[0]
            assert val == 0.0 or np.isnan(val), f"{col} = {val} for week 1"

    def test_yards_per_reception_computation(self):
        df = _make_player_games(n_weeks=5, receptions=5, receiving_yards=70)
        _compute_wr_features(df)
        # Week 3: rolling sum of [70, 70] = 140 yds / 10 receptions = 14.0
        week3 = df[df["week"] == 3]
        assert pytest.approx(week3["yards_per_reception_L3"].iloc[0], abs=0.01) == 14.0

    def test_reception_rate_computation(self):
        df = _make_player_games(n_weeks=5, receptions=5, targets=8)
        _compute_wr_features(df)
        week3 = df[df["week"] == 3]
        # 10 / 16 = 0.625
        assert pytest.approx(week3["reception_rate_L3"].iloc[0], abs=0.01) == 5 / 8

    def test_yards_per_target_computation(self):
        df = _make_player_games(n_weeks=5, targets=8, receiving_yards=80)
        _compute_wr_features(df)
        week3 = df[df["week"] == 3]
        # 160 / 16 = 10.0
        assert pytest.approx(week3["yards_per_target_L3"].iloc[0], abs=0.01) == 10.0

    def test_air_yards_per_target_computation(self):
        df = _make_player_games(n_weeks=5, targets=8, receiving_air_yards=100)
        _compute_wr_features(df)
        week3 = df[df["week"] == 3]
        # 200 / 16 = 12.5
        assert pytest.approx(week3["air_yards_per_target_L3"].iloc[0], abs=0.01) == 12.5

    def test_zero_targets_no_division_error(self):
        df = _make_player_games(n_weeks=4, targets=0, receptions=0, receiving_yards=0,
                                receiving_air_yards=0, receiving_yards_after_catch=0,
                                receiving_epa=0, receiving_first_downs=0)
        _compute_wr_features(df)
        for col in ["yards_per_target_L3", "reception_rate_L3",
                    "air_yards_per_target_L3", "receiving_epa_per_target_L3"]:
            assert not df[col].isin([np.inf, -np.inf]).any(), f"{col} has inf"

    def test_zero_receptions_yac(self):
        df = _make_player_games(n_weeks=4, receptions=0, receiving_yards_after_catch=0,
                                receiving_first_downs=0, receiving_yards=0)
        _compute_wr_features(df)
        assert not df["yac_per_reception_L3"].isin([np.inf, -np.inf]).any()
        assert not df["receiving_first_down_rate_L3"].isin([np.inf, -np.inf]).any()

    def test_team_target_share_single_player(self):
        """Solo WR on team should have target share close to 1.0."""
        df = _make_player_games(n_weeks=5, targets=8, recent_team="KC")
        _compute_wr_features(df)
        later = df[df["week"] >= 3]
        shares = later["team_wr_target_share_L3"].dropna()
        if len(shares) > 0:
            assert (shares <= 1.01).all()

    def test_team_target_share_two_players(self):
        """Two WRs should roughly split target share."""
        p1 = _make_player_games("W1", n_weeks=5, targets=8, recent_team="KC")
        p2 = _make_player_games("W2", n_weeks=5, targets=8, recent_team="KC")
        df = pd.concat([p1, p2], ignore_index=True)
        _compute_wr_features(df)
        later = df[(df["week"] >= 3) & (df["player_id"] == "W1")]
        shares = later["team_wr_target_share_L3"].dropna()
        if len(shares) > 0:
            assert all(0.4 <= s <= 0.6 for s in shares)

    def test_multiple_seasons_independent(self):
        s1 = _make_player_games(season=2022, n_weeks=3, targets=12, receiving_yards=120)
        s2 = _make_player_games(season=2023, n_weeks=3, targets=4, receiving_yards=30)
        df = pd.concat([s1, s2], ignore_index=True)
        _compute_wr_features(df)
        w1_2023 = df[(df["season"] == 2023) & (df["week"] == 1)]
        ypt = w1_2023["yards_per_target_L3"].iloc[0]
        assert ypt == 0.0 or np.isnan(ypt)


# ---------------------------------------------------------------------------
# fill_wr_nans
# ---------------------------------------------------------------------------

class TestFillWRNans:
    def _make_splits(self, train_vals, val_vals, test_vals, col="feat1"):
        train = pd.DataFrame({col: train_vals})
        val = pd.DataFrame({col: val_vals})
        test = pd.DataFrame({col: test_vals})
        return train, val, test

    def test_fills_nan_with_train_mean(self):
        train, val, test = self._make_splits([1.0, 2.0, 3.0], [np.nan], [np.nan])
        train, val, test = fill_wr_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0
        assert pytest.approx(test["feat1"].iloc[0]) == 2.0

    def test_replaces_inf_with_train_mean(self):
        train, val, test = self._make_splits([1.0, 3.0], [np.inf], [-np.inf])
        train, val, test = fill_wr_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0
        assert pytest.approx(test["feat1"].iloc[0]) == 2.0

    def test_train_inf_replaced_before_mean(self):
        train, val, test = self._make_splits([1.0, np.inf, 3.0], [np.nan], [np.nan])
        train, val, test = fill_wr_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0

    def test_no_nans_unchanged(self):
        train, val, test = self._make_splits([1.0, 2.0], [3.0], [4.0])
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
