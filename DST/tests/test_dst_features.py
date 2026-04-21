"""Tests for DST.dst_features — compute_dst_features and fill_dst_nans.

D/ST features are computed on the FULL team-week dataset before splitting
(like Kicker features). Groups by team+season (resets per-season, unlike K).

compute_dst_features requires pre-computed raw-stat columns (def_sacks,
def_ints, def_fumble_rec, def_fumbles_forced, def_blocked_kicks,
special_teams_tds, points_allowed, yards_allowed) plus ``fantasy_points``.
The ``make_team_games`` fixture (conftest.py) emits all of them.
"""

import numpy as np
import pandas as pd
import pytest

from DST.dst_features import compute_dst_features, fill_dst_nans

DST_FEATURE_COLS = [
    "sacks_L3",
    "sacks_L5",
    "ints_L3",
    "fumble_rec_L3",
    "forced_fumbles_L3",
    "blocked_kicks_L5",
    "pts_allowed_L3",
    "pts_allowed_L5",
    "yards_allowed_L3",
    "yards_allowed_L5",
    "yards_allowed_ewma",
    "dst_pts_L3",
    "dst_pts_L5",
    "dst_pts_L8",
    "pts_allowed_ewma",
    "dst_pts_ewma",
    "sack_trend",
    "turnover_trend",
    "pts_allowed_trend",
    "pts_allowed_std_L3",
    "dst_scoring_std_L3",
]


# ---------------------------------------------------------------------------
# compute_dst_features
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeDSTFeatures:
    def test_all_rolling_features_created(self, make_team_games):
        df = make_team_games()
        compute_dst_features(df)
        for col in DST_FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"

    def test_prior_season_features_created(self, make_team_games):
        """Prior-season features require >=2 seasons of data."""
        s1 = make_team_games(season=2022, n_weeks=5, def_sacks=3, points_allowed=18)
        s2 = make_team_games(season=2023, n_weeks=5, def_sacks=4, points_allowed=20)
        df = pd.concat([s1, s2], ignore_index=True)
        compute_dst_features(df)
        assert "prior_season_dst_pts_avg" in df.columns
        assert "prior_season_pts_allowed_avg" in df.columns
        # 2023 rows should have non-NaN prior-season values
        s2_rows = df[df["season"] == 2023]
        assert not s2_rows["prior_season_pts_allowed_avg"].isna().any()

    def test_first_week_features_are_zero(self, make_team_games):
        """Week 1 has no prior data — features should be 0 (filled)."""
        df = make_team_games(n_weeks=5)
        compute_dst_features(df)
        week1 = df[df["week"] == 1]
        for col in ["sacks_L3", "ints_L3", "pts_allowed_L3", "dst_pts_L3"]:
            val = week1[col].iloc[0]
            assert val == 0.0, f"{col} = {val} for week 1"

    def test_sacks_L3_computation(self, make_team_games):
        df = make_team_games(n_weeks=5, def_sacks=3)
        compute_dst_features(df)
        week3 = df[df["week"] == 3]
        # rolling mean of [3, 3] = 3
        assert pytest.approx(week3["sacks_L3"].iloc[0]) == 3.0

    def test_ints_L3_computation(self, make_team_games):
        df = make_team_games(n_weeks=5, def_ints=2)
        compute_dst_features(df)
        week3 = df[df["week"] == 3]
        assert pytest.approx(week3["ints_L3"].iloc[0]) == 2.0

    def test_fumble_rec_L3_computation(self, make_team_games):
        df = make_team_games(n_weeks=5, def_fumble_rec=1)
        compute_dst_features(df)
        week3 = df[df["week"] == 3]
        assert pytest.approx(week3["fumble_rec_L3"].iloc[0]) == 1.0

    def test_pts_allowed_L3(self, make_team_games):
        df = make_team_games(n_weeks=5, points_allowed=14)
        compute_dst_features(df)
        week3 = df[df["week"] == 3]
        assert pytest.approx(week3["pts_allowed_L3"].iloc[0]) == 14.0

    def test_trend_features_present(self, make_team_games):
        """sack_trend, turnover_trend, pts_allowed_trend should be computed."""
        df = make_team_games(n_weeks=8)
        compute_dst_features(df)
        for col in ["sack_trend", "turnover_trend", "pts_allowed_trend"]:
            assert col in df.columns

    def test_ewma_features_present(self, make_team_games):
        df = make_team_games(n_weeks=8)
        compute_dst_features(df)
        assert "pts_allowed_ewma" in df.columns
        assert "dst_pts_ewma" in df.columns

    def test_std_features_present(self, make_team_games):
        df = make_team_games(n_weeks=8)
        compute_dst_features(df)
        assert "pts_allowed_std_L3" in df.columns
        assert "dst_scoring_std_L3" in df.columns

    def test_multiple_teams_independent(self, make_team_games):
        """Each team's rolling window should be independent."""
        t1 = make_team_games("KC", n_weeks=5, def_sacks=4)
        t2 = make_team_games("SF", n_weeks=5, def_sacks=2)
        df = pd.concat([t1, t2], ignore_index=True)
        compute_dst_features(df)
        kc_w3 = df[(df["team"] == "KC") & (df["week"] == 3)]
        sf_w3 = df[(df["team"] == "SF") & (df["week"] == 3)]
        assert pytest.approx(kc_w3["sacks_L3"].iloc[0]) == 4.0
        assert pytest.approx(sf_w3["sacks_L3"].iloc[0]) == 2.0

    def test_per_season_grouping(self, make_team_games):
        """D/ST features SHOULD reset per-season (team+season groupby)."""
        s1 = make_team_games(season=2022, n_weeks=3, def_sacks=3)
        s2 = make_team_games(season=2023, n_weeks=3, def_sacks=1)
        df = pd.concat([s1, s2], ignore_index=True)
        compute_dst_features(df)
        # 2023 week 1 should NOT see 2022 data — sacks_L3 should be 0
        w1_2023 = df[(df["season"] == 2023) & (df["week"] == 1)]
        assert w1_2023["sacks_L3"].iloc[0] == 0.0

    def test_temp_columns_cleaned_up(self, make_team_games):
        """Temporary columns (_dst_total_pts, _turnovers) should be dropped."""
        df = make_team_games()
        compute_dst_features(df)
        temp_cols = [c for c in df.columns if c.startswith("_")]
        assert temp_cols == [], f"Temp columns left: {temp_cols}"


# ---------------------------------------------------------------------------
# fill_dst_nans
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFillDSTNans:
    def test_fills_nan_with_train_mean(self, make_splits):
        train, val, test = make_splits([1.0, 2.0, 3.0], [np.nan], [np.nan])
        train, val, test = fill_dst_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0
        assert pytest.approx(test["feat1"].iloc[0]) == 2.0

    def test_replaces_inf_with_train_mean(self, make_splits):
        train, val, test = make_splits([1.0, 3.0], [np.inf], [-np.inf])
        train, val, test = fill_dst_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 2.0
        assert pytest.approx(test["feat1"].iloc[0]) == 2.0

    def test_no_nans_unchanged(self, make_splits):
        train, val, test = make_splits([1.0, 2.0], [3.0], [4.0])
        train, val, test = fill_dst_nans(train, val, test, ["feat1"])
        assert pytest.approx(val["feat1"].iloc[0]) == 3.0
        assert pytest.approx(test["feat1"].iloc[0]) == 4.0

    def test_multiple_columns(self):
        train = pd.DataFrame({"f1": [1.0, 3.0], "f2": [10.0, 20.0]})
        val = pd.DataFrame({"f1": [np.nan], "f2": [np.nan]})
        test = pd.DataFrame({"f1": [5.0], "f2": [np.nan]})
        train, val, test = fill_dst_nans(train, val, test, ["f1", "f2"])
        assert pytest.approx(val["f1"].iloc[0]) == 2.0
        assert pytest.approx(val["f2"].iloc[0]) == 15.0
        assert pytest.approx(test["f1"].iloc[0]) == 5.0
        assert pytest.approx(test["f2"].iloc[0]) == 15.0
