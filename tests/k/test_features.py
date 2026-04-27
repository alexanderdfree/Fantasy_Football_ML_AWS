"""Tests for src.k.features — compute_features and fill_nans.

Kicker features are computed on the full dataset before splitting, and use
CROSS-SEASON rolling windows (all other positions reset per-season) because
kickers have stable multi-year careers and small sample sizes per season.

compute_features requires a pre-computed `fantasy_points` column (added by
compute_targets via the signed 4-head aggregation).
"""

import numpy as np
import pandas as pd
import pytest

from src.k.features import compute_features, fill_nans

FEATURE_COLS = [
    "fg_attempts_L3",
    "fg_accuracy_L5",
    "pat_volume_L3",
    "total_k_pts_L3",
    "long_fg_rate_L3",
    "k_pts_trend",
    "k_pts_std_L3",
    "avg_fg_distance_L3",
    "avg_fg_prob_L3",
    "fg_pct_40plus_L5",
    "q4_fg_rate_L5",
    "xp_accuracy_L5",
]


# ---------------------------------------------------------------------------
# compute_features
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeKFeatures:
    def test_all_features_created(self, make_kicker_games):
        df = make_kicker_games()
        compute_features(df)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"

    def test_first_week_features_are_zero(self, make_kicker_games):
        """Week 1 has no prior data — features should be 0 (filled)."""
        df = make_kicker_games(n_weeks=5)
        compute_features(df)
        week1 = df[df["week"] == 1]
        for col in ["fg_attempts_L3", "pat_volume_L3", "total_k_pts_L3"]:
            val = week1[col].iloc[0]
            assert val == 0.0, f"{col} = {val} for week 1"

    def test_fg_attempts_L3_computation(self, make_kicker_games):
        df = make_kicker_games(n_weeks=5, fg_att=3)
        compute_features(df)
        # Week 3: rolling mean of [3, 3] = 3
        week3 = df[df["week"] == 3]
        assert pytest.approx(week3["fg_attempts_L3"].iloc[0]) == 3.0

    def test_fg_accuracy_L5_computation(self, make_kicker_games):
        df = make_kicker_games(n_weeks=6, fg_att=4, fg_made=3)
        compute_features(df)
        # Week 3: rolling sum of [3, 3] made / [4, 4] att = 0.75
        week3 = df[df["week"] == 3]
        assert pytest.approx(week3["fg_accuracy_L5"].iloc[0], abs=0.01) == 0.75

    def test_pat_volume_L3(self, make_kicker_games):
        df = make_kicker_games(n_weeks=5, pat_att=3)
        compute_features(df)
        week3 = df[df["week"] == 3]
        assert pytest.approx(week3["pat_volume_L3"].iloc[0]) == 3.0

    def test_xp_accuracy_L5_computation(self, make_kicker_games):
        df = make_kicker_games(n_weeks=6, pat_att=4, pat_made=3)
        compute_features(df)
        week3 = df[df["week"] == 3]
        # rolling sum [3,3] made / [4,4] att = 0.75
        assert pytest.approx(week3["xp_accuracy_L5"].iloc[0], abs=0.01) == 0.75

    def test_zero_attempts_no_division_error(self, make_kicker_games):
        """Kicker with 0 attempts: rate features should be 0, not inf."""
        df = make_kicker_games(
            n_weeks=4,
            fg_att=0,
            fg_made=0,
            pat_att=0,
            pat_made=0,
            fg_made_40_49=0,
            fg_made_50_59=0,
            fg_made_60_=0,
            long_fg_att=0,
            long_fg_made=0,
            q4_fg_att=0,
            q4_fg_made=0,
        )
        compute_features(df)
        for col in [
            "fg_accuracy_L5",
            "long_fg_rate_L3",
            "fg_pct_40plus_L5",
            "q4_fg_rate_L5",
            "xp_accuracy_L5",
        ]:
            assert not df[col].isin([np.inf, -np.inf]).any(), f"{col} has inf"

    def test_cross_season_rolling(self, make_kicker_games):
        """Kicker features use cross-season rolling (no reset).

        Unlike other positions which reset per-season.
        """
        s1 = make_kicker_games(season=2022, n_weeks=3, fg_att=3)
        s2 = make_kicker_games(season=2023, n_weeks=3, fg_att=3)
        df = pd.concat([s1, s2], ignore_index=True)
        compute_features(df)
        # 2023 week 1 should SEE 2022 data (cross-season) — non-zero
        w1_2023 = df[(df["season"] == 2023) & (df["week"] == 1)]
        fg_att = w1_2023["fg_attempts_L3"].iloc[0]
        assert fg_att > 0, "Cross-season rolling should carry 2022 data into 2023"

    def test_multiple_kickers_independent(self, make_kicker_games):
        """Each kicker's history is independent."""
        k1 = make_kicker_games("K1", n_weeks=5, fg_att=4)
        k2 = make_kicker_games("K2", n_weeks=5, fg_att=2)
        df = pd.concat([k1, k2], ignore_index=True)
        compute_features(df)
        # K1 week 3 should have ~4 attempts (K1's history only)
        k1_w3 = df[(df["player_id"] == "K1") & (df["week"] == 3)]
        k2_w3 = df[(df["player_id"] == "K2") & (df["week"] == 3)]
        assert pytest.approx(k1_w3["fg_attempts_L3"].iloc[0]) == 4.0
        assert pytest.approx(k2_w3["fg_attempts_L3"].iloc[0]) == 2.0

    def test_temp_columns_cleaned_up(self, make_kicker_games):
        """Temporary computation columns (_k_total_pts, _long_fg_att) should be dropped."""
        df = make_kicker_games()
        compute_features(df)
        temp_cols = [c for c in df.columns if c.startswith("_")]
        assert temp_cols == [], f"Temp columns left: {temp_cols}"


# ---------------------------------------------------------------------------
# fill_nans
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFillKNans:
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
