"""Tests for shared.weather_features — get_weather_feature_columns and merge_schedule_features."""

from unittest.mock import patch

import numpy as np
import pytest

from src.shared.weather_features import (
    WEATHER_DROPS_BY_POSITION,
    WEATHER_FEATURES_ALL,
    get_weather_feature_columns,
    merge_schedule_features,
)


@pytest.fixture(autouse=True)
def _clear_schedule_cache():
    """Reset the module-level schedule cache between tests."""
    import src.shared.weather_features as wf

    wf._schedule_cache = None
    yield
    wf._schedule_cache = None


# ---------------------------------------------------------------------------
# get_weather_feature_columns
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetWeatherFeatureColumns:
    def test_qb_drops_is_grass(self):
        result = get_weather_feature_columns("QB", [])
        assert "is_grass" not in result
        assert "is_dome" in result

    def test_rb_drops_five_features(self):
        result = get_weather_feature_columns("RB", [])
        for dropped in WEATHER_DROPS_BY_POSITION["RB"]:
            assert dropped not in result
        # Remaining weather features should be present
        remaining = set(WEATHER_FEATURES_ALL) - WEATHER_DROPS_BY_POSITION["RB"]
        for col in remaining:
            assert col in result

    def test_te_drops(self):
        result = get_weather_feature_columns("TE", [])
        for dropped in WEATHER_DROPS_BY_POSITION["TE"]:
            assert dropped not in result

    def test_no_duplicates_with_base_cols(self):
        base = ["total_line", "feat_a"]
        result = get_weather_feature_columns("QB", base)
        assert result.count("total_line") == 1

    def test_unknown_position_no_drops(self):
        result = get_weather_feature_columns("FLEX", [])
        assert set(result) == set(WEATHER_FEATURES_ALL)


# ---------------------------------------------------------------------------
# merge_schedule_features
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMergeScheduleFeatures:
    @patch("src.shared.weather_features._load_schedules")
    def test_adds_weather_columns(self, mock_load, fake_schedules, player_df_factory):
        mock_load.return_value = fake_schedules
        df = player_df_factory("KC", n_weeks=4)
        result = merge_schedule_features(df)
        for col in WEATHER_FEATURES_ALL:
            assert col in result.columns, f"Missing column: {col}"

    @patch("src.shared.weather_features._load_schedules")
    def test_idempotent(self, mock_load, fake_schedules, player_df_factory):
        mock_load.return_value = fake_schedules
        df = player_df_factory("KC", n_weeks=4)
        result1 = merge_schedule_features(df)
        vals_before = result1["implied_total_x_wind"].values.copy()
        result2 = merge_schedule_features(result1)
        np.testing.assert_array_equal(result2["implied_total_x_wind"].values, vals_before)

    @patch("src.shared.weather_features._load_schedules")
    def test_dome_flags(self, mock_load, fake_schedules, player_df_factory):
        mock_load.return_value = fake_schedules
        df = player_df_factory("NO", n_weeks=2)  # NO plays in dome
        result = merge_schedule_features(df)
        assert (result["is_dome"] == 1).all()
        assert (result["temp_adjusted"] == 65.0).all()
        assert (result["wind_adjusted"] == 0.0).all()

    @patch("src.shared.weather_features._load_schedules")
    def test_implied_totals_math(self, mock_load, fake_schedules, player_df_factory):
        mock_load.return_value = fake_schedules
        # KC is home team with spread_line=-3.0, total_line=47.0
        df = player_df_factory("KC", n_weeks=1)
        result = merge_schedule_features(df)
        # implied_team_total = (47 - (-3)) / 2 = 25.0
        assert pytest.approx(result["implied_team_total"].iloc[0], abs=0.1) == 25.0
        # implied_opp_total = 47 - 25 = 22.0
        assert pytest.approx(result["implied_opp_total"].iloc[0], abs=0.1) == 22.0

    @patch("src.shared.weather_features._load_schedules")
    def test_unmatched_team_keeps_nan(self, mock_load, fake_schedules, player_df_factory):
        mock_load.return_value = fake_schedules
        df = player_df_factory("XYZ", n_weeks=2)  # team not in schedule
        result = merge_schedule_features(df)
        # Unmatched games should have NaN Vegas features so the error surfaces
        assert result["implied_team_total"].isna().all()
        assert result["total_line"].isna().all()
        # Interaction feature must also propagate NaN — silently filling with 0
        # hides the unmatched-game failure from downstream consumers.
        assert result["implied_opp_total"].isna().all()
        assert result["implied_total_x_wind"].isna().all()
