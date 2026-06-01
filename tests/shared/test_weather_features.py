"""Tests for src.shared.weather_features.merge_schedule_features."""

from unittest.mock import patch

import numpy as np
import pytest

from src.shared.weather_features import (
    WEATHER_FEATURES_ALL,
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

    @patch("src.shared.weather_features._load_schedules")
    def test_spread_line_and_div_game_survive_when_present_on_input(self, mock_load):
        """Regression test for H2.

        DST materializes ``spread_line`` and ``div_game`` on every row from the
        schedule (src/dst/data.py) before calling ``build_position_features``.
        Before the H2 fix, those bare-name columns collided with the merge's
        lookup columns; pandas suffix-renamed to ``_x``/``_y``; the merge-back
        loop only read bare names so it silently skipped them; and the
        cleanup loop unconditionally dropped the bare names. End state: both
        columns were missing on output, then zeroed by the catch-all backfill
        in ``build_position_features``. DST trained and served with
        ``spread_line == 0`` and ``div_game == 0`` for every row.
        """
        # Schedule with VARYING spread_line and div_game so we can assert
        # std() > 0 and div_game has both 0 and 1.
        import pandas as pd

        rows = []
        for week in range(1, 5):
            rows.append(
                {
                    "game_type": "REG",
                    "season": 2023,
                    "week": week,
                    "home_team": "KC",
                    "away_team": "SF",
                    "home_score": 24,
                    "away_score": 17,
                    "spread_line": -3.0 - week,  # -4, -5, -6, -7
                    "total_line": 47.0,
                    "roof": "outdoors",
                    "surface": "grass",
                    "temp": 72,
                    "wind": 8,
                    "home_rest": 7,
                    "away_rest": 7,
                    "div_game": week % 2,  # 1, 0, 1, 0
                }
            )
        mock_load.return_value = pd.DataFrame(rows)

        # DST-shape input: spread_line and div_game already present on df
        # (mirrors src/dst/data.py:340-360 materialization).
        df = pd.DataFrame(
            {
                "player_id": ["KC_DST"] * 4,
                "season": [2023] * 4,
                "week": list(range(1, 5)),
                "recent_team": ["KC"] * 4,
                "spread_line": [0.0] * 4,  # stale placeholder — must be overwritten
                "div_game": [0] * 4,  # stale placeholder — must be overwritten
            }
        )

        result = merge_schedule_features(df)

        assert "spread_line" in result.columns, "spread_line dropped (H2 regression)"
        assert "div_game" in result.columns, "div_game dropped (H2 regression)"

        # spread_line carries the merged varying values, not the placeholder 0.0
        assert result["spread_line"].std() > 0, (
            "spread_line is constant — merge did not overwrite the placeholder, "
            "or the column was zeroed by the cleanup-drop loop (H2 regression)"
        )
        # div_game contains both 0 and 1 from the schedule
        unique_div = set(int(v) for v in result["div_game"].unique())
        assert {0, 1}.issubset(unique_div), (
            f"div_game.unique() = {unique_div} missing 0 or 1 (H2 regression)"
        )
