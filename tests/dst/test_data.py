"""Tests for src.dst.data — filter_to_position (identity filter).

D/ST data is pre-built at team-level (via build_data), so filter_to_position
is an identity filter that returns a copy of the input.
"""

import pandas as pd
import pytest

from src.dst.data import filter_to_position


@pytest.mark.unit
class TestFilterToDST:
    """D/ST data is team-level and pre-built; filter_to_position is identity (copy)."""

    def test_returns_copy_of_input(self):
        df = pd.DataFrame(
            {
                "team": ["KC", "SF"],
                "season": [2023, 2023],
                "week": [1, 1],
                "def_sacks": [3, 4],
                "def_ints": [1, 2],
            }
        )
        result = filter_to_position(df)
        pd.testing.assert_frame_equal(result, df)

    def test_does_not_mutate_original(self):
        df = pd.DataFrame(
            {
                "team": ["KC"],
                "def_sacks": [3],
            }
        )
        original_cols = list(df.columns)
        original_len = len(df)
        _ = filter_to_position(df)
        assert list(df.columns) == original_cols
        assert len(df) == original_len

    def test_returned_is_independent_copy(self):
        """Mutating the result should not affect the input."""
        df = pd.DataFrame({"team": ["KC"], "def_sacks": [3]})
        result = filter_to_position(df)
        result["def_sacks"] = 99
        assert df["def_sacks"].iloc[0] == 3

    def test_empty_input(self):
        df = pd.DataFrame({"team": pd.Series(dtype=str), "def_sacks": pd.Series(dtype=int)})
        result = filter_to_position(df)
        assert len(result) == 0

    def test_preserves_all_columns(self, make_df):
        """Identity filter must not drop columns — DST has many team-level fields."""
        df = make_df(
            season=2023,
            week=1,
            opponent_team="LV",
            is_home=1,
            is_dome=0,
            div_game=1,
            rest_days=7,
            opp_qb_epa_L5=0.1,
            team="KC",
        )
        result = filter_to_position(df)
        assert set(result.columns) == set(df.columns)

    def test_multiple_teams_preserved(self):
        """Should preserve all 32 teams without filtering."""
        teams = ["KC", "SF", "BUF", "DAL", "PHI", "BAL", "MIA", "CIN"]
        df = pd.DataFrame(
            {
                "team": teams,
                "def_sacks": range(len(teams)),
            }
        )
        result = filter_to_position(df)
        assert len(result) == len(teams)
        assert set(result["team"]) == set(teams)
