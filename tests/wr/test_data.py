"""Tests for WR.wr_data — filter_to_wr and compute_team_wr_totals."""

import pandas as pd
import pytest

from src.wr.data import compute_team_wr_totals, filter_to_wr

# ---------------------------------------------------------------------------
# filter_to_wr
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFilterToWR:
    def test_filters_only_wr_rows(self, wr_position_df_factory):
        df = wr_position_df_factory(["QB", "RB", "WR", "WR", "TE"])
        result = filter_to_wr(df)
        assert len(result) == 2
        assert (result["position"] == "WR").all()

    def test_drops_position_encoding_columns(self, wr_position_df_factory):
        df = wr_position_df_factory(["WR", "WR"])
        result = filter_to_wr(df)
        for col in ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]:
            assert col not in result.columns

    def test_keeps_non_position_columns(self, wr_position_df_factory):
        df = wr_position_df_factory(["WR"])
        result = filter_to_wr(df)
        assert "receiving_yards" in result.columns

    def test_no_position_encoding_columns(self, wr_position_df_factory):
        df = wr_position_df_factory(["WR", "QB"], has_pos_cols=False)
        result = filter_to_wr(df)
        assert len(result) == 1

    def test_empty_result_when_no_wrs(self, wr_position_df_factory):
        df = wr_position_df_factory(["QB", "RB", "TE"])
        result = filter_to_wr(df)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_empty_input(self):
        df = pd.DataFrame({"position": pd.Series(dtype=str)})
        result = filter_to_wr(df)
        assert len(result) == 0

    def test_does_not_mutate_original(self, wr_position_df_factory):
        df = wr_position_df_factory(["WR", "QB"])
        original_cols = list(df.columns)
        _ = filter_to_wr(df)
        assert list(df.columns) == original_cols
        assert len(df) == 2


# ---------------------------------------------------------------------------
# compute_team_wr_totals
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeTeamWRTotals:
    def test_basic_aggregation(self):
        df = pd.DataFrame(
            {
                "recent_team": ["KC", "KC", "KC", "BUF"],
                "season": [2023, 2023, 2023, 2023],
                "week": [1, 1, 1, 1],
                "targets": [8, 10, 3, 7],
            }
        )
        result = compute_team_wr_totals(df)
        kc_row = result[result["recent_team"] == "KC"]
        assert kc_row["team_wr_targets"].values[0] == 21

    def test_multiple_weeks(self):
        df = pd.DataFrame(
            {
                "recent_team": ["KC", "KC"],
                "season": [2023, 2023],
                "week": [1, 2],
                "targets": [8, 10],
            }
        )
        result = compute_team_wr_totals(df)
        assert len(result) == 2

    def test_single_player_team(self):
        df = pd.DataFrame(
            {
                "recent_team": ["NYG"],
                "season": [2023],
                "week": [5],
                "targets": [12],
            }
        )
        result = compute_team_wr_totals(df)
        assert result["team_wr_targets"].values[0] == 12

    def test_zero_targets(self):
        df = pd.DataFrame(
            {
                "recent_team": ["LAR", "LAR"],
                "season": [2023, 2023],
                "week": [1, 1],
                "targets": [0, 0],
            }
        )
        result = compute_team_wr_totals(df)
        assert result["team_wr_targets"].values[0] == 0

    def test_output_columns(self):
        df = pd.DataFrame(
            {
                "recent_team": ["SF"],
                "season": [2023],
                "week": [1],
                "targets": [8],
            }
        )
        result = compute_team_wr_totals(df)
        assert set(result.columns) == {"recent_team", "season", "week", "team_wr_targets"}
