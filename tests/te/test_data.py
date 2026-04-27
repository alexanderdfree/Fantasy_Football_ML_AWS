"""Tests for TE.te_data — filter_to_te and compute_team_te_totals."""

import pandas as pd
import pytest

from src.te.data import compute_team_te_totals, filter_to_te

# ---------------------------------------------------------------------------
# filter_to_te
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFilterToTE:
    def test_filters_only_te_rows(self, te_position_df_factory):
        df = te_position_df_factory(["QB", "RB", "WR", "TE", "TE"])
        result = filter_to_te(df)
        assert len(result) == 2
        assert (result["position"] == "TE").all()

    def test_drops_position_encoding_columns(self, te_position_df_factory):
        df = te_position_df_factory(["TE", "TE"])
        result = filter_to_te(df)
        for col in ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]:
            assert col not in result.columns

    def test_keeps_non_position_columns(self, te_position_df_factory):
        df = te_position_df_factory(["TE"])
        result = filter_to_te(df)
        assert "receiving_yards" in result.columns

    def test_no_position_encoding_columns(self, te_position_df_factory):
        df = te_position_df_factory(["TE", "QB"], has_pos_cols=False)
        result = filter_to_te(df)
        assert len(result) == 1

    def test_empty_result_when_no_tes(self, te_position_df_factory):
        df = te_position_df_factory(["QB", "RB", "WR"])
        result = filter_to_te(df)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_empty_input(self):
        df = pd.DataFrame({"position": pd.Series(dtype=str)})
        result = filter_to_te(df)
        assert len(result) == 0

    def test_does_not_mutate_original(self, te_position_df_factory):
        df = te_position_df_factory(["TE", "QB"])
        original_cols = list(df.columns)
        _ = filter_to_te(df)
        assert list(df.columns) == original_cols
        assert len(df) == 2


# ---------------------------------------------------------------------------
# compute_team_te_totals
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeTeamTETotals:
    def test_basic_aggregation(self):
        df = pd.DataFrame(
            {
                "recent_team": ["KC", "KC", "BUF"],
                "season": [2023, 2023, 2023],
                "week": [1, 1, 1],
                "targets": [8, 3, 5],
            }
        )
        result = compute_team_te_totals(df)
        kc_row = result[result["recent_team"] == "KC"]
        assert kc_row["team_te_targets"].values[0] == 11

    def test_multiple_weeks(self):
        df = pd.DataFrame(
            {
                "recent_team": ["KC", "KC"],
                "season": [2023, 2023],
                "week": [1, 2],
                "targets": [8, 10],
            }
        )
        result = compute_team_te_totals(df)
        assert len(result) == 2

    def test_single_player_team(self):
        df = pd.DataFrame(
            {
                "recent_team": ["NYG"],
                "season": [2023],
                "week": [5],
                "targets": [6],
            }
        )
        result = compute_team_te_totals(df)
        assert result["team_te_targets"].values[0] == 6

    def test_zero_targets(self):
        df = pd.DataFrame(
            {
                "recent_team": ["LAR", "LAR"],
                "season": [2023, 2023],
                "week": [1, 1],
                "targets": [0, 0],
            }
        )
        result = compute_team_te_totals(df)
        assert result["team_te_targets"].values[0] == 0

    def test_output_columns(self):
        df = pd.DataFrame(
            {
                "recent_team": ["SF"],
                "season": [2023],
                "week": [1],
                "targets": [7],
            }
        )
        result = compute_team_te_totals(df)
        assert set(result.columns) == {"recent_team", "season", "week", "team_te_targets"}
