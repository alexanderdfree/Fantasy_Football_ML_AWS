"""Tests for WR.wr_data — filter_to_wr and compute_team_wr_totals."""

import pandas as pd
import pytest

from WR.wr_data import filter_to_wr, compute_team_wr_totals


# ---------------------------------------------------------------------------
# filter_to_wr
# ---------------------------------------------------------------------------

class TestFilterToWR:
    def _make_df(self, positions, has_pos_cols=True):
        data = {"position": positions, "receiving_yards": range(len(positions))}
        if has_pos_cols:
            data.update({
                "pos_QB": [1 if p == "QB" else 0 for p in positions],
                "pos_RB": [1 if p == "RB" else 0 for p in positions],
                "pos_WR": [1 if p == "WR" else 0 for p in positions],
                "pos_TE": [1 if p == "TE" else 0 for p in positions],
            })
        return pd.DataFrame(data)

    def test_filters_only_wr_rows(self):
        df = self._make_df(["QB", "RB", "WR", "WR", "TE"])
        result = filter_to_wr(df)
        assert len(result) == 2
        assert (result["position"] == "WR").all()

    def test_drops_position_encoding_columns(self):
        df = self._make_df(["WR", "WR"])
        result = filter_to_wr(df)
        for col in ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]:
            assert col not in result.columns

    def test_keeps_non_position_columns(self):
        df = self._make_df(["WR"])
        result = filter_to_wr(df)
        assert "receiving_yards" in result.columns

    def test_no_position_encoding_columns(self):
        df = self._make_df(["WR", "QB"], has_pos_cols=False)
        result = filter_to_wr(df)
        assert len(result) == 1

    def test_empty_result_when_no_wrs(self):
        df = self._make_df(["QB", "RB", "TE"])
        result = filter_to_wr(df)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_empty_input(self):
        df = pd.DataFrame({"position": pd.Series(dtype=str)})
        result = filter_to_wr(df)
        assert len(result) == 0

    def test_does_not_mutate_original(self):
        df = self._make_df(["WR", "QB"])
        original_cols = list(df.columns)
        _ = filter_to_wr(df)
        assert list(df.columns) == original_cols
        assert len(df) == 2


# ---------------------------------------------------------------------------
# compute_team_wr_totals
# ---------------------------------------------------------------------------

class TestComputeTeamWRTotals:
    def test_basic_aggregation(self):
        df = pd.DataFrame({
            "recent_team": ["KC", "KC", "KC", "BUF"],
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 1, 1, 1],
            "targets": [8, 10, 3, 7],
        })
        result = compute_team_wr_totals(df)
        kc_row = result[result["recent_team"] == "KC"]
        assert kc_row["team_wr_targets"].values[0] == 21

    def test_multiple_weeks(self):
        df = pd.DataFrame({
            "recent_team": ["KC", "KC"],
            "season": [2023, 2023],
            "week": [1, 2],
            "targets": [8, 10],
        })
        result = compute_team_wr_totals(df)
        assert len(result) == 2

    def test_single_player_team(self):
        df = pd.DataFrame({
            "recent_team": ["NYG"],
            "season": [2023],
            "week": [5],
            "targets": [12],
        })
        result = compute_team_wr_totals(df)
        assert result["team_wr_targets"].values[0] == 12

    def test_zero_targets(self):
        df = pd.DataFrame({
            "recent_team": ["LAR", "LAR"],
            "season": [2023, 2023],
            "week": [1, 1],
            "targets": [0, 0],
        })
        result = compute_team_wr_totals(df)
        assert result["team_wr_targets"].values[0] == 0

    def test_output_columns(self):
        df = pd.DataFrame({
            "recent_team": ["SF"],
            "season": [2023],
            "week": [1],
            "targets": [8],
        })
        result = compute_team_wr_totals(df)
        assert set(result.columns) == {
            "recent_team", "season", "week", "team_wr_targets"
        }
