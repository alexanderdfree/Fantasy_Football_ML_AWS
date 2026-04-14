"""Tests for RB.rb_data — filter_to_rb and compute_team_rb_totals."""

import pandas as pd
import pytest

from RB.rb_data import filter_to_rb, compute_team_rb_totals


# ---------------------------------------------------------------------------
# filter_to_rb
# ---------------------------------------------------------------------------

class TestFilterToRB:
    def _make_df(self, positions, has_pos_cols=True):
        """Helper: build a minimal DataFrame with position column + optional encoding."""
        data = {"position": positions, "rushing_yards": range(len(positions))}
        if has_pos_cols:
            data.update({
                "pos_QB": [1 if p == "QB" else 0 for p in positions],
                "pos_RB": [1 if p == "RB" else 0 for p in positions],
                "pos_WR": [1 if p == "WR" else 0 for p in positions],
                "pos_TE": [1 if p == "TE" else 0 for p in positions],
            })
        return pd.DataFrame(data)

    def test_filters_only_rb_rows(self):
        df = self._make_df(["QB", "RB", "WR", "RB", "TE"])
        result = filter_to_rb(df)
        assert len(result) == 2
        assert (result["position"] == "RB").all()

    def test_drops_position_encoding_columns(self):
        df = self._make_df(["RB", "RB"])
        result = filter_to_rb(df)
        for col in ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]:
            assert col not in result.columns

    def test_keeps_non_position_columns(self):
        df = self._make_df(["RB"])
        result = filter_to_rb(df)
        assert "rushing_yards" in result.columns

    def test_no_position_encoding_columns(self):
        """Should not crash when pos_XX columns are absent."""
        df = self._make_df(["RB", "QB"], has_pos_cols=False)
        result = filter_to_rb(df)
        assert len(result) == 1

    def test_empty_result_when_no_rbs(self):
        df = self._make_df(["QB", "WR", "TE"])
        result = filter_to_rb(df)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_empty_input(self):
        df = pd.DataFrame({"position": pd.Series(dtype=str)})
        result = filter_to_rb(df)
        assert len(result) == 0

    def test_does_not_mutate_original(self):
        df = self._make_df(["RB", "QB"])
        original_cols = list(df.columns)
        _ = filter_to_rb(df)
        assert list(df.columns) == original_cols
        assert len(df) == 2  # original unchanged


# ---------------------------------------------------------------------------
# compute_team_rb_totals
# ---------------------------------------------------------------------------

class TestComputeTeamRBTotals:
    def test_basic_aggregation(self):
        df = pd.DataFrame({
            "recent_team": ["KC", "KC", "KC", "BUF"],
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 1, 1, 1],
            "carries": [15, 8, 3, 20],
            "targets": [4, 6, 1, 5],
        })
        result = compute_team_rb_totals(df)
        kc_row = result[result["recent_team"] == "KC"]
        assert kc_row["team_rb_carries"].values[0] == 26
        assert kc_row["team_rb_targets"].values[0] == 11

    def test_multiple_weeks(self):
        df = pd.DataFrame({
            "recent_team": ["KC", "KC"],
            "season": [2023, 2023],
            "week": [1, 2],
            "carries": [15, 20],
            "targets": [4, 6],
        })
        result = compute_team_rb_totals(df)
        assert len(result) == 2  # one row per team-season-week

    def test_single_player_team(self):
        df = pd.DataFrame({
            "recent_team": ["NYG"],
            "season": [2023],
            "week": [5],
            "carries": [22],
            "targets": [7],
        })
        result = compute_team_rb_totals(df)
        assert result["team_rb_carries"].values[0] == 22

    def test_zero_carries_and_targets(self):
        df = pd.DataFrame({
            "recent_team": ["LAR", "LAR"],
            "season": [2023, 2023],
            "week": [1, 1],
            "carries": [0, 0],
            "targets": [0, 0],
        })
        result = compute_team_rb_totals(df)
        assert result["team_rb_carries"].values[0] == 0
        assert result["team_rb_targets"].values[0] == 0

    def test_output_columns(self):
        df = pd.DataFrame({
            "recent_team": ["SF"],
            "season": [2023],
            "week": [1],
            "carries": [10],
            "targets": [5],
        })
        result = compute_team_rb_totals(df)
        assert set(result.columns) == {
            "recent_team", "season", "week", "team_rb_carries", "team_rb_targets"
        }
