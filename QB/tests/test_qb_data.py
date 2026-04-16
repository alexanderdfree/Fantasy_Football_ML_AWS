"""Tests for QB.qb_data — filter_to_qb."""

import pandas as pd
import pytest

from QB.qb_data import filter_to_qb


# ---------------------------------------------------------------------------
# filter_to_qb
# ---------------------------------------------------------------------------

class TestFilterToQB:
    def _make_df(self, positions, has_pos_cols=True):
        data = {"position": positions, "passing_yards": range(len(positions))}
        if has_pos_cols:
            data.update({
                "pos_QB": [1 if p == "QB" else 0 for p in positions],
                "pos_RB": [1 if p == "RB" else 0 for p in positions],
                "pos_WR": [1 if p == "WR" else 0 for p in positions],
                "pos_TE": [1 if p == "TE" else 0 for p in positions],
            })
        return pd.DataFrame(data)

    def test_filters_only_qb_rows(self):
        df = self._make_df(["QB", "RB", "WR", "QB", "TE"])
        result = filter_to_qb(df)
        assert len(result) == 2
        assert (result["position"] == "QB").all()

    def test_drops_position_encoding_columns(self):
        df = self._make_df(["QB", "QB"])
        result = filter_to_qb(df)
        for col in ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]:
            assert col not in result.columns

    def test_keeps_non_position_columns(self):
        df = self._make_df(["QB"])
        result = filter_to_qb(df)
        assert "passing_yards" in result.columns

    def test_no_position_encoding_columns(self):
        """Should not crash when pos_XX columns are absent."""
        df = self._make_df(["QB", "RB"], has_pos_cols=False)
        result = filter_to_qb(df)
        assert len(result) == 1

    def test_empty_result_when_no_qbs(self):
        df = self._make_df(["RB", "WR", "TE"])
        result = filter_to_qb(df)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_empty_input(self):
        df = pd.DataFrame({"position": pd.Series(dtype=str)})
        result = filter_to_qb(df)
        assert len(result) == 0

    def test_does_not_mutate_original(self):
        df = self._make_df(["QB", "RB"])
        original_cols = list(df.columns)
        _ = filter_to_qb(df)
        assert list(df.columns) == original_cols
        assert len(df) == 2
