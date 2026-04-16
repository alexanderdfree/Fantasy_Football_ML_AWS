"""Tests for QB.qb_data — filter_to_qb."""

import pandas as pd
import pytest

from QB.qb_data import filter_to_qb


# ---------------------------------------------------------------------------
# filter_to_qb
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFilterToQB:
    def test_filters_only_qb_rows(self, make_df):
        df = make_df(["QB", "RB", "WR", "QB", "TE"])
        result = filter_to_qb(df)
        assert len(result) == 2
        assert (result["position"] == "QB").all()

    def test_drops_position_encoding_columns(self, make_df):
        df = make_df(["QB", "QB"])
        result = filter_to_qb(df)
        for col in ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]:
            assert col not in result.columns

    def test_keeps_non_position_columns(self, make_df):
        df = make_df(["QB"])
        result = filter_to_qb(df)
        assert "passing_yards" in result.columns

    def test_no_position_encoding_columns(self, make_df):
        """Should not crash when pos_XX columns are absent."""
        df = make_df(["QB", "RB"], has_pos_cols=False)
        result = filter_to_qb(df)
        assert len(result) == 1

    def test_empty_result_when_no_qbs(self, make_df):
        df = make_df(["RB", "WR", "TE"])
        result = filter_to_qb(df)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_empty_input(self):
        df = pd.DataFrame({"position": pd.Series(dtype=str)})
        result = filter_to_qb(df)
        assert len(result) == 0

    def test_does_not_mutate_original(self, make_df):
        df = make_df(["QB", "RB"])
        original_cols = list(df.columns)
        _ = filter_to_qb(df)
        assert list(df.columns) == original_cols
        assert len(df) == 2
