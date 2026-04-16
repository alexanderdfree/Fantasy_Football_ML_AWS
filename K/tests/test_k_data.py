"""Tests for K.k_data — filter_to_k (identity filter) and kicker_season_split."""

import pandas as pd
import pytest

from K.k_data import filter_to_k, kicker_season_split


# ---------------------------------------------------------------------------
# filter_to_k — identity filter
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFilterToK:
    """Kicker data is pre-filtered; filter_to_k is an identity (copy)."""

    def test_returns_copy_of_input(self):
        df = pd.DataFrame({
            "position": ["K", "K"],
            "fg_att": [3, 2],
            "player_id": ["K1", "K2"],
        })
        result = filter_to_k(df)
        pd.testing.assert_frame_equal(result, df)

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({
            "position": ["K"],
            "fg_att": [3],
        })
        original_cols = list(df.columns)
        original_len = len(df)
        _ = filter_to_k(df)
        assert list(df.columns) == original_cols
        assert len(df) == original_len

    def test_returned_is_independent_copy(self):
        """Mutating the result should not affect the input."""
        df = pd.DataFrame({"position": ["K"], "fg_att": [3]})
        result = filter_to_k(df)
        result["fg_att"] = 99
        assert df["fg_att"].iloc[0] == 3

    def test_empty_input(self):
        df = pd.DataFrame({"position": pd.Series(dtype=str), "fg_att": pd.Series(dtype=int)})
        result = filter_to_k(df)
        assert len(result) == 0

    def test_preserves_all_columns(self):
        """Identity filter must not drop columns."""
        df = pd.DataFrame({
            "player_id": ["K1"],
            "season": [2023],
            "week": [1],
            "fg_att": [3],
            "fg_made": [2],
            "pat_att": [3],
            "pat_made": [3],
            "roof": ["dome"],
        })
        result = filter_to_k(df)
        assert set(result.columns) == set(df.columns)


# ---------------------------------------------------------------------------
# kicker_season_split
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_season_df():
    """Multi-season kicker DataFrame spanning the train/val/test boundary."""
    rows = []
    for season in [2020, 2022, 2023, 2024, 2025]:
        for week in range(1, 4):
            rows.append({
                "player_id": "K1",
                "season": season,
                "week": week,
                "fg_att": 2,
            })
    return pd.DataFrame(rows)


@pytest.mark.unit
class TestKickerSeasonSplit:
    def test_split_boundaries(self, multi_season_df):
        """Train: 2015-2023, Val: 2024, Test: 2025."""
        train, val, test = kicker_season_split(multi_season_df)
        assert train["season"].max() <= 2023
        assert (val["season"] == 2024).all()
        assert (test["season"] == 2025).all()

    def test_all_rows_allocated(self, multi_season_df):
        train, val, test = kicker_season_split(multi_season_df)
        assert len(train) + len(val) + len(test) == len(multi_season_df)

    def test_train_contains_earlier_seasons(self, multi_season_df):
        train, _, _ = kicker_season_split(multi_season_df)
        assert 2020 in train["season"].values
        assert 2022 in train["season"].values
        assert 2023 in train["season"].values

    def test_returns_dataframes(self, multi_season_df):
        train, val, test = kicker_season_split(multi_season_df)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
