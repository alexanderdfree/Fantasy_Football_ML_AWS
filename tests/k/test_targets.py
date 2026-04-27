"""Tests for K.k_targets — 4-head compute_k_targets."""

import numpy as np
import pandas as pd
import pytest

from src.k.targets import compute_k_targets


def _make_k_row(**overrides):
    """Create a single-row kicker DataFrame with sensible defaults."""
    defaults = {
        "fg_yards_made": 0,
        "fg_missed": 0,
        "pat_made": 3,
        "pat_missed": 0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# compute_k_targets — 4-head decomposition
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeKTargets:
    def test_fg_yard_points_basic(self):
        """fg_yard_points = fg_yards_made * 0.1."""
        df = _make_k_row(fg_yards_made=100)
        result = compute_k_targets(df)
        assert pytest.approx(result["fg_yard_points"].iloc[0]) == 10.0

    def test_fg_yard_points_fractional(self):
        """Distance sums need not be whole multiples of 10."""
        df = _make_k_row(fg_yards_made=137)
        result = compute_k_targets(df)
        assert pytest.approx(result["fg_yard_points"].iloc[0]) == 13.7

    def test_fg_yard_points_zero(self):
        df = _make_k_row(fg_yards_made=0)
        result = compute_k_targets(df)
        assert pytest.approx(result["fg_yard_points"].iloc[0]) == 0.0

    def test_pat_points(self):
        df = _make_k_row(pat_made=5)
        result = compute_k_targets(df)
        assert pytest.approx(result["pat_points"].iloc[0]) == 5.0

    def test_fg_misses_raw_count(self):
        """fg_misses = fg_missed (non-negative raw count)."""
        df = _make_k_row(fg_missed=2)
        result = compute_k_targets(df)
        assert pytest.approx(result["fg_misses"].iloc[0]) == 2.0

    def test_xp_misses_raw_count(self):
        """xp_misses = pat_missed (non-negative raw count)."""
        df = _make_k_row(pat_missed=1)
        result = compute_k_targets(df)
        assert pytest.approx(result["xp_misses"].iloc[0]) == 1.0

    def test_all_four_heads_non_negative(self):
        """Every head is a non-negative value; sign is applied in fantasy_points."""
        df = _make_k_row(fg_yards_made=50, pat_made=2, fg_missed=3, pat_missed=1)
        result = compute_k_targets(df)
        for col in ["fg_yard_points", "pat_points", "fg_misses", "xp_misses"]:
            assert result[col].iloc[0] >= 0, f"{col} should be non-negative"

    def test_fantasy_points_signed_total(self):
        """fantasy_points = fg_yard_points + pat_points - fg_misses - xp_misses."""
        df = _make_k_row(
            fg_yards_made=100,  # 10.0 pts
            pat_made=3,  # 3.0 pts
            fg_missed=2,  # -2.0
            pat_missed=1,  # -1.0
        )
        result = compute_k_targets(df)
        # 10.0 + 3.0 - 2.0 - 1.0 = 10.0
        assert pytest.approx(result["fantasy_points"].iloc[0]) == 10.0

    def test_fantasy_points_perfect_game(self):
        """No misses, plenty of yards."""
        df = _make_k_row(
            fg_yards_made=150,  # 15.0 pts
            pat_made=5,  # 5.0 pts
            fg_missed=0,
            pat_missed=0,
        )
        result = compute_k_targets(df)
        assert pytest.approx(result["fantasy_points"].iloc[0]) == 20.0

    def test_fantasy_points_all_misses(self):
        """No makes, only misses → negative total."""
        df = _make_k_row(
            fg_yards_made=0,
            pat_made=0,
            fg_missed=3,
            pat_missed=1,
        )
        result = compute_k_targets(df)
        # 0 + 0 - 3 - 1 = -4.0
        assert pytest.approx(result["fantasy_points"].iloc[0]) == -4.0

    def test_nan_inputs_treated_as_zero(self):
        """All-NaN row yields zero across the 4 heads and fantasy_points."""
        df = pd.DataFrame(
            [
                {
                    "fg_yards_made": np.nan,
                    "fg_missed": np.nan,
                    "pat_made": np.nan,
                    "pat_missed": np.nan,
                }
            ]
        )
        result = compute_k_targets(df)
        assert result["fg_yard_points"].iloc[0] == 0.0
        assert result["pat_points"].iloc[0] == 0.0
        assert result["fg_misses"].iloc[0] == 0.0
        assert result["xp_misses"].iloc[0] == 0.0
        assert result["fantasy_points"].iloc[0] == 0.0

    def test_does_not_mutate_original(self):
        """compute_k_targets returns a new frame; input columns untouched."""
        df = _make_k_row(fg_yards_made=60, pat_made=2, fg_missed=1, pat_missed=0)
        original_cols = set(df.columns)
        original_copy = df.copy()
        _ = compute_k_targets(df)
        assert set(df.columns) == original_cols, "compute_k_targets mutated column set"
        pd.testing.assert_frame_equal(df, original_copy)

    def test_all_four_target_columns_present(self):
        """The 4 required target columns appear in the result."""
        df = _make_k_row()
        result = compute_k_targets(df)
        for col in ["fg_yard_points", "pat_points", "fg_misses", "xp_misses"]:
            assert col in result.columns, f"Missing target column: {col}"
        assert "fantasy_points" in result.columns
