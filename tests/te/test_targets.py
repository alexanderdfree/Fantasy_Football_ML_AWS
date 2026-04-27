"""Tests for src.te.targets — compute_targets emits 4 raw-stat columns."""

import numpy as np
import pandas as pd
import pytest

from src.config import SCORING_PPR
from src.data.loader import compute_fantasy_points
from src.te.targets import compute_targets

pytestmark = pytest.mark.unit


def _make_row(**overrides):
    """Create a single-row TE DataFrame with sensible defaults."""
    defaults = {
        "receiving_yards": 55,
        "rushing_yards": 0,
        "receptions": 4,
        "targets": 6,
        "receiving_tds": 1,
        "rushing_tds": 0,
        "sack_fumbles_lost": 0,
        "rushing_fumbles_lost": 0,
        "receiving_fumbles_lost": 0,
        "passing_yards": 0,
        "passing_tds": 0,
        "interceptions": 0,
        "fantasy_points": 0.0,
    }
    defaults.update(overrides)
    if "fantasy_points" not in overrides:
        defaults["fantasy_points"] = float(
            compute_fantasy_points(pd.DataFrame([defaults]), SCORING_PPR).iloc[0]
        )
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# compute_targets
# ---------------------------------------------------------------------------


class TestComputeTETargets:
    def test_receiving_tds_passthrough(self):
        df = _make_row(receiving_tds=2)
        result = compute_targets(df)
        assert result["receiving_tds"].iloc[0] == 2

    def test_receiving_yards_passthrough(self):
        df = _make_row(receiving_yards=87)
        result = compute_targets(df)
        assert pytest.approx(result["receiving_yards"].iloc[0]) == 87

    def test_receptions_passthrough(self):
        df = _make_row(receptions=6)
        result = compute_targets(df)
        assert result["receptions"].iloc[0] == 6

    def test_fumbles_lost_sums_all_three_categories(self):
        df = _make_row(rushing_fumbles_lost=1, receiving_fumbles_lost=1, sack_fumbles_lost=1)
        result = compute_targets(df)
        assert result["fumbles_lost"].iloc[0] == 3

    def test_all_nan_stats_treated_as_zero(self):
        df = pd.DataFrame(
            [
                {
                    "receiving_yards": np.nan,
                    "rushing_yards": np.nan,
                    "receptions": np.nan,
                    "targets": np.nan,
                    "receiving_tds": np.nan,
                    "rushing_tds": np.nan,
                    "sack_fumbles_lost": np.nan,
                    "rushing_fumbles_lost": np.nan,
                    "receiving_fumbles_lost": np.nan,
                    "passing_yards": np.nan,
                    "passing_tds": np.nan,
                    "interceptions": np.nan,
                    "fantasy_points": 0.0,
                }
            ]
        )
        result = compute_targets(df)
        assert result["receiving_tds"].iloc[0] == 0.0
        assert result["receiving_yards"].iloc[0] == 0.0
        assert result["receptions"].iloc[0] == 0.0
        assert result["fumbles_lost"].iloc[0] == 0.0

    def test_does_not_mutate_original(self):
        df = _make_row()
        original_cols = set(df.columns)
        _ = compute_targets(df)
        assert set(df.columns) == original_cols

    def test_zero_catch_game(self):
        """TE with 0 catches on blocking-heavy game."""
        df = _make_row(
            receptions=0, receiving_yards=0, receiving_tds=0, rushing_tds=0, rushing_yards=0
        )
        result = compute_targets(df)
        assert result["receiving_tds"].iloc[0] == 0
        assert result["receiving_yards"].iloc[0] == 0
        assert result["receptions"].iloc[0] == 0

    def test_big_te_game(self):
        df = _make_row(receptions=8, receiving_yards=120, receiving_tds=2)
        result = compute_targets(df)
        assert result["receptions"].iloc[0] == 8
        assert result["receiving_yards"].iloc[0] == 120
        assert result["receiving_tds"].iloc[0] == 2

    def test_aggregator_reproduces_te_fantasy_points(self):
        """Aggregating the 4 raw targets to fantasy points should match the
        TE-only slice of fantasy_points (no passing/rushing contributions)."""
        from src.shared.aggregate_targets import predictions_to_fantasy_points

        df = _make_row(receptions=6, receiving_yards=60, receiving_tds=1)
        result = compute_targets(df)
        preds = {
            t: result[t].values
            for t in ("receiving_tds", "receiving_yards", "receptions", "fumbles_lost")
        }
        total = predictions_to_fantasy_points("TE", preds, "ppr")
        # 6 rec × 1 + 60 yds × 0.1 + 1 TD × 6 = 18
        assert pytest.approx(total[0]) == 18.0

    def test_decomposition_discrepancy_emits_warning(self, capsys):
        """Rows whose ``fantasy_points`` cannot be reconstructed from the raw
        stats (data corruption upstream) should print a WARNING line."""
        # Why: pin fantasy_points to a value the 4 raw targets can't sum to;
        # _make_row's default stats reconstruct to 17.5 pts, far from 99.0.
        df = _make_row(fantasy_points=99.0)
        compute_targets(df)
        assert "target decomposition discrepancy" in capsys.readouterr().out
