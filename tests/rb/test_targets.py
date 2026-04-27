"""Tests for RB.rb_targets — compute_rb_targets (raw-stat targets)."""

import numpy as np
import pandas as pd
import pytest

from src.RB.rb_config import RB_TARGETS
from src.RB.rb_targets import compute_rb_targets


@pytest.mark.unit
class TestComputeRBTargets:
    def test_all_targets_present(self, make_rb_row):
        df = make_rb_row(rushing_yards=80, receiving_yards=40, receptions=3, rushing_tds=1)
        result = compute_rb_targets(df)
        for t in RB_TARGETS:
            assert t in result.columns, f"Missing target column: {t}"

    def test_rushing_yards_passthrough(self, make_rb_row):
        df = make_rb_row(rushing_yards=123)
        result = compute_rb_targets(df)
        assert pytest.approx(result["rushing_yards"].iloc[0]) == 123

    def test_receiving_yards_passthrough(self, make_rb_row):
        df = make_rb_row(receiving_yards=57)
        result = compute_rb_targets(df)
        assert pytest.approx(result["receiving_yards"].iloc[0]) == 57

    def test_receptions_passthrough(self, make_rb_row):
        df = make_rb_row(receptions=7)
        result = compute_rb_targets(df)
        assert pytest.approx(result["receptions"].iloc[0]) == 7

    def test_rushing_tds_passthrough(self, make_rb_row):
        df = make_rb_row(rushing_tds=2, receiving_tds=0)
        result = compute_rb_targets(df)
        assert pytest.approx(result["rushing_tds"].iloc[0]) == 2
        assert pytest.approx(result["receiving_tds"].iloc[0]) == 0

    def test_receiving_tds_passthrough(self, make_rb_row):
        df = make_rb_row(rushing_tds=0, receiving_tds=3)
        result = compute_rb_targets(df)
        assert pytest.approx(result["receiving_tds"].iloc[0]) == 3

    def test_fumbles_lost_sums_all_three_categories(self, make_rb_row):
        """fumbles_lost = sack_fumbles_lost + rushing_fumbles_lost + receiving_fumbles_lost."""
        df = make_rb_row(sack_fumbles_lost=1, rushing_fumbles_lost=1, receiving_fumbles_lost=1)
        result = compute_rb_targets(df)
        assert pytest.approx(result["fumbles_lost"].iloc[0]) == 3

    def test_fumbles_lost_sack_only(self, make_rb_row):
        """Rare trick-play sack fumble on an RB still counts toward fumbles_lost."""
        df = make_rb_row(sack_fumbles_lost=2, rushing_fumbles_lost=0, receiving_fumbles_lost=0)
        result = compute_rb_targets(df)
        assert pytest.approx(result["fumbles_lost"].iloc[0]) == 2

    def test_aggregator_matches_fantasy_points_check(self, make_rb_row):
        """fantasy_points_check (via aggregator) matches manual raw-stat scoring (PPR)."""
        df = make_rb_row(
            rushing_yards=80,
            receiving_yards=40,
            receptions=4,
            rushing_tds=1,
            receiving_tds=0,
            rushing_fumbles_lost=1,
        )
        result = compute_rb_targets(df)
        expected = 80 * 0.1 + 40 * 0.1 + 4 * 1.0 + 1 * 6 + 0 * 6 + 1 * -2
        assert pytest.approx(result["fantasy_points_check"].iloc[0]) == expected

    def test_all_nan_stats_treated_as_zero(self):
        df = pd.DataFrame(
            [
                {
                    "rushing_yards": np.nan,
                    "receiving_yards": np.nan,
                    "receptions": np.nan,
                    "rushing_tds": np.nan,
                    "receiving_tds": np.nan,
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
        result = compute_rb_targets(df)
        for t in RB_TARGETS:
            assert result[t].iloc[0] == 0.0

    def test_does_not_mutate_original(self, make_rb_row):
        df = make_rb_row()
        original_cols = set(df.columns)
        _ = compute_rb_targets(df)
        assert set(df.columns) == original_cols

    def test_large_game(self, make_rb_row):
        """Extreme stat line — no overflow."""
        df = make_rb_row(
            rushing_yards=300,
            receiving_yards=200,
            receptions=10,
            rushing_tds=4,
            receiving_tds=2,
        )
        result = compute_rb_targets(df)
        assert result["rushing_yards"].iloc[0] == 300
        assert result["receiving_yards"].iloc[0] == 200
        assert result["receptions"].iloc[0] == 10
        assert result["rushing_tds"].iloc[0] == 4
        assert result["receiving_tds"].iloc[0] == 2
