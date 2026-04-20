"""Tests for WR.wr_targets - compute_wr_targets (raw-stat targets)."""

import numpy as np
import pandas as pd
import pytest

from WR.wr_targets import compute_wr_targets


def _make_wr_row(**overrides):
    """Create a single-row WR DataFrame with sensible defaults."""
    defaults = {
        "receiving_yards": 80,
        "rushing_yards": 0,
        "receptions": 6,
        "targets": 8,
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
        fp = (
            defaults["receiving_yards"] * 0.1
            + defaults["receptions"] * 1.0  # PPR
            + defaults["rushing_yards"] * 0.1
            + (defaults["receiving_tds"] + defaults["rushing_tds"]) * 6
            + (
                defaults["sack_fumbles_lost"]
                + defaults["rushing_fumbles_lost"]
                + defaults["receiving_fumbles_lost"]
            )
            * -2
            + defaults["passing_yards"] * 0.04
            + defaults["passing_tds"] * 4
            + defaults["interceptions"] * -2
        )
        defaults["fantasy_points"] = fp
    return pd.DataFrame([defaults])


@pytest.mark.unit
class TestComputeWRTargets:
    def test_receiving_tds_identity(self):
        df = _make_wr_row(receiving_tds=2)
        result = compute_wr_targets(df)
        assert pytest.approx(result["receiving_tds"].iloc[0]) == 2.0

    def test_receiving_yards_identity(self):
        df = _make_wr_row(receiving_yards=95)
        result = compute_wr_targets(df)
        assert pytest.approx(result["receiving_yards"].iloc[0]) == 95.0

    def test_receptions_identity(self):
        df = _make_wr_row(receptions=7)
        result = compute_wr_targets(df)
        assert pytest.approx(result["receptions"].iloc[0]) == 7.0

    def test_fumbles_lost_is_rushing_plus_receiving(self):
        df = _make_wr_row(receiving_fumbles_lost=1, rushing_fumbles_lost=1)
        result = compute_wr_targets(df)
        assert pytest.approx(result["fumbles_lost"].iloc[0]) == 2.0

    def test_fumbles_lost_excludes_sack_fumbles(self):
        """Sack fumbles are a QB concept; WR fumbles_lost never includes them."""
        df = _make_wr_row(sack_fumbles_lost=1, rushing_fumbles_lost=0, receiving_fumbles_lost=0)
        result = compute_wr_targets(df)
        assert pytest.approx(result["fumbles_lost"].iloc[0]) == 0.0

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
        result = compute_wr_targets(df)
        assert result["receiving_tds"].iloc[0] == 0.0
        assert result["receiving_yards"].iloc[0] == 0.0
        assert result["receptions"].iloc[0] == 0.0
        assert result["fumbles_lost"].iloc[0] == 0.0

    def test_does_not_mutate_original(self):
        df = _make_wr_row()
        original_cols = set(df.columns)
        _ = compute_wr_targets(df)
        assert set(df.columns) == original_cols

    def test_zero_game(self):
        df = _make_wr_row(
            receptions=0,
            receiving_yards=0,
            receiving_tds=0,
            rushing_tds=0,
            rushing_yards=0,
        )
        result = compute_wr_targets(df)
        assert result["receiving_tds"].iloc[0] == 0.0
        assert result["receiving_yards"].iloc[0] == 0.0
        assert result["receptions"].iloc[0] == 0.0
        assert result["fumbles_lost"].iloc[0] == 0.0

    def test_big_game(self):
        df = _make_wr_row(
            receptions=10,
            receiving_yards=150,
            receiving_tds=2,
            rushing_yards=0,
            rushing_tds=0,
        )
        result = compute_wr_targets(df)
        assert result["receiving_tds"].iloc[0] == 2.0
        assert result["receiving_yards"].iloc[0] == 150.0
        assert result["receptions"].iloc[0] == 10.0
        assert result["fumbles_lost"].iloc[0] == 0.0
