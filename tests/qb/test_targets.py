"""Tests for src.qb.targets — compute_targets (raw-stat migration)."""

import numpy as np
import pandas as pd
import pytest

from src.qb.targets import compute_targets

TARGET_COLS = (
    "passing_yards",
    "rushing_yards",
    "passing_tds",
    "rushing_tds",
    "interceptions",
    "fumbles_lost",
)


def _make_row(**overrides):
    """Create a single-row QB DataFrame with sensible defaults."""
    defaults = {
        "passing_yards": 250,
        "rushing_yards": 20,
        "receiving_yards": 0,
        "receptions": 0,
        "passing_tds": 2,
        "rushing_tds": 0,
        "receiving_tds": 0,
        "interceptions": 1,
        "sack_fumbles_lost": 0,
        "rushing_fumbles_lost": 0,
        "receiving_fumbles_lost": 0,
        "fantasy_points": 0.0,
    }
    defaults.update(overrides)
    if "fantasy_points" not in overrides:
        fp = (
            defaults["passing_yards"] * 0.04
            + defaults["rushing_yards"] * 0.1
            + defaults["receiving_yards"] * 0.1
            + defaults["receptions"] * 1.0
            + defaults["passing_tds"] * 4
            + (defaults["rushing_tds"] + defaults["receiving_tds"]) * 6
            + defaults["interceptions"] * -2
            + (
                defaults["sack_fumbles_lost"]
                + defaults["rushing_fumbles_lost"]
                + defaults["receiving_fumbles_lost"]
            )
            * -2
        )
        defaults["fantasy_points"] = fp
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# compute_targets — 6 raw-stat columns
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeQBTargets:
    def test_all_six_columns_emitted(self):
        df = _make_row()
        result = compute_targets(df)
        for col in TARGET_COLS:
            assert col in result.columns, f"missing QB target column: {col}"

    def test_passing_yards_identity(self):
        df = _make_row(passing_yards=317)
        result = compute_targets(df)
        assert result["passing_yards"].iloc[0] == 317

    def test_rushing_yards_identity(self):
        df = _make_row(rushing_yards=42)
        result = compute_targets(df)
        assert result["rushing_yards"].iloc[0] == 42

    def test_passing_tds_identity(self):
        df = _make_row(passing_tds=3)
        result = compute_targets(df)
        assert result["passing_tds"].iloc[0] == 3

    def test_rushing_tds_identity(self):
        df = _make_row(rushing_tds=2)
        result = compute_targets(df)
        assert result["rushing_tds"].iloc[0] == 2

    def test_interceptions_identity(self):
        df = _make_row(interceptions=2)
        result = compute_targets(df)
        assert result["interceptions"].iloc[0] == 2

    def test_fumbles_lost_sums_all_three_categories(self):
        """QB fumbles_lost = sack_fumbles_lost + rushing_fumbles_lost + receiving_fumbles_lost."""
        df = _make_row(
            sack_fumbles_lost=1,
            rushing_fumbles_lost=1,
            receiving_fumbles_lost=1,
        )
        result = compute_targets(df)
        assert result["fumbles_lost"].iloc[0] == 3

    def test_fumbles_lost_sack_only(self):
        df = _make_row(sack_fumbles_lost=2, rushing_fumbles_lost=0)
        result = compute_targets(df)
        assert result["fumbles_lost"].iloc[0] == 2

    def test_fumbles_lost_rushing_only(self):
        df = _make_row(sack_fumbles_lost=0, rushing_fumbles_lost=1)
        result = compute_targets(df)
        assert result["fumbles_lost"].iloc[0] == 1

    def test_nan_fills_to_zero(self):
        df = pd.DataFrame(
            [
                {
                    "passing_yards": np.nan,
                    "rushing_yards": np.nan,
                    "receiving_yards": np.nan,
                    "receptions": np.nan,
                    "passing_tds": np.nan,
                    "rushing_tds": np.nan,
                    "receiving_tds": np.nan,
                    "interceptions": np.nan,
                    "sack_fumbles_lost": np.nan,
                    "rushing_fumbles_lost": np.nan,
                    "receiving_fumbles_lost": np.nan,
                    "fantasy_points": 0.0,
                }
            ]
        )
        result = compute_targets(df)
        for col in TARGET_COLS:
            assert result[col].iloc[0] == 0.0, f"{col} did not fill NaN to 0"

    def test_does_not_mutate_original(self):
        df = _make_row()
        original_cols = set(df.columns)
        _ = compute_targets(df)
        assert set(df.columns) == original_cols

    def test_zero_stat_game(self):
        df = _make_row(
            passing_yards=0,
            rushing_yards=0,
            passing_tds=0,
            rushing_tds=0,
            interceptions=0,
        )
        result = compute_targets(df)
        for col in TARGET_COLS:
            assert result[col].iloc[0] == 0.0

    def test_big_passing_game(self):
        df = _make_row(
            passing_yards=500, passing_tds=5, rushing_yards=10, rushing_tds=0, interceptions=0
        )
        result = compute_targets(df)
        assert result["passing_yards"].iloc[0] == 500
        assert result["rushing_yards"].iloc[0] == 10
        assert result["passing_tds"].iloc[0] == 5

    def test_sanity_check_no_warning_on_clean_input(self, capsys):
        """Aggregator on true targets should reproduce fantasy_points (ex-receiving)."""
        df = _make_row(
            passing_yards=275,
            passing_tds=2,
            rushing_yards=30,
            rushing_tds=1,
            interceptions=1,
        )
        _ = compute_targets(df)
        captured = capsys.readouterr()
        assert "discrepancy" not in captured.out
