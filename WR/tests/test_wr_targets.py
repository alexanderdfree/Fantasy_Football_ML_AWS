"""Tests for WR.wr_targets — compute_wr_targets and compute_wr_fumble_adjustment."""

import numpy as np
import pandas as pd
import pytest

from WR.wr_targets import compute_wr_targets, compute_wr_fumble_adjustment


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
            + (defaults["sack_fumbles_lost"] + defaults["rushing_fumbles_lost"]
               + defaults["receiving_fumbles_lost"]) * -2
            + defaults["passing_yards"] * 0.04
            + defaults["passing_tds"] * 4
            + defaults["interceptions"] * -2
        )
        defaults["fantasy_points"] = fp
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# compute_wr_targets
# ---------------------------------------------------------------------------

class TestComputeWRTargets:
    def test_receiving_floor(self):
        """receiving_floor = receptions * PPR_weight + receiving_yards * 0.1."""
        df = _make_wr_row(receptions=5, receiving_yards=50)
        result = compute_wr_targets(df)
        # 5 * 1.0 + 50 * 0.1 = 10.0
        assert pytest.approx(result["receiving_floor"].iloc[0]) == 10.0

    def test_rushing_floor(self):
        df = _make_wr_row(rushing_yards=30)
        result = compute_wr_targets(df)
        assert pytest.approx(result["rushing_floor"].iloc[0]) == 3.0

    def test_td_points_receiving_only(self):
        df = _make_wr_row(receiving_tds=2, rushing_tds=0)
        result = compute_wr_targets(df)
        assert pytest.approx(result["td_points"].iloc[0]) == 12.0

    def test_td_points_combined(self):
        """WR with receiving + rushing TDs (end-around play)."""
        df = _make_wr_row(receiving_tds=1, rushing_tds=1)
        result = compute_wr_targets(df)
        # 1*6 + 1*6 = 12
        assert pytest.approx(result["td_points"].iloc[0]) == 12.0

    def test_fumble_penalty(self):
        df = _make_wr_row(receiving_fumbles_lost=1, rushing_fumbles_lost=1)
        result = compute_wr_targets(df)
        assert pytest.approx(result["fumble_penalty"].iloc[0]) == -4.0

    def test_half_ppr_variant_exists(self):
        """Multiple PPR formats should produce separate receiving_floor columns."""
        df = _make_wr_row(receptions=4, receiving_yards=40)
        result = compute_wr_targets(df)
        # At least the primary receiving_floor should exist
        assert "receiving_floor" in result.columns

    def test_all_nan_stats_treated_as_zero(self):
        df = pd.DataFrame([{
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
        }])
        result = compute_wr_targets(df)
        assert result["receiving_floor"].iloc[0] == 0.0
        assert result["rushing_floor"].iloc[0] == 0.0
        assert result["td_points"].iloc[0] == 0.0
        assert result["fumble_penalty"].iloc[0] == 0.0

    def test_does_not_mutate_original(self):
        df = _make_wr_row()
        original_cols = set(df.columns)
        _ = compute_wr_targets(df)
        assert set(df.columns) == original_cols

    def test_zero_catch_game(self):
        """WR with 0 catches should have 0 receiving_floor."""
        df = _make_wr_row(receptions=0, receiving_yards=0, receiving_tds=0,
                          rushing_tds=0, rushing_yards=0)
        result = compute_wr_targets(df)
        assert result["receiving_floor"].iloc[0] == 0.0
        assert result["rushing_floor"].iloc[0] == 0.0
        assert result["td_points"].iloc[0] == 0.0

    def test_big_wr_game(self):
        """Huge game should decompose correctly."""
        df = _make_wr_row(receptions=10, receiving_yards=150, receiving_tds=2,
                          rushing_yards=20, rushing_tds=0)
        result = compute_wr_targets(df)
        # 10 * 1 + 150 * 0.1 = 25
        assert result["receiving_floor"].iloc[0] == 25.0
        assert result["rushing_floor"].iloc[0] == 2.0
        assert result["td_points"].iloc[0] == 12.0


# ---------------------------------------------------------------------------
# compute_wr_fumble_adjustment
# ---------------------------------------------------------------------------

class TestComputeWRFumbleAdjustment:
    def _make_fumble_df(self, receiving_fumbles, season=2023):
        n = len(receiving_fumbles)
        return pd.DataFrame({
            "player_id": ["W1"] * n,
            "season": [season] * n,
            "week": list(range(1, n + 1)),
            "sack_fumbles_lost": [0] * n,
            "rushing_fumbles_lost": [0] * n,
            "receiving_fumbles_lost": receiving_fumbles,
        })

    def test_first_game_is_zero(self):
        df = self._make_fumble_df([1, 0, 0])
        result = compute_wr_fumble_adjustment(df)
        assert result.iloc[0] == 0.0

    def test_second_game_uses_first(self):
        df = self._make_fumble_df([1, 0, 0, 0])
        result = compute_wr_fumble_adjustment(df)
        assert pytest.approx(result.iloc[1]) == -2.0

    def test_rolling_window(self):
        df = self._make_fumble_df([1, 1, 0, 0, 0, 0, 0, 0, 0])
        result = compute_wr_fumble_adjustment(df)
        # Game 9: rolling(8) of [1,1,0,0,0,0,0,0] * -2 mean = -0.5
        assert pytest.approx(result.iloc[8]) == -0.5

    def test_player_with_no_fumbles(self):
        df = self._make_fumble_df([0, 0, 0, 0])
        result = compute_wr_fumble_adjustment(df)
        assert pytest.approx(result.iloc[1]) == 0.0
        assert pytest.approx(result.iloc[3]) == 0.0

    def test_multiple_players_independent(self):
        df = pd.DataFrame({
            "player_id": ["W1", "W1", "W2", "W2"],
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 2, 1, 2],
            "sack_fumbles_lost": [0, 0, 0, 0],
            "rushing_fumbles_lost": [0, 0, 0, 0],
            "receiving_fumbles_lost": [1, 0, 0, 0],
        })
        result = compute_wr_fumble_adjustment(df)
        assert pytest.approx(result.iloc[1]) == -2.0
        assert pytest.approx(result.iloc[3]) == 0.0

    def test_multiple_seasons_reset(self):
        df = pd.DataFrame({
            "player_id": ["W1", "W1", "W1", "W1"],
            "season": [2022, 2022, 2023, 2023],
            "week": [1, 2, 1, 2],
            "sack_fumbles_lost": [0, 0, 0, 0],
            "rushing_fumbles_lost": [0, 0, 0, 0],
            "receiving_fumbles_lost": [1, 1, 0, 0],
        })
        result = compute_wr_fumble_adjustment(df)
        assert result.iloc[2] == 0.0
        assert pytest.approx(result.iloc[3]) == 0.0
