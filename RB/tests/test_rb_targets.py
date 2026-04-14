"""Tests for RB.rb_targets — compute_rb_targets and compute_fumble_adjustment."""

import numpy as np
import pandas as pd
import pytest

from RB.rb_targets import compute_rb_targets, compute_fumble_adjustment


def _make_rb_row(**overrides):
    """Create a single-row RB DataFrame with sensible defaults."""
    defaults = {
        "rushing_yards": 60,
        "receiving_yards": 30,
        "receptions": 3,
        "rushing_tds": 1,
        "receiving_tds": 0,
        "rushing_2pt_conversions": 0,
        "receiving_2pt_conversions": 0,
        "sack_fumbles_lost": 0,
        "rushing_fumbles_lost": 0,
        "passing_yards": 0,
        "passing_tds": 0,
        "interceptions": 0,
        "fantasy_points": 0.0,  # will be computed below
    }
    defaults.update(overrides)
    # Compute correct fantasy_points if not explicitly overridden
    if "fantasy_points" not in overrides:
        fp = (
            defaults["rushing_yards"] * 0.1
            + defaults["receptions"] * 1.0
            + defaults["receiving_yards"] * 0.1
            + (defaults["rushing_tds"] + defaults["receiving_tds"]) * 6
            + (defaults["rushing_2pt_conversions"] + defaults["receiving_2pt_conversions"]) * 2
            + (defaults["sack_fumbles_lost"] + defaults["rushing_fumbles_lost"]) * -2
            + defaults["passing_yards"] * 0.04
            + defaults["passing_tds"] * 4
            + defaults["interceptions"] * -2
        )
        defaults["fantasy_points"] = fp
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# compute_rb_targets
# ---------------------------------------------------------------------------

class TestComputeRBTargets:
    def test_rushing_floor(self):
        df = _make_rb_row(rushing_yards=100)
        result = compute_rb_targets(df)
        assert pytest.approx(result["rushing_floor"].iloc[0]) == 10.0

    def test_receiving_floor(self):
        df = _make_rb_row(receptions=5, receiving_yards=50)
        result = compute_rb_targets(df)
        # 5 * 1.0 + 50 * 0.1 = 10.0
        assert pytest.approx(result["receiving_floor"].iloc[0]) == 10.0

    def test_td_points_rushing_only(self):
        df = _make_rb_row(rushing_tds=2, receiving_tds=0)
        result = compute_rb_targets(df)
        assert pytest.approx(result["td_points"].iloc[0]) == 12.0

    def test_td_points_with_2pt_conversions(self):
        df = _make_rb_row(rushing_tds=1, receiving_tds=1, rushing_2pt_conversions=1)
        result = compute_rb_targets(df)
        # 1*6 + 1*6 + 1*2 = 14
        assert pytest.approx(result["td_points"].iloc[0]) == 14.0

    def test_fumble_penalty(self):
        df = _make_rb_row(sack_fumbles_lost=1, rushing_fumbles_lost=1)
        result = compute_rb_targets(df)
        assert pytest.approx(result["fumble_penalty"].iloc[0]) == -4.0

    def test_fantasy_points_check_matches(self):
        df = _make_rb_row(rushing_yards=80, receptions=4, receiving_yards=40,
                          rushing_tds=1, receiving_tds=0, rushing_fumbles_lost=1)
        result = compute_rb_targets(df)
        expected = (
            result["rushing_floor"].iloc[0]
            + result["receiving_floor"].iloc[0]
            + result["td_points"].iloc[0]
            + result["fumble_penalty"].iloc[0]
        )
        assert pytest.approx(result["fantasy_points_check"].iloc[0]) == expected

    def test_all_nan_stats_treated_as_zero(self):
        """Player with all NaN stats should produce zero targets."""
        df = pd.DataFrame([{
            "rushing_yards": np.nan,
            "receiving_yards": np.nan,
            "receptions": np.nan,
            "rushing_tds": np.nan,
            "receiving_tds": np.nan,
            "rushing_2pt_conversions": np.nan,
            "receiving_2pt_conversions": np.nan,
            "sack_fumbles_lost": np.nan,
            "rushing_fumbles_lost": np.nan,
            "passing_yards": np.nan,
            "passing_tds": np.nan,
            "interceptions": np.nan,
            "fantasy_points": 0.0,
        }])
        result = compute_rb_targets(df)
        assert result["rushing_floor"].iloc[0] == 0.0
        assert result["receiving_floor"].iloc[0] == 0.0
        assert result["td_points"].iloc[0] == 0.0
        assert result["fumble_penalty"].iloc[0] == 0.0

    def test_does_not_mutate_original(self):
        df = _make_rb_row()
        original_cols = set(df.columns)
        _ = compute_rb_targets(df)
        assert set(df.columns) == original_cols

    def test_zero_yard_game(self):
        """RB with 0 yards, 0 touches — everything should be 0."""
        df = _make_rb_row(
            rushing_yards=0, receiving_yards=0, receptions=0,
            rushing_tds=0, receiving_tds=0,
        )
        result = compute_rb_targets(df)
        assert result["rushing_floor"].iloc[0] == 0.0
        assert result["receiving_floor"].iloc[0] == 0.0
        assert result["td_points"].iloc[0] == 0.0

    def test_large_game(self):
        """Extreme stat line — no overflow."""
        df = _make_rb_row(rushing_yards=300, receiving_yards=200, receptions=10,
                          rushing_tds=4, receiving_tds=2)
        result = compute_rb_targets(df)
        assert result["rushing_floor"].iloc[0] == 30.0
        assert result["receiving_floor"].iloc[0] == 30.0  # 10 + 20
        assert result["td_points"].iloc[0] == 36.0  # 6*6


# ---------------------------------------------------------------------------
# compute_fumble_adjustment
# ---------------------------------------------------------------------------

class TestComputeFumbleAdjustment:
    def _make_fumble_df(self, player_fumbles, season=2023):
        """Create DataFrame with multiple games for one player."""
        n = len(player_fumbles)
        return pd.DataFrame({
            "player_id": ["P1"] * n,
            "season": [season] * n,
            "week": list(range(1, n + 1)),
            "sack_fumbles_lost": player_fumbles,
            "rushing_fumbles_lost": [0] * n,
        })

    def test_first_game_is_nan(self):
        """First game has no prior history — shift(1) produces NaN."""
        df = self._make_fumble_df([1, 0, 0])
        result = compute_fumble_adjustment(df)
        assert np.isnan(result.iloc[0])

    def test_second_game_uses_first(self):
        """Second game should see the first game's fumble."""
        df = self._make_fumble_df([1, 0, 0, 0])
        result = compute_fumble_adjustment(df)
        # Game 2 sees rolling mean of [1] -> rate = 1.0 -> penalty = -2.0
        assert pytest.approx(result.iloc[1]) == -2.0

    def test_rolling_window(self):
        """After several games, rolling mean averages correctly."""
        df = self._make_fumble_df([1, 1, 0, 0, 0, 0, 0, 0, 0])
        result = compute_fumble_adjustment(df)
        # Game 9 (index 8): shift(1) sees games 1-8, rolling(8) of [1,1,0,0,0,0,0,0] = 0.25
        assert pytest.approx(result.iloc[8]) == -0.5

    def test_player_with_no_fumbles(self):
        df = self._make_fumble_df([0, 0, 0, 0])
        result = compute_fumble_adjustment(df)
        # Game 2 onward should be 0.0 (0 fumbles * -2)
        assert pytest.approx(result.iloc[1]) == 0.0
        assert pytest.approx(result.iloc[3]) == 0.0

    def test_multiple_players(self):
        """Each player's fumble history is independent."""
        df = pd.DataFrame({
            "player_id": ["P1", "P1", "P2", "P2"],
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 2, 1, 2],
            "sack_fumbles_lost": [1, 0, 0, 0],
            "rushing_fumbles_lost": [0, 0, 0, 0],
        })
        result = compute_fumble_adjustment(df)
        # P1 game 2: sees [1] -> -2.0
        p1_game2 = result.iloc[1]
        assert pytest.approx(p1_game2) == -2.0
        # P2 game 2: sees [0] -> 0.0
        p2_game2 = result.iloc[3]
        assert pytest.approx(p2_game2) == 0.0

    def test_multiple_seasons_reset(self):
        """Fumble history should reset across seasons (grouped by season)."""
        df = pd.DataFrame({
            "player_id": ["P1", "P1", "P1", "P1"],
            "season": [2022, 2022, 2023, 2023],
            "week": [1, 2, 1, 2],
            "sack_fumbles_lost": [1, 1, 0, 0],
            "rushing_fumbles_lost": [0, 0, 0, 0],
        })
        result = compute_fumble_adjustment(df)
        # 2023 game 1: first game of new season, shift produces NaN
        assert np.isnan(result.iloc[2])
        # 2023 game 2: sees only [0] from 2023 week 1 -> 0.0
        assert pytest.approx(result.iloc[3]) == 0.0
