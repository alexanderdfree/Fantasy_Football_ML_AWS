"""Tests for RB.rb_targets — compute_rb_targets and compute_fumble_adjustment."""

import numpy as np
import pandas as pd
import pytest

from RB.rb_targets import compute_rb_targets, compute_fumble_adjustment


# ---------------------------------------------------------------------------
# compute_rb_targets
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeRBTargets:
    def test_rushing_floor(self, make_rb_row):
        df = make_rb_row(rushing_yards=100)
        result = compute_rb_targets(df)
        assert pytest.approx(result["rushing_floor"].iloc[0]) == 10.0

    def test_receiving_floor(self, make_rb_row):
        df = make_rb_row(receptions=5, receiving_yards=50)
        result = compute_rb_targets(df)
        # 5 * 1.0 + 50 * 0.1 = 10.0
        assert pytest.approx(result["receiving_floor"].iloc[0]) == 10.0

    def test_td_points_rushing_only(self, make_rb_row):
        df = make_rb_row(rushing_tds=2, receiving_tds=0)
        result = compute_rb_targets(df)
        assert pytest.approx(result["td_points"].iloc[0]) == 12.0

    def test_td_points_excludes_2pt_conversions(self, make_rb_row):
        """2pt conversions are excluded from td_points to match SCORING_PPR."""
        df = make_rb_row(rushing_tds=1, receiving_tds=1, rushing_2pt_conversions=1)
        result = compute_rb_targets(df)
        # 1*6 + 1*6 = 12 (2pt conversions intentionally excluded)
        assert pytest.approx(result["td_points"].iloc[0]) == 12.0

    def test_fumble_penalty(self, make_rb_row):
        df = make_rb_row(sack_fumbles_lost=1, rushing_fumbles_lost=1)
        result = compute_rb_targets(df)
        assert pytest.approx(result["fumble_penalty"].iloc[0]) == -4.0

    def test_fantasy_points_check_matches(self, make_rb_row):
        df = make_rb_row(rushing_yards=80, receptions=4, receiving_yards=40,
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
            "receiving_fumbles_lost": np.nan,
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

    def test_does_not_mutate_original(self, make_rb_row):
        df = make_rb_row()
        original_cols = set(df.columns)
        _ = compute_rb_targets(df)
        assert set(df.columns) == original_cols

    def test_zero_yard_game(self, make_rb_row):
        """RB with 0 yards, 0 touches — everything should be 0."""
        df = make_rb_row(
            rushing_yards=0, receiving_yards=0, receptions=0,
            rushing_tds=0, receiving_tds=0,
        )
        result = compute_rb_targets(df)
        assert result["rushing_floor"].iloc[0] == 0.0
        assert result["receiving_floor"].iloc[0] == 0.0
        assert result["td_points"].iloc[0] == 0.0

    def test_large_game(self, make_rb_row):
        """Extreme stat line — no overflow."""
        df = make_rb_row(rushing_yards=300, receiving_yards=200, receptions=10,
                         rushing_tds=4, receiving_tds=2)
        result = compute_rb_targets(df)
        assert result["rushing_floor"].iloc[0] == 30.0
        assert result["receiving_floor"].iloc[0] == 30.0  # 10 + 20
        assert result["td_points"].iloc[0] == 36.0  # 6*6


# ---------------------------------------------------------------------------
# compute_fumble_adjustment
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeFumbleAdjustment:
    def test_first_game_is_zero(self, make_fumble_df):
        """First game has no prior history — shift(1) produces NaN, filled to 0."""
        df = make_fumble_df([1, 0, 0])
        result = compute_fumble_adjustment(df)
        assert result.iloc[0] == 0.0

    def test_second_game_uses_first(self, make_fumble_df):
        """Second game should see the first game's fumble."""
        df = make_fumble_df([1, 0, 0, 0])
        result = compute_fumble_adjustment(df)
        # Game 2 sees rolling mean of [1] -> rate = 1.0 -> penalty = -2.0
        assert pytest.approx(result.iloc[1]) == -2.0

    def test_rolling_window(self, make_fumble_df):
        """After several games, rolling mean averages correctly."""
        df = make_fumble_df([1, 1, 0, 0, 0, 0, 0, 0, 0])
        result = compute_fumble_adjustment(df)
        # Game 9 (index 8): shift(1) sees games 1-8, rolling(8) of [1,1,0,0,0,0,0,0] = 0.25
        assert pytest.approx(result.iloc[8]) == -0.5

    def test_player_with_no_fumbles(self, make_fumble_df):
        df = make_fumble_df([0, 0, 0, 0])
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
            "receiving_fumbles_lost": [0, 0, 0, 0],
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
            "receiving_fumbles_lost": [0, 0, 0, 0],
        })
        result = compute_fumble_adjustment(df)
        # 2023 game 1: first game of new season, shift produces NaN, filled to 0
        assert result.iloc[2] == 0.0
        # 2023 game 2: sees only [0] from 2023 week 1 -> 0.0
        assert pytest.approx(result.iloc[3]) == 0.0
