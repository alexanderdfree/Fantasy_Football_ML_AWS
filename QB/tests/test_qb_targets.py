"""Tests for QB.qb_targets — compute_qb_targets and compute_qb_adjustment."""

import numpy as np
import pandas as pd
import pytest

from QB.qb_targets import compute_qb_adjustment, compute_qb_targets


def _make_qb_row(**overrides):
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
# compute_qb_targets
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeQBTargets:
    def test_passing_floor(self):
        df = _make_qb_row(passing_yards=300)
        result = compute_qb_targets(df)
        # 300 * 0.04 = 12.0
        assert pytest.approx(result["passing_floor"].iloc[0]) == 12.0

    def test_rushing_floor(self):
        df = _make_qb_row(rushing_yards=50)
        result = compute_qb_targets(df)
        assert pytest.approx(result["rushing_floor"].iloc[0]) == 5.0

    def test_td_points_passing(self):
        df = _make_qb_row(passing_tds=3, rushing_tds=0)
        result = compute_qb_targets(df)
        # 3 * 4 = 12
        assert pytest.approx(result["td_points"].iloc[0]) == 12.0

    def test_td_points_rushing(self):
        df = _make_qb_row(passing_tds=0, rushing_tds=2, receiving_tds=0)
        result = compute_qb_targets(df)
        assert pytest.approx(result["td_points"].iloc[0]) == 12.0

    def test_td_points_combined(self):
        df = _make_qb_row(passing_tds=2, rushing_tds=1, receiving_tds=1)
        result = compute_qb_targets(df)
        # 2*4 + 1*6 + 1*6 = 20
        assert pytest.approx(result["td_points"].iloc[0]) == 20.0

    def test_interception_penalty(self):
        df = _make_qb_row(interceptions=2)
        result = compute_qb_targets(df)
        assert pytest.approx(result["interception_penalty"].iloc[0]) == -4.0

    def test_fumble_penalty(self):
        df = _make_qb_row(sack_fumbles_lost=1, rushing_fumbles_lost=1)
        result = compute_qb_targets(df)
        assert pytest.approx(result["fumble_penalty"].iloc[0]) == -4.0

    def test_receiving_component(self):
        """QBs can occasionally catch passes (trick plays)."""
        df = _make_qb_row(receptions=1, receiving_yards=20)
        result = compute_qb_targets(df)
        # 1 * 1 + 20 * 0.1 = 3.0
        assert pytest.approx(result["receiving_component"].iloc[0]) == 3.0

    def test_fantasy_points_decomposition_matches(self):
        df = _make_qb_row(
            passing_yards=275, passing_tds=2, rushing_yards=30, rushing_tds=1, interceptions=1
        )
        result = compute_qb_targets(df)
        expected = (
            result["passing_floor"].iloc[0]
            + result["rushing_floor"].iloc[0]
            + result["td_points"].iloc[0]
            + result["interception_penalty"].iloc[0]
            + result["fumble_penalty"].iloc[0]
            + result["receiving_component"].iloc[0]
        )
        assert pytest.approx(df["fantasy_points"].iloc[0], abs=0.01) == expected

    def test_all_nan_stats_treated_as_zero(self):
        """Player with all NaN stats should produce zero targets."""
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
        result = compute_qb_targets(df)
        assert result["passing_floor"].iloc[0] == 0.0
        assert result["rushing_floor"].iloc[0] == 0.0
        assert result["td_points"].iloc[0] == 0.0
        assert result["interception_penalty"].iloc[0] == 0.0
        assert result["fumble_penalty"].iloc[0] == 0.0

    def test_does_not_mutate_original(self):
        df = _make_qb_row()
        original_cols = set(df.columns)
        _ = compute_qb_targets(df)
        assert set(df.columns) == original_cols

    def test_zero_stat_game(self):
        """QB that didn't play (all zeros) should have zero targets."""
        df = _make_qb_row(
            passing_yards=0,
            rushing_yards=0,
            passing_tds=0,
            rushing_tds=0,
            interceptions=0,
        )
        result = compute_qb_targets(df)
        assert result["passing_floor"].iloc[0] == 0.0
        assert result["rushing_floor"].iloc[0] == 0.0
        assert result["td_points"].iloc[0] == 0.0

    def test_big_passing_game(self):
        """Huge passing game should decompose correctly."""
        df = _make_qb_row(
            passing_yards=500, passing_tds=5, rushing_yards=10, rushing_tds=0, interceptions=0
        )
        result = compute_qb_targets(df)
        assert result["passing_floor"].iloc[0] == 20.0  # 500 * 0.04
        assert result["rushing_floor"].iloc[0] == 1.0
        assert result["td_points"].iloc[0] == 20.0  # 5*4


# ---------------------------------------------------------------------------
# compute_qb_adjustment
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeQBAdjustment:
    def _make_adj_df(self, ints, fumbles=None, receptions=None, recv_yds=None, season=2023):
        n = len(ints)
        if fumbles is None:
            fumbles = [0] * n
        if receptions is None:
            receptions = [0] * n
        if recv_yds is None:
            recv_yds = [0] * n
        return pd.DataFrame(
            {
                "player_id": ["QB1"] * n,
                "season": [season] * n,
                "week": list(range(1, n + 1)),
                "interceptions": ints,
                "sack_fumbles_lost": fumbles,
                "rushing_fumbles_lost": [0] * n,
                "receiving_fumbles_lost": [0] * n,
                "receptions": receptions,
                "receiving_yards": recv_yds,
            }
        )

    def test_first_game_is_zero(self):
        """First game has no prior history — shift produces NaN, filled to 0."""
        df = self._make_adj_df([1, 0, 0])
        result = compute_qb_adjustment(df)
        assert result.iloc[0] == 0.0

    def test_second_game_uses_first(self):
        """Second game sees the first game's stats via rolling L8."""
        df = self._make_adj_df([1, 0, 0, 0])
        result = compute_qb_adjustment(df)
        # Game 2: prior INT rate = 1 → -2 pts (from interceptions only)
        assert pytest.approx(result.iloc[1]) == -2.0

    def test_rolling_window_ints(self):
        """Rolling mean averages correctly."""
        df = self._make_adj_df([1, 1, 0, 0, 0, 0, 0, 0, 0])
        result = compute_qb_adjustment(df)
        # Game 9 (index 8): rolling(8) of [1,1,0,0,0,0,0,0] * -2 mean = -2*0.25 = -0.5
        assert pytest.approx(result.iloc[8]) == -0.5

    def test_fumbles_included(self):
        """Fumble adjustment should be included."""
        df = self._make_adj_df([0, 0], fumbles=[1, 0])
        result = compute_qb_adjustment(df)
        # Game 2: fumble rate = 1 → -2 pts
        assert pytest.approx(result.iloc[1]) == -2.0

    def test_receiving_component_included(self):
        """Receiving component (historical) should be added to adjustment."""
        df = self._make_adj_df([0, 0], receptions=[2, 0], recv_yds=[30, 0])
        result = compute_qb_adjustment(df)
        # Game 2: prior rec = 2*1 + 30*0.1 = 5 pts
        assert pytest.approx(result.iloc[1]) == 5.0

    def test_player_with_clean_record(self):
        df = self._make_adj_df([0, 0, 0, 0])
        result = compute_qb_adjustment(df)
        for i in range(len(df)):
            assert result.iloc[i] == 0.0

    def test_multiple_players_independent(self):
        """Each player's history is independent."""
        df = pd.DataFrame(
            {
                "player_id": ["QB1", "QB1", "QB2", "QB2"],
                "season": [2023, 2023, 2023, 2023],
                "week": [1, 2, 1, 2],
                "interceptions": [2, 0, 0, 0],
                "sack_fumbles_lost": [0, 0, 0, 0],
                "rushing_fumbles_lost": [0, 0, 0, 0],
                "receiving_fumbles_lost": [0, 0, 0, 0],
                "receptions": [0, 0, 0, 0],
                "receiving_yards": [0, 0, 0, 0],
            }
        )
        result = compute_qb_adjustment(df)
        # QB1 game 2: prior int=2 → -4 pts
        assert pytest.approx(result.iloc[1]) == -4.0
        # QB2 game 2: no INTs → 0
        assert pytest.approx(result.iloc[3]) == 0.0

    def test_multiple_seasons_reset(self):
        """History should reset across seasons (grouped by season)."""
        df = pd.DataFrame(
            {
                "player_id": ["QB1", "QB1", "QB1", "QB1"],
                "season": [2022, 2022, 2023, 2023],
                "week": [1, 2, 1, 2],
                "interceptions": [2, 2, 0, 0],
                "sack_fumbles_lost": [0, 0, 0, 0],
                "rushing_fumbles_lost": [0, 0, 0, 0],
                "receiving_fumbles_lost": [0, 0, 0, 0],
                "receptions": [0, 0, 0, 0],
                "receiving_yards": [0, 0, 0, 0],
            }
        )
        result = compute_qb_adjustment(df)
        # 2023 game 1: first game of new season → 0
        assert result.iloc[2] == 0.0
        # 2023 game 2: only sees 2023 week 1 (0 INTs) → 0
        assert pytest.approx(result.iloc[3]) == 0.0
