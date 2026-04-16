"""Tests for TE.te_targets — compute_te_targets and compute_te_fumble_adjustment."""

import numpy as np
import pandas as pd
import pytest

from TE.te_targets import compute_te_targets, compute_te_fumble_adjustment


def _make_te_row(**overrides):
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
# compute_te_targets
# ---------------------------------------------------------------------------

class TestComputeTETargets:
    def test_receiving_floor(self):
        df = _make_te_row(receptions=5, receiving_yards=50)
        result = compute_te_targets(df)
        # 5 * 1.0 + 50 * 0.1 = 10.0
        assert pytest.approx(result["receiving_floor"].iloc[0]) == 10.0

    def test_rushing_floor(self):
        """TEs occasionally rush — should still be computed."""
        df = _make_te_row(rushing_yards=15)
        result = compute_te_targets(df)
        assert pytest.approx(result["rushing_floor"].iloc[0]) == 1.5

    def test_td_points_receiving(self):
        df = _make_te_row(receiving_tds=2, rushing_tds=0)
        result = compute_te_targets(df)
        assert pytest.approx(result["td_points"].iloc[0]) == 12.0

    def test_td_points_rushing(self):
        """TE tackle-eligible TD (rare but happens)."""
        df = _make_te_row(receiving_tds=0, rushing_tds=1)
        result = compute_te_targets(df)
        assert pytest.approx(result["td_points"].iloc[0]) == 6.0

    def test_fumble_penalty(self):
        df = _make_te_row(receiving_fumbles_lost=1, sack_fumbles_lost=1)
        result = compute_te_targets(df)
        assert pytest.approx(result["fumble_penalty"].iloc[0]) == -4.0

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
        result = compute_te_targets(df)
        assert result["receiving_floor"].iloc[0] == 0.0
        assert result["rushing_floor"].iloc[0] == 0.0
        assert result["td_points"].iloc[0] == 0.0
        assert result["fumble_penalty"].iloc[0] == 0.0

    def test_does_not_mutate_original(self):
        df = _make_te_row()
        original_cols = set(df.columns)
        _ = compute_te_targets(df)
        assert set(df.columns) == original_cols

    def test_zero_catch_game(self):
        """TE with 0 catches on blocking-heavy game."""
        df = _make_te_row(receptions=0, receiving_yards=0, receiving_tds=0,
                          rushing_tds=0, rushing_yards=0)
        result = compute_te_targets(df)
        assert result["receiving_floor"].iloc[0] == 0.0
        assert result["td_points"].iloc[0] == 0.0

    def test_big_te_game(self):
        df = _make_te_row(receptions=8, receiving_yards=120, receiving_tds=2)
        result = compute_te_targets(df)
        # 8 + 12 = 20
        assert result["receiving_floor"].iloc[0] == 20.0
        assert result["td_points"].iloc[0] == 12.0


# ---------------------------------------------------------------------------
# compute_te_fumble_adjustment
# ---------------------------------------------------------------------------

class TestComputeTEFumbleAdjustment:
    def _make_fumble_df(self, fumbles, season=2023):
        n = len(fumbles)
        return pd.DataFrame({
            "player_id": ["T1"] * n,
            "season": [season] * n,
            "week": list(range(1, n + 1)),
            "sack_fumbles_lost": [0] * n,
            "rushing_fumbles_lost": [0] * n,
            "receiving_fumbles_lost": fumbles,
        })

    def test_first_game_is_zero(self):
        df = self._make_fumble_df([1, 0, 0])
        result = compute_te_fumble_adjustment(df)
        assert result.iloc[0] == 0.0

    def test_second_game_uses_first(self):
        df = self._make_fumble_df([1, 0, 0, 0])
        result = compute_te_fumble_adjustment(df)
        assert pytest.approx(result.iloc[1]) == -2.0

    def test_rolling_window(self):
        df = self._make_fumble_df([1, 1, 0, 0, 0, 0, 0, 0, 0])
        result = compute_te_fumble_adjustment(df)
        assert pytest.approx(result.iloc[8]) == -0.5

    def test_player_with_no_fumbles(self):
        df = self._make_fumble_df([0, 0, 0, 0])
        result = compute_te_fumble_adjustment(df)
        assert pytest.approx(result.iloc[1]) == 0.0
        assert pytest.approx(result.iloc[3]) == 0.0

    def test_multiple_players_independent(self):
        df = pd.DataFrame({
            "player_id": ["T1", "T1", "T2", "T2"],
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 2, 1, 2],
            "sack_fumbles_lost": [0, 0, 0, 0],
            "rushing_fumbles_lost": [0, 0, 0, 0],
            "receiving_fumbles_lost": [1, 0, 0, 0],
        })
        result = compute_te_fumble_adjustment(df)
        assert pytest.approx(result.iloc[1]) == -2.0
        assert pytest.approx(result.iloc[3]) == 0.0

    def test_multiple_seasons_reset(self):
        df = pd.DataFrame({
            "player_id": ["T1", "T1", "T1", "T1"],
            "season": [2022, 2022, 2023, 2023],
            "week": [1, 2, 1, 2],
            "sack_fumbles_lost": [0, 0, 0, 0],
            "rushing_fumbles_lost": [0, 0, 0, 0],
            "receiving_fumbles_lost": [1, 1, 0, 0],
        })
        result = compute_te_fumble_adjustment(df)
        assert result.iloc[2] == 0.0
        assert pytest.approx(result.iloc[3]) == 0.0
