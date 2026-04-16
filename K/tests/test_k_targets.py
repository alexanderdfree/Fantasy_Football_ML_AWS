"""Tests for K.k_targets — compute_k_targets and compute_k_miss_adjustment."""

import numpy as np
import pandas as pd
import pytest

from K.k_targets import compute_k_targets, compute_k_miss_adjustment


def _make_k_row(**overrides):
    """Create a single-row kicker DataFrame with sensible defaults."""
    defaults = {
        "fg_made_0_19": 0,
        "fg_made_20_29": 1,
        "fg_made_30_39": 1,
        "fg_made_40_49": 1,
        "fg_made_50_59": 0,
        "fg_made_60_": 0,
        "fg_missed": 0,
        "pat_made": 3,
        "pat_missed": 0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# compute_k_targets
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeKTargets:
    def test_fg_points_short_range(self):
        """FGs under 40yds = 3 pts each."""
        df = _make_k_row(fg_made_0_19=1, fg_made_20_29=1, fg_made_30_39=1,
                         fg_made_40_49=0, fg_made_50_59=0, fg_made_60_=0)
        result = compute_k_targets(df)
        # 3 FGs * 3 = 9
        assert pytest.approx(result["fg_points"].iloc[0]) == 9.0

    def test_fg_points_mid_range(self):
        """FGs 40-49yds = 4 pts each."""
        df = _make_k_row(fg_made_0_19=0, fg_made_20_29=0, fg_made_30_39=0,
                         fg_made_40_49=2, fg_made_50_59=0, fg_made_60_=0)
        result = compute_k_targets(df)
        # 2 * 4 = 8
        assert pytest.approx(result["fg_points"].iloc[0]) == 8.0

    def test_fg_points_long_range(self):
        """FGs 50+yds = 5 pts each."""
        df = _make_k_row(fg_made_0_19=0, fg_made_20_29=0, fg_made_30_39=0,
                         fg_made_40_49=0, fg_made_50_59=1, fg_made_60_=1)
        result = compute_k_targets(df)
        # (1 + 1) * 5 = 10
        assert pytest.approx(result["fg_points"].iloc[0]) == 10.0

    def test_fg_points_mixed(self):
        """Combined FG distribution."""
        df = _make_k_row(fg_made_0_19=1, fg_made_20_29=1, fg_made_30_39=1,
                         fg_made_40_49=1, fg_made_50_59=1, fg_made_60_=0)
        result = compute_k_targets(df)
        # (1+1+1)*3 + 1*4 + 1*5 = 18
        assert pytest.approx(result["fg_points"].iloc[0]) == 18.0

    def test_pat_points(self):
        df = _make_k_row(pat_made=5)
        result = compute_k_targets(df)
        assert pytest.approx(result["pat_points"].iloc[0]) == 5.0

    def test_miss_penalty(self):
        df = _make_k_row(fg_missed=2, pat_missed=1)
        result = compute_k_targets(df)
        # -2 - 1 = -3
        assert pytest.approx(result["miss_penalty"].iloc[0]) == -3.0

    def test_fantasy_points_override(self):
        """compute_k_targets overrides fantasy_points with K-specific formula."""
        df = _make_k_row(
            fg_made_0_19=0, fg_made_20_29=1, fg_made_30_39=0,
            fg_made_40_49=0, fg_made_50_59=0, fg_made_60_=0,
            pat_made=2, fg_missed=1, pat_missed=0,
        )
        # expected: 1*3 (FG 20-29yd) + 2*1 (PAT) + (-1) (miss) = 4
        result = compute_k_targets(df)
        assert pytest.approx(result["fantasy_points"].iloc[0]) == 4.0

    def test_all_nan_treated_as_zero(self):
        df = pd.DataFrame([{
            "fg_made_0_19": np.nan,
            "fg_made_20_29": np.nan,
            "fg_made_30_39": np.nan,
            "fg_made_40_49": np.nan,
            "fg_made_50_59": np.nan,
            "fg_made_60_": np.nan,
            "fg_missed": np.nan,
            "pat_made": np.nan,
            "pat_missed": np.nan,
        }])
        result = compute_k_targets(df)
        assert result["fg_points"].iloc[0] == 0.0
        assert result["pat_points"].iloc[0] == 0.0
        assert result["miss_penalty"].iloc[0] == 0.0

    def test_does_not_mutate_original(self):
        df = _make_k_row()
        original_cols = set(df.columns)
        _ = compute_k_targets(df)
        assert set(df.columns) == original_cols

    def test_perfect_game(self):
        """Kicker's best possible ~7 FG / 5 PAT game."""
        df = _make_k_row(fg_made_0_19=0, fg_made_20_29=2, fg_made_30_39=2,
                         fg_made_40_49=2, fg_made_50_59=1, fg_made_60_=0,
                         pat_made=5, fg_missed=0, pat_missed=0)
        result = compute_k_targets(df)
        # (2+2)*3 + 2*4 + 1*5 = 25 fg_pts + 5 pat = 30 total
        assert pytest.approx(result["fg_points"].iloc[0]) == 25.0
        assert pytest.approx(result["pat_points"].iloc[0]) == 5.0
        assert pytest.approx(result["fantasy_points"].iloc[0]) == 30.0

    def test_missed_everything_game(self):
        """Rough game — missed FGs + PATs."""
        df = _make_k_row(fg_made_0_19=0, fg_made_20_29=0, fg_made_30_39=0,
                         fg_made_40_49=0, fg_made_50_59=0, fg_made_60_=0,
                         pat_made=0, fg_missed=3, pat_missed=1)
        result = compute_k_targets(df)
        assert pytest.approx(result["fg_points"].iloc[0]) == 0.0
        assert pytest.approx(result["pat_points"].iloc[0]) == 0.0
        # -3 - 1 = -4
        assert pytest.approx(result["miss_penalty"].iloc[0]) == -4.0


# ---------------------------------------------------------------------------
# compute_k_miss_adjustment
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeKMissAdjustment:
    def _make_miss_df(self, fg_missed, pat_missed=None, player_ids=None, seasons=None):
        n = len(fg_missed)
        if pat_missed is None:
            pat_missed = [0] * n
        if player_ids is None:
            player_ids = ["K1"] * n
        if seasons is None:
            seasons = [2023] * n
        return pd.DataFrame({
            "player_id": player_ids,
            "season": seasons,
            "week": list(range(1, n + 1)),
            "fg_missed": fg_missed,
            "pat_missed": pat_missed,
        })

    def test_first_game_is_zero(self):
        """First game has no prior history — shift produces NaN, filled to 0."""
        df = self._make_miss_df([1, 0, 0])
        result = compute_k_miss_adjustment(df)
        assert result.iloc[0] == 0.0

    def test_second_game_uses_first(self):
        """Second game sees first game's misses."""
        df = self._make_miss_df([1, 0, 0, 0])
        result = compute_k_miss_adjustment(df)
        # Prior avg miss = 1, penalty = 1 * -1 = -1
        assert pytest.approx(result.iloc[1]) == -1.0

    def test_rolling_window(self):
        """Rolling mean across 8-game window."""
        df = self._make_miss_df([1, 1, 0, 0, 0, 0, 0, 0, 0])
        result = compute_k_miss_adjustment(df)
        # Game 9: rolling(8) of [1,1,0,0,0,0,0,0] mean = 0.25, *-1 = -0.25
        assert pytest.approx(result.iloc[8]) == -0.25

    def test_combined_fg_and_pat_misses(self):
        """Both FG and PAT misses count."""
        df = self._make_miss_df([1, 0], pat_missed=[1, 0])
        result = compute_k_miss_adjustment(df)
        # Prior total misses = 2, penalty = -2
        assert pytest.approx(result.iloc[1]) == -2.0

    def test_kicker_with_no_misses(self):
        df = self._make_miss_df([0, 0, 0, 0])
        result = compute_k_miss_adjustment(df)
        for i in range(len(df)):
            assert result.iloc[i] == 0.0

    def test_multiple_kickers_independent(self):
        df = self._make_miss_df(
            [2, 0, 0, 0],
            player_ids=["K1", "K1", "K2", "K2"],
        )
        # Re-index weeks
        df["week"] = [1, 2, 1, 2]
        result = compute_k_miss_adjustment(df)
        # K1 game 2: prior miss = 2, penalty = -2
        assert pytest.approx(result.iloc[1]) == -2.0
        # K2 game 2: no misses yet, penalty = 0
        assert pytest.approx(result.iloc[3]) == 0.0

    def test_cross_season_rolling(self):
        """Kicker miss adjustment uses CROSS-SEASON rolling (no season reset).

        Kickers have stable careers and small sample sizes; cross-season
        windows provide more signal than single-season ones.
        """
        df = self._make_miss_df(
            [1, 1, 0, 0],
            seasons=[2022, 2022, 2023, 2023],
        )
        df["week"] = [1, 2, 1, 2]
        result = compute_k_miss_adjustment(df)
        # 2023 game 1: rolling(8) of [1, 1] prior misses (cross-season!) = 1.0, penalty = -1
        assert pytest.approx(result.iloc[2]) == -1.0
        # 2023 game 2: rolling(8) of [1, 1, 0] = 2/3, penalty = -2/3
        assert pytest.approx(result.iloc[3], abs=0.01) == -2 / 3
