"""Tests for DST.dst_targets — compute_dst_targets and compute_dst_adjustment.

D/ST targets use team-level aggregated data. Special care is taken with
pts_allowed_bonus (tiered scoring) and the _dst_adjustment which contains
def_tds + safeties (excluded from trainable targets due to nflreadr data gap).

Uses the session-scoped ``make_df`` fixture from conftest.py to avoid
duplicating single-row DST frame construction across every test.
"""

import numpy as np
import pandas as pd
import pytest

from DST.dst_targets import compute_dst_adjustment, compute_dst_targets

# ---------------------------------------------------------------------------
# compute_dst_targets
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeDSTTargets:
    def test_defensive_scoring(self, make_df):
        """defensive_scoring = sacks * 1 + ints * 2 + fumble_rec * 2."""
        df = make_df(def_sacks=4, def_ints=2, def_fumble_rec=1)
        result = compute_dst_targets(df)
        # 4*1 + 2*2 + 1*2 = 10
        assert pytest.approx(result["defensive_scoring"].iloc[0]) == 10.0

    def test_td_points_special_teams_only(self, make_df):
        """td_points = special_teams_tds * 6 (NOT def_tds!)."""
        df = make_df(special_teams_tds=1, def_tds=1)
        result = compute_dst_targets(df)
        # Only ST TDs count — def_tds go to _dst_adjustment
        assert pytest.approx(result["td_points"].iloc[0]) == 6.0

    def test_pts_allowed_bonus_shutout(self, make_df):
        """0 points allowed → +10 bonus."""
        df = make_df(points_allowed=0)
        result = compute_dst_targets(df)
        assert pytest.approx(result["pts_allowed_bonus"].iloc[0]) == 10.0

    def test_pts_allowed_bonus_low(self, make_df):
        """1-6 points allowed → +7 bonus."""
        df = make_df(points_allowed=3)
        result = compute_dst_targets(df)
        assert pytest.approx(result["pts_allowed_bonus"].iloc[0]) == 7.0

    def test_pts_allowed_bonus_medium_low(self, make_df):
        """7-13 points → +4 bonus."""
        df = make_df(points_allowed=10)
        result = compute_dst_targets(df)
        assert pytest.approx(result["pts_allowed_bonus"].iloc[0]) == 4.0

    def test_pts_allowed_bonus_14_20(self, make_df):
        """14-20 points → +1 bonus."""
        df = make_df(points_allowed=17)
        result = compute_dst_targets(df)
        assert pytest.approx(result["pts_allowed_bonus"].iloc[0]) == 1.0

    def test_pts_allowed_bonus_21_27(self, make_df):
        """21-27 points → 0 bonus."""
        df = make_df(points_allowed=24)
        result = compute_dst_targets(df)
        assert pytest.approx(result["pts_allowed_bonus"].iloc[0]) == 0.0

    def test_pts_allowed_bonus_28_34(self, make_df):
        """28-34 points → -1 bonus."""
        df = make_df(points_allowed=30)
        result = compute_dst_targets(df)
        assert pytest.approx(result["pts_allowed_bonus"].iloc[0]) == -1.0

    def test_pts_allowed_bonus_35plus(self, make_df):
        """35+ points → -4 bonus (blowout loss)."""
        df = make_df(points_allowed=42)
        result = compute_dst_targets(df)
        assert pytest.approx(result["pts_allowed_bonus"].iloc[0]) == -4.0

    def test_dst_adjustment_contains_def_tds(self, make_df):
        """def_tds + safeties belong in adjustment, not targets."""
        df = make_df(def_tds=1, def_safeties=1)
        result = compute_dst_targets(df)
        # 1*6 + 1*2 = 8
        assert pytest.approx(result["_dst_adjustment"].iloc[0]) == 8.0

    def test_fantasy_points_sum_complete(self, make_df):
        """fantasy_points = targets + adjustment."""
        df = make_df(
            def_sacks=3,
            def_ints=1,
            def_fumble_rec=1,
            special_teams_tds=1,
            points_allowed=10,
            def_tds=1,
            def_safeties=0,
        )
        result = compute_dst_targets(df)
        # def_scoring: 3+2+2=7
        # td_points: 6
        # pts_allowed_bonus: 4 (10 pts in 7-13 tier)
        # adjustment: 6
        # total: 23
        assert pytest.approx(result["fantasy_points"].iloc[0]) == 23.0

    def test_all_nan_stats_treated_as_zero(self):
        """NaN stats become 0. pts_allowed defaults to 21 (middle tier → 0 bonus)."""
        df = pd.DataFrame(
            [
                {
                    "def_sacks": np.nan,
                    "def_ints": np.nan,
                    "def_fumble_rec": np.nan,
                    "points_allowed": np.nan,  # → 21 default → 0 bonus
                    "special_teams_tds": np.nan,
                    "def_tds": np.nan,
                    "def_safeties": np.nan,
                }
            ]
        )
        result = compute_dst_targets(df)
        assert result["defensive_scoring"].iloc[0] == 0.0
        assert result["td_points"].iloc[0] == 0.0
        # NaN → 21 default → 21-27 tier → 0
        assert pytest.approx(result["pts_allowed_bonus"].iloc[0]) == 0.0

    def test_does_not_mutate_original(self, make_df):
        df = make_df()
        original_cols = set(df.columns)
        _ = compute_dst_targets(df)
        assert set(df.columns) == original_cols

    def test_dominant_defense_game(self, make_df):
        """Dream D/ST game — shutout with multiple scores."""
        df = make_df(
            def_sacks=6,
            def_ints=3,
            def_fumble_rec=2,
            special_teams_tds=1,
            points_allowed=0,
        )
        result = compute_dst_targets(df)
        # 6 + 6 + 4 = 16
        assert result["defensive_scoring"].iloc[0] == 16.0
        # +10 shutout bonus
        assert result["pts_allowed_bonus"].iloc[0] == 10.0
        # +6 ST TD
        assert result["td_points"].iloc[0] == 6.0


# ---------------------------------------------------------------------------
# compute_dst_adjustment
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeDSTAdjustment:
    def test_returns_precomputed_column(self, make_df):
        """compute_dst_adjustment returns the _dst_adjustment column set by targets."""
        df = make_df(def_tds=1, def_safeties=0)
        df_with_targets = compute_dst_targets(df)
        adj = compute_dst_adjustment(df_with_targets)
        # def_tds * 6 = 6
        assert pytest.approx(adj.iloc[0]) == 6.0

    def test_without_precomputed_column(self, make_df):
        """If _dst_adjustment is missing, returns zeros."""
        df = make_df()
        # Call directly without compute_dst_targets first
        adj = compute_dst_adjustment(df)
        assert (adj == 0.0).all()
        assert len(adj) == len(df)

    def test_includes_safeties(self, make_df):
        """Safeties contribute 2 pts each to adjustment."""
        df = make_df(def_tds=0, def_safeties=2)
        df_with_targets = compute_dst_targets(df)
        adj = compute_dst_adjustment(df_with_targets)
        # 2 * 2 = 4
        assert pytest.approx(adj.iloc[0]) == 4.0

    def test_zero_when_no_defensive_tds_or_safeties(self, make_df):
        df = make_df(def_tds=0, def_safeties=0)
        df_with_targets = compute_dst_targets(df)
        adj = compute_dst_adjustment(df_with_targets)
        assert pytest.approx(adj.iloc[0]) == 0.0

    def test_combined_def_tds_and_safeties(self, make_df):
        df = make_df(def_tds=2, def_safeties=1)
        df_with_targets = compute_dst_targets(df)
        adj = compute_dst_adjustment(df_with_targets)
        # 2*6 + 1*2 = 14
        assert pytest.approx(adj.iloc[0]) == 14.0
