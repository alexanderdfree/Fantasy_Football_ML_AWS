"""Tests for src.dst.targets — compute_targets and tier-bonus helpers.

D/ST targets are 10 raw NFL stat counts. Fantasy points are computed after
prediction by summing the linear scoring coefficients (sacks*1, INT*2, ...)
and applying the PA/YA tier bonuses.

Uses the session-scoped ``make_df`` fixture from conftest.py to avoid
duplicating single-row DST frame construction across every test.
"""

import numpy as np
import pandas as pd
import pytest

from src.dst.targets import (
    _pts_allowed_to_bonus,
    _yds_allowed_to_bonus,
    compute_targets,
)

# ---------------------------------------------------------------------------
# compute_targets
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeDSTTargets:
    def test_raw_counts_passed_through(self, make_df):
        """The 10 raw-stat columns pass through unchanged for non-NaN inputs."""
        df = make_df(
            def_sacks=4,
            def_ints=2,
            def_fumble_rec=1,
            def_fumbles_forced=3,
            def_safeties=1,
            def_tds=2,
            def_blocked_kicks=1,
            special_teams_tds=1,
        )
        result = compute_targets(df)
        assert result["def_sacks"].iloc[0] == 4
        assert result["def_ints"].iloc[0] == 2
        assert result["def_fumble_rec"].iloc[0] == 1
        assert result["def_fumbles_forced"].iloc[0] == 3
        assert result["def_safeties"].iloc[0] == 1
        assert result["def_tds"].iloc[0] == 2
        assert result["def_blocked_kicks"].iloc[0] == 1
        assert result["special_teams_tds"].iloc[0] == 1

    def test_points_allowed_copied_raw(self, make_df):
        """points_allowed is the raw value — tier mapping only affects fantasy_points."""
        df = make_df(points_allowed=24)
        result = compute_targets(df)
        assert pytest.approx(result["points_allowed"].iloc[0]) == 24.0

    def test_yards_allowed_copied_raw(self, make_df):
        """yards_allowed is the raw value."""
        df = make_df(yards_allowed=412)
        result = compute_targets(df)
        assert pytest.approx(result["yards_allowed"].iloc[0]) == 412.0

    def test_fantasy_points_full_sum(self, make_df):
        """fantasy_points = linear raw-stat combo + tier-mapped PA + tier-mapped YA."""
        df = make_df(
            def_sacks=3,
            def_ints=1,
            def_fumble_rec=1,
            def_fumbles_forced=2,
            def_safeties=0,
            def_tds=1,
            special_teams_tds=1,
            def_blocked_kicks=0,
            points_allowed=10,
            yards_allowed=220,
        )
        result = compute_targets(df)
        # linear = 3*1 + 1*2 + 1*2 + 2*1 + 0*2 + 1*6 + 1*6 + 0*2 = 21
        # PA tier (10 → 7-13) = +4
        # YA tier (220 → 200-299) = +2
        # total = 21 + 4 + 2 = 27
        assert pytest.approx(result["fantasy_points"].iloc[0]) == 27.0

    def test_dominant_defense_game(self, make_df):
        """Dream D/ST game — shutout with extreme stats."""
        df = make_df(
            def_sacks=6,
            def_ints=3,
            def_fumble_rec=2,
            def_fumbles_forced=4,
            def_safeties=1,
            def_tds=2,
            special_teams_tds=1,
            def_blocked_kicks=1,
            points_allowed=0,
            yards_allowed=80,
        )
        result = compute_targets(df)
        # linear = 6 + 6 + 4 + 4 + 2 + 12 + 6 + 2 = 42
        # PA tier (0 → +10), YA tier (<100 → +5)
        # total = 42 + 10 + 5 = 57
        assert pytest.approx(result["fantasy_points"].iloc[0]) == 57.0

    def test_all_nan_stats_fall_back_to_defaults(self):
        """NaN stats → zero; PA defaults to 21 (0 bonus), YA defaults to 350 (-1 bonus)."""
        df = pd.DataFrame(
            [
                {
                    "def_sacks": np.nan,
                    "def_ints": np.nan,
                    "def_fumble_rec": np.nan,
                    "def_fumbles_forced": np.nan,
                    "def_safeties": np.nan,
                    "def_tds": np.nan,
                    "special_teams_tds": np.nan,
                    "def_blocked_kicks": np.nan,
                    "points_allowed": np.nan,
                    "yards_allowed": np.nan,
                }
            ]
        )
        result = compute_targets(df)
        for col in [
            "def_sacks",
            "def_ints",
            "def_fumble_rec",
            "def_fumbles_forced",
            "def_safeties",
            "def_tds",
            "def_blocked_kicks",
            "special_teams_tds",
        ]:
            assert result[col].iloc[0] == 0.0
        assert result["points_allowed"].iloc[0] == 21.0
        assert result["yards_allowed"].iloc[0] == 350.0
        # fantasy_points: linear=0, PA(21)=0, YA(350)=-1 = -1
        assert pytest.approx(result["fantasy_points"].iloc[0]) == -1.0

    def test_does_not_mutate_original(self, make_df):
        df = make_df()
        original_cols = set(df.columns)
        _ = compute_targets(df)
        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# _yds_allowed_to_bonus — tier boundary sweep
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestYdsAllowedToBonus:
    """Exhaustive tier-boundary sweep for the yards-allowed lookup."""

    @pytest.mark.parametrize(
        "ya,expected",
        [
            (0, 5),
            (99, 5),
            (100, 3),
            (150, 3),
            (199, 3),
            (200, 2),
            (250, 2),
            (299, 2),
            (300, 0),
            (325, 0),
            (349, 0),
            (350, -1),
            (375, -1),
            (399, -1),
            (400, -3),
            (425, -3),
            (449, -3),
            (450, -5),
            (500, -5),
            (600, -5),
            (800, -5),
        ],
    )
    def test_tier_boundaries(self, ya, expected):
        assert _yds_allowed_to_bonus(ya) == expected


# ---------------------------------------------------------------------------
# _pts_allowed_to_bonus — keep coverage parallel to the new YA sweep
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPtsAllowedToBonus:
    @pytest.mark.parametrize(
        "pa,expected",
        [
            (0, 10),
            (1, 7),
            (6, 7),
            (7, 4),
            (13, 4),
            (14, 1),
            (20, 1),
            (21, 0),
            (27, 0),
            (28, -1),
            (34, -1),
            (35, -4),
            (55, -4),
        ],
    )
    def test_tier_boundaries(self, pa, expected):
        assert _pts_allowed_to_bonus(pa) == expected
