"""Shared fixtures for DST tests.

The standard make_sim_df / make_test_df / make_tensors / make_splits fixtures
get installed by ``register_standard_fixtures`` (see
``tests/shared/position_fixtures.py``); DST is team-level so ``id_prefix``
is ``"TEAM"`` (each ``player_id`` is a team code). DST-specific helpers
for single-row target rows, multi-week team frames, and the tiny synthetic
pipeline dataset stay here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dst.config import POSITION_CONFIG
from tests.shared.position_fixtures import (
    register_position_markers,
    register_standard_fixtures,
)

# DST scoring scale: team defenses typically score 5-15 fantasy pts/week.
SCORING_SCALE = 15


def pytest_configure(config):
    register_position_markers(config)


register_standard_fixtures(
    globals(),
    scoring_scale=SCORING_SCALE,
    id_prefix="TEAM",
    targets=POSITION_CONFIG.targets,
    stat_col="passing_yards",  # DST tests don't use make_position_df
    rng_kind="legacy",
)


# ---------------------------------------------------------------------------
# DST-specific fixtures (not generalisable across positions)
# ---------------------------------------------------------------------------


def _build_dst_row(**overrides) -> pd.DataFrame:
    """Single-row DST DataFrame with sensible defaults; safe to mutate in-test."""
    row = {
        "def_sacks": 3,
        "def_ints": 1,
        "def_fumble_rec": 1,
        "def_fumbles_forced": 1,
        "def_blocked_kicks": 0,
        "points_allowed": 17,
        "yards_allowed": 330,
        "special_teams_tds": 0,
        "def_tds": 0,
        "def_safeties": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


@pytest.fixture(scope="session")
def make_df():
    """Factory: build a single-row DST DataFrame with sensible defaults."""
    return _build_dst_row


@pytest.fixture(scope="session")
def make_team_games():
    """Factory: multi-week DST DataFrame for one team.

    ``compute_features`` reads the raw-stat columns + ``fantasy_points``
    for rolling aggregates. ``fantasy_points`` is populated with a placeholder
    linear sum (no PA/YA tier bonus) since the tests that use this fixture
    care about rolling windows, not exact tier scoring.
    """

    def _factory(
        team: str = "KC",
        n_weeks: int = 6,
        season: int = 2023,
        def_sacks: int = 3,
        def_ints: int = 1,
        def_fumble_rec: int = 1,
        def_fumbles_forced: int = 1,
        def_blocked_kicks: int = 0,
        def_tds: int = 0,
        def_safeties: int = 0,
        points_allowed: int = 17,
        yards_allowed: int = 330,
        special_teams_tds: int = 0,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "team": [team] * n_weeks,
                "season": [season] * n_weeks,
                "week": list(range(1, n_weeks + 1)),
                "def_sacks": [def_sacks] * n_weeks,
                "def_ints": [def_ints] * n_weeks,
                "def_fumble_rec": [def_fumble_rec] * n_weeks,
                "def_fumbles_forced": [def_fumbles_forced] * n_weeks,
                "def_blocked_kicks": [def_blocked_kicks] * n_weeks,
                "def_tds": [def_tds] * n_weeks,
                "def_safeties": [def_safeties] * n_weeks,
                "points_allowed": [points_allowed] * n_weeks,
                "yards_allowed": [yards_allowed] * n_weeks,
                "special_teams_tds": [special_teams_tds] * n_weeks,
            }
        )
        # Placeholder fantasy_points (linear-only) — real tier scoring happens
        # in compute_targets. This column feeds rolling-feature windows.
        df["fantasy_points"] = (
            df["def_sacks"]
            + df["def_ints"] * 2
            + df["def_fumble_rec"] * 2
            + df["def_fumbles_forced"]
            + df["def_safeties"] * 2
            + df["def_tds"] * 6
            + df["special_teams_tds"] * 6
            + df["def_blocked_kicks"] * 2
        )
        return df

    return _factory


# ---------------------------------------------------------------------------
# Synthetic pipeline dataset — used by E2E and regression tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_dataset():
    """Deterministic tiny DST dataset for E2E / regression tests.

    32 teams x 4 seasons x 17 weeks = 2176 team-week rows.  Targets,
    features, and the min_games-per-season filter will reduce this to a
    few hundred trainable rows — plenty for a 1-epoch smoke test, tiny
    enough to finish in < 20s.
    """
    return _build_tiny_dataset(seed=42)


def _build_tiny_dataset(seed: int = 42) -> pd.DataFrame:
    """Build the synthetic DST dataset.  Separated so the E2E reproducibility
    test can rebuild it without touching fixture caching."""
    rng = np.random.RandomState(seed)

    # Standard 32-team league; 2 training seasons + 1 val + 1 test.
    teams = [
        "KC",
        "SF",
        "BUF",
        "DAL",
        "PHI",
        "BAL",
        "MIA",
        "CIN",
        "JAX",
        "NE",
        "NYG",
        "NYJ",
        "CLE",
        "PIT",
        "HOU",
        "IND",
        "TEN",
        "DEN",
        "LAC",
        "LV",
        "DET",
        "CHI",
        "MIN",
        "GB",
        "ATL",
        "CAR",
        "NO",
        "TB",
        "WAS",
        "SEA",
        "ARI",
        "LAR",
    ]
    seasons = [2022, 2023, 2024, 2025]
    weeks = list(range(1, 18))  # 17-week regular season

    rows = []
    for season in seasons:
        for team in teams:
            # Opponents: sample without replacement per team, mod to 17
            opp_pool = [t for t in teams if t != team]
            rng.shuffle(opp_pool)
            for i, week in enumerate(weeks):
                opponent = opp_pool[i % len(opp_pool)]
                rows.append(
                    {
                        "team": team,
                        "season": season,
                        "week": week,
                        "opponent_team": opponent,
                        # Defensive stats — integer-valued, Poisson-ish
                        "def_sacks": int(rng.poisson(2.5)),
                        "def_ints": int(rng.poisson(0.8)),
                        "def_fumble_rec": int(rng.poisson(0.6)),
                        "def_fumbles_forced": int(rng.poisson(1.0)),
                        "def_blocked_kicks": int(rng.binomial(1, 0.1)),
                        "def_tds": int(rng.binomial(1, 0.05)),
                        "def_safeties": int(rng.binomial(1, 0.02)),
                        "special_teams_tds": int(rng.binomial(1, 0.04)),
                        # Points allowed — Poisson centered at 22
                        "points_allowed": int(max(0, rng.poisson(22))),
                        # Yards allowed — Poisson centered at 350
                        "yards_allowed": int(max(0, rng.poisson(350))),
                        # Context
                        "is_home": int(i % 2 == 0),
                        "is_dome": int(rng.binomial(1, 0.35)),
                        "rest_days": int(rng.choice([6, 7, 7, 7, 10])),
                        "div_game": int(rng.binomial(1, 0.38)),
                        "spread_line": float(rng.normal(0, 5)),
                        "total_line": float(rng.normal(45, 4)),
                        # Opponent-offense rolling signals (rough substitutes)
                        "opp_scoring_L3": float(rng.normal(22, 5)),
                        "opp_scoring_L5": float(rng.normal(22, 4)),
                        "opp_turnovers_L5": float(rng.normal(1.4, 0.4)),
                        "opp_sacks_allowed_L5": float(rng.normal(2.5, 0.7)),
                        "opp_qb_epa_L5": float(rng.normal(0, 0.2)),
                        "opp_qb_int_rate_L5": float(rng.normal(0.025, 0.015)),
                        "opp_qb_sack_rate_L5": float(rng.normal(0.07, 0.02)),
                        "opp_qb_rush_yds_L5": float(rng.normal(15, 10)),
                        # Per-game opp stats (attention history sequence)
                        "opp_scoring": float(max(0, rng.normal(22, 10))),
                        "opp_fumbles": float(max(0, rng.poisson(0.7))),
                        "opp_interceptions": float(max(0, rng.poisson(0.8))),
                        "opp_qb_epa": float(rng.normal(0, 5)),
                    }
                )

    df = pd.DataFrame(rows)
    # Pipeline-compatible extras (mimic build_data's post-processing)
    df["player_id"] = df["team"]
    df["player_display_name"] = df["team"] + " D/ST"
    df["player_name"] = df["team"]
    df["recent_team"] = df["team"]
    df["position"] = "DST"
    df["headshot_url"] = ""
    df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)
    return df
