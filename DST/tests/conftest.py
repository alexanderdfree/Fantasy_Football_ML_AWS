"""Shared fixtures and pytest configuration for DST tests.

Promotes the helpers that used to be duplicated across each
``DST/tests/test_dst_*.py`` file into session-scoped factory fixtures:

* ``make_df``        — single-row team-game frame with the DST target inputs
  (``def_sacks``, ``points_allowed``, etc.).  Replaces ``_make_dst_row``.
* ``make_team_games`` — multi-week frame for one team with pre-computed
  targets.  Replaces ``_make_team_games`` (kept name split so the intent is
  obvious — ``make_df`` = single-row, ``make_team_games`` = multi-week).
* ``make_splits``    — (train, val, test) single-column DataFrames for
  ``fill_dst_nans`` tests.
* ``make_sim_df``    — team-level weekly simulation frame (``player_id`` =
  team code) used by backtest/evaluation tests.  DST is team-level, so the
  scoring scale is 5–15 fantasy points/week (vs ~20 for QB).
* ``make_test_df``   — player-level ranking frame used by
  ``compute_ranking_metrics`` tests.
* ``make_tensors``   — per-target torch tensors for ``MultiTargetLoss``
  tests (includes the 'total' aux target).

The fixtures use the ``factory`` pattern — the fixture returns a callable,
so each test gets a fresh object with its own keyword arguments.  Scope is
``session`` because the callables themselves are stateless and cheap to
reuse; the DataFrames they produce are freshly materialised per call and
safe to mutate.

Also registers the ``unit`` / ``integration`` / ``e2e`` / ``regression``
markers so tests can be filtered (``pytest -m unit``) without triggering
``PytestUnknownMarkWarning``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path so `DST.*` and `shared.*` imports
# work when pytest is invoked from an arbitrary directory.  Mirrors the
# convention used by shared/tests/conftest.py and batch/tests/conftest.py.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# pytest markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register markers so `pytest -m unit` works without warnings."""
    config.addinivalue_line("markers", "unit: fast unit tests (<1s each)")
    config.addinivalue_line(
        "markers",
        "integration: multi-component tests that exercise shared modules together",
    )
    config.addinivalue_line(
        "markers",
        "e2e: full-pipeline smoke tests (run_pipeline end-to-end)",
    )
    config.addinivalue_line(
        "markers",
        "regression: model-quality threshold tests (MAE/R2 on a stable slice)",
    )


# ---------------------------------------------------------------------------
# Factory fixtures — DataFrame / tensor builders
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def make_df():
    """Factory: build a single-row DST DataFrame with sensible defaults.

    Mirrors the old ``_make_dst_row`` helper from test_dst_targets.py.
    Any keyword argument overrides the default for that field.  Returns
    a fresh DataFrame per call — safe to mutate in-test.
    """
    def _factory(**overrides) -> pd.DataFrame:
        row = {
            "def_sacks": 3,
            "def_ints": 1,
            "def_fumble_rec": 1,
            "points_allowed": 17,
            "special_teams_tds": 0,
            "def_tds": 0,
            "def_safeties": 0,
        }
        row.update(overrides)
        return pd.DataFrame([row])

    return _factory


@pytest.fixture(scope="session")
def make_team_games():
    """Factory: multi-week DST DataFrame for one team, with targets pre-computed.

    Replaces ``_make_team_games`` from test_dst_features.py.  Includes the
    three DST target columns (``defensive_scoring``, ``td_points``,
    ``pts_allowed_bonus``) because ``compute_dst_features`` depends on them
    for the rolling/EWMA features over total DST points.
    """
    def _factory(
        team: str = "KC",
        n_weeks: int = 6,
        season: int = 2023,
        def_sacks: int = 3,
        def_ints: int = 1,
        def_fumble_rec: int = 1,
        points_allowed: int = 17,
        special_teams_tds: int = 0,
    ) -> pd.DataFrame:
        df = pd.DataFrame({
            "team": [team] * n_weeks,
            "season": [season] * n_weeks,
            "week": list(range(1, n_weeks + 1)),
            "def_sacks": [def_sacks] * n_weeks,
            "def_ints": [def_ints] * n_weeks,
            "def_fumble_rec": [def_fumble_rec] * n_weeks,
            "points_allowed": [points_allowed] * n_weeks,
            "special_teams_tds": [special_teams_tds] * n_weeks,
        })
        df["defensive_scoring"] = (
            df["def_sacks"] + df["def_ints"] * 2 + df["def_fumble_rec"] * 2
        )
        df["td_points"] = df["special_teams_tds"] * 6
        df["pts_allowed_bonus"] = 1  # placeholder (real tiering in compute_dst_targets)
        return df

    return _factory


@pytest.fixture(scope="session")
def make_splits():
    """Factory: (train, val, test) single-column DataFrames for NaN-fill tests."""
    def _factory(train_vals, val_vals, test_vals, col: str = "feat1"):
        train = pd.DataFrame({col: train_vals})
        val = pd.DataFrame({col: val_vals})
        test = pd.DataFrame({col: test_vals})
        return train, val, test

    return _factory


@pytest.fixture(scope="session")
def make_sim_df():
    """Factory: team-level weekly simulation DataFrame for backtest tests.

    DST is team-level — ``player_id`` holds the team code (``TEAM1`` ..).
    Fantasy-point scale is 5–15 per week (team-level D/ST scoring), not
    the 20+ range that applies to QB/RB/WR.  The fixture injects two
    noisy prediction columns (``pred_ridge``, ``pred_nn``) to support
    multi-model backtest tests.
    """
    def _factory(n_weeks: int = 4, n_players: int = 15, seed: int = 42):
        rng = np.random.RandomState(seed)
        rows = []
        for week in range(1, n_weeks + 1):
            for pid in range(1, n_players + 1):
                fp = rng.rand() * 15  # team-level DST scale
                rows.append({
                    "week": week,
                    "player_id": f"TEAM{pid}",  # team code — DST player_id convention
                    "fantasy_points": fp,
                    "pred_ridge": fp + rng.randn() * 2,
                    "pred_nn": fp + rng.randn() * 3,
                })
        return pd.DataFrame(rows)

    return _factory


@pytest.fixture(scope="session")
def make_test_df():
    """Factory: player-level ranking DataFrame for compute_ranking_metrics tests."""
    def _factory(n_weeks: int = 3, n_players: int = 15, seed: int = 42):
        rng = np.random.RandomState(seed)
        rows = []
        for week in range(1, n_weeks + 1):
            for pid in range(1, n_players + 1):
                rows.append({
                    "week": week,
                    "player_id": f"TEAM{pid}",
                    "pred_total": rng.rand() * 15,
                    "fantasy_points": rng.rand() * 15,
                })
        return pd.DataFrame(rows)

    return _factory


@pytest.fixture(scope="session")
def make_tensors():
    """Factory: per-target torch tensors for MultiTargetLoss tests.

    Returns (preds, targets) — each a dict keyed by the three DST targets
    plus ``total`` (the aux-loss anchor).  Imports torch lazily so collecting
    tests that don't need tensors doesn't pay the torch-import cost.
    """
    import torch

    DST_TARGETS = ["defensive_scoring", "td_points", "pts_allowed_bonus"]

    def _factory(n: int = 10, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        preds = {t: torch.randn(n) for t in DST_TARGETS}
        preds["total"] = torch.randn(n)
        targets = {t: torch.randn(n) for t in DST_TARGETS}
        targets["total"] = torch.randn(n)
        return preds, targets

    return _factory


# ---------------------------------------------------------------------------
# Synthetic pipeline dataset — used by E2E and regression tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_dst_dataset():
    """Deterministic tiny DST dataset for E2E / regression tests.

    32 teams x 4 seasons x 17 weeks = 2176 team-week rows.  Targets,
    features, and the min_games-per-season filter will reduce this to a
    few hundred trainable rows — plenty for a 1-epoch smoke test, tiny
    enough to finish in < 20s.  Seed is pinned; regenerating the fixture
    yields identical frames.

    Shape contract:
    * Every NFL team plays all 17 regular-season weeks in every season.
    * ``team`` / ``recent_team`` / ``player_id`` all agree (DST is
      team-level; ``player_id`` = team abbr, per ``build_dst_data``).
    * ``fantasy_points`` is NOT added here — the pipeline derives it from
      ``compute_dst_targets`` after we patch the scraper.
    """
    return _build_tiny_dst_dataset(seed=42)


def _build_tiny_dst_dataset(seed: int = 42) -> pd.DataFrame:
    """Build the synthetic DST dataset.  Separated so the E2E reproducibility
    test can rebuild it without touching fixture caching."""
    rng = np.random.RandomState(seed)

    # Standard 32-team league; 2 training seasons + 1 val + 1 test.
    teams = [
        "KC", "SF", "BUF", "DAL", "PHI", "BAL", "MIA", "CIN",
        "JAX", "NE", "NYG", "NYJ", "CLE", "PIT", "HOU", "IND",
        "TEN", "DEN", "LAC", "LV", "DET", "CHI", "MIN", "GB",
        "ATL", "CAR", "NO", "TB", "WAS", "SEA", "ARI", "LAR",
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
                rows.append({
                    "team": team,
                    "season": season,
                    "week": week,
                    "opponent_team": opponent,
                    # Defensive stats — integer-valued, Poisson-ish
                    "def_sacks": int(rng.poisson(2.5)),
                    "def_ints": int(rng.poisson(0.8)),
                    "def_fumble_rec": int(rng.poisson(0.6)),
                    "def_tds": int(rng.binomial(1, 0.05)),
                    "def_safeties": int(rng.binomial(1, 0.02)),
                    "special_teams_tds": int(rng.binomial(1, 0.04)),
                    # Points allowed — Poisson centered at 22
                    "points_allowed": int(max(0, rng.poisson(22))),
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
                })

    df = pd.DataFrame(rows)
    # Pipeline-compatible extras (mimic build_dst_data's post-processing)
    df["player_id"] = df["team"]
    df["player_display_name"] = df["team"] + " D/ST"
    df["player_name"] = df["team"]
    df["recent_team"] = df["team"]
    df["position"] = "DST"
    df["headshot_url"] = ""
    df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)
    return df
