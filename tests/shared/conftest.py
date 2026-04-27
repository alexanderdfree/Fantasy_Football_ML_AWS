"""Test fixtures for tests/shared/.

Hosts the session-scoped base fixtures reused across the new E2E suite:

- ``tiny_synthetic_games`` — 50-player x 2-season x 17-week weekly-game DataFrame
  with the columns ``filter_to_{pos}()`` expects.
- ``tiny_model_artifact`` — pre-trained Ridge + scaler saved once per session.
- ``frozen_rng`` — seeds numpy, torch, python ``random``, and ``PYTHONHASHSEED``.

Also hosts the fixtures promoted from the duplicated helpers previously inlined
across ``tests/shared/test_*.py`` (error-analysis, weather, training data
factories). Project-root sys.path wiring and pytest-marker registration live in
the repo-root ``conftest.py`` so this file doesn't duplicate them.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Deterministic synthetic game DataFrame
# ---------------------------------------------------------------------------

# Column schema used by ``filter_to_{qb,rb,wr,te}`` and the downstream
# ``src.shared.pipeline`` feature-builder. Keep this minimal — only the columns
# the filters + engineer actually read.
_SYNTHETIC_COLUMNS = [
    "player_id",
    "player_name",
    "position",
    "recent_team",
    "opponent_team",
    "season",
    "week",
    "fantasy_points",
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "fumbles_lost",
    "snap_pct",
    "is_home",
]

_POSITIONS = ("QB", "RB", "WR", "TE")
_TEAMS = ("KC", "SF", "BUF", "PHI", "DAL", "GB", "MIA", "CIN")


def _build_tiny_synthetic_games(seed: int = 42) -> pd.DataFrame:
    """Generate 50 players x 2 seasons x 17 weeks of deterministic weekly records."""
    rng = np.random.default_rng(seed)
    n_players = 50
    seasons = (2022, 2023)
    weeks = tuple(range(1, 18))

    rows: list[dict] = []
    for pid in range(n_players):
        position = _POSITIONS[pid % len(_POSITIONS)]
        team = _TEAMS[pid % len(_TEAMS)]
        player_id = f"P{pid:03d}"
        name = f"Player {pid}"

        base_fp = {"QB": 18.0, "RB": 11.0, "WR": 10.0, "TE": 7.0}[position]

        for season in seasons:
            for week in weeks:
                opp = _TEAMS[(pid + week + season) % len(_TEAMS)]
                if opp == team:
                    opp = _TEAMS[(_TEAMS.index(team) + 1) % len(_TEAMS)]
                noise = float(rng.normal(0.0, 3.0))
                fp = max(0.0, base_fp + noise)

                row = {
                    "player_id": player_id,
                    "player_name": name,
                    "position": position,
                    "recent_team": team,
                    "opponent_team": opp,
                    "season": int(season),
                    "week": int(week),
                    "fantasy_points": fp,
                    "attempts": 32 if position == "QB" else 0,
                    "completions": 21 if position == "QB" else 0,
                    "passing_yards": 240.0 if position == "QB" else 0.0,
                    "passing_tds": 2 if position == "QB" else 0,
                    "interceptions": 1 if position == "QB" else 0,
                    "sacks": 2 if position == "QB" else 0,
                    "carries": 15 if position == "RB" else 0,
                    "rushing_yards": 65.0 if position == "RB" else 0.0,
                    "rushing_tds": 1 if position == "RB" else 0,
                    "targets": {"QB": 0, "RB": 3, "WR": 8, "TE": 5}[position],
                    "receptions": {"QB": 0, "RB": 2, "WR": 5, "TE": 3}[position],
                    "receiving_yards": {"QB": 0.0, "RB": 18.0, "WR": 65.0, "TE": 35.0}[position],
                    "receiving_tds": 1 if position in {"WR", "TE"} else 0,
                    "receiving_air_yards": {"QB": 0.0, "RB": 10.0, "WR": 80.0, "TE": 40.0}[
                        position
                    ],
                    "fumbles_lost": 0,
                    "snap_pct": 0.75,
                    "is_home": int(week % 2 == 0),
                }
                rows.append(row)

    return pd.DataFrame(rows, columns=_SYNTHETIC_COLUMNS)


@pytest.fixture(scope="session")
def tiny_synthetic_games() -> pd.DataFrame:
    """Session-scoped 50x2x17 synthetic weekly game DataFrame (seed=42)."""
    return _build_tiny_synthetic_games(seed=42)


# ---------------------------------------------------------------------------
# Tiny pre-trained Ridge artifact
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_model_artifact(tmp_path_factory) -> Path:
    """A pre-trained tiny Ridge model saved once per session.

    Returns the directory where ``scaler.pkl`` and ``ridge_model.pkl`` live
    (the same layout ``RidgeModel.save()`` writes).
    """
    from src.models.linear import RidgeModel

    rng = np.random.default_rng(42)
    X = rng.normal(size=(128, 6)).astype(np.float64)
    coef = np.array([0.4, -0.3, 0.2, 0.1, 0.05, -0.15])
    y = X @ coef + rng.normal(scale=0.1, size=X.shape[0])

    model = RidgeModel(alpha=1.0)
    model.fit(X, y)

    model_dir = tmp_path_factory.mktemp("tiny_model")
    model.save(model_dir=str(model_dir))
    return model_dir


# ---------------------------------------------------------------------------
# Frozen RNG context
# ---------------------------------------------------------------------------


@pytest.fixture
def frozen_rng():
    """Seed every RNG we care about, yield, then restore.

    Usage::

        def test_something(frozen_rng):
            frozen_rng(seed=42)
            ...  # numpy/torch/random/PYTHONHASHSEED are all deterministic
    """
    saved_hashseed = os.environ.get("PYTHONHASHSEED")

    def _seed(seed: int = 42) -> int:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    yield _seed

    if saved_hashseed is None:
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        os.environ["PYTHONHASHSEED"] = saved_hashseed


# ---------------------------------------------------------------------------
# Shared fixtures — error_analysis / stratification
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def error_df_factory():
    """Factory for synthetic error-analysis DataFrames.

    Returns a callable that builds a fresh copy so tests can mutate safely.
    Uses a fixed seed so rows are deterministic.
    """

    def _make(n=100):
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "player_id": [f"P{i}" for i in range(n)],
                "week": rng.integers(1, 18, size=n),
                "snap_pct": rng.random(n) * 100,
                "opp_def_rank_vs_pos": rng.integers(1, 33, size=n),
                "is_home": rng.choice([0, 1], size=n),
                "rushing_tds": rng.choice([0.0, 1.0, 2.0], size=n, p=[0.6, 0.3, 0.1]),
                "rolling_std_fantasy_points_L3": rng.random(n) * 5,
                "fantasy_points": rng.random(n) * 20,
                "pred_total": rng.random(n) * 20,
            }
        )

    return _make


# ---------------------------------------------------------------------------
# Shared fixtures — weather features
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fake_schedules():
    """Minimal schedule data covering teams/seasons used across tests.

    Session-scoped: same DataFrame is reused — tests must not mutate it.
    """
    rows = []
    for season in [2022, 2023]:
        for week in range(1, 19):
            rows.append(
                {
                    "game_type": "REG",
                    "season": season,
                    "week": week,
                    "away_team": "SF",
                    "home_team": "KC",
                    "home_score": 24,
                    "away_score": 17,
                    "spread_line": -3.0,
                    "total_line": 47.0,
                    "roof": "outdoors",
                    "surface": "grass",
                    "temp": 72,
                    "wind": 8,
                    "home_rest": 7,
                    "away_rest": 7,
                    "div_game": 0,
                }
            )
            rows.append(
                {
                    "game_type": "REG",
                    "season": season,
                    "week": week,
                    "away_team": "DAL",
                    "home_team": "NO",
                    "home_score": 21,
                    "away_score": 20,
                    "spread_line": -1.0,
                    "total_line": 50.0,
                    "roof": "dome",
                    "surface": "a_turf",
                    "temp": 72,
                    "wind": 0,
                    "home_rest": 7,
                    "away_rest": 7,
                    "div_game": 0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def player_df_factory():
    """Factory for small player DataFrames used in merge tests."""

    def _make(team="KC", n_weeks=4, season=2023):
        return pd.DataFrame(
            {
                "player_id": ["P1"] * n_weeks,
                "season": [season] * n_weeks,
                "week": list(range(1, n_weeks + 1)),
                "recent_team": [team] * n_weeks,
                "fantasy_points": [10.0] * n_weeks,
            }
        )

    return _make


# ---------------------------------------------------------------------------
# Shared fixtures — training (history dataset / collate / dataloaders)
# ---------------------------------------------------------------------------

# Legacy default — retained for backward compatibility with shared tests that
# exercise MultiTargetLoss / MultiHeadNetWithHistory against the pre-migration
# fantasy-point-component target schema. New shared tests should prefer
# ``TARGETS_RB_RAW`` and the ``tiny_synthetic_rb_raw`` fixture below.
TARGETS_DEFAULT = ["rushing_yards", "receiving_yards", "rushing_tds"]

# Raw-stat RB target list (matches ``RB_TARGETS`` in ``RB/rb_config.py`` after
# the target migration). Use this for new shared tests that want to mirror the
# production target schema without depending on per-position config imports.
TARGETS_RB_RAW = [
    "rushing_tds",
    "receiving_tds",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "fumbles_lost",
]


@pytest.fixture
def history_batch_factory():
    """Factory producing a list of (static, history, targets) tuples.

    Used by collate and dataloader tests. Function-scoped because the
    returned tensors can be consumed/moved by callers.
    """

    def _make(seq_lens, static_dim=4, game_dim=3):
        batch = []
        for slen in seq_lens:
            static = torch.randn(static_dim)
            history = torch.randn(slen, game_dim)
            targets = {"t1": torch.tensor(1.0)}
            batch.append((static, history, targets))
        return batch

    return _make


@pytest.fixture
def history_data_factory():
    """Factory producing synthetic (X_static, X_history, y_dict) training data.

    Uses the global numpy RNG so callers can seed externally via
    ``np.random.seed(...)`` before invoking the factory.
    """

    def _make(n, static_dim=5, game_dim=3, targets=TARGETS_DEFAULT):
        X_s = np.random.randn(n, static_dim).astype(np.float32)
        X_h = [
            np.random.randn(np.random.randint(1, 8), game_dim).astype(np.float32) for _ in range(n)
        ]
        y = {t: np.random.randn(n).astype(np.float32) for t in targets}
        y["total"] = sum(y[t] for t in targets)
        return X_s, X_h, y

    return _make


@pytest.fixture
def tiny_synthetic_rb_raw(history_data_factory):
    """Synthetic RB training triple with raw-stat targets (post-migration).

    Wraps ``history_data_factory`` pre-configured with ``TARGETS_RB_RAW`` so
    shared tests that want the new target schema can take this fixture
    directly instead of threading the target list through every call.
    """

    def _make(n, static_dim=5, game_dim=3):
        return history_data_factory(n, static_dim, game_dim, targets=TARGETS_RB_RAW)

    return _make
