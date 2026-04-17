"""Test fixtures for shared/ tests.

Also hosts the session-scoped base fixtures that the new E2E suite reuses
across every position:

- ``tiny_synthetic_games`` — a 50-player x 2-season x 17-week weekly-game
  DataFrame with the columns ``filter_to_{pos}()`` expects (position,
  player_id, season, week, recent_team, opponent_team, and the core fantasy
  stats).
- ``tiny_model_artifact`` — a pre-trained Ridge model + scaler saved once
  per session; the fixture returns the directory path.
- ``frozen_rng`` — seeds numpy, torch, python ``random`` and sets
  ``PYTHONHASHSEED`` so hash-ordered containers are deterministic.

Project root is still put on ``sys.path`` here for any consumer that invokes
``pytest shared/tests/...`` directly without running through the root
conftest.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Deterministic synthetic game DataFrame
# ---------------------------------------------------------------------------

# Column schema used by ``filter_to_{qb,rb,wr,te}`` and the downstream
# ``shared.pipeline`` feature-builder. Keep this minimal — only the columns
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
    "fantasy_points_floor",
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

        # Position-specific baseline fantasy output, jittered deterministically.
        base_fp = {"QB": 18.0, "RB": 11.0, "WR": 10.0, "TE": 7.0}[position]

        for season in seasons:
            for week in weeks:
                opp = _TEAMS[(pid + week + season) % len(_TEAMS)]
                if opp == team:
                    opp = _TEAMS[(_TEAMS.index(team) + 1) % len(_TEAMS)]
                noise = float(rng.normal(0.0, 3.0))
                fp = max(0.0, base_fp + noise)

                # Position-shaped stat lines. Zero-out irrelevant stats so
                # ``filter_to_{pos}`` callers can still compute totals.
                row = {
                    "player_id": player_id,
                    "player_name": name,
                    "position": position,
                    "recent_team": team,
                    "opponent_team": opp,
                    "season": int(season),
                    "week": int(week),
                    "fantasy_points": fp,
                    "fantasy_points_floor": fp * 0.8,
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
                    "receiving_air_yards": {"QB": 0.0, "RB": 10.0, "WR": 80.0, "TE": 40.0}[position],
                    "fumbles_lost": 0,
                    "snap_pct": 0.75,
                    "is_home": int(week % 2 == 0),
                }
                rows.append(row)

    df = pd.DataFrame(rows, columns=_SYNTHETIC_COLUMNS)
    return df


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
        try:
            import torch  # Local import keeps import-time cost out of collection.
        except ImportError:
            pass
        else:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        return seed

    yield _seed

    if saved_hashseed is None:
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        os.environ["PYTHONHASHSEED"] = saved_hashseed
