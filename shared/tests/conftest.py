"""Test fixtures and marker registration for shared/ tests.

Ensures the project root is importable, registers shared pytest markers,
and provides reusable session-scoped fixtures for data and helpers that
were previously duplicated across test modules.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------
# Registered locally so the shared/tests suite can run standalone. Unit 7
# also registers these globally in pyproject.toml; pytest tolerates the
# duplicate registration.

def pytest_configure(config):
    for marker in [
        "unit: fast unit tests (<1s each)",
        "integration: multi-component tests (<10s)",
        "e2e: full-pipeline tests (<60s)",
        "regression: model quality thresholds",
        "slow: excluded from default run",
    ]:
        config.addinivalue_line("markers", marker)


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
        return pd.DataFrame({
            "player_id": [f"P{i}" for i in range(n)],
            "week": rng.integers(1, 18, size=n),
            "snap_pct": rng.random(n) * 100,
            "opp_def_rank_vs_pos": rng.integers(1, 33, size=n),
            "is_home": rng.choice([0, 1], size=n),
            "td_points": rng.choice([0.0, 6.0, 12.0], size=n, p=[0.6, 0.3, 0.1]),
            "rolling_std_fantasy_points_L3": rng.random(n) * 5,
            "fantasy_points": rng.random(n) * 20,
            "pred_total": rng.random(n) * 20,
        })
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
            rows.append({
                "game_type": "REG", "season": season, "week": week,
                "away_team": "SF", "home_team": "KC",
                "home_score": 24, "away_score": 17,
                "spread_line": -3.0, "total_line": 47.0,
                "roof": "outdoors", "surface": "grass",
                "temp": 72, "wind": 8,
                "home_rest": 7, "away_rest": 7, "div_game": 0,
            })
            rows.append({
                "game_type": "REG", "season": season, "week": week,
                "away_team": "DAL", "home_team": "NO",
                "home_score": 21, "away_score": 20,
                "spread_line": -1.0, "total_line": 50.0,
                "roof": "dome", "surface": "a_turf",
                "temp": 72, "wind": 0,
                "home_rest": 7, "away_rest": 7, "div_game": 0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def player_df_factory():
    """Factory for small player DataFrames used in merge tests."""
    def _make(team="KC", n_weeks=4, season=2023):
        return pd.DataFrame({
            "player_id": ["P1"] * n_weeks,
            "season": [season] * n_weeks,
            "week": list(range(1, n_weeks + 1)),
            "recent_team": [team] * n_weeks,
            "fantasy_points": [10.0] * n_weeks,
        })
    return _make


# ---------------------------------------------------------------------------
# Shared fixtures — training (history dataset / collate / dataloaders)
# ---------------------------------------------------------------------------

TARGETS_DEFAULT = ["rushing_floor", "receiving_floor", "td_points"]


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
            np.random.randn(np.random.randint(1, 8), game_dim).astype(np.float32)
            for _ in range(n)
        ]
        y = {t: np.random.randn(n).astype(np.float32) for t in targets}
        y["total"] = sum(y[t] for t in targets)
        return X_s, X_h, y
    return _make
