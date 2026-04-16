"""Test fixtures and pytest config for K (Kicker) tests.

Promotes the per-file `_make_*` helpers into session-scoped fixtures and
registers the shared pytest markers (unit / integration / e2e / regression).

Kicker scoring scale: total fantasy points typically 5-15 per game; FG points
dominate, PATs are small and correlated with team TDs. Synthetic data reflects
that scale.
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
# Pytest markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register the markers used across K tests so `--strict-markers` is clean."""
    config.addinivalue_line("markers", "unit: fast unit test (< 1s each)")
    config.addinivalue_line(
        "markers", "integration: multi-component test (< 10s each)"
    )
    config.addinivalue_line("markers", "e2e: full-pipeline test (< 60s each)")
    config.addinivalue_line(
        "markers", "regression: model quality thresholds (may need fixture data)"
    )


# ---------------------------------------------------------------------------
# Kicker-specific constants
# ---------------------------------------------------------------------------

# Kickers use 2 prediction targets; unlike other positions which have 3.
K_TARGETS = ["fg_points", "pat_points"]
K_LOSS_WEIGHTS = {"fg_points": 1.0, "pat_points": 1.0}

# Kickers typically score 5-15 fantasy points per game; FG driven.
K_SCORING_SCALE = 12.0


# ---------------------------------------------------------------------------
# Backtest / simulation fixtures
# ---------------------------------------------------------------------------

def _build_sim_df(n_weeks: int, n_players: int, seed: int = 42) -> pd.DataFrame:
    """Build a kicker weekly DataFrame with Ridge / NN predictions.

    Each row = one kicker-week with noisy predictions around `fantasy_points`.
    Scale matches kicker fantasy output (rand * 12 ~ 5-15 typical).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(1, n_weeks + 1):
        for pid in range(1, n_players + 1):
            fp = float(rng.random() * K_SCORING_SCALE)
            rows.append({
                "week": week,
                "player_id": f"K{pid}",
                "fantasy_points": fp,
                "pred_ridge": fp + float(rng.standard_normal() * 2),
                "pred_nn": fp + float(rng.standard_normal() * 3),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def make_sim_df():
    """Factory fixture: build a simulated kicker DataFrame.

    Usage:
        def test_foo(make_sim_df):
            df = make_sim_df(n_weeks=3, n_players=15)
    """
    return _build_sim_df


@pytest.fixture(scope="session")
def sim_df(make_sim_df):
    """Default 4-week x 15-kicker simulation DataFrame."""
    return make_sim_df(n_weeks=4, n_players=15)


# ---------------------------------------------------------------------------
# Evaluation / ranking fixtures
# ---------------------------------------------------------------------------

def _build_test_df(n_weeks: int, n_players: int, seed: int = 42) -> pd.DataFrame:
    """Build a DataFrame suitable for compute_ranking_metrics."""
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(1, n_weeks + 1):
        for pid in range(1, n_players + 1):
            rows.append({
                "week": week,
                "player_id": f"K{pid}",
                "pred_total": float(rng.random() * K_SCORING_SCALE),
                "fantasy_points": float(rng.random() * K_SCORING_SCALE),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def make_test_df():
    """Factory fixture: build a ranking-test DataFrame."""
    return _build_test_df


@pytest.fixture(scope="session")
def test_df(make_test_df):
    """Default 3-week x 15-kicker test DataFrame for ranking tests."""
    return make_test_df(n_weeks=3, n_players=15)


# ---------------------------------------------------------------------------
# Training tensor fixtures
# ---------------------------------------------------------------------------

def _build_tensors(n: int = 10, seed: int = 42):
    """Build (preds, targets) tensor dicts for loss-function tests."""
    torch.manual_seed(seed)
    preds = {t: torch.randn(n) for t in K_TARGETS}
    preds["total"] = torch.randn(n)
    targets = {t: torch.randn(n) for t in K_TARGETS}
    targets["total"] = torch.randn(n)
    return preds, targets


@pytest.fixture
def make_tensors():
    """Factory fixture (function-scoped): fresh tensors per test to avoid mutation.

    Tensor fixtures are function-scoped because backward() accumulates grads —
    sharing would leak state across tests.
    """
    return _build_tensors


# ---------------------------------------------------------------------------
# NaN-fill split fixtures
# ---------------------------------------------------------------------------

def _build_splits(train_vals, val_vals, test_vals, col: str = "feat1"):
    """Build 3-split DataFrames with one feature column."""
    return (
        pd.DataFrame({col: train_vals}),
        pd.DataFrame({col: val_vals}),
        pd.DataFrame({col: test_vals}),
    )


@pytest.fixture(scope="session")
def make_splits():
    """Factory fixture: build (train, val, test) DataFrames for NaN tests."""
    return _build_splits


# ---------------------------------------------------------------------------
# Kicker games fixture for feature tests
# ---------------------------------------------------------------------------

def _build_kicker_games(
    player_id: str = "K1",
    n_weeks: int = 6,
    season: int = 2023,
    fg_att: int = 3,
    fg_made: int = 2,
    pat_att: int = 3,
    pat_made: int = 3,
    fg_made_40_49: int = 1,
    fg_made_50_59: int = 0,
    fg_made_60_: int = 0,
    fg_missed_40_49: int = 0,
    fg_missed_50_59: int = 0,
    fg_missed_60_: int = 0,
    avg_fg_distance: float = 35.0,
    avg_fg_prob: float = 0.85,
    long_fg_att: int = 1,
    long_fg_made: int = 1,
    q4_fg_att: int = 1,
    q4_fg_made: int = 1,
) -> pd.DataFrame:
    """Build multi-week kicker data for feature tests.

    Includes pre-computed targets (fg_points, pat_points, miss_penalty) — the
    feature pipeline reads them to build rolling total-points statistics.
    """
    df = pd.DataFrame({
        "player_id": [player_id] * n_weeks,
        "season": [season] * n_weeks,
        "week": list(range(1, n_weeks + 1)),
        "fg_att": [fg_att] * n_weeks,
        "fg_made": [fg_made] * n_weeks,
        "pat_att": [pat_att] * n_weeks,
        "pat_made": [pat_made] * n_weeks,
        "fg_made_40_49": [fg_made_40_49] * n_weeks,
        "fg_made_50_59": [fg_made_50_59] * n_weeks,
        "fg_made_60_": [fg_made_60_] * n_weeks,
        "fg_missed_40_49": [fg_missed_40_49] * n_weeks,
        "fg_missed_50_59": [fg_missed_50_59] * n_weeks,
        "fg_missed_60_": [fg_missed_60_] * n_weeks,
        "avg_fg_distance": [avg_fg_distance] * n_weeks,
        "avg_fg_prob": [avg_fg_prob] * n_weeks,
        "long_fg_att": [long_fg_att] * n_weeks,
        "long_fg_made": [long_fg_made] * n_weeks,
        "q4_fg_att": [q4_fg_att] * n_weeks,
        "q4_fg_made": [q4_fg_made] * n_weeks,
    })
    df["fg_points"] = 0
    df["pat_points"] = df["pat_made"]
    df["miss_penalty"] = 0
    return df


@pytest.fixture(scope="session")
def make_kicker_games():
    """Factory fixture: build multi-week kicker games DataFrame."""
    return _build_kicker_games


# ---------------------------------------------------------------------------
# Tiny synthetic pipeline dataset for E2E + regression tests
# ---------------------------------------------------------------------------

def _build_tiny_k_dataset(
    n_players: int = 50,
    n_seasons: int = 2,
    n_weeks: int = 17,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a tiny synthetic kicker dataset suitable for E2E + regression tests.

    Players get stable "team" assignments and realistic-scale kicker stats.
    Schedule-merge is short-circuited via `_schedule_merged=True` so the
    synthetic data does not need to match real NFL schedule rows.
    Cross-season history is preserved so rolling features fire.

    Total rows = n_players * n_seasons * n_weeks (default 50*2*17 = 1700).
    """
    rng = np.random.default_rng(seed)
    # Kickers span seasons in K_SEASONS starting 2015; use two adjacent
    # seasons close to the real split so the pipeline's season filtering works
    # out of the box (train: <=2023, val: 2024, test: 2025).
    base_season = 2023
    seasons = list(range(base_season, base_season + n_seasons + 1))  # +1 for test

    teams = ["KC", "SF", "BUF", "MIA", "DAL", "PHI", "BAL", "CIN",
             "DET", "GB", "MIN", "CHI", "SEA", "LAR", "ARI", "NO"]

    rows = []
    for pid in range(1, n_players + 1):
        team = teams[pid % len(teams)]
        # Stable per-player skill: leaks into targets so features have signal.
        player_skill = float(np.clip(rng.normal(0.80, 0.06), 0.55, 0.97))
        player_volume = float(np.clip(rng.normal(2.5, 0.6), 0.5, 4.5))
        for season in seasons:
            for week in range(1, n_weeks + 1):
                # Skill-driven attempts: higher-volume kickers attempt more.
                fg_att = int(np.clip(rng.poisson(player_volume), 0, 6))
                fg_made = int(rng.binomial(fg_att, player_skill))
                fg_missed = fg_att - fg_made
                # Distance distribution: bucket the made FGs
                if fg_made == 0:
                    fg_made_short = fg_made_mid = fg_made_long_50 = fg_made_60 = 0
                else:
                    probs = rng.random(fg_made)
                    fg_made_short = int((probs < 0.5).sum())  # 0-39
                    fg_made_mid = int(((probs >= 0.5) & (probs < 0.80)).sum())  # 40-49
                    fg_made_long_50 = int(((probs >= 0.80) & (probs < 0.97)).sum())
                    fg_made_60 = int((probs >= 0.97).sum())

                pat_att = int(rng.integers(1, 5))
                pat_made = int(rng.binomial(pat_att, 0.96))
                pat_missed = pat_att - pat_made

                # Q4 and clutch subsets: draw counts so that made <= att <= total.
                q4_fg_att = int(rng.integers(0, fg_att + 1))
                q4_fg_made = int(rng.binomial(q4_fg_att, player_skill))
                clutch_fg_att = int(rng.integers(0, fg_att + 1))
                clutch_fg_made = int(rng.binomial(clutch_fg_att, player_skill))

                # Long (40+) subset: the made counts in the 40-49/50-59/60+
                # buckets by definition sum to at most fg_made. The attempted
                # long FGs are the made-longs plus some fraction of misses.
                long_fg_made_val = fg_made_mid + fg_made_long_50 + fg_made_60
                long_fg_att_val = long_fg_made_val + max(
                    0, int(round(fg_missed * 0.4))
                )

                rows.append({
                    "player_id": f"K{pid:03d}",
                    "player_name": f"Kicker{pid}",
                    "recent_team": team,
                    "position": "K",
                    "season_type": "REG",
                    "season": season,
                    "week": week,
                    "fg_att": fg_att,
                    "fg_made": fg_made,
                    "fg_missed": fg_missed,
                    "fg_made_0_19": 0,
                    "fg_made_20_29": fg_made_short // 2,
                    "fg_made_30_39": fg_made_short - fg_made_short // 2,
                    "fg_made_40_49": fg_made_mid,
                    "fg_made_50_59": fg_made_long_50,
                    "fg_made_60_": fg_made_60,
                    "fg_missed_40_49": max(0, (fg_missed * 2) // 5),
                    "fg_missed_50_59": max(0, (fg_missed * 2) // 5),
                    "fg_missed_60_": 0,
                    "pat_att": pat_att,
                    "pat_made": pat_made,
                    "pat_missed": pat_missed,
                    # PBP-derived columns (feature inputs)
                    "avg_fg_distance": float(rng.normal(38, 4)) if fg_att else 0.0,
                    "max_fg_distance": float(rng.normal(48, 5)) if fg_att else 0.0,
                    "avg_fg_prob": float(np.clip(rng.normal(0.82, 0.05), 0, 1)),
                    "clutch_fg_att": clutch_fg_att,
                    "clutch_fg_made": clutch_fg_made,
                    "q4_fg_att": q4_fg_att,
                    "q4_fg_made": q4_fg_made,
                    "long_fg_att": long_fg_att_val,
                    "long_fg_made": long_fg_made_val,
                    "game_wind": float(rng.normal(8, 5)),
                    "game_temp": float(rng.normal(60, 15)),
                    "roof": "outdoors",
                    "surface": "grass",
                    "is_dome": 0,
                    # Schedule-merged features (pre-filled so merge is skipped)
                    "is_home": int(rng.integers(0, 2)),
                    "total_line": float(rng.normal(45, 5)),
                    "implied_team_total": float(rng.normal(22, 4)),
                    "implied_opp_total": float(rng.normal(22, 4)),
                    "is_grass": 1,
                    "temp_adjusted": 65.0,
                    "wind_adjusted": 0.0,
                    "is_divisional": 0,
                    "days_rest_improved": 7,
                    "rest_advantage": 0,
                    "implied_total_x_wind": 22.0,
                    "_schedule_merged": True,
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_k_dataset():
    """Session-scoped tiny kicker dataset (50 players x 3 seasons x 17 weeks).

    Returns a single DataFrame covering 2023-2025 so pipeline season splits
    produce non-empty train/val/test partitions.
    """
    return _build_tiny_k_dataset(n_players=50, n_seasons=2, n_weeks=17, seed=42)


@pytest.fixture(scope="session")
def tiny_k_splits(tiny_k_dataset):
    """Session-scoped train/val/test splits from the tiny kicker dataset.

    Split mirrors the real pipeline: train <= 2023, val = 2024, test = 2025.
    """
    df = tiny_k_dataset
    train = df[df["season"] <= 2023].copy()
    val = df[df["season"] == 2024].copy()
    test = df[df["season"] == 2025].copy()
    return train, val, test
