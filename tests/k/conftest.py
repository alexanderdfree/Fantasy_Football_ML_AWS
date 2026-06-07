"""Shared fixtures for K (Kicker) tests.

The standard make_sim_df / sim_df / make_test_df / test_df / make_tensors /
make_splits fixtures get installed by ``register_standard_fixtures`` (see
``tests/shared/position_fixtures.py``); only kicker-specific helpers remain
here — ``make_kicker_games`` (per-week stat frame for feature tests) and
the tiny synthetic dataset used by E2E / regression tests.
"""

import numpy as np
import pandas as pd
import pytest

from src.k.config import POSITION_CONFIG
from tests.shared.position_fixtures import (
    register_position_markers,
    register_standard_fixtures,
)

# Kickers typically score 5-15 fantasy points per game; FG driven.
SCORING_SCALE = 12.0


def pytest_configure(config):
    register_position_markers(config)


register_standard_fixtures(
    globals(),
    scoring_scale=SCORING_SCALE,
    id_prefix="K",
    targets=POSITION_CONFIG.targets,
    stat_col="passing_yards",  # K uses the default; not consumed by K tests
    rng_kind="default",
    install_default_shortcuts=True,
)


# ---------------------------------------------------------------------------
# K-specific fixtures (not generic)
# ---------------------------------------------------------------------------


def _build_k_row(**overrides) -> pd.DataFrame:
    """Single-row kicker DataFrame with sensible defaults."""
    defaults = {
        "fg_yards_made": 0,
        "fg_missed": 0,
        "pat_made": 3,
        "pat_missed": 0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


def _build_games(
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

    Includes a pre-computed `fantasy_points` column — compute_features reads
    it to build the rolling total-points statistics. After the 4-head target
    refactor, the signed fantasy total is the canonical rolling input.
    """
    df = pd.DataFrame(
        {
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
        }
    )
    df["fantasy_points"] = df["pat_made"].astype(float)
    return df


@pytest.fixture(scope="session")
def make_kicker_games():
    """Factory fixture: build multi-week kicker games DataFrame."""
    return _build_games


# ---------------------------------------------------------------------------
# Tiny synthetic pipeline dataset for E2E + regression tests
# ---------------------------------------------------------------------------


def _build_tiny_dataset(
    n_players: int = 50,
    n_seasons: int = 10,
    n_weeks: int = 17,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a tiny synthetic kicker dataset suitable for E2E + regression tests.

    Players get stable "team" assignments and realistic-scale kicker stats.
    Schedule-merge is short-circuited via `_schedule_merged=True` so the
    synthetic data does not need to match real NFL schedule rows.
    Cross-season history is preserved so rolling features fire.

    Covers seasons [base_season, base_season + n_seasons] inclusive — with
    defaults that's 2015-2025 (11 seasons), matching the real SEASONS
    range. Multi-season training data is required for the pipeline's
    expanding-window Ridge CV tuning to build non-empty folds.
    """
    rng = np.random.default_rng(seed)
    # Match the real kicker dataset: 2015-2025, split train<=2023 / val=2024 /
    # test=2025. Multiple seasons in train are required for expanding-window
    # CV (`src/shared/pipeline._build_expanding_cv_folds`) to produce non-empty
    # folds; with a single train season, every fold is skipped and downstream
    # `np.mean([])` warnings fire.
    base_season = 2015
    seasons = list(range(base_season, base_season + n_seasons + 1))  # +1 for test

    teams = [
        "KC",
        "SF",
        "BUF",
        "MIA",
        "DAL",
        "PHI",
        "BAL",
        "CIN",
        "DET",
        "GB",
        "MIN",
        "CHI",
        "SEA",
        "LAR",
        "ARI",
        "NO",
    ]

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
                # Long (40+) subset: the made counts in the 40-49/50-59/60+
                # buckets by definition sum to at most fg_made. The attempted
                # long FGs are the made-longs plus some fraction of misses.
                long_fg_made_val = fg_made_mid + fg_made_long_50 + fg_made_60
                long_fg_att_val = long_fg_made_val + max(0, int(round(fg_missed * 0.4)))

                # fg_yards_made: bucket-midpoint approximation of the sum of
                # made-FG kick distances. Drives fg_yard_points target = sum * 0.1.
                # Using midpoints (25 / 35 / 45 / 55 / 62) keeps the target in
                # a realistic 0-20pt-per-game range for the E2E synthetic fixture.
                fg_yards_made_val = (
                    (fg_made_short // 2) * 25  # 20-29 bucket midpoint
                    + (fg_made_short - fg_made_short // 2) * 35  # 30-39 bucket midpoint
                    + fg_made_mid * 45  # 40-49 bucket midpoint
                    + fg_made_long_50 * 55  # 50-59 bucket midpoint
                    + fg_made_60 * 62  # 60+ bucket representative
                )

                rows.append(
                    {
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
                        "fg_yards_made": fg_yards_made_val,
                        "pat_att": pat_att,
                        "pat_made": pat_made,
                        "pat_missed": pat_missed,
                        # PBP-derived columns (feature inputs)
                        "avg_fg_distance": float(rng.normal(38, 4)) if fg_att else 0.0,
                        "avg_fg_prob": float(np.clip(rng.normal(0.82, 0.05), 0, 1)),
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
                        "_schedule_merged": True,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_dataset():
    """Session-scoped tiny kicker dataset (50 players x 11 seasons x 17 weeks).

    Returns a single DataFrame covering 2015-2025 so pipeline season splits
    produce non-empty train/val/test partitions AND train has enough unique
    seasons (9) for expanding-window Ridge CV to build non-empty folds.
    """
    return _build_tiny_dataset(n_players=50, n_seasons=10, n_weeks=17, seed=42)


@pytest.fixture(scope="session")
def tiny_splits(tiny_dataset):
    """Session-scoped train/val/test splits from the tiny kicker dataset.

    Split mirrors the real pipeline: train <= 2023, val = 2024, test = 2025.
    """
    df = tiny_dataset
    train = df[df["season"] <= 2023].copy()
    val = df[df["season"] == 2024].copy()
    test = df[df["season"] == 2025].copy()
    return train, val, test


def _build_tiny_kicks(weekly_df: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
    """Expand each weekly row into per-kick records.

    Produces `fg_att` FG rows and `pat_att` XP rows per weekly row, with random
    distances/probabilities consistent with the aggregate stats. Kept simple —
    exact bucket alignment isn't required for unit tests, just enough structural
    fidelity that the attention pipeline can consume it.

    Each kick carries a synthetic per-game ``play_id`` increment so the
    production code path that branches on ``'play_id' in kicks_df.columns``
    (see ``src/k/features.py::build_nested_kick_history``) is exercised
    rather than silently falling back to insertion-order sorting.
    """
    rng = np.random.default_rng(seed)
    kicks: list[dict] = []
    for _, row in weekly_df.iterrows():
        pid = row["player_id"]
        season = row["season"]
        week = row["week"]
        game_wind = float(row.get("game_wind", 0) or 0)
        is_home = int(row.get("is_home", 0) or 0)
        # Per-game play counter so kicks within a single (pid, season, week)
        # have a monotonically increasing play_id — matches PBP's stable
        # per-play sequence number used by ``build_nested_kick_history`` for
        # deterministic most-recent truncation.
        play_id_counter = 0

        fg_att = int(row.get("fg_att", 0) or 0)
        fg_made = int(row.get("fg_made", 0) or 0)
        made_flags = np.concatenate(
            [np.ones(fg_made, dtype=int), np.zeros(fg_att - fg_made, dtype=int)]
        )
        rng.shuffle(made_flags)
        for made in made_flags:
            distance = float(np.clip(rng.normal(38, 6), 17, 66))
            kicks.append(
                {
                    "player_id": pid,
                    "season": season,
                    "week": week,
                    "play_id": play_id_counter,
                    "is_fg": 1,
                    "is_xp": 0,
                    "kick_distance": distance,
                    "kick_made": int(made),
                    "fg_prob": float(np.clip(rng.normal(0.82, 0.06), 0.1, 0.99)),
                    "is_q4": int(rng.integers(0, 2)),
                    "score_diff": float(rng.normal(0, 8)),
                    "game_wind": game_wind,
                    "is_home": is_home,
                }
            )
            play_id_counter += 1

        pat_att = int(row.get("pat_att", 0) or 0)
        pat_made = int(row.get("pat_made", 0) or 0)
        xp_flags = np.concatenate(
            [np.ones(pat_made, dtype=int), np.zeros(pat_att - pat_made, dtype=int)]
        )
        rng.shuffle(xp_flags)
        for made in xp_flags:
            kicks.append(
                {
                    "player_id": pid,
                    "season": season,
                    "week": week,
                    "play_id": play_id_counter,
                    "is_fg": 0,
                    "is_xp": 1,
                    "kick_distance": 0.0,
                    "kick_made": int(made),
                    "fg_prob": 0.0,
                    "is_q4": int(rng.integers(0, 2)),
                    "score_diff": 0.0,
                    "game_wind": game_wind,
                    "is_home": is_home,
                }
            )
            play_id_counter += 1
    return pd.DataFrame(kicks)


@pytest.fixture(scope="session")
def make_tiny_k_kicks():
    """Factory: expand a weekly kicker DataFrame into per-kick rows."""
    return _build_tiny_kicks


@pytest.fixture(scope="session")
def tiny_kicks(tiny_dataset):
    """Session-scoped per-kick records matching tiny_dataset."""
    return _build_tiny_kicks(tiny_dataset, seed=123)
