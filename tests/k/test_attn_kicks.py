"""Tests for K attention-NN per-kick history building.

Covers:
  - The build_nested_kick_history helper: shape, masking, truncation, and
    per-row leakage guards (week-N history only sees weeks < N).
  - The per-kick fixture factory: one row per attempt, correct is_fg/is_xp flags.
"""

import numpy as np
import pandas as pd
import pytest

from src.k.config import ATTN_KICK_STATS
from src.k.data import reconstruct_kicker_kicks_from_pbp
from src.k.features import build_nested_kick_history


def _weekly(player_id: str, season: int, weeks: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [player_id] * len(weeks),
            "season": [season] * len(weeks),
            "week": weeks,
        }
    )


def _kick(player_id: str, season: int, week: int, **overrides) -> dict:
    base = {
        "player_id": player_id,
        "season": season,
        "week": week,
        "is_fg": 1,
        "is_xp": 0,
        "kick_distance": 35.0,
        "kick_made": 1,
        "fg_prob": 0.85,
        "is_q4": 0,
        "score_diff": 0.0,
        "game_wind": 0.0,
        "is_home": 1,
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestBuildNestedKickHistory:
    def test_output_shapes(self):
        weekly = _weekly("K1", 2023, [1, 2, 3])
        kicks = pd.DataFrame([_kick("K1", 2023, 1), _kick("K1", 2023, 2)])
        X, outer, inner = build_nested_kick_history(
            weekly, kicks, ATTN_KICK_STATS, max_games=5, max_kicks_per_game=4
        )
        assert X.shape == (3, 5, 4, len(ATTN_KICK_STATS))
        assert outer.shape == (3, 5)
        assert inner.shape == (3, 5, 4)

    def test_first_week_empty_history(self):
        """The first game of a season has no prior games — all masks False."""
        weekly = _weekly("K1", 2023, [1, 2])
        kicks = pd.DataFrame([_kick("K1", 2023, 1), _kick("K1", 2023, 1)])
        X, outer, inner = build_nested_kick_history(
            weekly, kicks, ATTN_KICK_STATS, max_games=3, max_kicks_per_game=3
        )
        # Row 0 = week 1 — nothing prior
        assert not outer[0].any()
        assert not inner[0].any()
        assert np.all(X[0] == 0)
        # Row 1 = week 2 — one prior game with 2 kicks
        assert outer[1, 0] and not outer[1, 1:].any()
        assert inner[1, 0, 0] and inner[1, 0, 1]
        assert not inner[1, 0, 2]

    def test_no_current_week_leakage(self):
        """Row for week N must only see kicks with kicks.week < N."""
        weekly = _weekly("K1", 2023, [3])
        kicks = pd.DataFrame(
            [
                _kick("K1", 2023, 1, kick_distance=20.0),
                _kick("K1", 2023, 2, kick_distance=30.0),
                _kick("K1", 2023, 3, kick_distance=40.0),  # current week — leakage
                _kick("K1", 2023, 4, kick_distance=50.0),  # future — leakage
            ]
        )
        X, outer, inner = build_nested_kick_history(
            weekly, kicks, ATTN_KICK_STATS, max_games=5, max_kicks_per_game=5
        )
        distance_idx = ATTN_KICK_STATS.index("kick_distance")
        # Two prior games (weeks 1 and 2)
        assert outer[0, 0] and outer[0, 1]
        assert not outer[0, 2:].any()
        # Distances must be 20 (week 1) and 30 (week 2) — no 40 or 50
        seen = {X[0, 0, 0, distance_idx], X[0, 1, 0, distance_idx]}
        assert seen == {20.0, 30.0}, f"leaked distances: {seen}"

    def test_outer_ordering_oldest_first(self):
        """Earliest prior game sits in slot 0, most recent in the last real slot."""
        weekly = _weekly("K1", 2023, [4])
        kicks = pd.DataFrame(
            [
                _kick("K1", 2023, 1, kick_distance=20.0),
                _kick("K1", 2023, 2, kick_distance=30.0),
                _kick("K1", 2023, 3, kick_distance=40.0),
            ]
        )
        X, outer, inner = build_nested_kick_history(
            weekly, kicks, ATTN_KICK_STATS, max_games=5, max_kicks_per_game=2
        )
        distance_idx = ATTN_KICK_STATS.index("kick_distance")
        assert X[0, 0, 0, distance_idx] == 20.0  # oldest
        assert X[0, 1, 0, distance_idx] == 30.0
        assert X[0, 2, 0, distance_idx] == 40.0  # most recent

    def test_outer_truncation_keeps_most_recent(self):
        """When prior-game count > max_games, the oldest games are dropped."""
        weekly = _weekly("K1", 2023, [8])
        # 7 prior games, weeks 1..7
        kicks = pd.DataFrame(
            [_kick("K1", 2023, w, kick_distance=float(w * 10)) for w in range(1, 8)]
        )
        X, outer, inner = build_nested_kick_history(
            weekly, kicks, ATTN_KICK_STATS, max_games=3, max_kicks_per_game=1
        )
        distance_idx = ATTN_KICK_STATS.index("kick_distance")
        # Only 3 most recent games kept (weeks 5, 6, 7), oldest-first
        assert outer[0, 0] and outer[0, 1] and outer[0, 2]
        assert X[0, 0, 0, distance_idx] == 50.0
        assert X[0, 1, 0, distance_idx] == 60.0
        assert X[0, 2, 0, distance_idx] == 70.0

    def test_inner_truncation_keeps_most_recent_kick(self):
        """When a game has > max_kicks_per_game kicks, the oldest inside-game kicks drop."""
        weekly = _weekly("K1", 2023, [2])
        kicks = pd.DataFrame(
            [_kick("K1", 2023, 1, kick_distance=float(d)) for d in [10, 20, 30, 40, 50]]
        )
        X, outer, inner = build_nested_kick_history(
            weekly, kicks, ATTN_KICK_STATS, max_games=2, max_kicks_per_game=2
        )
        distance_idx = ATTN_KICK_STATS.index("kick_distance")
        # Most recent 2 kicks from week 1 should land in slot 0
        assert outer[0, 0]
        assert inner[0, 0, 0] and inner[0, 0, 1]
        kept = {X[0, 0, 0, distance_idx], X[0, 0, 1, distance_idx]}
        assert kept == {40.0, 50.0}, f"inner truncation kept wrong distances: {kept}"

    def test_inner_truncation_uses_play_id_when_present(self):
        """play_id is the deterministic within-game sort key. Truncation must
        keep the highest play_ids (latest kicks of the game), regardless of
        the row order in kicks_df."""
        weekly = _weekly("K1", 2023, [2])
        # Shuffle row order intentionally — sort must rely on play_id, not insertion order.
        kicks = pd.DataFrame(
            [
                _kick("K1", 2023, 1, kick_distance=30.0, play_id=300),  # 3rd
                _kick("K1", 2023, 1, kick_distance=10.0, play_id=100),  # 1st
                _kick("K1", 2023, 1, kick_distance=50.0, play_id=500),  # 5th
                _kick("K1", 2023, 1, kick_distance=20.0, play_id=200),  # 2nd
                _kick("K1", 2023, 1, kick_distance=40.0, play_id=400),  # 4th
            ]
        )
        X, outer, inner = build_nested_kick_history(
            weekly, kicks, ATTN_KICK_STATS, max_games=2, max_kicks_per_game=2
        )
        distance_idx = ATTN_KICK_STATS.index("kick_distance")
        # Highest 2 play_ids (400, 500) → distances 40.0 and 50.0
        kept = {X[0, 0, 0, distance_idx], X[0, 0, 1, distance_idx]}
        assert kept == {40.0, 50.0}, (
            f"inner truncation should pick highest-play_id kicks; got {kept}"
        )

    def test_different_players_dont_cross_contaminate(self):
        weekly = _weekly("K1", 2023, [2])
        kicks = pd.DataFrame(
            [
                _kick("K2", 2023, 1, kick_distance=20.0),  # different player
                _kick("K1", 2023, 1, kick_distance=30.0),
            ]
        )
        X, outer, inner = build_nested_kick_history(
            weekly, kicks, ATTN_KICK_STATS, max_games=3, max_kicks_per_game=3
        )
        distance_idx = ATTN_KICK_STATS.index("kick_distance")
        assert outer[0, 0]
        assert X[0, 0, 0, distance_idx] == 30.0
        assert not inner[0, 0, 1:].any()

    def test_different_seasons_dont_cross_contaminate(self):
        weekly = _weekly("K1", 2023, [2])
        kicks = pd.DataFrame(
            [
                _kick("K1", 2022, 5, kick_distance=20.0),  # prior season
                _kick("K1", 2023, 1, kick_distance=30.0),
            ]
        )
        X, outer, inner = build_nested_kick_history(
            weekly, kicks, ATTN_KICK_STATS, max_games=3, max_kicks_per_game=3
        )
        distance_idx = ATTN_KICK_STATS.index("kick_distance")
        assert outer[0, 0] and not outer[0, 1:].any()
        assert X[0, 0, 0, distance_idx] == 30.0

    def test_empty_kicks_df(self):
        """No kick records at all — all masks False, output all zeros."""
        weekly = _weekly("K1", 2023, [1, 2])
        kicks = pd.DataFrame(columns=["player_id", "season", "week", *ATTN_KICK_STATS])
        X, outer, inner = build_nested_kick_history(
            weekly, kicks, ATTN_KICK_STATS, max_games=2, max_kicks_per_game=2
        )
        assert X.shape == (2, 2, 2, len(ATTN_KICK_STATS))
        assert not outer.any()
        assert not inner.any()

    def test_empty_weekly_df(self):
        weekly = pd.DataFrame(columns=["player_id", "season", "week"])
        kicks = pd.DataFrame(
            [_kick("K1", 2023, 1)], columns=["player_id", "season", "week", *ATTN_KICK_STATS]
        )
        X, outer, inner = build_nested_kick_history(
            weekly, kicks, ATTN_KICK_STATS, max_games=2, max_kicks_per_game=2
        )
        assert X.shape == (0, 2, 2, len(ATTN_KICK_STATS))
        assert outer.shape == (0, 2)

    def test_missing_kick_stat_raises(self):
        weekly = _weekly("K1", 2023, [2])
        kicks = pd.DataFrame([_kick("K1", 2023, 1)]).drop(columns=["fg_prob"])
        with pytest.raises(KeyError, match="fg_prob"):
            build_nested_kick_history(
                weekly, kicks, ATTN_KICK_STATS, max_games=2, max_kicks_per_game=2
            )


@pytest.mark.unit
class TestReconstructKickerKicksFromPbp:
    def test_empty_seasons_returns_empty_dataframe_with_schema(self):
        """An empty seasons list must short-circuit cleanly without
        ``IndexError`` on ``seasons[0]`` and must return the documented schema."""
        df = reconstruct_kicker_kicks_from_pbp([])
        assert len(df) == 0
        for col in (
            "player_id",
            "season",
            "week",
            "play_id",
            "is_fg",
            "is_xp",
            "kick_distance",
            "kick_made",
            "fg_prob",
            "is_q4",
            "score_diff",
            "game_wind",
        ):
            assert col in df.columns, f"empty-seasons schema missing {col}"


@pytest.mark.unit
class TestKickFixture:
    def test_fixture_one_row_per_attempt(self, make_kicker_games, make_tiny_k_kicks):
        weekly = make_kicker_games(n_weeks=3, fg_att=2, pat_att=3)
        kicks = make_tiny_k_kicks(weekly)
        per_week = kicks.groupby("week").size()
        assert all(per_week == 5)  # 2 FG + 3 XP per game

    def test_fixture_is_fg_xp_flags_consistent(self, make_kicker_games, make_tiny_k_kicks):
        weekly = make_kicker_games(n_weeks=2, fg_att=1, fg_made=1, pat_att=1, pat_made=1)
        kicks = make_tiny_k_kicks(weekly)
        # Each row is exactly one of FG or XP
        assert ((kicks["is_fg"] + kicks["is_xp"]) == 1).all()
        # XP rows have distance=0 and fg_prob=0
        xps = kicks[kicks["is_xp"] == 1]
        assert (xps["kick_distance"] == 0.0).all()
        assert (xps["fg_prob"] == 0.0).all()
