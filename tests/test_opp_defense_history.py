"""Tests for the opp-defense attention-history builders in
``src/features/engineer.py``.

Covers:
- Per-game aggregation shape and stat correctness.
- Sequence builder padding, mask, oldest→newest ordering.
- Strict leakage guard (target week's own defense stats must NOT appear in
  the sequence) and season-boundary behaviour.
- Robustness to missing columns / empty frames.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    OPP_DEFENSE_HISTORY_STATS,
    build_opp_defense_history_arrays,
    build_opp_defense_per_game_df,
)


def _synthetic_all_position_df(
    teams=("X", "Y"),
    weeks=(1, 2, 3, 4),
    players=("p1", "p2"),
    season=2023,
):
    """Two defenses, four weeks, two offensive players each week.

    Each player-row contributes fixed stats so the per-game aggregate
    equals ``n_players * stat_per_player``, which makes the expected values
    below easy to verify by hand.
    """
    rows = []
    for team in teams:
        for week in weeks:
            for pid in players:
                rows.append(
                    {
                        "player_id": pid,
                        "position": "QB",
                        "opponent_team": team,
                        "recent_team": "Z" if team == "X" else "W",
                        "season": season,
                        "week": week,
                        "sacks": 1.0,
                        "passing_yards": 200.0,
                        "passing_tds": 1.0,
                        "interceptions": 1.0,
                        "rushing_yards": 30.0,
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.unit
class TestBuildOppDefensePerGameDf:
    def test_aggregation_shape_and_columns(self):
        df = _synthetic_all_position_df()
        out = build_opp_defense_per_game_df(df)
        # 2 defenses × 4 weeks = 8 rows.
        assert len(out) == 8
        for col in OPP_DEFENSE_HISTORY_STATS:
            assert col in out.columns
        assert set(out.columns) >= {"opponent_team", "season", "week"}

    def test_aggregated_stats_match_sum(self):
        df = _synthetic_all_position_df()
        out = build_opp_defense_per_game_df(df).sort_values(["opponent_team", "season", "week"])
        # Each defense sees 2 players × 1 sack = 2 sacks per week, etc.
        assert (out["def_sacks"] == 2.0).all()
        assert (out["def_pass_yds_allowed"] == 400.0).all()
        assert (out["def_pass_td_allowed"] == 2.0).all()
        assert (out["def_ints"] == 2.0).all()
        assert (out["def_rush_yds_allowed"] == 60.0).all()

    def test_missing_columns_returns_empty_frame(self):
        df = pd.DataFrame({"player_id": ["p1"], "season": [2023], "week": [1]})
        out = build_opp_defense_per_game_df(df)
        assert len(out) == 0
        # Schema is still present so downstream callers don't KeyError.
        for col in OPP_DEFENSE_HISTORY_STATS:
            assert col in out.columns


@pytest.mark.unit
class TestBuildOppDefenseHistoryArrays:
    def test_shape_and_mask_monotonic(self):
        df = _synthetic_all_position_df()
        per_game = build_opp_defense_per_game_df(df)
        X_opp, mask = build_opp_defense_history_arrays(
            df, per_game, OPP_DEFENSE_HISTORY_STATS, max_seq_len=5
        )
        assert X_opp.shape == (len(df), 5, len(OPP_DEFENSE_HISTORY_STATS))
        assert mask.shape == (len(df), 5)
        # Each player-row pulls its opponent's prior games this season —
        # monotone in week: week W has W-1 prior games.
        seq_lens = mask.sum(axis=1)
        for i in range(len(df)):
            expected = df.iloc[i]["week"] - 1
            assert seq_lens[i] == expected, f"row {i} week={df.iloc[i]['week']}: got {seq_lens[i]}"

    def test_no_leakage_current_week_absent(self):
        """The defense's target-week row must NOT appear in the sequence.

        Build a per-game frame where each week's stats are *distinct* so we
        can spot a leaked week by value.
        """
        rows = []
        for team in ["X"]:
            for week in range(1, 5):
                rows.append(
                    {
                        "opponent_team": team,
                        "season": 2023,
                        "week": week,
                        **{s: float(week * 10) for s in OPP_DEFENSE_HISTORY_STATS},
                    }
                )
        per_game = pd.DataFrame(rows)
        df = pd.DataFrame([{"player_id": "p1", "opponent_team": "X", "season": 2023, "week": 3}])
        X_opp, mask = build_opp_defense_history_arrays(
            df, per_game, OPP_DEFENSE_HISTORY_STATS, max_seq_len=5
        )
        # 2 prior games (weeks 1 & 2); week 3's stat (=30) must be absent.
        assert mask[0].sum() == 2
        values = X_opp[0, :2, 0]  # first stat = def_sacks
        assert 30.0 not in values, f"week-3 value leaked into history: {values}"
        # Oldest → newest ordering: week 1 (=10) first, week 2 (=20) second.
        np.testing.assert_array_equal(values, [10.0, 20.0])

    def test_season_boundary_isolation(self):
        """A defense's games from a prior season must NOT appear in the
        sequence for a target-week row in the current season."""
        rows = [
            {
                "opponent_team": "X",
                "season": 2022,
                "week": w,
                **{s: 999.0 for s in OPP_DEFENSE_HISTORY_STATS},
            }
            for w in range(1, 5)
        ] + [
            {
                "opponent_team": "X",
                "season": 2023,
                "week": w,
                **{s: float(w) for s in OPP_DEFENSE_HISTORY_STATS},
            }
            for w in range(1, 5)
        ]
        per_game = pd.DataFrame(rows)
        df = pd.DataFrame([{"player_id": "p1", "opponent_team": "X", "season": 2023, "week": 3}])
        X_opp, mask = build_opp_defense_history_arrays(
            df, per_game, OPP_DEFENSE_HISTORY_STATS, max_seq_len=5
        )
        assert mask[0].sum() == 2
        values = X_opp[0, :2, 0]
        assert 999.0 not in values, "prior-season value leaked across season boundary"
        np.testing.assert_array_equal(values, [1.0, 2.0])

    def test_truncation_keeps_most_recent(self):
        """When prior games > max_seq_len, the sequence keeps the MOST RECENT
        games (tail), matching ``build_game_history_arrays`` semantics."""
        rows = [
            {
                "opponent_team": "X",
                "season": 2023,
                "week": w,
                **{s: float(w) for s in OPP_DEFENSE_HISTORY_STATS},
            }
            for w in range(1, 11)
        ]
        per_game = pd.DataFrame(rows)
        df = pd.DataFrame([{"player_id": "p1", "opponent_team": "X", "season": 2023, "week": 10}])
        X_opp, mask = build_opp_defense_history_arrays(
            df, per_game, OPP_DEFENSE_HISTORY_STATS, max_seq_len=3
        )
        assert mask[0].sum() == 3
        # Weeks 7, 8, 9 kept (oldest dropped); weeks 1-6 absent.
        np.testing.assert_array_equal(X_opp[0, :3, 0], [7.0, 8.0, 9.0])

    def test_empty_per_game_returns_zero_filled(self):
        df = pd.DataFrame([{"player_id": "p1", "opponent_team": "X", "season": 2023, "week": 3}])
        empty = pd.DataFrame(
            columns=["opponent_team", "season", "week"] + OPP_DEFENSE_HISTORY_STATS
        )
        X_opp, mask = build_opp_defense_history_arrays(
            df, empty, OPP_DEFENSE_HISTORY_STATS, max_seq_len=5
        )
        assert X_opp.shape == (1, 5, len(OPP_DEFENSE_HISTORY_STATS))
        assert mask.sum() == 0

    def test_stat_subset_column_intersection(self):
        """Requesting a stat that isn't in the lookup frame silently drops
        it instead of raising — same robustness as build_game_history_arrays."""
        df = _synthetic_all_position_df()
        per_game = build_opp_defense_per_game_df(df)
        X_opp, _ = build_opp_defense_history_arrays(
            df.iloc[:4], per_game, ["def_sacks", "non_existent_stat"], max_seq_len=5
        )
        # Only def_sacks survives the intersection.
        assert X_opp.shape == (4, 5, 1)
