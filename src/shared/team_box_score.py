"""Per-game team box-score columns for the attention NN's history sequence.

The attention branch consumes per-historical-game tokens; today those tokens
carry only the player's own stats. RB usage is driven heavily by game script
(team pass/rush split, points scored, opponent points), so this module
materialises seven team-level box-score columns plus one opponent column on
every player-week row. ``build_game_history_arrays`` then pulls them into the
history tensor automatically whenever a position's ``ATTN_HISTORY_STATS``
lists them.

Per-row semantics: each row's box-score columns describe *that row's own
game*. When the row later participates as a history token for a future game,
the columns describe that historical game's box score — naturally lagged by
the same shift-by-1 convention the rest of the history pipeline uses.
"""

from __future__ import annotations

import pandas as pd

from src.config import CACHE_DIR, SEASONS
from src.data.loader import load_team_week_stats
from src.shared.weather_features import _load_schedules

# Columns produced per (season, week, team) before the per-row merge.
TEAM_BOX_SCORE_FEATURES: list[str] = [
    "team_pass_attempts",
    "team_completions",
    "team_passing_yards",
    "team_rush_attempts",
    "team_rushing_yards",
    "team_points_scored",
    "team_turnovers",
]

# Opponent-side columns the merge attaches keyed on opponent_team. Kept to a
# single column on purpose: combined with team_points_scored it yields the
# in-game score margin (the dominant game-script signal for RB usage), while
# the opponent's offensive splits would largely duplicate the team's own
# defensive context that the parallel opp-defense attention branch already
# learns.
OPP_BOX_SCORE_FEATURES: list[str] = ["opp_team_points_scored"]

# Intentionally NOT module-level-cached: tests monkeypatch ``_load_schedules``
# per-suite (DST, RB E2E, etc.), and a cached lookup derived from one test's
# synthetic schedule would leak into the next test's run. Per-call rebuild is
# fast — both upstream loaders (``_load_schedules`` and
# ``load_team_week_stats``) hit their own caches, and the lookup build itself
# is millisecond-scale on the full 14-season frame.


def _build_team_box_score_lookup() -> pd.DataFrame:
    """Return one row per ``(season, week, team)`` with the seven team box-score
    columns plus ``opp_team_points_scored``.

    Sources:
      * nflverse ``stats_team_week`` parquet for offensive volume/efficiency
        (attempts, completions, passing_yards, carries, rushing_yards) and
        the four turnover components (passing_interceptions,
        rushing_fumbles_lost, receiving_fumbles_lost, sack_fumbles_lost).
      * Schedules parquet for the two scoring columns (team_points_scored
        from the team's side of the score, opp_team_points_scored from the
        opponent's side).
    """
    try:
        schedules_reg = _load_schedules()
    except FileNotFoundError:
        # Synthetic test fixtures pre-stamp ``_schedule_merged=True`` and
        # short-circuit the schedule load entirely; the parquet is absent
        # at the test's cwd. Degrade to an empty lookup so the merge yields
        # zero-filled columns — same pattern downstream callers already
        # tolerate for padded history positions.
        return pd.DataFrame(
            columns=["season", "week", "team"] + TEAM_BOX_SCORE_FEATURES + OPP_BOX_SCORE_FEATURES
        )

    if {"home_score", "away_score"}.issubset(schedules_reg.columns):
        home_score = schedules_reg[
            ["season", "week", "home_team", "home_score", "away_score"]
        ].copy()
        home_score.columns = [
            "season",
            "week",
            "team",
            "team_points_scored",
            "opp_team_points_scored",
        ]
        away_score = schedules_reg[
            ["season", "week", "away_team", "away_score", "home_score"]
        ].copy()
        away_score.columns = [
            "season",
            "week",
            "team",
            "team_points_scored",
            "opp_team_points_scored",
        ]
        points = pd.concat([home_score, away_score], ignore_index=True).drop_duplicates(
            subset=["season", "week", "team"]
        )
    else:
        # Synthetic test schedules sometimes omit the score columns; fall
        # back to a frame keyed on the (season, week, team) lookup keys with
        # zero-filled scoring so the merge still yields columns of the right
        # shape. The same pattern any caller relies on: the history sequence
        # will see zeros for missing matches, indistinguishable from padding.
        teams = pd.concat(
            [
                schedules_reg[["season", "week", "home_team"]].rename(
                    columns={"home_team": "team"}
                ),
                schedules_reg[["season", "week", "away_team"]].rename(
                    columns={"away_team": "team"}
                ),
            ],
            ignore_index=True,
        ).drop_duplicates()
        points = teams.copy()
        points["team_points_scored"] = 0.0
        points["opp_team_points_scored"] = 0.0

    team_stats = load_team_week_stats(SEASONS, cache_dir=CACHE_DIR)
    if team_stats.empty:
        # Box-score frame still carries the scoring columns even when team_stats
        # is unavailable — score margin alone is useful and matches the
        # graceful-degradation pattern in ``build_opp_defense_per_game_df``.
        out = points.copy()
        for col in TEAM_BOX_SCORE_FEATURES:
            if col not in out.columns:
                out[col] = 0.0
        return out[["season", "week", "team"] + TEAM_BOX_SCORE_FEATURES + OPP_BOX_SCORE_FEATURES]

    ts = team_stats.copy()
    turnover_cols = [
        "passing_interceptions",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "sack_fumbles_lost",
    ]
    ts["_turnovers"] = sum(ts[col].fillna(0) for col in turnover_cols)

    box = ts[
        [
            "team",
            "season",
            "week",
            "attempts",
            "completions",
            "passing_yards",
            "carries",
            "rushing_yards",
            "_turnovers",
        ]
    ].rename(
        columns={
            "attempts": "team_pass_attempts",
            "completions": "team_completions",
            "passing_yards": "team_passing_yards",
            "carries": "team_rush_attempts",
            "rushing_yards": "team_rushing_yards",
            "_turnovers": "team_turnovers",
        }
    )

    merged = box.merge(points, on=["season", "week", "team"], how="outer")
    return merged[["season", "week", "team"] + TEAM_BOX_SCORE_FEATURES + OPP_BOX_SCORE_FEATURES]


def merge_team_box_score_features(df: pd.DataFrame, label: str | None = None) -> pd.DataFrame:
    """Merge team and opponent per-game box-score columns onto a player-week frame.

    Two merges:
      * ``team_*`` columns keyed on ``(season, week, recent_team)``.
      * ``opp_team_points_scored`` keyed on ``(season, week, opponent_team)``.

    Idempotent — a second call short-circuits via the ``_team_box_score_merged``
    sentinel column so per-split callers in ``build_position_features`` don't
    re-do the join.

    NaN matches (synthetic test data outside the schedule cache; pre-2012
    rows) are zero-filled in place — ``build_game_history_arrays`` already
    zero-fills missing history slots, so consistent zero handling here keeps
    real-game tokens distinguishable from padding only by the mask, not by
    a stray NaN.
    """
    if "_team_box_score_merged" in df.columns:
        return df

    lookup = _build_team_box_score_lookup()

    team_lookup = lookup[["season", "week", "team"] + TEAM_BOX_SCORE_FEATURES].rename(
        columns={"team": "recent_team"}
    )
    opp_lookup = lookup[["season", "week", "team"] + OPP_BOX_SCORE_FEATURES].rename(
        columns={"team": "opponent_team"}
    )

    n_before = len(df)
    df_merged = df.merge(team_lookup, on=["season", "week", "recent_team"], how="left")
    if "opponent_team" in df_merged.columns:
        df_merged = df_merged.merge(opp_lookup, on=["season", "week", "opponent_team"], how="left")
    else:
        df_merged["opp_team_points_scored"] = 0.0

    if len(df_merged) != n_before:
        df_merged = df_merged.drop_duplicates(subset=["player_id", "season", "week"], keep="first")

    new_cols = TEAM_BOX_SCORE_FEATURES + OPP_BOX_SCORE_FEATURES
    for col in new_cols:
        values = df_merged[col].fillna(0.0).to_numpy()
        df[col] = values

    n_missing = (
        df_merged[TEAM_BOX_SCORE_FEATURES[0]].isna().sum()
        if TEAM_BOX_SCORE_FEATURES[0] in df_merged.columns
        else 0
    )
    if n_missing > 0:
        tag = f" [{label}]" if label else ""
        print(f"  WARNING:{tag} {n_missing} rows have no team box-score match (filled with 0)")

    df["_team_box_score_merged"] = True
    return df
