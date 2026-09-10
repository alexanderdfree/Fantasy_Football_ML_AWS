"""Recover played offensive games that have no weekly statistical record."""

from __future__ import annotations

import pandas as pd

from src.data.nflcom_loader import schedule_team_code_normalization

_KEYS = ["player_id", "season", "week"]
_OFFENSIVE_POSITIONS = {"QB", "RB", "WR", "TE"}
# An absent stat line on a verified played game means no recorded events.
# Rates/averages stay missing: no attempts does not establish a zero success rate.
_EVENT_TOTALS = (
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "sack_yards",
    "sack_fumbles",
    "sack_fumbles_lost",
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_first_downs",
    "passing_epa",
    "passing_2pt_conversions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles",
    "rushing_fumbles_lost",
    "rushing_first_downs",
    "rushing_epa",
    "rushing_2pt_conversions",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_fumbles",
    "receiving_fumbles_lost",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_first_downs",
    "receiving_epa",
    "receiving_2pt_conversions",
    "special_teams_tds",
    "fantasy_points",
    "fantasy_points_ppr",
)


def restore_offensive_appearances(
    weekly: pd.DataFrame, snaps: pd.DataFrame, roster_positions: pd.DataFrame
) -> pd.DataFrame:
    """Append missing REG player-weeks with a known identity and offensive snaps.

    ``snaps`` has already been bridged from PFR to GSIS. Restore only games
    represented in the weekly feed, so a missing source season/game never turns
    into fabricated zero-stat labels. Existing statistical rows are untouched.
    The caller merges snap percentages and other game context after restoration.
    """
    required = {"gsis_id", "season", "week", "team", "opponent", "game_type", "offense_snaps"}
    if not required.issubset(snaps) or "season_type" not in weekly:
        return weekly
    played = snaps.loc[
        snaps["game_type"].eq("REG") & snaps["offense_snaps"].gt(0) & snaps["gsis_id"].notna()
    ].rename(columns={"gsis_id": "player_id", "team": "recent_team", "opponent": "opponent_team"})
    if played.empty:
        return weekly
    played = played.drop(columns=["position"], errors="ignore").merge(
        roster_positions, on=["player_id", "season"], how="inner", validate="many_to_one"
    )
    played = played.loc[played["position"].isin(_OFFENSIVE_POSITIONS)].copy()
    norm = schedule_team_code_normalization()
    for col in ("recent_team", "opponent_team"):
        played[col] = played[col].replace(norm)
    games = weekly.loc[weekly["season_type"].eq("REG"), ["season", "week", "recent_team"]].copy()
    games["recent_team"] = games["recent_team"].replace(norm)
    played = played.merge(
        games.drop_duplicates(), on=["season", "week", "recent_team"], how="inner"
    )
    existing = pd.MultiIndex.from_frame(weekly[_KEYS])
    played = played.loc[~pd.MultiIndex.from_frame(played[_KEYS]).isin(existing)]
    # Duplicate source records must not multiply the player-week population.
    played = played.sort_values(
        [*_KEYS, "offense_snaps", "recent_team"], ascending=[True, True, True, False, True]
    ).drop_duplicates(_KEYS)
    if played.empty:
        return weekly

    restored = played.reindex(columns=weekly.columns).copy()
    restored["season_type"] = "REG"
    for col in _EVENT_TOTALS:
        if col in restored:
            restored[col] = 0.0
    if "player" in played:
        for col in ("player_name", "player_display_name"):
            if col in restored:
                restored[col] = played["player"]
    print(f"  Restored {len(restored)} played offensive player-weeks with no statistical record")
    return pd.concat([weekly, restored], ignore_index=True)
