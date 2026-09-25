"""Complete D/ST touchdown and blocked-punt counts from nflverse play-by-play.

Team stats separate fumble-return touchdowns from ``def_tds`` without identifying
whether the recovery was offensive. Reconstruct the existing defensive and
special-teams TD targets from scoring plays instead of adding ambiguous columns.
The compact team-week cache is built with the training inputs, never in serving.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path

import pandas as pd

from src import config
from src.data import nfl_source
from src.data.cache_io import atomic_write_parquet
from src.data.external_sources import _seasons_cache_signature
from src.shared.weather_features import TEAM_CODE_NORMALIZATION

SCORING_KEYS = ("team", "season", "week")
SCORING_COLUMNS = ("def_tds", "special_teams_tds", "def_punt_blocks")
PBP_COLUMNS = (
    "season",
    "season_type",
    "week",
    "game_id",
    "play_id",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "td_team",
    "touchdown",
    "punt_attempt",
    "kickoff_attempt",
    "field_goal_attempt",
    "extra_point_attempt",
    "two_point_attempt",
    "punt_blocked",
    "play_type",
)


def aggregate_dst_scoring_events(pbp: pd.DataFrame) -> pd.DataFrame:
    """Return complete regular-season team-week counts, including zero games.

    Defensive TDs score for the opponent of the initial possession team.
    Kicking plays count in the special-teams target, including kicking-team
    recoveries and blocked-FG returns. Offensive own-fumble TDs and conversion
    attempts do not count. Each scoring play belongs to exactly one TD target.
    """
    required = set(PBP_COLUMNS) - {"play_type"}
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"D/ST scoring PBP is missing columns: {sorted(missing)}")
    pbp = pbp.loc[pbp["season_type"].eq("REG")].copy()
    if pbp.empty:
        return pd.DataFrame(columns=[*SCORING_KEYS, *SCORING_COLUMNS])
    pbp = pbp.drop_duplicates(["game_id", "play_id"])
    for column in ("home_team", "away_team", "posteam", "defteam", "td_team"):
        pbp[column] = pbp[column].replace(TEAM_CODE_NORMALIZATION)
    games = pd.concat(
        [
            pbp[["season", "week", side]].rename(columns={side: "team"})
            for side in ("home_team", "away_team")
        ],
        ignore_index=True,
    ).drop_duplicates(list(SCORING_KEYS))
    valid = ~pbp[["extra_point_attempt", "two_point_attempt"]].eq(1).any(axis=1)
    if "play_type" in pbp:
        valid &= pbp["play_type"].ne("no_play")
    special_teams = pbp[["punt_attempt", "kickoff_attempt", "field_goal_attempt"]].eq(1).any(axis=1)
    touchdowns = valid & pbp["touchdown"].eq(1) & pbp["td_team"].notna()
    defensive = pbp["posteam"].notna() & pbp["td_team"].ne(pbp["posteam"])
    event_masks = {
        "def_tds": (touchdowns & defensive & ~special_teams, "td_team"),
        "special_teams_tds": (touchdowns & special_teams, "td_team"),
        "def_punt_blocks": (valid & pbp["punt_blocked"].eq(1), "defteam"),
    }
    for column, (mask, team_column) in event_masks.items():
        counts = (
            pbp.loc[mask]
            .groupby([team_column, "season", "week"])
            .size()
            .rename(column)
            .reset_index()
            .rename(columns={team_column: "team"})
        )
        games = games.merge(counts, on=list(SCORING_KEYS), how="left", validate="one_to_one")
        games[column] = games[column].fillna(0).astype("int32")
    return (
        games[[*SCORING_KEYS, *SCORING_COLUMNS]]
        .sort_values(list(SCORING_KEYS))
        .reset_index(drop=True)
    )


def _valid_cache(frame: pd.DataFrame, seasons: list[int]) -> bool:
    required = {*SCORING_KEYS, *SCORING_COLUMNS}
    return (
        required.issubset(frame.columns)
        and set(frame["season"]) == set(seasons)
        and not frame.duplicated(list(SCORING_KEYS)).any()
        and frame[list(SCORING_KEYS)].notna().all().all()
        and frame[list(SCORING_COLUMNS)].ge(0).all().all()
    )


def load_dst_scoring_events(
    seasons: list[int], cache_dir: str | Path | None = None, *, allow_fetch: bool = True
) -> pd.DataFrame:
    """Load or build a versioned compact cache; cache-only callers never fetch.

    Missing/failed seasons are errors and never publish a partial cache. CI's
    training-input builder prewarms this cache; serving calls with
    ``allow_fetch=False`` and consumes the same released input as training.
    """
    seasons = sorted({int(season) for season in seasons})
    if not seasons:
        raise ValueError("D/ST scoring requires at least one season")
    directory = Path(config.CACHE_DIR if cache_dir is None else cache_dir)
    path = directory / f"dst_scoring_pbp_v1_{_seasons_cache_signature(seasons)}.parquet"
    if path.exists():
        frame = pd.read_parquet(path)
        if _valid_cache(frame, seasons):
            return frame
    if not allow_fetch:
        raise FileNotFoundError(
            f"Complete D/ST scoring cache unavailable at {path}; rebuild/reload the training data release"
        )

    from src.data.release import assert_source_fetch_allowed

    assert_source_fetch_allowed(path)

    def fetch(season: int) -> pd.DataFrame:
        frame = aggregate_dst_scoring_events(nfl_source.pbp_data([season], PBP_COLUMNS))
        if frame.empty or set(frame["season"]) != {season}:
            raise ValueError(f"D/ST scoring PBP has no complete regular-season data for {season}")
        return frame

    with ThreadPoolExecutor(max_workers=min(3, len(seasons))) as pool:
        futures = [pool.submit(copy_context().run, fetch, season) for season in seasons]
        frame = pd.concat((future.result() for future in futures), ignore_index=True)
    if not _valid_cache(frame, seasons):
        raise ValueError("Invalid D/ST scoring PBP aggregation; refusing to cache")
    atomic_write_parquet(frame, str(path), index=False)
    return frame
