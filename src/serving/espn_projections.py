"""Historical ESPN projections for comparisons, using the project's scoring rules.

The public stock-PPR league returns a whole season in ``kona_player_info``.
Only weekly projection entries are retained; actuals and season totals never
enter the frame. Cache each season separately so subset requests cannot collide.
The 2023 Week 1 archive is incomplete and excluded across all positions.
Historical values are not documented as immutable kickoff snapshots.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR
from src.data import nfl_source
from src.data.cache_io import atomic_write_parquet
from src.serving import espn_live
from src.shared.aggregate_targets import (
    DST_TARGETS,
    K_TARGETS,
    POSITION_TARGET_MAP,
    predictions_to_fantasy_points,
)

ESPN_MIN_SEASON = 2018
ESPN_EXCLUDED_WEEKS = frozenset({(2023, 1)})
ESPN_NOTE = (
    "ESPN weekly raw-stat projections, rescored with our position-specific rules "
    "(including field-goal yardage and DST tiers), rather than ESPN's stock points. "
    "Archive available from 2018; incomplete 2023 Week 1 is excluded. Missing "
    "projections stay missing. Historical kickoff-snapshot provenance is unverified."
)
_POS_MAP = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
# Stable ESPN franchise IDs; relocation aliases normalized to nflverse codes.
_TEAM_MAP = dict(
    zip(
        (*range(1, 31), 33, 34),
        [
            "ATL",
            "BUF",
            "CHI",
            "CIN",
            "CLE",
            "DAL",
            "DEN",
            "DET",
            "GB",
            "TEN",
            "IND",
            "KC",
            "LV",
            "LA",
            "MIA",
            "MIN",
            "NE",
            "NO",
            "NYG",
            "NYJ",
            "PHI",
            "ARI",
            "PIT",
            "LAC",
            "SF",
            "SEA",
            "TB",
            "WAS",
            "CAR",
            "JAX",
            "BAL",
            "HOU",
        ],
        strict=True,
    )
)
# ESPN's scoring-stat IDs, also documented in espn-api/football/constant.py.
_OFFENSE_STATS = {
    "passing_yards": "3",
    "passing_tds": "4",
    "interceptions": "20",
    "rushing_yards": "24",
    "rushing_tds": "25",
    "receptions": "53",
    "receiving_yards": "42",
    "receiving_tds": "43",
    "fumbles_lost": "72",
}
_DST_STATS = {
    "def_sacks": "99",
    "def_ints": "95",
    "def_fumble_rec": "96",
    "def_fumbles_forced": "106",
    "def_safeties": "98",
    "def_blocked_kicks": "97",
    "points_allowed": "120",
    "yards_allowed": "127",
}
_REQUIRED_STATS = {"K": {"214"}, "DST": {"120", "127"}}
_TARGETS = tuple(dict.fromkeys((*_OFFENSE_STATS, *K_TARGETS, *DST_TARGETS)))
_KEYS = ["player_id", "season", "week"]
_COLUMNS = ["espn_id", "player_name", "position", "season", "week", *_TARGETS]


def _normalize_season(payload: dict, season: int) -> pd.DataFrame:
    """Decode ESPN's raw statistics without treating placeholders as forecasts."""
    players = payload.get("players")
    if not isinstance(players, list):
        raise ValueError("ESPN response has no players list")
    if len(players) >= espn_live._FANTASY_FILTER_LIMIT:
        raise ValueError("ESPN player response reached the limit; possible truncated pool")
    rows = []
    for entry in players:
        player = entry.get("player") or {}
        pos = _POS_MAP.get(player.get("defaultPositionId"))
        if pos is None:
            continue
        espn_id = (
            _TEAM_MAP.get(player.get("proTeamId"))
            if pos == "DST"
            else espn_live._norm_espn_id(player.get("id"))
        )
        if not espn_id:
            continue
        for split in player.get("stats") or []:
            week = split.get("scoringPeriodId", 0)
            if (
                split.get("seasonId") != season
                or split.get("statSourceId") != 1
                or split.get("statSplitTypeId") != 1
                or not 1 <= week <= (17 if season < 2021 else 18)
                or (season, week) in ESPN_EXCLUDED_WEEKS
            ):
                continue
            stats = split.get("stats") or {}
            if pos in _REQUIRED_STATS and not _REQUIRED_STATS[pos].issubset(stats):
                continue
            if pos in POSITION_TARGET_MAP and not any(
                _OFFENSE_STATS[target] in stats for target in POSITION_TARGET_MAP[pos]
            ):
                continue
            # Sparse absent fields are zero; malformed/nonfinite values fail the
            # season instead of being cached as fabricated zero projections.
            raw = {key: float(value) for key, value in stats.items()}
            if not all(np.isfinite(value) for value in raw.values()):
                raise ValueError(f"Nonfinite ESPN projection: {season}/{week}/{espn_id}")
            targets = dict.fromkeys(_TARGETS, 0.0)
            if pos == "K":
                targets.update(
                    fg_yard_points=raw["214"] * 0.1,
                    pat_points=raw.get("86", 0.0),
                    fg_misses=raw.get("85", 0.0),
                    xp_misses=raw.get("88", 0.0),
                )
            elif pos == "DST":
                targets.update({name: raw.get(key, 0.0) for name, key in _DST_STATS.items()})
                # 94 excludes blocked-kick return TDs (93). 105 already combines
                # defense + special teams, so using it would count returns twice.
                targets["def_tds"] = raw.get("94", 0.0) + raw.get("93", 0.0)
                targets["special_teams_tds"] = raw.get("101", 0.0) + raw.get("102", 0.0)
            else:
                targets.update({name: raw.get(key, 0.0) for name, key in _OFFENSE_STATS.items()})
            if not any(targets.values()):
                continue
            rows.append(
                {
                    "espn_id": espn_id,
                    "player_name": player.get("fullName", ""),
                    "position": pos,
                    "season": season,
                    "week": week,
                    **targets,
                }
            )
    frame = pd.DataFrame(rows, columns=_COLUMNS)
    if frame.duplicated(["espn_id", "season", "week"]).any():
        raise ValueError("Duplicate ESPN player-week projections")
    return frame


def load_espn_projections(
    seasons: Sequence[int],
    cache_dir: str = CACHE_DIR,
    force_refresh: bool = False,
    *,
    weeks: Sequence[int] | None = None,
    reader=None,
) -> pd.DataFrame:
    """Fetch/cache raw season frames, then select requested regular-season weeks."""
    seasons = sorted({int(s) for s in seasons})
    if not seasons or seasons[0] < ESPN_MIN_SEASON:
        raise ValueError(f"ESPN requires nonempty seasons from {ESPN_MIN_SEASON} onward")
    selected_weeks = None if weeks is None else {int(w) for w in weeks}
    if selected_weeks is not None and (
        not selected_weeks or not selected_weeks <= set(range(1, 19))
    ):
        raise ValueError("weeks must be a nonempty subset of 1..18")
    reader = reader or espn_live._get_json
    headers = {
        "X-Fantasy-Filter": json.dumps(
            {
                "players": {
                    "limit": espn_live._FANTASY_FILTER_LIMIT,
                    "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
                }
            }
        )
    }

    def load_season(season):
        path = Path(cache_dir) / f"espn_projections_v1_{season}.parquet"
        if path.exists() and not force_refresh:
            return pd.read_parquet(path)
        frame = _normalize_season(
            reader(espn_live._fantasy_projections_url(season, 1), headers=headers), season
        )
        if frame.empty:
            raise RuntimeError(f"No ESPN projections for {season}; not caching an empty response")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(frame, path)
        return frame

    with ThreadPoolExecutor(max_workers=min(3, len(seasons))) as pool:
        frame = pd.concat(pool.map(load_season, seasons), ignore_index=True)
    if selected_weeks is not None:
        frame = frame[frame["week"].isin(selected_weeks)]
    return frame.sort_values(["season", "week", "position", "espn_id"]).reset_index(drop=True)


def load_espn_with_gsis_id(
    seasons: Sequence[int],
    cache_dir: str = CACHE_DIR,
    force_refresh: bool = False,
    *,
    weeks: Sequence[int] | None = None,
    reader=None,
    player_ids_loader=None,
) -> pd.DataFrame:
    """Bridge athlete IDs through nflverse; DST keeps its normalized franchise key."""
    frame = load_espn_projections(seasons, cache_dir, force_refresh, weeks=weeks, reader=reader)
    ids = (player_ids_loader or nfl_source.player_ids)()
    bridge = ids[["espn_id", "gsis_id"]].dropna().copy()
    bridge["espn_id"] = bridge["espn_id"].map(espn_live._norm_espn_id)
    bridge = bridge.drop_duplicates("espn_id")
    frame = frame.merge(bridge, on="espn_id", how="left", validate="many_to_one")
    frame = frame.rename(columns={"gsis_id": "player_id"})
    dst = frame["position"] == "DST"
    frame.loc[dst, "player_id"] = frame.loc[dst, "espn_id"]
    matched = frame["player_id"].notna()
    print(f"ESPN player-id join: {matched.sum()}/{len(frame)} rows matched")
    return frame.loc[matched].reset_index(drop=True)


def project_espn_to_fantasy(raw: pd.DataFrame, pos: str, scoring_format: str) -> pd.DataFrame:
    """Rescore every position through the same aggregator as model predictions."""
    frame = raw.loc[(raw["position"] == pos) & raw["player_id"].notna()]
    if frame.empty:
        return pd.DataFrame(columns=[*_KEYS, "espn_pred_total"])
    targets = K_TARGETS if pos == "K" else DST_TARGETS if pos == "DST" else POSITION_TARGET_MAP[pos]
    out = frame[_KEYS].copy()
    out["espn_pred_total"] = predictions_to_fantasy_points(
        pos,
        {name: frame[name].to_numpy() for name in targets},
        scoring_format,
    )
    return out
