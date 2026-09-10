"""Recover lagging live-season QBR from the same ESPN source as nflverse.

The archive remains authoritative for values it already validated. This
CI-builder-only fallback fills missing completed player-games; it never writes
historical caches or substitutes a different quarterback statistic.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from src.data import nfl_source
from src.data.external_sources import QBR_FEATURE_COLUMNS, bridge_qbr_to_gsis
from src.serving import espn_live

_QBR_URL = "https://site.web.api.espn.com/apis/fitt/v3/sports/football/nfl/qbr"
_KEYS = ["player_id", "season", "week"]
_STAT_NAMES = {"schedAdjQBR": "qbr_total", "qbpaa": "pts_added"}


def _parse_qbr(payload: dict, season: int, week: int) -> pd.DataFrame:
    """Parse the provider's named QBR statistics, rejecting ambiguous records."""
    selected = payload["currentValues"]
    if (
        int(selected["season"]) != season
        or int(selected["week"]) != week
        or int(selected["seasontype"]) != 2
        or selected["qbrType"] != "weeks"
    ):
        raise ValueError("ESPN QBR response does not match requested regular-season week")
    if int(payload.get("pagination", {}).get("pages", 1)) > 1:
        raise ValueError("ESPN QBR response is incomplete (multiple pages)")
    headers = {c["name"]: c["names"] for c in payload["categories"]}
    rows = []
    for record in payload["athletes"]:
        game = record["game"]
        if int(game["weekNumber"]) != week:
            raise ValueError("ESPN QBR game week disagrees with requested week")
        date = pd.to_datetime(game["date"], utc=True, errors="coerce")
        if pd.isna(date) or date > pd.Timestamp.now("UTC"):
            raise ValueError("ESPN QBR game date is missing or in the future")
        stats = {}
        for category in record["categories"]:
            names = headers[category["name"]]
            values = category["totals"]
            if len(names) != len(values):
                raise ValueError("ESPN QBR stat names and values have different lengths")
            stats.update(zip(names, values, strict=True))
        row = {
            "season": season,
            "season_type": "Regular",
            "game_week": week,
            "game_id": str(game["id"]),
            "player_id": record["athlete"]["id"],
            **{target: float(stats[source]) for source, target in _STAT_NAMES.items()},
        }
        if not all(np.isfinite(row[c]) for c in QBR_FEATURE_COLUMNS):
            raise ValueError("ESPN QBR contains a nonfinite statistic")
        if not 0 <= row["qbr_total"] <= 100:
            raise ValueError("ESPN Total QBR is outside its 0-100 scale")
        rows.append(row)
    result = pd.DataFrame(
        rows,
        columns=[
            "season",
            "season_type",
            "game_week",
            "game_id",
            "player_id",
            *QBR_FEATURE_COLUMNS,
        ],
    ).drop_duplicates()
    if result.duplicated(["player_id", "season", "game_week"], keep=False).any():
        raise ValueError("ESPN QBR has conflicting duplicate player-week observations")
    return result


def recover_qbr(
    current: pd.DataFrame, schedules: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict]:
    """Fill missing live QB QBR/points-added using validated completed games.

    ESPN's qualified weekly feed need not include every backup QB. Coverage
    therefore describes observed QB player-games, not an assertion that missing
    QBR means zero. Failures leave the existing trained missing-data encoding
    intact. Returned metadata distinguishes recovered, partial and absent data.
    """
    result = current.copy()
    for column in QBR_FEATURE_COLUMNS:
        if column not in result:
            result[column] = np.nan
    completed = schedules.loc[
        schedules["season"].eq(season)
        & schedules["game_type"].eq("REG")
        & schedules["home_score"].notna()
        & schedules["away_score"].notna()
        & pd.to_datetime(schedules["gameday"], utc=True, errors="coerce").le(
            pd.Timestamp.now("UTC")
        )
    ].copy()
    completed["_game_id"] = pd.to_numeric(completed["espn"], errors="coerce").astype("Int64")
    teams = pd.concat(
        [
            completed[["season", "week", "_game_id", side]].rename(columns={side: "recent_team"})
            for side in ("home_team", "away_team")
        ],
        ignore_index=True,
    ).dropna(subset=["_game_id"])
    eligible = result.loc[result["position"].eq("QB") & result["season"].eq(season)].merge(
        teams, on=["season", "week", "recent_team"], how="inner", validate="many_to_one"
    )
    eligible = eligible.drop_duplicates(_KEYS)
    needed = eligible[list(QBR_FEATURE_COLUMNS)].isna().any(axis=1)
    weeks = sorted(eligible.loc[needed, "week"].astype(int).unique().tolist())
    metadata = {
        "source": "ESPN QBR (nflverse upstream)",
        "url": _QBR_URL,
        "retrieved_at": None,
        "season": season,
        "eligible_qb_rows": len(eligible),
        "observed_before_rows": int((~needed).sum()),
        "observed_rows": int((~needed).sum()),
        "recovered_rows": 0,
        "recovered_cells": 0,
        "requested_weeks": weeks,
        "validated_games": [],
        "rejected_rows": 0,
        "errors": [],
    }

    def status() -> str:
        if not len(eligible):
            return "not_required"
        if metadata["observed_rows"] == len(eligible):
            return "available"
        return "partial" if metadata["observed_rows"] else "unavailable"

    if not weeks:
        metadata["status"] = status()
        return result, metadata

    def fetch(week: int) -> pd.DataFrame:
        url = (
            _QBR_URL
            + "?"
            + urlencode(
                {
                    "qbrType": "weeks",
                    "seasontype": 2,
                    # Short appearances still belong in game-history tokens.
                    # Qualification is a leaderboard filter, not data validity.
                    "isqualified": "false",
                    "season": season,
                    "week": week,
                    "limit": 100,
                }
            )
        )
        return _parse_qbr(espn_live._get_json(url), season, week)

    frames = []
    with ThreadPoolExecutor(max_workers=min(4, len(weeks))) as pool:
        futures = {pool.submit(fetch, week): week for week in weeks}
        for future in as_completed(futures):
            week = futures[future]
            try:
                raw = future.result()
                valid = nfl_source._validated_qbr_games(raw, completed)
                metadata["rejected_rows"] += len(raw) - len(valid)
                frames.append(valid)
            except Exception as exc:  # network/provider schema boundary
                metadata["errors"].append({"week": week, "error": str(exc)[:300]})
    metadata["retrieved_at"] = datetime.now(UTC).isoformat()
    metadata["errors"].sort(key=lambda e: e["week"])
    if frames:
        raw = pd.concat(frames, ignore_index=True)
        if not raw.empty:
            try:
                ids = nfl_source.player_ids()
                # The normal bridge drops game_id. Carry it through by group
                # so an otherwise valid QBR from the wrong game cannot fill a
                # player's actual weekly observation after the identity join.
                bridged = pd.concat(
                    [
                        bridge_qbr_to_gsis(group, ids).assign(_game_id=int(game_id))
                        for game_id, group in raw.groupby("game_id")
                    ],
                    ignore_index=True,
                )
                matched = eligible[_KEYS + ["_game_id"]].merge(
                    bridged, on=_KEYS + ["_game_id"], how="inner", validate="one_to_one"
                )
                metadata["validated_games"] = sorted(matched["_game_id"].astype(str).unique())
                recovered = matched.set_index(_KEYS)
                result_keys = pd.MultiIndex.from_frame(result[_KEYS])
                changed = pd.Series(False, index=result.index)
                for column in QBR_FEATURE_COLUMNS:
                    incoming = pd.Series(
                        recovered[column].reindex(result_keys).to_numpy(), index=result.index
                    )
                    fill = result[column].isna() & incoming.notna() & result["position"].eq("QB")
                    result.loc[fill, column] = incoming.loc[fill]
                    metadata["recovered_cells"] += int(fill.sum())
                    changed |= fill
                metadata["recovered_rows"] = int(changed.sum())
                observed = result.set_index(_KEYS)[list(QBR_FEATURE_COLUMNS)]
                metadata["observed_rows"] = int(
                    observed.reindex(pd.MultiIndex.from_frame(eligible[_KEYS]))
                    .notna()
                    .all(axis=1)
                    .sum()
                )
            except Exception as exc:  # external identity/schema boundary
                metadata["errors"].append({"week": None, "error": str(exc)[:300]})
    metadata["status"] = status()
    return result, metadata
