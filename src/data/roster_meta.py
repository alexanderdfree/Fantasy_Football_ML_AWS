"""Player age + rookie-status enrichment for serving row payloads.

Historical sources are the already-synced ``data/raw`` caches (no new fetches at
serving time). Live artifact builders may instead supply freshly fetched weekly
rosters and schedules without modifying those frozen historical caches.
The seasonal rosters parquet carries ``birth_date`` / ``entry_year`` /
``rookie_year`` per (player, season) — note ``loader._fetch_rosters`` stringifies
object columns, so dates/years must be re-parsed with ``errors="coerce"`` — and
the schedules parquet supplies each (season, week, team) game date so ``age`` is
as-of kickoff, not as-of today.

Best-effort by design: a missing/stale parquet leaves the ``age`` /
``is_rookie`` columns as NaN (the frontend feature-detects and hides the Age /
Rookies filters), never raises into the serving data build. DST rows are
team-level units with no roster identity, so they stay NaN too.

``is_rookie`` is carried as a float (1.0 / 0.0 / NaN) in the frame so the
predictions-cache parquet round-trip stays clean; ``serialization`` converts to
bool/None at the JSON boundary.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, SEASONS

logger = logging.getLogger(__name__)

_meta_lock = threading.Lock()
_meta_cache: pd.DataFrame | None = None
_gameday_cache: pd.DataFrame | None = None


def _rosters_path() -> str:
    return os.path.join(CACHE_DIR, f"rosters_{SEASONS[0]}_{SEASONS[-1]}.parquet")


def _schedules_path() -> str:
    return os.path.join(CACHE_DIR, f"schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet")


def _roster_meta(rosters: pd.DataFrame) -> pd.DataFrame:
    """Normalize seasonal or weekly rosters to unique player-season metadata."""
    rosters = rosters.copy()
    if "player_id" not in rosters and "gsis_id" in rosters:
        rosters["player_id"] = rosters["gsis_id"]
    if not {"player_id", "season"}.issubset(rosters):
        raise ValueError("rosters lack player_id/season columns")
    rosters["season"] = pd.to_numeric(rosters["season"], errors="coerce").astype("Int64")
    rosters = rosters.dropna(subset=["player_id", "season"])
    if "week" in rosters:
        # Weekly snapshots repeat identities, including players changing teams.
        # Metadata is player-season level; choose the newest supplied snapshot.
        rosters["week"] = pd.to_numeric(rosters["week"], errors="coerce")
        rosters = rosters.sort_values("week", kind="stable", na_position="first")
    meta = rosters.drop_duplicates(["player_id", "season"], keep="last").copy()
    missing = pd.Series(np.nan, index=meta.index)
    # Stringified by the loader's parquet-serialization coercion —
    # "None"/"nan" strings coerce to NaT/NaN here.
    meta["birth_date"] = pd.to_datetime(meta.get("birth_date", missing), errors="coerce")
    entry = pd.to_numeric(meta.get("entry_year", missing), errors="coerce")
    rookie = pd.to_numeric(meta.get("rookie_year", missing), errors="coerce")
    meta["entry_year"] = entry.fillna(rookie)
    return meta[["player_id", "season", "birth_date", "entry_year"]]


def _gameday_map(schedules: pd.DataFrame) -> pd.DataFrame:
    """Normalize cached or freshly fetched schedules to team game dates."""
    need = {"season", "week", "gameday", "home_team", "away_team"}
    if not need.issubset(schedules):
        raise ValueError(f"schedules lack columns: {sorted(need - set(schedules.columns))}")
    gd = pd.concat(
        [
            schedules[["season", "week", "gameday", side]].rename(columns={side: "team"})
            for side in ("home_team", "away_team")
        ],
        ignore_index=True,
    )
    gd["gameday"] = pd.to_datetime(gd["gameday"], errors="coerce")
    gd["season"] = pd.to_numeric(gd["season"], errors="coerce").astype("Int64")
    gd["week"] = pd.to_numeric(gd["week"], errors="coerce").astype("Int64")
    return gd.dropna(subset=["gameday"]).drop_duplicates(["season", "week", "team"])


def _live_roster_meta(roster: pd.DataFrame) -> pd.DataFrame:
    """Season-bound ESPN display metadata, independent of model input features."""
    if not {"player_id", "roster_season"}.issubset(roster):
        return pd.DataFrame()
    meta = roster.rename(columns={"roster_season": "season"}).copy()
    meta["season"] = pd.to_numeric(meta["season"], errors="coerce").astype("Int64")
    meta = meta.dropna(subset=["player_id", "season"]).drop_duplicates(
        ["player_id", "season"], keep="last"
    )
    missing = pd.Series(np.nan, index=meta.index)
    # ESPN's exact DOB uses a timestamp, not an age rounded as of fetch time.
    meta["live_birth_date"] = pd.to_datetime(
        meta.get("roster_birth_date", missing), errors="coerce", utc=True
    ).dt.tz_localize(None)
    experience = pd.to_numeric(meta.get("roster_experience_years", missing), errors="coerce")
    debut = pd.to_numeric(meta.get("roster_debut_year", missing), errors="coerce")
    # Zero experience denotes a rookie on the current ESPN roster (e.g. the
    # 2026 Raiders roster's Mike Washington Jr.). Do not subtract experience
    # from season to invent an entry year: accrued seasons can have gaps.
    meta["live_is_rookie"] = experience.eq(0).astype(float).where(experience.ge(0))
    # A previous NFL debut establishes veteran status if experience is absent.
    # A current debut alone is insufficient: a player can have sat out a year.
    meta.loc[meta["live_is_rookie"].isna() & debut.lt(meta["season"]), "live_is_rookie"] = 0.0
    return meta[["player_id", "season", "live_birth_date", "live_is_rookie"]]


def load_roster_meta() -> pd.DataFrame:
    """(player_id, season) → birth_date (datetime64), entry_year (float).

    ``entry_year`` falls back to ``rookie_year`` when absent — both mean "the
    season the player entered the league" in the nflverse roster schema.
    Memoized; returns an empty typed frame when the parquet is unavailable.
    """
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache
    with _meta_lock:
        if _meta_cache is not None:
            return _meta_cache
        empty = pd.DataFrame(
            {
                "player_id": pd.Series(dtype=str),
                "season": pd.Series(dtype="int64"),
                "birth_date": pd.Series(dtype="datetime64[ns]"),
                "entry_year": pd.Series(dtype=float),
            }
        )
        path = _rosters_path()
        try:
            _meta_cache = _roster_meta(pd.read_parquet(path))
        except Exception as exc:  # noqa: BLE001 — data-source boundary, degrade gracefully
            logger.warning("roster meta unavailable (%s): %s", path, exc)
            _meta_cache = empty
        return _meta_cache


def load_gameday_map() -> pd.DataFrame:
    """(season, week, team) → gameday (datetime64), from the schedules cache."""
    global _gameday_cache
    if _gameday_cache is not None:
        return _gameday_cache
    with _meta_lock:
        if _gameday_cache is not None:
            return _gameday_cache
        empty = pd.DataFrame(
            {
                "season": pd.Series(dtype="int64"),
                "week": pd.Series(dtype="int64"),
                "team": pd.Series(dtype=str),
                "gameday": pd.Series(dtype="datetime64[ns]"),
            }
        )
        path = _schedules_path()
        try:
            _gameday_cache = _gameday_map(pd.read_parquet(path))
        except Exception as exc:  # noqa: BLE001 — data-source boundary, degrade gracefully
            logger.warning("gameday map unavailable (%s): %s", path, exc)
            _gameday_cache = empty
        return _gameday_cache


def attach_age_and_rookie(
    results: pd.DataFrame,
    *,
    rosters: pd.DataFrame | None = None,
    schedules: pd.DataFrame | None = None,
    live_roster: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add ``age`` (float years at kickoff) + ``is_rookie`` (1.0/0.0/NaN) columns.

    Joins roster meta on (player_id, season) and gamedays on (season, week,
    recent_team). Rows without a roster match (DST team units, unknown ids)
    stay NaN. When a row's gameday is missing (bye normalization, schedule
    gaps), age falls back to a nominal Dec 1 of the season — a ±few-months
    approximation that never shifts the bucket by more than one year.

    Artifact builders can supply current-season ``rosters`` and ``schedules``;
    omitted arguments retain the historical cache reads. Explicit empty inputs
    mean unavailable metadata and never fall back to a different season's data.
    Optional ESPN ``live_roster`` supplies exact DOB and current-roster rookie
    status only where nflverse metadata is missing, matched to its source season.
    """
    results = results.copy()
    if not {"player_id", "season", "week"}.issubset(results.columns):
        results["age"] = np.nan
        results["is_rookie"] = np.nan
        return results

    if rosters is None:
        meta = load_roster_meta()
    elif rosters.empty:
        meta = pd.DataFrame()
    else:
        try:
            meta = _roster_meta(rosters)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("supplied roster meta unavailable: %s", exc)
            meta = pd.DataFrame()
    merged = results[["player_id", "season", "week"]].copy()
    if "recent_team" in results.columns:
        merged["team"] = results["recent_team"].values
    else:
        merged["team"] = pd.NA
    merged["season"] = pd.to_numeric(merged["season"], errors="coerce").astype("Int64")
    merged["week"] = pd.to_numeric(merged["week"], errors="coerce").astype("Int64")
    if meta.empty:
        merged = merged.reset_index(drop=True)
        merged["birth_date"] = pd.NaT
        merged["entry_year"] = np.nan
    else:
        merged = merged.merge(meta, on=["player_id", "season"], how="left", validate="many_to_one")
    merged["live_is_rookie"] = np.nan
    if live_roster is not None and not live_roster.empty:
        live_meta = _live_roster_meta(live_roster)
        if not live_meta.empty:
            merged = merged.drop(columns="live_is_rookie").merge(
                live_meta, on=["player_id", "season"], how="left", validate="many_to_one"
            )
            merged["birth_date"] = merged["birth_date"].fillna(merged["live_birth_date"])

    if schedules is None:
        gd = load_gameday_map()
    elif schedules.empty:
        gd = pd.DataFrame()
    else:
        try:
            gd = _gameday_map(schedules)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("supplied gameday map unavailable: %s", exc)
            gd = pd.DataFrame()
    if not gd.empty:
        gd = gd.copy()
        gd["season"] = pd.to_numeric(gd["season"], errors="coerce").astype("Int64")
        merged = merged.merge(gd, on=["season", "week", "team"], how="left", validate="many_to_one")
    else:
        merged["gameday"] = pd.NaT

    season_years = pd.to_numeric(merged["season"], errors="coerce").astype(float)
    fallback = pd.to_datetime(
        {"year": season_years.fillna(SEASONS[-1]).astype(int), "month": 12, "day": 1},
        errors="coerce",
    )
    asof = merged["gameday"].fillna(fallback)
    birth = merged["birth_date"]
    birthday_pending = (asof.dt.month < birth.dt.month) | (
        (asof.dt.month == birth.dt.month) & (asof.dt.day < birth.dt.day)
    )
    # Calendar age increments on the birthday, including leap-year boundaries;
    # flooring elapsed days / 365.25 can be a year wrong on the birthday itself.
    age = asof.dt.year - birth.dt.year - birthday_pending.astype(int)
    results["age"] = age.to_numpy()
    rookie = pd.Series(
        np.where(
            merged["entry_year"].notna() & season_years.notna(),
            (merged["entry_year"] == season_years).astype(float),
            np.nan,
        ),
        index=merged.index,
    )
    results["is_rookie"] = rookie.fillna(merged["live_is_rookie"]).to_numpy()
    if "position" in results:
        results.loc[results["position"].eq("DST"), ["age", "is_rookie"]] = np.nan
    return results


def reset_caches() -> None:
    """Test hook: drop the memoized parquet reads."""
    global _meta_cache, _gameday_cache
    with _meta_lock:
        _meta_cache = None
        _gameday_cache = None
