"""Player age + rookie-status enrichment for serving row payloads.

Sources are the already-synced ``data/raw`` caches (no new fetches at serving
time): the seasonal rosters parquet carries ``birth_date`` / ``entry_year`` /
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
            cols = ["player_id", "season", "birth_date", "entry_year", "rookie_year"]
            rosters = pd.read_parquet(path)
            present = [c for c in cols if c in rosters.columns]
            rosters = rosters[present]
            if "player_id" not in rosters.columns or "season" not in rosters.columns:
                raise ValueError(f"rosters cache lacks id/season columns: {present}")
            meta = rosters.drop_duplicates(["player_id", "season"]).copy()
            # Stringified by the loader's parquet-serialization coercion —
            # "None"/"nan" strings coerce to NaT/NaN here.
            meta["birth_date"] = pd.to_datetime(meta.get("birth_date"), errors="coerce")
            entry = pd.to_numeric(meta.get("entry_year"), errors="coerce")
            rookie = pd.to_numeric(meta.get("rookie_year"), errors="coerce")
            meta["entry_year"] = entry.fillna(rookie) if rookie is not None else entry
            meta["season"] = pd.to_numeric(meta["season"], errors="coerce").astype("Int64")
            _meta_cache = meta[["player_id", "season", "birth_date", "entry_year"]]
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
            sched = pd.read_parquet(path)
            need = {"season", "week", "gameday", "home_team", "away_team"}
            if not need.issubset(sched.columns):
                raise ValueError(
                    f"schedules cache lacks columns: {sorted(need - set(sched.columns))}"
                )
            frames = []
            for side in ("home_team", "away_team"):
                part = sched[["season", "week", "gameday", side]].rename(columns={side: "team"})
                frames.append(part)
            gd = pd.concat(frames, ignore_index=True)
            gd["gameday"] = pd.to_datetime(gd["gameday"], errors="coerce")
            gd = gd.dropna(subset=["gameday"]).drop_duplicates(["season", "week", "team"])
            _gameday_cache = gd[["season", "week", "team", "gameday"]]
        except Exception as exc:  # noqa: BLE001 — data-source boundary, degrade gracefully
            logger.warning("gameday map unavailable (%s): %s", path, exc)
            _gameday_cache = empty
        return _gameday_cache


def attach_age_and_rookie(results: pd.DataFrame) -> pd.DataFrame:
    """Add ``age`` (float years at kickoff) + ``is_rookie`` (1.0/0.0/NaN) columns.

    Joins roster meta on (player_id, season) and gamedays on (season, week,
    recent_team). Rows without a roster match (DST team units, unknown ids)
    stay NaN. When a row's gameday is missing (bye normalization, schedule
    gaps), age falls back to a nominal Dec 1 of the season — a ±few-months
    approximation that never shifts the bucket by more than one year.
    """
    results = results.copy()
    if not {"player_id", "season", "week"}.issubset(results.columns):
        results["age"] = np.nan
        results["is_rookie"] = np.nan
        return results

    meta = load_roster_meta()
    if meta.empty:
        results["age"] = np.nan
        results["is_rookie"] = np.nan
        return results

    merged = results[["player_id", "season", "week"]].copy()
    if "recent_team" in results.columns:
        merged["team"] = results["recent_team"].values
    else:
        merged["team"] = pd.NA
    merged["season"] = pd.to_numeric(merged["season"], errors="coerce").astype("Int64")
    merged = merged.merge(meta, on=["player_id", "season"], how="left")

    gd = load_gameday_map()
    if not gd.empty:
        gd = gd.copy()
        gd["season"] = pd.to_numeric(gd["season"], errors="coerce").astype("Int64")
        merged = merged.merge(gd, on=["season", "week", "team"], how="left")
    else:
        merged["gameday"] = pd.NaT

    season_years = pd.to_numeric(merged["season"], errors="coerce").astype(float)
    fallback = pd.to_datetime(
        {"year": season_years.fillna(SEASONS[-1]).astype(int), "month": 12, "day": 1},
        errors="coerce",
    )
    asof = merged["gameday"].fillna(fallback)
    age_days = (asof - merged["birth_date"]).dt.days
    results["age"] = np.floor(age_days / 365.25)
    results["is_rookie"] = np.where(
        merged["entry_year"].notna() & season_years.notna(),
        (merged["entry_year"] == season_years).astype(float),
        np.nan,
    )
    return results


def reset_caches() -> None:
    """Test hook: drop the memoized parquet reads."""
    global _meta_cache, _gameday_cache
    with _meta_lock:
        _meta_cache = None
        _gameday_cache = None
