"""Fetch + cache + gsis-join Sleeper (RotoWire) weekly NFL projections.

Source: Sleeper's undocumented projections endpoint
``https://api.sleeper.app/projections/nfl/{season}/{week}`` (free, no auth). Every
record carries ``company: "rotowire"`` — so this is **one** additional expert
(RotoWire), not a consensus. Offense only (QB/RB/WR/TE) for now; Sleeper also
serves DST/K but those need team-keying / are totals-only and are out of scope.

Lives under ``src/analysis/`` (not ``src/data/``) on purpose: ``src/data/`` is a
global retrain trigger in ``src/scripts/scope_positions.py`` and this loader is
analysis-only. Mirrors the cache + network-defensiveness idiom of
``src/data/nflcom_loader.py``.

Two public entry points (parallel to ``nflcom_loader``):

    load_sleeper_projections(seasons, ...) -> pd.DataFrame
        One row per (sleeper_player_id, position, season, week). Raw stats mapped
        to our internal target names. Cached to
        ``data/raw/sleeper_projections_v1_{min}_{max}_{weeks}.parquet``.

    load_sleeper_with_gsis_id(seasons, ...) -> pd.DataFrame
        Same frame, joined to ``player_id`` (gsis_id) via the nflverse
        ``ff_playerids`` crosswalk (``nfl_source.player_ids()``), the same bridge
        pattern used for ESPN-QBR (``external_sources.py``) and PFR (``loader.py``).

PROVENANCE CAVEAT: Sleeper does not document whether these historical projections
are the as-of-kickoff snapshot or a later backfill. Spot evidence (fractional
expected-value stats that do not match actuals) suggests genuine pre-game
projections, but callers should sanity-check RotoWire's error magnitude against a
known expert (NFL.com) before trusting the comparison — see the comparison
script's provenance gate.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

import pandas as pd

from src.config import CACHE_DIR
from src.data import nfl_source

SLEEPER_BASE = "https://api.sleeper.app/projections/nfl"
SLEEPER_OFFENSE_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")
"""Positions ingested. Sleeper also serves K/DEF, but K is totals-only and DEF
needs team-keying — both out of scope for this offense-only cut."""

SLEEPER_DEFAULT_WEEKS = tuple(range(1, 19))  # 18-week regular season since 2021.
_CACHE_VERSION = "v1"
# Sleeper returns byte-identical placeholder junk (only adp_dd_ppr) for 2016/2017;
# genuine weekly projections start in 2018.
_MIN_SEASON = 2018
_RETRY_BACKOFF_S = 0.5
_PROVIDER = "rotowire"
_REQUEST_TIMEOUT_S = 30

# Sleeper ``stats`` field -> our internal target name. Shared across offensive
# positions; fields a position doesn't carry (e.g. a QB's rec_yd) fill to 0.
SLEEPER_STAT_MAP: dict[str, str] = {
    "pass_yd": "passing_yards",
    "pass_td": "passing_tds",
    "pass_int": "interceptions",
    "rush_yd": "rushing_yards",
    "rush_td": "rushing_tds",
    "rec": "receptions",
    "rec_yd": "receiving_yards",
    "rec_td": "receiving_tds",
    "fum_lost": "fumbles_lost",
}
_ALL_TARGET_COLUMNS: tuple[str, ...] = tuple(SLEEPER_STAT_MAP.values())


# ---------- Network ----------------------------------------------------------


def _default_reader(url: str) -> list:
    """Fetch a Sleeper projections URL and return the decoded JSON list."""
    req = urllib.request.Request(url, headers={"User-Agent": "fantasy-ml-research"})
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
        return json.load(resp)


def _projection_url(season: int, week: int, positions: Sequence[str]) -> str:
    pos_q = "".join(f"&position[]={p}" for p in positions)
    return f"{SLEEPER_BASE}/{season}/{week}?season_type=regular{pos_q}"


def _read_one_week(
    season: int,
    week: int,
    positions: Sequence[str],
    *,
    reader=_default_reader,
    max_retries: int = 1,
    backoff_s: float = _RETRY_BACKOFF_S,
) -> list | None:
    """Fetch one (season, week) projection list from Sleeper.

    Returns ``None`` on 404 / persistent connection error / empty payload. Retries
    transient errors (URLError / 5xx) once after ``backoff_s``. 404 is not retried
    (it's the expected signal for a week that doesn't exist yet). ``reader`` is
    injectable for tests — pass a stub returning a ``list`` to avoid network.
    """
    url = _projection_url(season, week, positions)
    for attempt in range(max_retries + 1):
        try:
            records = reader(url)
        except urllib.error.HTTPError as e:
            if getattr(e, "code", None) == 404:
                print(f"  WARN sleeper: skip {season} W{week} (404 / not found)")
                return None
            if attempt < max_retries:
                print(f"  WARN sleeper: transient HTTP {e.code} on {season} W{week}; retrying")
                time.sleep(backoff_s)
                continue
            print(f"  WARN sleeper: skip {season} W{week} (HTTP {e.code} after retry)")
            return None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_retries:
                print(f"  WARN sleeper: transient {type(e).__name__} on {season} W{week}; retrying")
                time.sleep(backoff_s)
                continue
            print(f"  WARN sleeper: skip {season} W{week} ({type(e).__name__} after retry)")
            return None
        else:
            return records if records else None
    return None


# ---------- Normalization ----------------------------------------------------


def _normalize_week(records: list, season: int, week: int) -> pd.DataFrame:
    """Map a (season, week) Sleeper record list to our normalized projection frame.

    Output columns: season, week, position, sleeper_player_id, player_name, team,
    company, sleeper_projected_pts, plus every target in ``_ALL_TARGET_COLUMNS``
    (0-filled where a position doesn't carry the stat). Non-offense / position-less
    records are dropped.
    """
    rows: list[dict] = []
    for rec in records:
        player = rec.get("player") or {}
        pos = player.get("position")
        if pos not in SLEEPER_OFFENSE_POSITIONS:
            continue
        stats = rec.get("stats") or {}
        row = {
            "season": int(rec.get("season", season)),
            "week": int(rec.get("week", week)),
            "position": pos,
            "sleeper_player_id": str(rec.get("player_id")),
            "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
            "team": player.get("team_abbr") or player.get("team") or "",
            "company": rec.get("company"),
            "sleeper_projected_pts": float(stats.get("pts_ppr") or 0.0),
        }
        for src_field, target in SLEEPER_STAT_MAP.items():
            row[target] = float(stats.get(src_field) or 0.0)
        rows.append(row)

    if not rows:
        cols = [
            "season",
            "week",
            "position",
            "sleeper_player_id",
            "player_name",
            "team",
            "company",
            "sleeper_projected_pts",
            *_ALL_TARGET_COLUMNS,
        ]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def _validate_seasons(seasons: Sequence[int]) -> list[int]:
    seasons = sorted({int(s) for s in seasons})
    if not seasons:
        raise ValueError("seasons must be a non-empty sequence of ints")
    too_early = [s for s in seasons if s < _MIN_SEASON]
    if too_early:
        raise ValueError(
            f"Sleeper projections are only genuine from {_MIN_SEASON} onward "
            f"(earlier seasons return placeholder data); got {too_early}"
        )
    return seasons


def _weeks_signature(weeks: Sequence[int]) -> str:
    uniq = sorted({int(w) for w in weeks})
    return (
        f"w{uniq[0]}-{uniq[-1]}"
        if uniq == list(range(uniq[0], uniq[-1] + 1))
        else f"w{'_'.join(map(str, uniq))}"
    )


# ---------- Public entry points ----------------------------------------------


def load_sleeper_projections(
    seasons: Sequence[int],
    cache_dir: str = CACHE_DIR,
    force_refresh: bool = False,
    *,
    weeks: Sequence[int] | None = None,
    positions: Sequence[str] = SLEEPER_OFFENSE_POSITIONS,
    reader=_default_reader,
) -> pd.DataFrame:
    """Fetch + cache Sleeper projections for one or more seasons (offense only).

    Cache: ``{cache_dir}/sleeper_projections_{_CACHE_VERSION}_{min}_{max}_{weeks}.parquet``.
    Per-(season, week) HTTP errors are logged + skipped rather than fatal. ``reader``
    is injectable for tests.
    """
    seasons = _validate_seasons(seasons)
    weeks_to_try = tuple(weeks) if weeks is not None else SLEEPER_DEFAULT_WEEKS
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = (
        f"{cache_dir}/sleeper_projections_{_CACHE_VERSION}"
        f"_{min(seasons)}_{max(seasons)}_{_weeks_signature(weeks_to_try)}.parquet"
    )
    if os.path.exists(cache_path) and not force_refresh:
        return pd.read_parquet(cache_path)

    parts: list[pd.DataFrame] = []
    for season in seasons:
        for week in weeks_to_try:
            records = _read_one_week(season, week, positions, reader=reader)
            if records is None:
                continue
            frame = _normalize_week(records, season, week)
            if not frame.empty:
                parts.append(frame)

    if not parts:
        # Don't poison the cache with an empty frame — bare retry next call.
        raise RuntimeError(
            f"No Sleeper projection rows fetched for seasons={seasons}; "
            "check the endpoint or network access."
        )
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["season", "week", "position", "player_name"]).reset_index(drop=True)
    df.to_parquet(cache_path)
    return df


def load_sleeper_with_gsis_id(
    seasons: Sequence[int],
    cache_dir: str = CACHE_DIR,
    force_refresh: bool = False,
    *,
    weeks: Sequence[int] | None = None,
    reader=_default_reader,
    player_ids_loader=None,
) -> pd.DataFrame:
    """Augment ``load_sleeper_projections`` with internal ``player_id`` (gsis_id).

    Joins Sleeper's ``sleeper_player_id`` to ``gsis_id`` via the nflverse
    ``ff_playerids`` crosswalk (``nfl_source.player_ids()``) — the same bridge
    pattern used for ESPN-QBR and PFR elsewhere in the repo. Rows that don't map
    keep ``player_id = NaN`` and are dropped by the downstream inner join.

    ``player_ids_loader`` is injectable for tests (default ``nfl_source.player_ids``).
    """
    proj = load_sleeper_projections(
        seasons, cache_dir=cache_dir, force_refresh=force_refresh, weeks=weeks, reader=reader
    )
    if player_ids_loader is None:
        player_ids_loader = nfl_source.player_ids
    ids = player_ids_loader()

    bridge = ids[["sleeper_id", "gsis_id"]].dropna().drop_duplicates(subset=["sleeper_id"])
    # ff_playerids stores sleeper_id as float (e.g. 4881.0); Sleeper's API keys are
    # string ints ("4881"). Normalize both to str-of-int for the merge.
    bridge = bridge.assign(sleeper_player_id=bridge["sleeper_id"].astype("int64").astype(str))

    merged = proj.merge(
        bridge[["sleeper_player_id", "gsis_id"]], on="sleeper_player_id", how="left"
    )
    merged = merged.rename(columns={"gsis_id": "player_id"})

    n_total = len(merged)
    n_matched = int(merged["player_id"].notna().sum())
    rate = n_matched / n_total if n_total else 0.0
    print(f"\nSleeper gsis_id join: {n_matched}/{n_total} = {rate:.1%} matched ({_PROVIDER})")
    return merged
