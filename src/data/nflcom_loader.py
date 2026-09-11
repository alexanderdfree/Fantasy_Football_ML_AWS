"""Fetch + cache + name-normalize NFL.com weekly projection CSVs.

Source: https://github.com/hvpkod/NFL-Data — community archive of NFL.com's
official weekly fantasy projections (2015-2025), one CSV per (year, week,
position).

Provides two public entry points:

    load_nflcom_projections(seasons, ...) -> pd.DataFrame
        One row per (player_name, position, season, week). Raw stats are mapped
        to our internal target names (passing_yards, rushing_tds, etc.). Cached
        to ``data/raw/nflcom_projections_v2_{seasons_sig}_{weeks_sig}.parquet``
        (a contiguous season range renders as ``{min}_{max}``; a sparse list
        adds a disambiguating hash — see ``_seasons_cache_signature``).

    load_nflcom_with_gsis_id(seasons, ...) -> pd.DataFrame
        Same frame, joined to ``player_id`` (gsis_id) via roster lookup. Cached
        separately. Raises if global match rate falls below ``min_match_rate``.

Mirrors the cache idiom of ``src/data/loader.py``: parquet under
``data/raw/`` (gitignored), idempotent, season-range suffix in the filename.
``_CACHE_VERSION`` is baked into the filename so future schema changes can
invalidate old caches without surgery.
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd

from src.config import CACHE_DIR
from src.data import nfl_source
from src.data.cache_io import atomic_write_parquet
from src.data.external_sources import _seasons_cache_signature
from src.data.identity import (
    TEAM_CODE_MAP as TEAM_CODE_MAP,
)
from src.data.identity import (
    normalize_player_name as normalize_player_name,
)
from src.data.identity import (
    normalize_team_code as normalize_team_code,
)
from src.data.identity import (
    schedule_team_code_normalization as schedule_team_code_normalization,
)

NFLCOM_BASE = "https://raw.githubusercontent.com/hvpkod/NFL-Data/main/NFL-data-Players"
NFLCOM_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K")
"""Positions available in upstream archive. There is no DST/Defense file."""

NFLCOM_DEFAULT_WEEKS = tuple(range(1, 19))  # NFL regular season is 18 weeks since 2021.
# v2 retains cross-position offense stats; v1 caches already lost those values.
_CACHE_VERSION = "v2"
_FETCH_COMPLETE_ATTR = "nflcom_fetch_complete_v1"

# Network defensiveness: 404 is expected (late-season weeks) and not retried.
# Other transient failures (5xx, ECONNRESET, DNS blips) are retried once after
# a short backoff before giving up. CLAUDE.md: "network/data-source boundaries
# are real and should be defensive."
_RETRY_BACKOFF_S = 0.5
_MAX_FETCH_WORKERS = 8  # Concurrent (year, week, position) HTTP fetches.

# NFL.com 'Fum' column is total fumbles; our internal `fumbles_lost` target is
# the lost subset. League-average lost rate is ~50%; impact on FP is ~0.1 pt/game
# so this approximation is fine. Documented here so it shows up in code search.
_FUM_LOST_RATIO = 0.5


# Map NFL.com column names -> our internal target names, per position.
# QB/RB/WR/TE share the offensive-stat schema and we map every relevant column.
# K's projected file is per-distance-bucket FG/PAT counts, which doesn't align
# with our K raw-stat targets — only `nflcom_projected_pts` (their PlayerWeekProjectedPts)
# is reusable downstream.
_OFFENSE_COLUMN_MAP = {
    "PassingYDS": "passing_yards",
    "PassingTD": "passing_tds",
    "PassingInt": "interceptions",
    "RushingYDS": "rushing_yards",
    "RushingTD": "rushing_tds",
    "ReceivingRec": "receptions",
    "ReceivingYDS": "receiving_yards",
    "ReceivingTD": "receiving_tds",
    "Fum": "fumbles_lost",  # × _FUM_LOST_RATIO at ingestion
}
NFLCOM_COLUMN_MAP: dict[str, dict[str, str]] = {
    pos: dict(_OFFENSE_COLUMN_MAP) for pos in ("QB", "RB", "WR", "TE")
} | {
    "K": {},
}

# Uniform raw-stat shape, with absent source columns filled with zero.
_ALL_TARGET_COLUMNS = {
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "receptions",
    "fumbles_lost",
}


def _team_abbr_normalize(team: str | None) -> str:
    """Back-compat alias for :func:`normalize_team_code`.

    Kept so callers within this module keep working without churn; new
    callers (including cross-bundle imports) should use
    :func:`normalize_team_code` directly — it's the canonical project-wide
    helper.
    """
    return normalize_team_code(team)


def _weeks_cache_signature(weeks: tuple[int, ...] | list[int]) -> str:
    """Stable, filename-safe signature of a week selection for cache keys.

    A contiguous range renders as the readable ``w{min}-{max}`` (e.g. the
    default 1-18 weeks -> ``w1-18``); any non-contiguous / sparse selection
    renders as ``w{min}-{max}-{len}-{8-hex hash}`` so the filename stays bounded
    while remaining unique per distinct selection. Order- and duplicate-
    insensitive: ``(3, 1, 2)`` and ``(1, 2, 3)`` produce the same signature.
    """
    uniq = sorted(set(int(w) for w in weeks))
    if not uniq:
        return "wnone"
    lo, hi = uniq[0], uniq[-1]
    if uniq == list(range(lo, hi + 1)):
        return f"w{lo}-{hi}"
    digest = hashlib.sha1(",".join(map(str, uniq)).encode()).hexdigest()[:8]
    return f"w{lo}-{hi}-{len(uniq)}-{digest}"


def _projection_url(year: int, week: int, position: str) -> str:
    return f"{NFLCOM_BASE}/{year}/{week}/projected/{position}_projected.csv"


def _is_404(err: Exception) -> bool:
    """True iff the exception is an HTTPError with status 404."""
    return isinstance(err, HTTPError) and getattr(err, "code", None) == 404


class _ProjectionFetchUnavailable(RuntimeError):
    """A transient fetch failed, so the returned coverage cannot be cached."""


def _read_complete_cache(path):
    if os.path.exists(path):
        cached = pd.read_parquet(path)
        if cached.attrs.get(_FETCH_COMPLETE_ATTR) is True:
            return cached
    return None


def _read_one_projection(
    year: int,
    week: int,
    position: str,
    *,
    reader=pd.read_csv,
    max_retries: int = 1,
    backoff_s: float = _RETRY_BACKOFF_S,
) -> pd.DataFrame | None:
    """Fetch one (year, week, position) CSV from upstream.

    Returns ``None`` on 404 / empty file; transient failures raise. Logs a
    warning so operators see late-season weeks dropping out rather than silently
    shrinking the frame.

    Retries non-404 transient errors (URLError, 5xx, etc.) once after
    ``backoff_s`` seconds. 404s are not retried (they are the expected signal
    for a week that doesn't exist yet).

    ``reader`` injectable for tests — pass a stub to avoid network.
    """
    url = _projection_url(year, week, position)
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            df = reader(url)
        except (HTTPError, URLError, TimeoutError, FileNotFoundError) as e:
            last_err = e
            if _is_404(e) or isinstance(e, FileNotFoundError):
                # 404 is expected — no retry, just log and skip.
                print(f"  WARN nflcom: skip {position} {year} W{week} (404 / not found)")
                return None
            if attempt < max_retries:
                # Transient — back off and retry once.
                print(
                    f"  WARN nflcom: transient {type(e).__name__} on {position} "
                    f"{year} W{week}; retrying in {backoff_s}s"
                )
                time.sleep(backoff_s)
                continue
            print(
                f"  WARN nflcom: skip {position} {year} W{week} "
                f"({type(e).__name__} after {max_retries} retry)"
            )
            raise _ProjectionFetchUnavailable(f"{position} {year} W{week}: {e}") from e
        except pd.errors.EmptyDataError:
            print(f"  WARN nflcom: skip {position} {year} W{week} (empty CSV)")
            return None
        else:
            if df.empty:
                return None
            df = df.copy()
            df["season"] = year
            df["week"] = week
            df["position"] = position
            return df
    # Unreachable — both branches above exit explicitly. Defensive return for
    # the type-checker.
    if last_err is not None:
        return None
    return None


def _numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Coerce ``df[col]`` to numeric, returning an all-NaN Series if absent.

    ``pd.to_numeric(df.get(col))`` returns a *scalar* ``np.nan`` when ``col`` is
    missing (``df.get`` yields ``None``), and any chained ``.fillna()`` then
    raises ``AttributeError`` on the scalar. Guard the existence check so a
    missing upstream column degrades to an all-NaN column of the right length
    instead of crashing the whole fetch.
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _normalize_one_position(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Apply per-position column mapping + universal columns.

    Output schema (all positions): season, week, position, nflcom_player_id,
    player_name, team, opponent, nflcom_projected_pts, nflcom_projected_rank,
    plus scoring columns from ``_ALL_TARGET_COLUMNS`` (0 for absent source stats).
    """
    out = pd.DataFrame(
        {
            "season": df["season"].astype(int),
            "week": df["week"].astype(int),
            "position": df["position"],
            "nflcom_player_id": df["PlayerId"].astype(str),
            "player_name": df["PlayerName"].astype(str),
            # Don't .astype(str) before normalizing — that would coerce NaN to
            # the literal string "nan" (which then survives team-canonicalization
            # as "NAN"). Pass the original series and let _team_abbr_normalize
            # detect float NaN via pd.isna and emit "" instead.
            "team": df["Team"].map(_team_abbr_normalize),
            "opponent": df["PlayerOpponent"].map(_team_abbr_normalize),
            "nflcom_projected_pts": _numeric_col(df, "PlayerWeekProjectedPts").fillna(0.0),
            "nflcom_projected_rank": _numeric_col(df, "ProjectedRank"),
        }
    )
    column_map = NFLCOM_COLUMN_MAP[position]
    for src_col, target_col in column_map.items():
        vals = _numeric_col(df, src_col).fillna(0.0)
        if target_col == "fumbles_lost":
            vals = vals * _FUM_LOST_RATIO
        out[target_col] = vals.astype(float)
    # K has no normalized offense stats, but keeps the same frame schema.
    for col in _ALL_TARGET_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    return out


def load_nflcom_projections(
    seasons: list[int],
    weeks: tuple[int, ...] | list[int] | None = None,
    cache_dir: str = CACHE_DIR,
    force_refresh: bool = False,
    *,
    reader=pd.read_csv,
) -> pd.DataFrame:
    """Fetch + cache projected CSVs for one or more seasons.

    Cache: ``{cache_dir}/nflcom_projections_{_CACHE_VERSION}_{seasons_sig}_{weeks_sig}.parquet``
    (a contiguous season range renders as ``{min}_{max}`` — byte-identical to the
    legacy key — while a sparse list adds a disambiguating hash so it can't
    collide with the contiguous superset's cache). Subsequent calls with the
    same season range and ``force_refresh=False`` skip the network entirely.

    Network behaviour mirrors ``src/data/loader.py``: ``pd.read_parquet/csv(url)``
    with no ``requests`` dependency. Per-(year, week, position) HTTP errors are
    logged + skipped rather than fatal.
    """
    if not seasons:
        raise ValueError("seasons must be a non-empty list of ints")
    weeks_to_try = tuple(weeks) if weeks is not None else NFLCOM_DEFAULT_WEEKS
    os.makedirs(cache_dir, exist_ok=True)
    # Include a weeks signature in the cache key. Without it, a first call with
    # the default 1-18 range and a later call with a narrower (or different)
    # week range collide on the same file, so the later call silently returns
    # the earlier frame's week coverage. The full week range is canonicalized
    # (de-duped + ordered) so equivalent-but-unsorted inputs hash to one key.
    weeks_sig = _weeks_cache_signature(weeks_to_try)
    # Signature the seasons dimension too (contiguous -> byte-identical {min}_{max},
    # sparse -> disambiguated hash) so a sparse list like [2015, 2025] can't collide
    # on the contiguous superset's cache file and silently return partial coverage
    # (the #488 defect class, fixed here to mirror loader.py/external_sources.py).
    seasons_sig = _seasons_cache_signature(seasons)
    cache_path = (
        f"{cache_dir}/nflcom_projections_{_CACHE_VERSION}_{seasons_sig}_{weeks_sig}.parquet"
    )
    if not force_refresh:
        cached = _read_complete_cache(cache_path)
        if cached is not None:
            return cached

    # Parallelize the (year, week, position) fetch fan-out. Each task is one
    # HTTP GET, so I/O-bound — threads beat sequential by ~5-10x for typical
    # year-ranges. Workers cap is small enough to not anger raw.githubusercontent.com.
    tasks = [
        (year, week, position)
        for year in seasons
        for week in weeks_to_try
        for position in NFLCOM_POSITIONS
    ]
    parts: list[pd.DataFrame] = []
    fetch_complete = True
    with ThreadPoolExecutor(max_workers=_MAX_FETCH_WORKERS) as executor:
        futures = {
            executor.submit(_read_one_projection, year, week, position, reader=reader): (
                year,
                week,
                position,
            )
            for year, week, position in tasks
        }
        for future in as_completed(futures):
            year, week, position = futures[future]
            try:
                raw = future.result()
            except _ProjectionFetchUnavailable:
                fetch_complete = False
                continue
            if raw is None:
                continue
            parts.append(_normalize_one_position(raw, position))

    if not parts:
        # Don't poison the cache with an empty frame — bare retry next call.
        raise RuntimeError(
            f"No NFL.com projection rows fetched for seasons={seasons}; "
            "check upstream URL or network access."
        )
    df = pd.concat(parts, ignore_index=True)
    # Sort for deterministic cache contents (parallel fetch returns rows in
    # nondeterministic order); makes diffs across re-fetches stable.
    df = df.sort_values(["season", "week", "position", "player_name"]).reset_index(drop=True)
    df.attrs[_FETCH_COMPLETE_ATTR] = fetch_complete
    if fetch_complete:
        atomic_write_parquet(df, cache_path)
    return df


def _build_roster_lookup(rosters: pd.DataFrame) -> pd.DataFrame:
    """Reduce a rosters frame to ``(norm_name, season, team, position) -> player_id``.

    nflverse's ``import_seasonal_rosters`` schema may vary; we tolerate
    ``team_abbr`` / ``team`` / ``recent_team`` and prefer the first that exists.
    """
    if "player_id" not in rosters.columns:
        raise ValueError("rosters frame must have player_id (gsis_id)")
    # Find the right name column.
    name_col = next(
        (c for c in ("player_name", "full_name", "player_display_name") if c in rosters.columns),
        None,
    )
    if name_col is None:
        raise ValueError(
            "rosters frame must have one of: player_name, full_name, player_display_name"
        )
    team_col = next(
        (c for c in ("team_abbr", "team", "recent_team") if c in rosters.columns),
        None,
    )
    if team_col is None:
        raise ValueError("rosters frame must have one of: team_abbr, team, recent_team")
    pos_col = "position" if "position" in rosters.columns else None
    if pos_col is None:
        raise ValueError("rosters frame must have a 'position' column")

    lookup = pd.DataFrame(
        {
            "norm_name": rosters[name_col].map(normalize_player_name),
            "season": rosters["season"].astype(int),
            "team": rosters[team_col].map(_team_abbr_normalize),
            "position": rosters[pos_col].astype(str),
            "player_id": rosters["player_id"].astype(str),
        }
    )
    # Drop rows with no name or no id — they can't contribute to a join.
    lookup = lookup[(lookup["norm_name"] != "") & (lookup["player_id"] != "")]
    # Dedup on the full key (a few players appear twice in a season's roster
    # snapshot — different team rows after a trade); keep the first.
    lookup = lookup.drop_duplicates(subset=["norm_name", "season", "team", "position"])
    return lookup.reset_index(drop=True)


def _format_unmatched_diagnostic(unmatched: pd.DataFrame, top_n: int = 5) -> str:
    if unmatched.empty:
        return ""
    counts = (
        unmatched.groupby(["player_name", "position", "team"], dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(top_n)
    )
    lines = [f"  - {name} ({pos}, {team}) × {n}" for (name, pos, team), n in counts.items()]
    return "\n".join(lines)


def load_nflcom_with_gsis_id(
    seasons: list[int],
    cache_dir: str = CACHE_DIR,
    force_refresh: bool = False,
    *,
    weeks: tuple[int, ...] | list[int] | None = None,
    rosters: pd.DataFrame | None = None,
    min_match_rate: float = 0.90,
    reader=pd.read_csv,
) -> pd.DataFrame:
    """Augment ``load_nflcom_projections`` with internal ``player_id`` (gsis_id).

    Join strategy (in order):
      1. (norm_name, season, team, position) — exact match
      2. (norm_name, season, position)       — fallback when team disagrees
         (mid-season trades, NFL.com bye-week roster snapshots, etc.)

    Diagnostics:
      - Match rate logged per (year, position).
      - Top-5 unmatched names logged for inspection.
      - Raises ``RuntimeError`` if the global match rate < ``min_match_rate``.

    Cache: ``{cache_dir}/nflcom_projections_joined_{_CACHE_VERSION}_{seasons_sig}_{rate_sig}.parquet``
    (a contiguous season range renders as ``{min}_{max}``; a sparse list adds a
    disambiguating hash — see ``_seasons_cache_signature``).
    """
    if not seasons:
        raise ValueError("seasons must be a non-empty list of ints")
    os.makedirs(cache_dir, exist_ok=True)
    # Encode ``min_match_rate`` in the cache key. A cache hit returns the joined
    # frame without recomputing the match rate, so the RuntimeError guard below
    # only fires on a cold cache. Without the threshold in the key, a first
    # caller at min_match_rate=0.80 caches an 0.85-rate result and a later
    # caller passing min_match_rate=0.99 silently receives it as if its
    # (stricter) threshold had passed. Quantize to whole percent so equivalent
    # floats (0.90 vs 0.9000001) share one cache file.
    rate_sig = f"mr{int(round(min_match_rate * 100))}"
    # Signature the seasons dimension (contiguous -> byte-identical {min}_{max},
    # sparse -> disambiguated hash) so a sparse season list can't collide on the
    # contiguous superset's cache file (the #488 defect class).
    seasons_sig = _seasons_cache_signature(seasons)
    # `weeks` joined the cache key later (upcoming-week fetches ask for a
    # single week): default-None keeps the historical filename so warm
    # full-range caches stay valid; an explicit week range gets its own file
    # (same collision rationale as load_nflcom_projections' weeks_sig).
    weeks_part = f"_w{_weeks_cache_signature(tuple(weeks))}" if weeks is not None else ""
    cache_path = (
        f"{cache_dir}/nflcom_projections_joined_{_CACHE_VERSION}"
        f"_{seasons_sig}_{rate_sig}{weeks_part}.parquet"
    )
    # ``rosters`` is the one input not captured by the cache key (seasons +
    # version + match-rate). A caller-supplied override must bypass the cache
    # read or write so it cannot replace a default-rosters cache; with the
    # default (``rosters is None``) the rosters are derived from ``seasons``, so
    # the key is complete and the cache hit is correct. (#439)
    default_rosters = rosters is None
    if not force_refresh and default_rosters:
        cached = _read_complete_cache(cache_path)
        if cached is not None and "player_id" in cached and len(cached):
            valid = cached["player_id"].notna() & ~cached["player_id"].astype(str).str.lower().isin(
                ["nan", "none", "<na>", ""]
            )
            if valid.mean() >= min_match_rate:
                return cached

    proj = load_nflcom_projections(
        seasons, weeks=weeks, cache_dir=cache_dir, force_refresh=force_refresh, reader=reader
    )
    fetch_complete = proj.attrs.get(_FETCH_COMPLETE_ATTR) is True

    if rosters is None:
        rosters = nfl_source.rosters(list(seasons))
    lookup = _build_roster_lookup(rosters)

    proj = proj.copy()
    proj["norm_name"] = proj["player_name"].map(normalize_player_name)

    # Primary join: name + season + team + position.
    primary = proj.merge(
        lookup[["norm_name", "season", "team", "position", "player_id"]],
        on=["norm_name", "season", "team", "position"],
        how="left",
    )

    # Fallback: for rows still unmatched, try (norm_name, season, position) —
    # but ONLY when that key maps to a single distinct player_id. When two
    # players share the same normalized name + position in a season (it
    # happens — multiple "Mike Williams" types), neither row should silently
    # adopt one of them; leave them unmatched and let the diagnostics surface.
    unmatched_mask = primary["player_id"].isna()
    if unmatched_mask.any():
        unique_keys = (
            lookup.groupby(["norm_name", "season", "position"])["player_id"]
            .nunique(dropna=True)
            .reset_index(name="_n_distinct")
        )
        unique_keys = unique_keys[unique_keys["_n_distinct"] == 1][
            ["norm_name", "season", "position"]
        ]
        fallback_lookup = (
            lookup.drop(columns=["team"])
            .merge(unique_keys, on=["norm_name", "season", "position"], how="inner")
            .drop_duplicates(subset=["norm_name", "season", "position"])
        )
        # Preserve row identity through the merge: pandas.merge does NOT
        # guarantee row-order, so we carry the original index through and
        # assign back by index rather than by position.
        unmatched_rows = primary.loc[unmatched_mask].drop(columns=["player_id"])
        fallback = (
            unmatched_rows.reset_index()
            .merge(
                fallback_lookup[["norm_name", "season", "position", "player_id"]],
                on=["norm_name", "season", "position"],
                how="left",
            )
            .set_index("index")
        )
        primary.loc[fallback.index, "player_id"] = fallback["player_id"]

    # Diagnostics
    n_total = len(primary)
    n_matched = int(primary["player_id"].notna().sum())
    match_rate = n_matched / n_total if n_total else 0.0

    print(f"\nNFL.com gsis_id join: {n_matched}/{n_total} = {match_rate:.1%} matched")
    by_year_pos = (
        primary.assign(_matched=primary["player_id"].notna())
        .groupby(["season", "position"])["_matched"]
        .agg(["sum", "count"])
    )
    for (year, pos), row in by_year_pos.iterrows():
        print(f"  {year} {pos}: {int(row['sum'])}/{int(row['count'])}")

    unmatched = primary.loc[primary["player_id"].isna(), ["player_name", "position", "team"]]
    if not unmatched.empty:
        print("Top-5 unmatched names:")
        print(_format_unmatched_diagnostic(unmatched))

    if match_rate < min_match_rate:
        raise RuntimeError(
            f"NFL.com gsis_id match rate {match_rate:.1%} < min_match_rate "
            f"{min_match_rate:.1%}. Top-5 unmatched:\n" + _format_unmatched_diagnostic(unmatched)
        )

    primary = primary.drop(columns=["norm_name"])
    primary.attrs[_FETCH_COMPLETE_ATTR] = fetch_complete
    if fetch_complete and default_rosters:
        atomic_write_parquet(primary, cache_path)
    return primary
