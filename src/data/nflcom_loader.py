"""Fetch + cache + name-normalize NFL.com weekly projection CSVs.

Source: https://github.com/hvpkod/NFL-Data — community archive of NFL.com's
official weekly fantasy projections (2015-2025), one CSV per (year, week,
position).

Provides two public entry points:

    load_nflcom_projections(seasons, ...) -> pd.DataFrame
        One row per (player_name, position, season, week). Raw stats are mapped
        to our internal target names (passing_yards, rushing_tds, etc.). Cached
        to ``data/raw/nflcom_projections_v1_{min}_{max}.parquet``.

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
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError

import nfl_data_py as nfl
import numpy as np
import pandas as pd

from src.config import CACHE_DIR

NFLCOM_BASE = "https://raw.githubusercontent.com/hvpkod/NFL-Data/main/NFL-data-Players"
NFLCOM_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K")
"""Positions available in upstream archive. There is no DST/Defense file."""

NFLCOM_DEFAULT_WEEKS = tuple(range(1, 19))  # NFL regular season is 18 weeks since 2021.
_CACHE_VERSION = "v1"

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

_SUFFIX_TOKENS = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

# Map NFL.com column names -> our internal target names, per position.
# QB/RB/WR/TE share the offensive-stat schema and we map every relevant column.
# K's projected file is per-distance-bucket FG/PAT counts, which doesn't align
# with our K raw-stat targets — only `nflcom_projected_pts` (their PlayerWeekProjectedPts)
# is reusable downstream.
NFLCOM_COLUMN_MAP: dict[str, dict[str, str]] = {
    "QB": {
        "PassingYDS": "passing_yards",
        "PassingTD": "passing_tds",
        "PassingInt": "interceptions",
        "RushingYDS": "rushing_yards",
        "RushingTD": "rushing_tds",
        "Fum": "fumbles_lost",  # × _FUM_LOST_RATIO at ingestion
    },
    "RB": {
        "RushingYDS": "rushing_yards",
        "RushingTD": "rushing_tds",
        "ReceivingRec": "receptions",
        "ReceivingYDS": "receiving_yards",
        "ReceivingTD": "receiving_tds",
        "Fum": "fumbles_lost",
    },
    "WR": {
        "ReceivingRec": "receptions",
        "ReceivingYDS": "receiving_yards",
        "ReceivingTD": "receiving_tds",
        "Fum": "fumbles_lost",
    },
    "TE": {
        "ReceivingRec": "receptions",
        "ReceivingYDS": "receiving_yards",
        "ReceivingTD": "receiving_tds",
        "Fum": "fumbles_lost",
    },
    "K": {},
}

# Empty placeholder columns to fill on positions that don't carry a stat (e.g.
# QB rows: receiving_*=0). Lets the per-position aggregator reuse a uniform shape.
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

# Historical -> canonical NFL team-abbr mapping. Both NFL.com and nflverse have
# historically inconsistent codes; canonicalize one side so the join works.
#
# This is the **canonical project-wide team-code normalization base map** — the
# single source of truth for relocation/abbr aliases. It targets the
# *NFL.com/roster* join universe, which canonicalizes the Rams to ``"LAR"``
# (``import_seasonal_rosters`` and NFL.com projection CSVs). Callers should
# import ``TEAM_CODE_MAP`` / call ``normalize_team_code(code)`` rather than
# hand-maintaining a parallel dictionary.
#
# Join-universe caveat (do NOT "fix" by collapsing the two): the nflverse
# *schedule* + *weekly* releases use ``"LA"`` for the Rams (verified across
# 2016-2025: ``import_schedules`` and ``import_weekly_data`` both emit ``LA``,
# never ``LAR``). A merge that maps the schedule's ``LA`` to ``LAR`` while the
# player frame still carries ``LA`` silently misses every Rams row. The
# schedule-join consumers therefore derive their normalization from this base
# via ``schedule_team_code_normalization()`` below, which remaps the Rams back
# to ``LA`` and drops the historical ``STL`` to ``LA`` (not ``LAR``). That
# helper is the *one* place the schedule-universe variant is defined;
# ``src/shared/weather_features._TEAM_CODE_NORMALIZATION`` is built from it.
TEAM_CODE_MAP: dict[str, str] = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "WSH": "WAS",
    "JAX": "JAX",
    "JAC": "JAX",
    "LA": "LAR",
}
# Back-compat alias for the original private name.
_TEAM_CANONICAL = TEAM_CODE_MAP


def normalize_team_code(code: str | None) -> str:
    """Map a historical NFL team code to its current canonical abbreviation.

    Canonical project-wide helper for team-code normalization — importable
    from ``src.data.nflcom_loader`` by any bundle that needs to align
    franchise codes across data sources (NFL.com projections, nflverse
    schedules/PBP, internal stats). Examples:

        >>> normalize_team_code("OAK")
        'LV'
        >>> normalize_team_code("STL")
        'LAR'
        >>> normalize_team_code("@OAK")  # NFL.com prefixes opponent for away games
        'LV'
        >>> normalize_team_code(None)
        ''

    Parameters
    ----------
    code : str | None
        The raw team code. ``None`` / NaN / empty string returns ``""``.
        A leading ``"@"`` (NFL.com away-game prefix) is stripped.
        Unknown codes pass through unchanged (after upper-case + strip).

    Returns
    -------
    str
        The canonical NFL team abbreviation, or ``""`` for missing input.
    """
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return ""
    s = str(code).strip().upper()
    # NFL.com sometimes prefixes opponent with '@' for away games — strip.
    s = s.lstrip("@")
    return TEAM_CODE_MAP.get(s, s)


def schedule_team_code_normalization() -> dict[str, str]:
    """Return the relocation map for the *nflverse schedule/weekly* join universe.

    Derived from :data:`TEAM_CODE_MAP` (the single source of truth) with the
    one documented join-direction difference: nflverse schedules and weekly
    data canonicalize the Rams to ``"LA"`` (not ``"LAR"``), so the historical
    ``STL`` maps to ``LA`` and the modern ``LA`` is left untouched (no
    ``LA -> LAR`` rewrite, which would break the Rams join against player rows
    that already carry ``LA``). The ``WSH/JAX/JAC`` entries are dropped because
    nflverse already emits ``WAS``/``JAX`` consistently — only the three
    relocated franchises ever differ between the schedule's historical codes
    and the player frame's modern codes.

    Consumed by ``src.shared.weather_features._TEAM_CODE_NORMALIZATION`` so the
    schedule-side normalization has exactly one definition.
    """
    base = {k: v for k, v in TEAM_CODE_MAP.items() if k in ("OAK", "SD", "STL")}
    base["STL"] = "LA"  # nflverse schedule/weekly uses LA for the Rams, not LAR.
    return base


def normalize_player_name(name: str | None) -> str:
    """Canonicalize a player name for cross-source joining.

    - Lowercase, strip leading/trailing whitespace.
    - Drop trailing suffix tokens (Jr, Sr, II, III, IV, V).
    - Drop punctuation entirely (apostrophes, periods, hyphens collapse to "").
    - Collapse internal whitespace.

    Examples:
        "Patrick Mahomes II"   -> "patrick mahomes"
        "Marvin Harrison Jr."  -> "marvin harrison"
        "Ja'Marr Chase"        -> "jamarr chase"
        "A.J. Brown"           -> "aj brown"
        "Foo  Bar"             -> "foo bar"
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).strip().lower()
    if not s:
        return ""
    # Drop punctuation. Keep whitespace and ascii letters/digits.
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    # Tokenize, drop trailing suffix tokens (only at the end; "II Smith" stays).
    tokens = s.split()
    while tokens and tokens[-1] in _SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


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

    Returns ``None`` on 404 / persistent connection errors / empty file. Logs a
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
        except (HTTPError, URLError, FileNotFoundError) as e:
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
            return None
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
    plus per-target columns from ``_ALL_TARGET_COLUMNS`` (filled with 0 where the
    position doesn't carry that stat).
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
    # Fill missing target columns (e.g. QB rows have no receiving_*) with 0.
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

    Cache: ``{cache_dir}/nflcom_projections_{_CACHE_VERSION}_{min}_{max}.parquet``.
    Subsequent calls with the same season range and ``force_refresh=False``
    skip the network entirely.

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
    cache_path = (
        f"{cache_dir}/nflcom_projections_{_CACHE_VERSION}"
        f"_{min(seasons)}_{max(seasons)}_{weeks_sig}.parquet"
    )
    if os.path.exists(cache_path) and not force_refresh:
        return pd.read_parquet(cache_path)

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
            raw = future.result()
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
    df.to_parquet(cache_path)
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

    Cache: ``{cache_dir}/nflcom_projections_joined_{_CACHE_VERSION}_{min}_{max}.parquet``.
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
    cache_path = (
        f"{cache_dir}/nflcom_projections_joined_{_CACHE_VERSION}"
        f"_{min(seasons)}_{max(seasons)}_{rate_sig}.parquet"
    )
    if os.path.exists(cache_path) and not force_refresh:
        return pd.read_parquet(cache_path)

    proj = load_nflcom_projections(
        seasons, cache_dir=cache_dir, force_refresh=force_refresh, reader=reader
    )

    if rosters is None:
        rosters = nfl.import_seasonal_rosters(list(seasons))
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
    primary.to_parquet(cache_path)
    return primary
