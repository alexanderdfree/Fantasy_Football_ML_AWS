"""Fetch + cache + gsis-join Sleeper (RotoWire) weekly NFL projections.

Source: Sleeper's undocumented projections endpoint
``https://api.sleeper.app/projections/nfl/{season}/{week}`` (free, no auth). Every
record carries ``company: "rotowire"`` — so this is **one** additional expert
(RotoWire), not a consensus. Covers offense (QB/RB/WR/TE, joined via the gsis
crosswalk) and DST (team-keyed); K is totals-only and out of scope.

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
"""Offensive positions ingested via the sleeper_id -> gsis_id crosswalk join."""

# Sleeper serves DST under the "DEF" code, keyed by team abbrev (player_id = team,
# no gsis). K is totals-only (not decomposable to our distance-based targets) and
# stays out of scope.
_SLEEPER_DEF_CODE = "DEF"
SLEEPER_FETCH_POSITIONS: tuple[str, ...] = (*SLEEPER_OFFENSE_POSITIONS, _SLEEPER_DEF_CODE)

SLEEPER_DEFAULT_WEEKS = tuple(range(1, 19))  # 18-week regular season since 2021.
_CACHE_VERSION = "v2"  # bumped: cache now also carries DST rows + DST target columns.
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

# Sleeper DST ``stats`` field -> our DST target name. DST records are team-keyed
# and routed through the bespoke DST aggregator. Sleeper covers all 10 DST_TARGETS
# (special_teams_tds <- st_td); values here must equal aggregate_targets.DST_TARGETS.
SLEEPER_DST_STAT_MAP: dict[str, str] = {
    "sack": "def_sacks",
    "int": "def_ints",
    "fum_rec": "def_fumble_rec",
    "ff": "def_fumbles_forced",
    "safe": "def_safeties",
    "def_td": "def_tds",
    "blk_kick": "def_blocked_kicks",
    "st_td": "special_teams_tds",
    "pts_allow": "points_allowed",
    "yds_allow": "yards_allowed",
}

# Sleeper uses "LAR" for the Rams; nflverse schedules (and the model's DST
# player_id, which is the team abbrev) use "LA". Every other team matches. This is
# a pure rename — consistent across all seasons since 2016, NOT a relocation remap
# — so it's safe to apply for any 2018+ season.
_DST_TEAM_FIXUP: dict[str, str] = {"LAR": "LA"}

_ALL_TARGET_COLUMNS: tuple[str, ...] = (
    *SLEEPER_STAT_MAP.values(),
    *SLEEPER_DST_STAT_MAP.values(),
)


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
    (0-filled where a position doesn't carry the stat). Offense records are keyed by
    Sleeper player id; DST ("DEF") records are team-keyed (player_id = team, mapped
    to the nflverse convention) and emitted with position "DST". Placeholder /
    position-less records are dropped.
    """
    rows: list[dict] = []
    for rec in records:
        player = rec.get("player") or {}
        raw_pos = player.get("position")
        stats = rec.get("stats") or {}
        if raw_pos in SLEEPER_OFFENSE_POSITIONS:
            stat_map = SLEEPER_STAT_MAP
            # Drop unprojected roster placeholders (no pts_ppr, no mapped stat):
            # Sleeper returns a record for EVERY rostered player, and scoring a
            # placeholder as a confident 0.0 projection contaminates the comparison
            # (NFL.com's curated archive has no such rows).
            if "pts_ppr" not in stats and not any(k in stats for k in stat_map):
                continue
            row = {
                "position": raw_pos,
                "sleeper_player_id": str(rec.get("player_id")),
                "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                # The populated team is at the record top level; player.team_abbr is
                # often null in the payload — prefer the record's value.
                "team": rec.get("team") or player.get("team_abbr") or player.get("team") or "",
            }
        elif raw_pos == _SLEEPER_DEF_CODE:
            stat_map = SLEEPER_DST_STAT_MAP
            if "pts_ppr" not in stats and not any(k in stats for k in stat_map):
                continue
            # DST is team-keyed: Sleeper's player_id IS the team abbrev. Map to the
            # nflverse convention (LAR -> LA) so it joins the model's DST player_id.
            team = _DST_TEAM_FIXUP.get(str(rec.get("player_id")), str(rec.get("player_id")))
            row = {"position": "DST", "sleeper_player_id": team, "player_name": team, "team": team}
        else:
            continue
        row["season"] = int(rec.get("season", season))
        row["week"] = int(rec.get("week", week))
        row["company"] = rec.get("company")
        row["sleeper_projected_pts"] = float(stats.get("pts_ppr") or 0.0)
        for src_field, target in stat_map.items():
            row[target] = float(stats.get(src_field) or 0.0)
        rows.append(row)

    base_cols = [
        "season",
        "week",
        "position",
        "sleeper_player_id",
        "player_name",
        "team",
        "company",
        "sleeper_projected_pts",
    ]
    if not rows:
        return pd.DataFrame(columns=[*base_cols, *_ALL_TARGET_COLUMNS])
    df = pd.DataFrame(rows)
    # Offense rows lack DST targets and vice versa -> NaN after the build; 0-fill so
    # every target column is present and numeric for the per-position aggregator.
    for col in _ALL_TARGET_COLUMNS:
        df[col] = df[col].fillna(0.0) if col in df.columns else 0.0
    return df


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
    positions: Sequence[str] = SLEEPER_FETCH_POSITIONS,
    reader=_default_reader,
) -> pd.DataFrame:
    """Fetch + cache Sleeper projections for one or more seasons (offense + DST).

    Cache: ``{cache_dir}/sleeper_projections_{_CACHE_VERSION}_{min}_{max}_{weeks}_{positions}.parquet``.
    Per-(season, week) HTTP errors are logged + skipped rather than fatal. ``reader``
    is injectable for tests.
    """
    seasons = _validate_seasons(seasons)
    weeks_to_try = tuple(weeks) if weeks is not None else SLEEPER_DEFAULT_WEEKS
    os.makedirs(cache_dir, exist_ok=True)
    # Include a positions token so a narrower fetch can't be served a wider cache
    # (or vice versa) under the same key.
    pos_sig = "-".join(sorted(positions))
    cache_path = (
        f"{cache_dir}/sleeper_projections_{_CACHE_VERSION}"
        f"_{min(seasons)}_{max(seasons)}_{_weeks_signature(weeks_to_try)}_{pos_sig}.parquet"
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
    """Augment ``load_sleeper_projections`` with internal ``player_id``.

    Offense rows: ``sleeper_player_id`` is bridged to ``gsis_id`` via the nflverse
    ``ff_playerids`` crosswalk (``nfl_source.player_ids()``) — the same pattern used
    for ESPN-QBR / PFR. DST rows are **team-keyed** (no gsis): ``player_id`` is the
    nflverse-normalized team abbrev, which joins the model's DST ``player_id``
    directly. Offense rows that don't map keep ``player_id = NaN`` and are dropped by
    the downstream inner join.

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

    is_dst = proj["position"] == "DST"
    offense = (
        proj[~is_dst]
        .merge(bridge[["sleeper_player_id", "gsis_id"]], on="sleeper_player_id", how="left")
        .rename(columns={"gsis_id": "player_id"})
    )
    dst = proj[is_dst].copy()
    dst["player_id"] = dst["sleeper_player_id"]  # team abbrev, already normalized
    merged = pd.concat([offense, dst], ignore_index=True)

    n_off = len(offense)
    n_matched = int(offense["player_id"].notna().sum())
    rate = n_matched / n_off if n_off else 0.0
    print(
        f"\nSleeper gsis_id join: {n_matched}/{n_off} = {rate:.1%} offense matched "
        f"({_PROVIDER}); DST team-keyed (n={len(dst)})"
    )
    return merged
