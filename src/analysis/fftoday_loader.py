"""Fetch + cache + name-normalize FFToday weekly projection pages.

Source: https://www.fftoday.com/rankings/playerwkproj.php?Season=Y&GameWeek=W&PosID=P
— FFToday's free public weekly projection grids. Unlike FantasyPros' historical
pages (which return only currently-rostered players with *current* teams, a
survivorship-biased pool — see ``todo/new-sources-research-2026-06.md``), FFToday's
``Season=`` param serves the **genuine historical slate** with correct historical
teams and since-retired players. Archive floor is 2010; verified back to 2013.

This is an **expert-comparison benchmark** (a third independent projector to score
the model against), NOT a model feature — training on expert points would relearn a
consensus we are trying to beat.

Two public entry points, mirroring ``src/data/nflcom_loader.py``:

    load_fftoday_projections(seasons, ...) -> pd.DataFrame
        One row per (player_name, position, season, week); raw stats mapped to our
        internal target names. Cached to
        ``data/raw/fftoday_projections_v1_{min}_{max}_{weeks}.parquet``.

    load_fftoday_with_gsis_id(seasons, ...) -> pd.DataFrame
        Same frame joined to ``player_id`` (gsis_id) via the roster name+team
        lookup. Raises if the global match rate falls below ``min_match_rate``.

Coverage: QB/RB/WR/TE (offense). K's grid is per-count FG/XP (no yardage), which
doesn't map to our distance-weighted K targets, and there is no DST grid — both are
out of scope here, matching the report's "5/6 positions, no DST" finding.
"""

from __future__ import annotations

import html as _html
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError

import pandas as pd

from src.config import CACHE_DIR
from src.data import nfl_source
from src.data.cache_io import atomic_write_parquet

# Reuse the canonical name/team normalizers + roster lookup + diagnostic from the
# NFL.com loader (same name-keyed bridge problem; no native player id). Importing
# the shared helpers keeps one source of truth for name/team canonicalization.
from src.data.nflcom_loader import (
    _build_roster_lookup,
    _format_unmatched_diagnostic,
    _team_abbr_normalize,
    normalize_player_name,
)

FFTODAY_BASE = "https://www.fftoday.com/rankings/playerwkproj.php"

# FFToday position ids. Offense only: K (PosID=80) is per-count FG/XP without
# yardage (no clean map to our K targets) and there is no DST grid.
FFTODAY_POS_IDS: dict[str, int] = {"QB": 10, "RB": 20, "WR": 30, "TE": 40}
FFTODAY_POSITIONS: tuple[str, ...] = tuple(FFTODAY_POS_IDS)

FFTODAY_DEFAULT_WEEKS = tuple(range(1, 19))
_CACHE_VERSION = "v1"
_MIN_SEASON = 2010  # FFToday weekly-projection archive floor.

_RETRY_BACKOFF_S = 0.5
_MAX_FETCH_WORKERS = 6  # Be polite to fftoday.com.
_UA = "Mozilla/5.0 (compatible; ff-ml-research/1.0; benchmark-only)"

# Per-position map: internal target name -> 0-based data-cell index in a FFToday
# projection row. Row cells are [Chg, Player, Team, Opp, <stat cols...>, FPts].
# QB:  Chg Player Team Opp | Comp Att pYd pTD INT | rAtt rYd rTD | FPts
# RB/WR/TE: Chg Player Team Opp | rAtt rYd rTD | Rec recYd recTD | FPts
_FFTODAY_STAT_INDEX: dict[str, dict[str, int]] = {
    "QB": {
        "passing_yards": 6,
        "passing_tds": 7,
        "interceptions": 8,
        "rushing_yards": 10,
        "rushing_tds": 11,
    },
    "RB": {
        "rushing_yards": 5,
        "rushing_tds": 6,
        "receptions": 7,
        "receiving_yards": 8,
        "receiving_tds": 9,
    },
    "WR": {
        "rushing_yards": 5,
        "rushing_tds": 6,
        "receptions": 7,
        "receiving_yards": 8,
        "receiving_tds": 9,
    },
    "TE": {
        "rushing_yards": 5,
        "rushing_tds": 6,
        "receptions": 7,
        "receiving_yards": 8,
        "receiving_tds": 9,
    },
}

# All internal target columns (0-filled where a position doesn't carry the stat)
# so the frame has a uniform shape the aggregator can consume. FFToday has no
# fumble column, so ``fumbles_lost`` is always 0 (a ~0.1 pt/game contributor).
_ALL_TARGET_COLUMNS = (
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "receptions",
    "fumbles_lost",
)

# A player data row is uniquely identified by a link into /stats/players/<id>/<name>.
_PLAYER_ANCHOR = re.compile(
    r"<a\s+href=[\"']/stats/players/(\d+)/[^\"']*[\"'][^>]*>(.*?)</a>", re.I | re.S
)
_TD_CELL = re.compile(r"<td\b[^>]*>(.*?)</td>", re.I | re.S)
_TR_ROW = re.compile(r"<tr\b[^>]*>.*?</tr>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")


def _default_html_reader(url: str) -> str:
    """Fetch a URL and return decoded HTML text (injectable for tests)."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted host)
        return resp.read().decode("utf-8", errors="replace")


def _text(cell_html: str) -> str:
    return _html.unescape(_TAG.sub(" ", cell_html)).strip()


def _projection_url(year: int, week: int, position: str) -> str:
    return f"{FFTODAY_BASE}?Season={year}&GameWeek={week}&PosID={FFTODAY_POS_IDS[position]}"


def _num(cells: list[str], idx: int) -> float:
    if idx < 0 or idx >= len(cells):
        return 0.0
    try:
        return float(cells[idx].replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def _parse_projection_html(page: str, position: str) -> pd.DataFrame:
    """Parse one FFToday projection page into normalized target rows.

    A row is a ``<tr>`` whose Player cell links to ``/stats/players/<id>/<name>``.
    Header/section rows have no such link and are skipped. FPts is the last cell.
    """
    stat_index = _FFTODAY_STAT_INDEX[position]
    records: list[dict] = []
    for row in _TR_ROW.findall(page):
        anchor = _PLAYER_ANCHOR.search(row)
        if anchor is None:
            continue  # header / spacer row
        cells = [_text(c) for c in _TD_CELL.findall(row)]
        if len(cells) < 5:
            continue
        name = _html.unescape(_TAG.sub("", anchor.group(2))).strip()
        if not name:
            continue
        rec: dict = {
            "player_name": name,
            "fftoday_player_id": anchor.group(1),
            "team": cells[2] if len(cells) > 2 else "",
            "opponent": cells[3] if len(cells) > 3 else "",
            "fftoday_projected_pts": _num(cells, len(cells) - 1),
        }
        for target in _ALL_TARGET_COLUMNS:
            rec[target] = 0.0
        for target, idx in stat_index.items():
            rec[target] = _num(cells, idx)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _read_one_projection(
    year: int,
    week: int,
    position: str,
    *,
    reader=_default_html_reader,
    max_retries: int = 1,
    backoff_s: float = _RETRY_BACKOFF_S,
) -> pd.DataFrame | None:
    """Fetch + parse one (year, week, position) page. ``None`` on 404/empty/error.

    Mirrors the defensive behaviour of ``nflcom_loader._read_one_projection``:
    404 is expected for weeks that don't exist and is not retried; other transient
    errors retry once. ``reader`` is injectable so tests avoid the network.
    """
    url = _projection_url(year, week, position)
    for attempt in range(max_retries + 1):
        try:
            page = reader(url)
        except (HTTPError, URLError, FileNotFoundError) as e:
            code = getattr(e, "code", None)
            if code == 404 or isinstance(e, FileNotFoundError):
                print(f"  WARN fftoday: skip {position} {year} W{week} (404 / not found)")
                return None
            if attempt < max_retries:
                print(
                    f"  WARN fftoday: transient {type(e).__name__} on {position} "
                    f"{year} W{week}; retrying in {backoff_s}s"
                )
                time.sleep(backoff_s)
                continue
            print(f"  WARN fftoday: skip {position} {year} W{week} ({type(e).__name__})")
            return None
        else:
            df = _parse_projection_html(page, position)
            if df.empty:
                return None
            df["season"] = int(year)
            df["week"] = int(week)
            df["position"] = position
            return df
    return None


def load_fftoday_projections(
    seasons: list[int],
    weeks: tuple[int, ...] | list[int] | None = None,
    cache_dir: str = CACHE_DIR,
    force_refresh: bool = False,
    *,
    reader=_default_html_reader,
) -> pd.DataFrame:
    """Fetch + cache FFToday weekly projections for one or more seasons.

    Cache: ``{cache_dir}/fftoday_projections_{ver}_{min}_{max}_{weeks}.parquet``.
    """
    if not seasons:
        raise ValueError("seasons must be a non-empty list of ints")
    bad = [s for s in seasons if s < _MIN_SEASON]
    if bad:
        raise ValueError(f"FFToday archive starts at {_MIN_SEASON}; got {sorted(bad)}")
    weeks_to_try = tuple(weeks) if weeks is not None else FFTODAY_DEFAULT_WEEKS
    os.makedirs(cache_dir, exist_ok=True)
    lo, hi = min(weeks_to_try), max(weeks_to_try)
    weeks_sig = (
        f"w{lo}-{hi}"
        if list(weeks_to_try) == list(range(lo, hi + 1))
        else f"w{lo}-{hi}-{len(set(weeks_to_try))}"
    )
    cache_path = (
        f"{cache_dir}/fftoday_projections_{_CACHE_VERSION}"
        f"_{min(seasons)}_{max(seasons)}_{weeks_sig}.parquet"
    )
    if os.path.exists(cache_path) and not force_refresh:
        return pd.read_parquet(cache_path)

    tasks = [
        (year, week, position)
        for year in seasons
        for week in weeks_to_try
        for position in FFTODAY_POSITIONS
    ]
    parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=_MAX_FETCH_WORKERS) as executor:
        futures = {
            executor.submit(_read_one_projection, y, w, p, reader=reader): (y, w, p)
            for (y, w, p) in tasks
        }
        for future in as_completed(futures):
            raw = future.result()
            if raw is not None:
                parts.append(raw)

    if not parts:
        raise RuntimeError(
            f"No FFToday projection rows fetched for seasons={seasons}; "
            "check upstream URL or network access."
        )
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["season", "week", "position", "player_name"]).reset_index(drop=True)
    atomic_write_parquet(df, cache_path)
    return df


def load_fftoday_with_gsis_id(
    seasons: list[int],
    cache_dir: str = CACHE_DIR,
    force_refresh: bool = False,
    *,
    rosters: pd.DataFrame | None = None,
    min_match_rate: float = 0.90,
    reader=_default_html_reader,
) -> pd.DataFrame:
    """Augment ``load_fftoday_projections`` with internal ``player_id`` (gsis_id).

    Two-stage name join, identical strategy to ``nflcom_loader.load_nflcom_with_gsis_id``
    (FFToday carries the correct historical team, so the primary team-aware join is
    reliable): (1) (norm_name, season, team, position); (2) fallback
    (norm_name, season, position) when that key maps to a single distinct id. Raises
    if the global match rate < ``min_match_rate``.
    """
    if not seasons:
        raise ValueError("seasons must be a non-empty list of ints")
    os.makedirs(cache_dir, exist_ok=True)
    rate_sig = f"mr{int(round(min_match_rate * 100))}"
    cache_path = (
        f"{cache_dir}/fftoday_projections_joined_{_CACHE_VERSION}"
        f"_{min(seasons)}_{max(seasons)}_{rate_sig}.parquet"
    )
    if os.path.exists(cache_path) and not force_refresh and rosters is None:
        return pd.read_parquet(cache_path)

    proj = load_fftoday_projections(
        seasons, cache_dir=cache_dir, force_refresh=force_refresh, reader=reader
    )
    if rosters is None:
        rosters = nfl_source.rosters(list(seasons))
    lookup = _build_roster_lookup(rosters)

    proj = proj.copy()
    proj["team"] = proj["team"].map(_team_abbr_normalize)
    proj["opponent"] = proj["opponent"].map(_team_abbr_normalize)
    proj["norm_name"] = proj["player_name"].map(normalize_player_name)

    primary = proj.merge(
        lookup[["norm_name", "season", "team", "position", "player_id"]],
        on=["norm_name", "season", "team", "position"],
        how="left",
    )

    unmatched_mask = primary["player_id"].isna()
    if unmatched_mask.any():
        unique_keys = (
            lookup.groupby(["norm_name", "season", "position"])["player_id"]
            .nunique(dropna=True)
            .reset_index(name="_n")
        )
        unique_keys = unique_keys[unique_keys["_n"] == 1][["norm_name", "season", "position"]]
        fallback_lookup = (
            lookup.drop(columns=["team"])
            .merge(unique_keys, on=["norm_name", "season", "position"], how="inner")
            .drop_duplicates(subset=["norm_name", "season", "position"])
        )
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

    n_total = len(primary)
    n_matched = int(primary["player_id"].notna().sum())
    match_rate = n_matched / n_total if n_total else 0.0
    print(f"\nFFToday gsis_id join: {n_matched}/{n_total} = {match_rate:.1%} matched")
    unmatched = primary.loc[primary["player_id"].isna(), ["player_name", "position", "team"]]
    if not unmatched.empty:
        print("Top-5 unmatched names:")
        print(_format_unmatched_diagnostic(unmatched))
    if match_rate < min_match_rate:
        raise RuntimeError(
            f"FFToday gsis_id match rate {match_rate:.1%} < min_match_rate "
            f"{min_match_rate:.1%}. Top-5 unmatched:\n" + _format_unmatched_diagnostic(unmatched)
        )

    primary = primary.drop(columns=["norm_name"])
    # Drop rows that never matched a gsis_id — the comparison joins on player_id and
    # an NaN id can't pair to a model prediction anyway (mirrors Sleeper/NFL.com).
    primary = primary[primary["player_id"].notna()].reset_index(drop=True)
    atomic_write_parquet(primary, cache_path)
    return primary
