"""Serving-safe expert projection helpers.

The analysis package is deliberately excluded from the production Docker image,
so serving cannot import the offline comparison scripts. This module carries the
small projection helpers needed by the live Season Leaders surface.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.config import CACHE_DIR
from src.data import nfl_source
from src.shared.aggregate_targets import POSITION_TARGET_MAP, predictions_to_fantasy_points

_EXPERT_KEY_COLS = ["player_id", "season", "week"]


def project_nflcom_to_fantasy(
    nflcom_df: pd.DataFrame, pos: str, scoring_format: str = "ppr"
) -> pd.DataFrame:
    """Score NFL.com projected raw stats on this app's fantasy-point scale."""
    value_col = "nflcom_pred_total"
    if nflcom_df is None or nflcom_df.empty or "position" not in nflcom_df.columns:
        return pd.DataFrame(columns=[*_EXPERT_KEY_COLS, value_col])

    pos_df = nflcom_df[(nflcom_df["position"] == pos) & nflcom_df["player_id"].notna()].copy()
    if pos_df.empty:
        return pd.DataFrame(columns=[*_EXPERT_KEY_COLS, value_col])

    out = pos_df[_EXPERT_KEY_COLS].copy()
    if pos == "K":
        out[value_col] = pd.to_numeric(
            pos_df.get("nflcom_projected_pts", np.nan), errors="coerce"
        ).to_numpy()
        return out

    targets = list(POSITION_TARGET_MAP.get(pos, {}))
    if not targets:
        return pd.DataFrame(columns=[*_EXPERT_KEY_COLS, value_col])
    pred_dict = {}
    for target in targets:
        if target in pos_df.columns:
            pred_dict[target] = (
                pd.to_numeric(pos_df[target], errors="coerce").fillna(0.0).to_numpy()
            )
        else:
            pred_dict[target] = np.zeros(len(pos_df), dtype=float)

    out[value_col] = predictions_to_fantasy_points(pos, pred_dict, scoring_format)
    return out


SLEEPER_BASE = "https://api.sleeper.app/projections/nfl"
SLEEPER_OFFENSE_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")
_SLEEPER_DEF_CODE = "DEF"
SLEEPER_FETCH_POSITIONS: tuple[str, ...] = (*SLEEPER_OFFENSE_POSITIONS, _SLEEPER_DEF_CODE)
SLEEPER_DEFAULT_WEEKS = tuple(range(1, 19))
_CACHE_VERSION = "v2"
_MIN_SEASON = 2018
_RETRY_BACKOFF_S = 0.5
_REQUEST_TIMEOUT_S = 30

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

_DST_TEAM_FIXUP: dict[str, str] = {"LAR": "LA"}
_ALL_TARGET_COLUMNS: tuple[str, ...] = (
    *SLEEPER_STAT_MAP.values(),
    *SLEEPER_DST_STAT_MAP.values(),
)


def _default_sleeper_reader(url: str) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "fantasy-ml-serving"})
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
        return json.load(resp)


def _projection_url(season: int, week: int, positions: Sequence[str]) -> str:
    pos_q = "".join(f"&position[]={p}" for p in positions)
    return f"{SLEEPER_BASE}/{season}/{week}?season_type=regular{pos_q}"


def _read_one_sleeper_week(
    season: int,
    week: int,
    positions: Sequence[str],
    *,
    reader=_default_sleeper_reader,
    max_retries: int = 1,
    backoff_s: float = _RETRY_BACKOFF_S,
) -> list | None:
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


def _normalize_sleeper_week(records: list, season: int, week: int) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in records:
        player = rec.get("player") or {}
        raw_pos = player.get("position")
        stats = rec.get("stats") or {}
        if raw_pos in SLEEPER_OFFENSE_POSITIONS:
            stat_map = SLEEPER_STAT_MAP
            if "pts_ppr" not in stats and not any(k in stats for k in stat_map):
                continue
            row = {
                "position": raw_pos,
                "sleeper_player_id": str(rec.get("player_id")),
                "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                "team": rec.get("team") or player.get("team_abbr") or player.get("team") or "",
            }
        elif raw_pos == _SLEEPER_DEF_CODE:
            stat_map = SLEEPER_DST_STAT_MAP
            if "pts_ppr" not in stats and not any(k in stats for k in stat_map):
                continue
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
    for col in _ALL_TARGET_COLUMNS:
        df[col] = df[col].fillna(0.0) if col in df.columns else 0.0
    return df


def _validate_sleeper_seasons(seasons: Sequence[int]) -> list[int]:
    seasons = sorted({int(s) for s in seasons})
    if not seasons:
        raise ValueError("seasons must be a non-empty sequence of ints")
    too_early = [s for s in seasons if s < _MIN_SEASON]
    if too_early:
        raise ValueError(
            f"Sleeper projections are only genuine from {_MIN_SEASON} onward; got {too_early}"
        )
    return seasons


def _weeks_signature(weeks: Sequence[int]) -> str:
    uniq = sorted({int(w) for w in weeks})
    return (
        f"w{uniq[0]}-{uniq[-1]}"
        if uniq == list(range(uniq[0], uniq[-1] + 1))
        else f"w{'_'.join(map(str, uniq))}"
    )


def load_sleeper_projections(
    seasons: Sequence[int],
    cache_dir: str = CACHE_DIR,
    force_refresh: bool = False,
    *,
    weeks: Sequence[int] | None = None,
    positions: Sequence[str] = SLEEPER_FETCH_POSITIONS,
    reader=_default_sleeper_reader,
) -> pd.DataFrame:
    """Fetch + cache Sleeper/RotoWire projections for offense and DST."""
    seasons = _validate_sleeper_seasons(seasons)
    weeks_to_try = tuple(weeks) if weeks is not None else SLEEPER_DEFAULT_WEEKS
    os.makedirs(cache_dir, exist_ok=True)
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
            records = _read_one_sleeper_week(season, week, positions, reader=reader)
            if records is None:
                continue
            frame = _normalize_sleeper_week(records, season, week)
            if not frame.empty:
                parts.append(frame)

    if not parts:
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
    reader=_default_sleeper_reader,
    player_ids_loader=None,
) -> pd.DataFrame:
    """Augment Sleeper/RotoWire projections with internal ``player_id``."""
    proj = load_sleeper_projections(
        seasons, cache_dir=cache_dir, force_refresh=force_refresh, weeks=weeks, reader=reader
    )
    if player_ids_loader is None:
        player_ids_loader = nfl_source.player_ids
    ids = player_ids_loader()

    bridge = ids[["sleeper_id", "gsis_id"]].dropna().drop_duplicates(subset=["sleeper_id"])
    bridge = bridge.assign(sleeper_player_id=bridge["sleeper_id"].astype("int64").astype(str))

    is_dst = proj["position"] == "DST"
    offense = (
        proj[~is_dst]
        .merge(bridge[["sleeper_player_id", "gsis_id"]], on="sleeper_player_id", how="left")
        .rename(columns={"gsis_id": "player_id"})
    )
    dst = proj[is_dst].copy()
    dst["player_id"] = dst["sleeper_player_id"]
    merged = pd.concat([offense, dst], ignore_index=True)

    n_off = len(offense)
    n_matched = int(offense["player_id"].notna().sum())
    rate = n_matched / n_off if n_off else 0.0
    print(
        f"\nSleeper gsis_id join: {n_matched}/{n_off} = {rate:.1%} offense matched "
        f"(rotowire); DST team-keyed (n={len(dst)})"
    )
    return merged
