"""External per-game / player-level signal sources: ff_opportunity, ESPN QBR,
and player contracts.

Each loader returns a *merge-ready* frame keyed to the canonical weekly join
keys and caches to ``{cache_dir}/{name}_{min}_{max}.parquet`` (mirrors the
season-suffixed cache convention in :mod:`src.data.loader` and
:mod:`src.data.redzone_pbp`). ``src/data/loader.py`` calls these inside
``load_raw_data``'s ``ThreadPoolExecutor`` and left-merges the result.

All raw fetches go through :mod:`src.data.nfl_source` (the nflreadpy→pandas
boundary): ``nfl_source.ff_opportunity`` / ``nfl_source.contracts`` /
``nfl_source.player_ids`` wrap nflreadpy loaders, and ``nfl_source.qbr_weekly``
reads the espnscrapeR-data CSV (nflreadpy has no QBR loader).

Three sources, three shapes:

* **ff_opportunity** (ffverse "expected points") — per-(player_id, season, week)
  *expected* stats modeled from in-game opportunity. ``player_id`` is gsis, so
  it joins straight onto ``weekly``.
* **ESPN QBR** — per-game opponent-adjusted quarterback rating. ESPN-id keyed;
  bridged to gsis via the ``player_ids`` crosswalk shared with the snap-count
  merge. QB-only; non-QB rows stay NaN (only QB config consumes it).
* **contracts** — OTC historical contracts (one row per contract) collapsed to
  the contract *in effect as of each season*. ``apy_cap_pct`` (APY as a share
  of the salary cap) is the cross-era-normalized headline.

Leakage note: the per-game ff_opp/QBR columns are consumed by the attention NN
*history* branch (prior in-season games only — see
``src.features.engineer.build_game_history_arrays``) and by prior-season
aggregates (``season + 1`` shift); neither path lets the current game's value
predict itself. ff_opp expected fantasy points are used as a *feature*, never a
*target* — targets remain raw stats.
"""

from __future__ import annotations

import hashlib
import os

import pandas as pd
import pyarrow.parquet as pq

from src.config import CACHE_DIR
from src.data import nfl_source

# --- ff_opportunity (ffverse expected points) ------------------------------
# Union of the expected-stat columns the skill positions wire into their
# attention history sequences, plus ``total_fantasy_points_exp`` for the
# prior-season opportunity prior. Each position whitelists the subset relevant
# to its targets (QB: pass/rush; RB: rush/rec; WR/TE: rec).
FF_OPP_FEATURE_COLUMNS: tuple[str, ...] = (
    "pass_yards_gained_exp",
    "pass_touchdown_exp",
    "pass_interception_exp",
    "rush_yards_gained_exp",
    "rush_touchdown_exp",
    "rec_yards_gained_exp",
    "rec_touchdown_exp",
    "receptions_exp",
    "rec_first_down_exp",
    "total_fantasy_points_exp",
)

# --- ESPN QBR --------------------------------------------------------------
QBR_FEATURE_COLUMNS: tuple[str, ...] = ("qbr_total", "pts_added")

# --- contracts -------------------------------------------------------------
CONTRACT_FEATURE_COLUMNS: tuple[str, ...] = (
    "contract_apy_cap_pct",
    "contract_guaranteed",
    "contract_years_remaining",
    "contract_age",
)

# Per-game external stats that :mod:`src.features.engineer` rolls into
# ``prior_season_mean_{stat}`` aggregates for the static branch. Contracts are
# already player-season state, so they are NOT aggregated here.
EXTERNAL_PRIOR_STATS: tuple[str, ...] = (*FF_OPP_FEATURE_COLUMNS, *QBR_FEATURE_COLUMNS)


def _seasons_cache_signature(seasons: list[int]) -> str:
    """Filename-safe signature of a season selection for cache keys.

    A contiguous range renders as ``{min}_{max}`` — byte-identical to the legacy
    ``seasons[0]_seasons[-1]`` key, so the production full-range cache filename
    (e.g. ``ff_opportunity_2012_2025.parquet``) is unchanged. A sparse /
    non-contiguous selection adds ``_{len}_{8-hex}`` so two different selections
    sharing min/max can't collide on one cache file. Order- and
    duplicate-insensitive. Mirrors ``nflcom_loader._weeks_cache_signature``.
    """
    uniq = sorted(set(int(s) for s in seasons))
    if not uniq:
        return "none"
    lo, hi = uniq[0], uniq[-1]
    if uniq == list(range(lo, hi + 1)):
        return f"{lo}_{hi}"
    digest = hashlib.sha1(",".join(map(str, uniq)).encode()).hexdigest()[:8]
    return f"{lo}_{hi}_{len(uniq)}_{digest}"


def _cached_parquet_has_columns(path: str, required: tuple[str, ...]) -> bool:
    """False (-> regenerate) if the cached parquet is missing any required
    column. A column added to a feature tuple after the cache was written would
    otherwise be left-merged then ``fillna(0)``'d to all-zeros downstream. Reads
    only the parquet schema (no row data). Mirrors
    ``redzone_pbp._cached_rz_pbp_is_current``.
    """
    missing = set(required) - set(pq.read_schema(path).names)
    if missing:
        print(f"  Stale cache at {path} missing {sorted(missing)}; regenerating")
        return False
    return True


def _coerce_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``(player_id, season, week)`` to the canonical nflverse-weekly
    dtypes (player_id str, season/week int) so the loader merges don't fail on
    dtype skew. ff_opportunity ships ``season`` as str and ``week`` as float,
    which pandas refuses to merge against weekly's int columns. Applied on both
    the fresh-build and cache-read paths so a stale cache can't reintroduce it.
    """
    if df.empty:
        return df
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(str)
    for key in ("season", "week"):
        if key in df.columns:
            df[key] = pd.to_numeric(df[key], errors="coerce").astype("int64")
    return df


def load_ff_opportunity(seasons: list[int], cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Per-(player_id, season, week) ff_opportunity expected-stat columns.

    Fetched via ``nfl_source.ff_opportunity`` (nflreadpy's ``load_ff_opportunity``)
    and cached. Returns the merge keys + ``FF_OPP_FEATURE_COLUMNS``. On a fetch
    failure (network boundary) returns an empty frame so the cold build still
    succeeds and the loader backfills the columns with 0.
    """
    path = f"{cache_dir}/ff_opportunity_{_seasons_cache_signature(seasons)}.parquet"
    keep = ["player_id", "season", "week", *FF_OPP_FEATURE_COLUMNS]
    if os.path.exists(path) and _cached_parquet_has_columns(path, FF_OPP_FEATURE_COLUMNS):
        return _coerce_merge_keys(pd.read_parquet(path))
    try:
        df = nfl_source.ff_opportunity(list(seasons))
    except Exception as e:
        print(f"WARNING: ff_opportunity fetch failed ({e}); skipping")
        return pd.DataFrame(columns=keep)
    cols = [c for c in keep if c in df.columns]
    # One row per player-game (defensive against duplicate source rows so the
    # downstream left-merge can't fan out).
    out = df[cols].drop_duplicates(subset=["player_id", "season", "week"], keep="first")
    out = _coerce_merge_keys(out)
    out.to_parquet(path)
    return out


def _fetch_qbr_weekly_raw(seasons: list[int]) -> pd.DataFrame:
    """Raw weekly ESPN QBR (still ESPN-id keyed), subset to the bridge columns.

    Not cached on its own — :func:`load_qbr_weekly` caches the bridged,
    merge-ready result. Subset keeps the frame small and free of the
    mixed-type columns that don't round-trip through parquet.
    """
    qbr = nfl_source.qbr_weekly(list(seasons))
    keep = [
        c
        for c in (
            "season",
            "season_type",
            "game_week",
            "week_num",
            "player_id",
            *QBR_FEATURE_COLUMNS,
        )
        if c in qbr.columns
    ]
    return qbr[keep]


def bridge_qbr_to_gsis(qbr: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    """Map ESPN-id weekly QBR onto gsis ``player_id`` + ``(season, week)``.

    Regular season only. Returns merge-ready ``[player_id, season, week]`` +
    ``QBR_FEATURE_COLUMNS``. Rows that don't bridge to a gsis id are dropped
    (they simply won't merge — the weekly row keeps NaN QBR).
    """
    keep_cols = ["player_id", "season", "week", *QBR_FEATURE_COLUMNS]
    if qbr.empty or ids.empty or "espn_id" not in ids.columns:
        return pd.DataFrame(columns=keep_cols)
    q = qbr.copy()
    # QBR carries playoff rows too; keep regular season only.
    if "season_type" in q.columns:
        q = q[q["season_type"].astype(str).str.contains("Regular", case=False, na=False)]
    # Week: prefer the numeric week_num, fall back to game_week.
    wk = q["week_num"] if "week_num" in q.columns else q["game_week"]
    q = q.assign(week=pd.to_numeric(wk, errors="coerce")).dropna(subset=["week"])
    q["week"] = q["week"].astype(int)
    # ESPN id -> gsis id bridge (dedupe on espn_id so the merge can't fan out).
    x = ids[["espn_id", "gsis_id"]].dropna().drop_duplicates(subset=["espn_id"], keep="first")
    x = x.assign(espn_id=pd.to_numeric(x["espn_id"], errors="coerce")).dropna(subset=["espn_id"])
    q["espn_player_id"] = pd.to_numeric(q["player_id"], errors="coerce")
    q = (
        q.drop(columns=["player_id"])
        .merge(x, left_on="espn_player_id", right_on="espn_id", how="left")
        .dropna(subset=["gsis_id"])
        .rename(columns={"gsis_id": "player_id"})
    )
    for c in QBR_FEATURE_COLUMNS:
        if c not in q.columns:
            q[c] = float("nan")
    return q[keep_cols].drop_duplicates(subset=["player_id", "season", "week"], keep="first")


def load_qbr_weekly(seasons: list[int], cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Merge-ready weekly QBR keyed to gsis ``player_id`` + ``(season, week)``.

    Fetches the raw ESPN-id QBR + the id crosswalk and bridges to gsis, then
    caches the bridged result. Returns ``[player_id, season, week]`` +
    ``QBR_FEATURE_COLUMNS`` (empty if the source is unavailable).
    """
    path = f"{cache_dir}/qbr_weekly_{_seasons_cache_signature(seasons)}.parquet"
    keep = ["player_id", "season", "week", *QBR_FEATURE_COLUMNS]
    if os.path.exists(path) and _cached_parquet_has_columns(path, QBR_FEATURE_COLUMNS):
        return _coerce_merge_keys(pd.read_parquet(path))
    try:
        raw = _fetch_qbr_weekly_raw(seasons)
        ids = nfl_source.player_ids()
    except Exception as e:
        # Supplementary QB-only signal: a fetch failure degrades to "no QBR"
        # (loader leaves the columns NaN) rather than crashing the shared
        # load_raw_data pull for all six positions.
        print(f"WARNING: QBR fetch failed ({e}); skipping")
        return pd.DataFrame(columns=keep)
    out = _coerce_merge_keys(bridge_qbr_to_gsis(raw, ids))
    out.to_parquet(path)
    return out


def derive_active_contracts(contracts: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Collapse one-row-per-contract OTC data to the contract in effect as of
    each season (latest ``year_signed`` **strictly before** season), per gsis
    player.

    Returns ``[player_id, season]`` + ``CONTRACT_FEATURE_COLUMNS``. A
    (player, season) with no contract signed before that season is absent
    (the downstream left-merge + ``fillna(0)`` handles it).

    **Leakage (#645):** OTC stores only the integer ``year_signed``, with no
    sub-season date. A contract signed in season ``S`` may be an offseason deal
    (known before week 1) or a mid-season extension / free-agent signing (an
    October QB/RB/WR extension) that is *not* known before the early weeks of
    ``S`` — and the contract APY feature is a per-``(player, season)`` static
    value applied to every week of the season. Attaching a season-``S`` contract
    to season ``S`` therefore leaks the team's in-season reaction to the player
    into predictions of that season's earlier weeks. Because the integer year
    can't distinguish the two cases, a contract is treated as effective only
    from ``year_signed + 1`` (``year_signed < season``): by the start of the
    following season any signing — offseason or mid-season — is definitely
    known. This defers a genuinely-known offseason signing by one season (the
    conservative price of integer-only resolution) but keeps every prior-year
    contract: ``merge_asof`` still resolves to the most recent
    ``year_signed < season``.
    """
    base_cols = ["player_id", "season", *CONTRACT_FEATURE_COLUMNS]
    need = {"gsis_id", "year_signed", "years", "guaranteed", "apy_cap_pct"}
    if contracts.empty or not need.issubset(contracts.columns):
        return pd.DataFrame(columns=base_cols)
    c = contracts.loc[
        contracts["gsis_id"].notna() & contracts["year_signed"].notna(),
        ["gsis_id", "year_signed", "years", "guaranteed", "apy_cap_pct"],
    ].copy()
    c["year_signed"] = c["year_signed"].astype("int64")
    # Effective the season AFTER signing — see the leakage note above. merge_asof
    # then joins on this (so season S only sees contracts signed in S-1 or
    # earlier); ``contract_age``/``contract_years_remaining`` below still derive
    # from the true ``year_signed``.
    c["effective_season"] = c["year_signed"] + 1
    c = c.sort_values("effective_season")
    grid = pd.DataFrame(
        [(g, s) for g in c["gsis_id"].unique() for s in seasons],
        columns=["gsis_id", "season"],
    )
    grid["season"] = grid["season"].astype("int64")
    grid = grid.sort_values("season")
    merged = pd.merge_asof(
        grid,
        c,
        left_on="season",
        right_on="effective_season",
        by="gsis_id",
        direction="backward",
    ).dropna(subset=["year_signed"])
    merged["contract_age"] = merged["season"] - merged["year_signed"]
    merged["contract_years_remaining"] = (merged["years"] - merged["contract_age"]).clip(lower=0)
    merged = merged.rename(
        columns={
            "apy_cap_pct": "contract_apy_cap_pct",
            "guaranteed": "contract_guaranteed",
            "gsis_id": "player_id",
        }
    )
    return merged[base_cols]


def load_contracts(seasons: list[int], cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Active-as-of-season contract attributes per (player_id, season), cached."""
    path = f"{cache_dir}/contracts_{_seasons_cache_signature(seasons)}.parquet"
    if os.path.exists(path) and _cached_parquet_has_columns(path, CONTRACT_FEATURE_COLUMNS):
        return _coerce_merge_keys(pd.read_parquet(path))
    try:
        raw = nfl_source.contracts()
    except Exception as e:
        # Supplementary signal: degrade to "no contract" (loader fills 0)
        # rather than crashing the shared load_raw_data pull.
        print(f"WARNING: contracts fetch failed ({e}); skipping")
        return pd.DataFrame(columns=["player_id", "season", *CONTRACT_FEATURE_COLUMNS])
    out = _coerce_merge_keys(derive_active_contracts(raw, list(seasons)))
    out.to_parquet(path)
    return out
