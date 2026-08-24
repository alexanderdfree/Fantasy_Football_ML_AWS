"""Live upcoming-week predictions: frame builder, inference, cache, and poller.

The forward-looking slate (matchups, Vegas lines, active rosters, injuries)
comes from ESPN (:mod:`src.serving.espn_live`); historical context stays on
nflverse. To featurize the upcoming week **byte-identically to training** (the
anti-drift guarantee), we splice synthetic player-week rows for the upcoming
slate onto the full nflverse history and run the real offline pipeline
(``load_raw_data`` -> ``preprocess`` -> ``engineer.build_features``), then feed
the upcoming slice to the existing per-position serving inference
(``core._apply_position_models``). The artifact is **built by a scheduled CI
job** (``.github/workflows/refresh-upcoming-week.yml``) and uploaded to S3;
serving only **downloads** it — a background poller syncs the latest from S3,
it does not build it in the serving container (ADR-0018; the in-container build
was reverted in #1069 -> #1076 because a 2-worker task OOMs running
``load_raw_data`` + ``build_features`` + inference).

Skill positions only (QB/RB/WR/TE). K/DST use separate pipelines and land in a
follow-up.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
from datetime import UTC, datetime

import numpy as np
import pandas as pd

import src.serving.app as app_pkg
import src.serving.roster_meta as roster_meta
from src.config import CACHE_DIR, SEASONS
from src.data import nfl_source
from src.data.loader import load_raw_data
from src.data.nflcom_loader import load_nflcom_with_gsis_id
from src.data.preprocessing import preprocess
from src.features.engineer import build_features
from src.serving import core, espn_live, live_sources
from src.serving.expert_sources import load_sleeper_with_gsis_id
from src.serving.serialization import (
    _MODEL_PRED_PREFIXES,
    _bool_or_none,
    _int_or_none,
    _pred_col,
    _round_or_none,
    _safe_num,
    _safe_str,
)
from src.shared import weather_features

UPCOMING_POSITIONS = ("QB", "RB", "WR", "TE")
_VALID_SCORING = ("ppr", "half_ppr", "standard")

_ARTIFACT_NAME = "upcoming_week.json"
_ESPN_HEADSHOT = "https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png"

# The artifact is built OUT of serving (a scheduled CI job — see
# .github/workflows/refresh-upcoming-week.yml) and uploaded to S3; the serving
# container only DOWNLOADS + serves it. Running the full data+feature build in
# the 2-worker serving task OOM'd it (PBP rebuild + 2025-PBP download), so the
# build never runs here. This is the ADR-0018 CI-artifact path.
_ENV_BUCKET = "FF_MODEL_S3_BUCKET"
_ENV_PREFIX = "FF_MODEL_S3_PREFIX"
# How often the serving container re-pulls the artifact from S3 (a cheap GET);
# 0 disables. Default 10 min so a fresh CI build shows up without a redeploy.
_DOWNLOAD_INTERVAL_S = int(os.environ.get("FF_UPCOMING_SYNC_INTERVAL_S", "600"))

# Per-player, slowly-changing attributes that load_raw_data merges per game but
# the synthetic upcoming rows lack (no current-week merge exists yet). They're
# pass-through model features (build_features doesn't derive anything from them),
# so we carry each player's most-recent value forward onto the upcoming week.
# Left NaN they fill to a constant downstream, which makes the attention models
# blind to depth — most visibly the QB head, which then can't separate starters
# from backups and roughly doubles every projection (verified 2026 W1).
_CARRYFORWARD_ATTRS = (
    "depth_chart_rank",
    "contract_apy_cap_pct",
    "contract_guaranteed",
    "contract_years_remaining",
    "contract_age",
)
# Defaults for rows still missing after carry-forward (rookies / new players) and
# for current-health columns (which are about *this* week, not the past).
# depth_chart_rank uses the TRAINING sentinel -1 (loader.py fills unknown ranks
# with -1), which build_position_features remaps to the train-mean of real ranks
# (~0 after standardization) — the same neutral value training gives unknown
# players. A literal 3.0 is NOT remapped: it standardizes as a real rank-3
# (buried), a train/serve mismatch that most hurt the depth-sensitive QB head
# (#1270). game_status=1 (active) / practice_status=2 (full) ARE the neutral
# train modes (96% / 95% of rows), so they stay.
_CONTEXT_DEFAULTS = {"depth_chart_rank": -1.0, "game_status": 1.0, "practice_status": 2.0}

# Memoized history (the build_features INPUT frame). nflverse history is stable,
# so build it once per process and reuse across polls.
_history_lock = threading.Lock()
_history_cache: pd.DataFrame | None = None

# Last-built input signature → skip the heavy rebuild when nothing changed.
_state_lock = threading.Lock()
_last_signature: str | None = None


def _artifact_path() -> str:
    return os.path.join(core._PREDICTIONS_CACHE_DIR, _ARTIFACT_NAME)


def _schedules_path() -> str:
    return os.path.join(CACHE_DIR, f"schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet")


# --------------------------------------------------------------------------
# History (build_features input) — memoized
# --------------------------------------------------------------------------
def _load_history() -> pd.DataFrame:
    """Preprocessed full-history frame (the ``build_features`` input).

    Equivalent to refresh-splits' ``preprocess(load_raw_data())`` — runs off the
    warm ``data/raw`` caches present in the serving container. Memoized: the
    historical frame never changes between polls.
    """
    global _history_cache
    if _history_cache is not None:
        return _history_cache
    with _history_lock:
        if _history_cache is not None:
            return _history_cache
        _history_cache = preprocess(load_raw_data(list(SEASONS)))
        return _history_cache


# --------------------------------------------------------------------------
# Schedule-cache augmentation (Risk #1)
# --------------------------------------------------------------------------
def _augment_schedules_cache(sched_rows: pd.DataFrame) -> None:
    """Merge the upcoming season's schedule rows into the on-disk schedules
    parquet so ``build_features`` computes ``implied_team_total`` for the new
    week.

    ``engineer.build_features`` reads ``schedules_{lo}_{hi}.parquet`` directly
    (and ``weather_features._load_schedules`` caches it), keyed on the
    ``SEASONS`` range — the upcoming season is absent, so the implied total (the
    top Week-1 feature) would silently NaN→0. We splice the ESPN-derived rows in
    (dedupe on ``game_id``), atomically, and reset the module cache. Idempotent;
    the new-season rows are inert for historical features (keyed by season/week).
    No ``src/shared`` edit, so no retrain is triggered.
    """
    path = _schedules_path()
    if sched_rows is None or sched_rows.empty or not os.path.exists(path):
        return
    existing = pd.read_parquet(path)
    new = sched_rows.reindex(columns=existing.columns)
    keep = existing[~existing["game_id"].isin(sched_rows["game_id"])]
    combined = pd.concat([keep, new], ignore_index=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        combined.to_parquet(tmp)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    # Drop the module-cached schedules so consumers re-read the augmented file.
    weather_features._schedule_cache = None


# --------------------------------------------------------------------------
# Frame assembly
# --------------------------------------------------------------------------
def _build_skeleton(
    season: int, week: int, slate: pd.DataFrame, roster: pd.DataFrame, history_columns
) -> pd.DataFrame:
    """Synthetic player-week rows for the upcoming slate, shaped like a
    preprocessed history row (raw stats absent → NaN; features come from prior
    games via ``build_features``'s shift/rolling, so the current row's own NaNs
    are never self-referenced)."""
    matchup = slate[["recent_team", "opponent_team", "is_home"]].drop_duplicates("recent_team")
    skel = roster.merge(matchup, on="recent_team", how="inner")
    skel["season"] = season
    skel["week"] = week
    skel["season_type"] = "REG"
    # Align to the history schema so concat doesn't introduce ragged columns;
    # absent stat columns become NaN (handled downstream).
    skel = skel.reindex(columns=list(dict.fromkeys(list(history_columns) + list(skel.columns))))
    return skel


def build_upcoming_week_frame(
    season: int,
    week: int,
    slate: pd.DataFrame,
    roster: pd.DataFrame,
    injuries_df: pd.DataFrame | None = None,
    rosters_df: pd.DataFrame | None = None,
    depth_chart_ranks: dict[str, float] | None = None,
    game_status_map: dict[str, float] | None = None,
    practice_status_map: dict[str, float] | None = None,
    contract_features: pd.DataFrame | None = None,
):
    """Featurize the upcoming (season, week) via the real offline pipeline.

    Returns the ``build_features`` output rows for the current season *through*
    that (season, week): the upcoming week's synthetic rows (context-filled)
    plus the season's completed REG weeks. The completed weeks matter because
    ``core._apply_position_models`` rebuilds every within-season lookback from
    the frame it receives — the attention game-history arrays, the
    ``shift(1)``-rolling L3 features, QB's ``season_starts_to_date`` — so a
    frame holding only week W served every mid-season week as a Week-1
    cold start (#1411). The artifact still serializes only the week-W rows
    (``run_upcoming_inference`` slices predictions, not inputs).
    ``injuries_df`` (ESPN Out/Doubtful) and ``rosters_df``
    (weekly RES/INA reserve/inactive, ``nfl_source.rosters_weekly``) together
    size the role-inheritance vacancy out-set exactly as the training splits do
    (refresh-splits passes both) — omitting ``rosters_df`` shrinks the out-set
    and drifts ``inherited_opportunity`` / ``is_top_available`` (#1277).
    Both ``None`` → the feature degrades to 0. ``depth_chart_ranks`` / ``game_status_map`` /
    ``practice_status_map`` / ``contract_features`` are the live role/health/
    contract signals applied in ``_fill_current_week_context`` (``None`` →
    carry-forward / defaults only).
    """
    history = _load_history()
    skel = _build_skeleton(season, week, slate, roster, history.columns)
    combined = pd.concat([history, skel], ignore_index=True)
    featurized = build_features(combined, injuries_df=injuries_df, rosters_df=rosters_df)
    if not featurized.index.is_unique:
        # _apply_position_models writes predictions by index; a duplicated index
        # would land week-W predictions on context rows (and vice versa).
        raise RuntimeError("build_features returned a non-unique index")
    in_season = featurized["season"] == season
    sl = featurized[in_season & (featurized["week"] == week)].copy()
    sl = _fill_current_week_context(
        sl, history, depth_chart_ranks, game_status_map, practice_status_map, contract_features
    )
    # Season-to-date context is REG-only already: preprocess() filters history to
    # REG and _build_skeleton stamps REG on the synthetic rows.
    ctx = featurized[in_season & (featurized["week"] < week)]
    if week > 1 and ctx.empty:
        # Mid-season with no completed-week rows (e.g. nflverse weekly lagging)
        # degrades to exactly the #1411 one-week cold-start — make it visible.
        print(
            f"[upcoming_week] WARNING: no season-to-date rows for {season} W{week}; attention history/rollups will cold-start"
        )
    # No ignore_index: _apply_position_models writes predictions by index, so
    # both slices must keep their original (disjoint) ``featurized`` indices.
    return pd.concat([ctx, sl])


def _fill_current_week_context(
    slice_df: pd.DataFrame,
    history: pd.DataFrame,
    depth_chart_ranks: dict[str, float] | None = None,
    game_status_map: dict[str, float] | None = None,
    practice_status_map: dict[str, float] | None = None,
    contract_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Populate the current-week contextual columns the synthetic rows lack.

    Precedence per column: **live signal → carry-forward → default**.
    ``depth_chart_rank`` (live ESPN depth chart) and ``contract_*`` (live nflverse
    OTC derive) win over the carried value; ``game_status`` (ESPN injuries) and
    ``practice_status`` (nflverse injuries / Sleeper fallback) override the healthy
    default for flagged players. The remaining ``_CARRYFORWARD_ATTRS`` carry
    forward. All are pass-through features, so filling the built slice is
    equivalent to filling pre-build.
    """
    # --- live signals that should beat the (possibly stale) carry-forward ---
    if depth_chart_ranks and "depth_chart_rank" in slice_df.columns:
        live = slice_df["player_id"].map(depth_chart_ranks)
        slice_df["depth_chart_rank"] = live.where(live.notna(), slice_df["depth_chart_rank"])
    if contract_features is not None and not contract_features.empty:
        for col in live_sources.CONTRACT_FEATURE_COLUMNS:
            if col in slice_df.columns and col in contract_features.columns:
                live = slice_df["player_id"].map(contract_features[col])
                slice_df[col] = live.where(live.notna(), slice_df[col])
    # --- carry slowly-changing attrs forward for anything still missing ---
    present = [c for c in _CARRYFORWARD_ATTRS if c in history.columns and c in slice_df.columns]
    if present:
        latest = (
            history.sort_values(["player_id", "season", "week"])
            .groupby("player_id")[present]
            .last()
        )
        for c in present:
            carried = slice_df["player_id"].map(latest[c])
            slice_df[c] = slice_df[c].where(slice_df[c].notna(), carried)
    # --- live current-week health overrides the healthy default ---
    if game_status_map and "game_status" in slice_df.columns:
        gs = slice_df["player_id"].map(game_status_map)
        slice_df["game_status"] = gs.where(gs.notna(), slice_df["game_status"])
    if practice_status_map and "practice_status" in slice_df.columns:
        ps = slice_df["player_id"].map(practice_status_map)
        slice_df["practice_status"] = ps.where(ps.notna(), slice_df["practice_status"])
    for col, default in _CONTEXT_DEFAULTS.items():
        if col in slice_df.columns:
            slice_df[col] = slice_df[col].fillna(default)
    return slice_df


# --------------------------------------------------------------------------
# Inference (reuse core._apply_position_models verbatim)
# --------------------------------------------------------------------------
def run_upcoming_inference(
    featurized: pd.DataFrame,
    roster: pd.DataFrame,
    slate: pd.DataFrame,
    season: int,
    week: int,
):
    """Run the four skill-position models over the season-to-date frame.

    ``featurized`` carries the current season's completed weeks alongside the
    synthetic week-W rows (see ``build_upcoming_week_frame``); the whole frame
    goes through ``_apply_position_models`` so the per-position lookback
    rebuild sees real history (#1411), and only the (season, week) rows are
    returned for serialization. Builds a SEPARATE results frame (never touches
    the prod 2025 ``_cache["results"]``), indexed to ``featurized`` so
    ``_apply_position_models`` writes land on the right rows. Returns the results
    frame with per-format prediction columns + display/matchup fields, plus
    ``actual_*`` left null (no games played yet).
    """
    core._ensure_base_data()

    keep = [
        c
        for c in (
            "player_id",
            "position",
            "recent_team",
            "season",
            "week",
            "opponent_team",
            "is_home",
            "implied_team_total",
        )
        if c in featurized.columns
    ]
    results = featurized[keep].copy()

    # Index-preserving display + matchup enrichment (.map keeps the index that
    # _apply_position_models writes against).
    name_map = dict(
        zip(roster["player_id"], roster.get("espn_name", roster["player_id"]), strict=False)
    )
    head_map = {
        pid: _ESPN_HEADSHOT.format(espn_id=eid)
        for pid, eid in zip(roster["player_id"], roster.get("espn_id", []), strict=False)
        if eid
    }
    spread_map = dict(zip(slate["recent_team"], slate["spread_line"], strict=False))
    total_map = dict(zip(slate["recent_team"], slate["total_line"], strict=False))
    results["player_display_name"] = results["player_id"].map(name_map)
    results["headshot_url"] = results["player_id"].map(head_map).fillna("")
    results["spread_line"] = results["recent_team"].map(spread_map)
    results["total_line"] = results["recent_team"].map(total_map)

    # Pre-init prediction columns to NaN (mirror core._load_base_data_locked) so
    # a position whose models fail to load renders "--".
    for fmt in _VALID_SCORING:
        for prefix in _MODEL_PRED_PREFIXES:
            results[_pred_col(prefix, fmt)] = np.nan
    for prefix in _MODEL_PRED_PREFIXES:
        results[f"{prefix}_pred"] = np.nan

    splits = app_pkg._cache.get("splits", {})
    for pos in UPCOMING_POSITIONS:
        if pos not in splits:
            continue
        train, val, _ = splits[pos]
        try:
            core._apply_position_models(train, val, featurized, pos, results)
        except Exception as e:  # noqa: BLE001 - one position's failure must not sink the rest
            print(f"[upcoming_week] {pos} inference failed: {e!r}")
    # The season-to-date context rows exist only to give the per-position
    # lookback rebuild real history; only week W is served.
    return results[(results["season"] == season) & (results["week"] == week)].copy()


# --------------------------------------------------------------------------
# Expert projections for the upcoming slate (NFL.com + RotoWire + ESPN)
# --------------------------------------------------------------------------
_UPCOMING_EXPERTS = ("nflcom", "rotowire", "espn")

# ESPN's fantasy appliedTotal is PPR (stock league ``leaguedefaults/3``); the
# three formats differ only by the reception weight — identical to
# src/config.py's SCORING_STANDARD/HALF_PPR/PPR "receptions" values — so the
# non-PPR formats derive exactly by removing the reception share.
_ESPN_RECEPTION_WEIGHT = {"ppr": 1.0, "half_ppr": 0.5, "standard": 0.0}


def _fetch_upcoming_expert_frames(
    season: int, week: int, *, nflcom_loader=None, rotowire_loader=None, espn_loader=None
):
    """Best-effort raw expert projection frames for one (season, week).

    Runs only in the CI artifact builder (serving just downloads the artifact).
    NFL.com's archive publishes the current week's CSVs pre-game but not
    guaranteed; Sleeper serves the upcoming week live; ESPN's fantasy API
    serves weekly point projections year-round. Any source failing (or empty)
    degrades to None → its expert column stays null → the frontend renders
    "--" and hides that column (per-source gating).
    """
    if nflcom_loader is None:
        nflcom_loader = load_nflcom_with_gsis_id
    if rotowire_loader is None:
        rotowire_loader = load_sleeper_with_gsis_id
    if espn_loader is None:
        espn_loader = espn_live.fetch_fantasy_projections
    raw_nflcom = None
    try:
        raw_nflcom = nflcom_loader([season], weeks=[week], force_refresh=True)
    except Exception as e:  # noqa: BLE001 - expert data is optional
        print(f"[upcoming_week] NFL.com projections unavailable: {e!r}")
    if raw_nflcom is not None and (raw_nflcom.empty or "position" not in raw_nflcom.columns):
        raw_nflcom = None
    raw_rotowire = None
    try:
        raw_rotowire = rotowire_loader([season], weeks=[week], force_refresh=True)
    except Exception as e:  # noqa: BLE001 - expert data is optional
        print(f"[upcoming_week] RotoWire projections unavailable: {e!r}")
    if raw_rotowire is not None and (raw_rotowire.empty or "position" not in raw_rotowire.columns):
        raw_rotowire = None
    raw_espn = None
    try:
        raw_espn = espn_loader(season, week)
    except Exception as e:  # noqa: BLE001 - expert data is optional
        print(f"[upcoming_week] ESPN projections unavailable: {e!r}")
    if raw_espn is not None and (raw_espn.empty or "player_id" not in raw_espn.columns):
        raw_espn = None
    return raw_nflcom, raw_rotowire, raw_espn


def _expert_digest(raw_nflcom, raw_rotowire, raw_espn) -> str:
    """Compact change-detection digest of the expert frames for _input_signature.

    Row count + numeric-content sum per source: cheap, order-invariant, and
    moves whenever a projection value or the covered player set changes.
    """
    parts = []
    for name, df in (("nfl", raw_nflcom), ("rw", raw_rotowire), ("espn", raw_espn)):
        if df is None or df.empty:
            parts.append(f"{name}:none")
            continue
        num = df.select_dtypes("number")
        total = round(float(num.fillna(0).to_numpy().sum()), 2) if not num.empty else 0.0
        parts.append(f"{name}:{len(df)}:{total}")
    return "|".join(parts)


def _espn_points_for_format(raw_espn: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """Convert the ESPN PPR frame to one scoring format's join frame.

    ``espn_ppr_total`` is ESPN's own PPR appliedTotal; subtracting the non-PPR
    share of projected receptions lands the other formats exactly (verified
    against ESPN's standard stock league — see espn_live._STAT_ID_RECEPTIONS).
    Returns the ``_assign_expert_totals`` shape: key cols + espn_pred_total.
    """
    out = raw_espn[["player_id", "season", "week"]].copy()
    rec = pd.to_numeric(raw_espn["espn_receptions"], errors="coerce").fillna(0.0)
    ppr = pd.to_numeric(raw_espn["espn_ppr_total"], errors="coerce")
    out["espn_pred_total"] = ppr - (1.0 - _ESPN_RECEPTION_WEIGHT[fmt]) * rec
    return out


def _apply_upcoming_experts(results: pd.DataFrame, raw_nflcom, raw_rotowire, raw_espn) -> None:
    """Project + join the expert frames onto the upcoming results in place.

    Mirrors core._apply_expert_predictions (same projector + key-join helpers)
    scoped to the upcoming positions; per-source/position failures degrade to
    null columns, never break the artifact build. ESPN needs no per-position
    pass: its frame is already fantasy points keyed by player/season/week.
    """
    for source in _UPCOMING_EXPERTS:
        for fmt in _VALID_SCORING:
            results[_pred_col(source, fmt)] = np.nan
    for fmt in _VALID_SCORING:
        for pos in UPCOMING_POSITIONS:
            if raw_nflcom is not None:
                try:
                    nfl = core.project_nflcom_to_fantasy(raw_nflcom, pos, fmt)
                    core._assign_expert_totals(results, "nflcom", fmt, nfl, "nflcom_pred_total")
                except Exception as e:  # noqa: BLE001 - one source/position can degrade
                    print(f"[upcoming_week] NFL.com {pos}/{fmt} projection failed: {e!r}")
            if raw_rotowire is not None:
                try:
                    rw = core._project_rotowire_to_fantasy(raw_rotowire, pos, fmt)
                    core._assign_expert_totals(results, "rotowire", fmt, rw, "rotowire_pred_total")
                except Exception as e:  # noqa: BLE001 - one source/position can degrade
                    print(f"[upcoming_week] RotoWire {pos}/{fmt} projection failed: {e!r}")
        if raw_espn is not None:
            try:
                esp = _espn_points_for_format(raw_espn, fmt)
                core._assign_expert_totals(results, "espn", fmt, esp, "espn_pred_total")
            except Exception as e:  # noqa: BLE001 - one source can degrade
                print(f"[upcoming_week] ESPN {fmt} projection failed: {e!r}")


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------
def _results_to_upcoming_rows(results: pd.DataFrame, scoring: str) -> list[dict]:
    """Serialize the results frame into homepage rows for one scoring format.

    Mirrors ``serialization._records_to_player_rows`` (so the frontend can reuse
    its renderer + player modal) but adds matchup/Vegas fields and forces
    ``actual: null`` (no games played yet).
    """
    pred_keys = {prefix: _pred_col(prefix, scoring) for prefix in _MODEL_PRED_PREFIXES}
    rows: list[dict] = []
    for r in results.to_dict(orient="records"):
        spread = r.get("spread_line")
        total = r.get("total_line")
        rows.append(
            {
                "player_id": _safe_str(r.get("player_id")),
                "name": _safe_str(r.get("player_display_name")),
                "position": _safe_str(r.get("position")),
                "team": _safe_str(r.get("recent_team")),
                "opponent": _safe_str(r.get("opponent_team")),
                "is_home": int(r["is_home"]) if pd.notna(r.get("is_home")) else None,
                "spread_line": _round_or_none(spread),
                "total_line": _round_or_none(total),
                "implied_team_total": _round_or_none(r.get("implied_team_total")),
                "actual": None,
                "ridge_pred": _safe_num(r.get(pred_keys["ridge"])),
                "nn_pred": _safe_num(r.get(pred_keys["nn"])),
                "attn_nn_pred": _safe_num(r.get(pred_keys["attn_nn"])),
                "lgbm_pred": _safe_num(r.get(pred_keys["lgbm"])),
                "nflcom_pred": _safe_num(r.get(_pred_col("nflcom", scoring))),
                "rotowire_pred": _safe_num(r.get(_pred_col("rotowire", scoring))),
                "espn_pred": _safe_num(r.get(_pred_col("espn", scoring))),
                "headshot": _safe_str(r.get("headshot_url", "")),
                "age": _int_or_none(r.get("age")),
                "is_rookie": _bool_or_none(r.get("is_rookie")),
            }
        )
    return rows


def _build_artifact(season: int, week: int, results: pd.DataFrame) -> dict:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "available": True,
        "season": int(season),
        "week": int(week),
        "week_label": f"Week {week} · {season}",
        "no_actuals": True,
        "positions": list(UPCOMING_POSITIONS),
        "degraded_positions": core._degraded_positions(),
        "scoring": {fmt: _results_to_upcoming_rows(results, scoring=fmt) for fmt in _VALID_SCORING},
    }


def _input_signature(
    season,
    week,
    slate: pd.DataFrame,
    roster: pd.DataFrame,
    depth_chart_ranks: dict[str, float] | None = None,
    game_status_map: dict[str, float] | None = None,
    practice_status_map: dict[str, float] | None = None,
    contract_features: pd.DataFrame | None = None,
    rosters_df: pd.DataFrame | None = None,
    expert_digest: str = "",
) -> str:
    """Stable hash of (models + slate lines + roster id-set + live depth chart +
    injury/practice statuses + contracts + weekly RES/INA out-set) — recompute
    only on a real change (model swap, line/roster move, depth-chart shuffle,
    injury/practice update, contract change, reserve/inactive move)."""
    try:
        model_fp = core._compute_models_fingerprint()
    except Exception:  # noqa: BLE001
        model_fp = ""
    slate_part = slate.sort_values("recent_team")[
        ["recent_team", "opponent_team", "is_home", "spread_line", "total_line"]
    ].to_csv(index=False)
    roster_part = ",".join(sorted(roster["player_id"].astype(str)))
    depth_part = ",".join(f"{k}:{v}" for k, v in sorted((depth_chart_ranks or {}).items()))
    inj_part = ",".join(f"{k}:{v}" for k, v in sorted((game_status_map or {}).items()))
    prac_part = ",".join(f"{k}:{v}" for k, v in sorted((practice_status_map or {}).items()))
    if (
        contract_features is not None
        and not contract_features.empty
        and "contract_apy_cap_pct" in contract_features.columns
    ):
        apy_sum = round(float(contract_features["contract_apy_cap_pct"].fillna(0).sum()), 4)
        contract_part = f"{len(contract_features)}:{apy_sum}"
    else:
        contract_part = ""
    if (
        rosters_df is not None
        and not rosters_df.empty
        and {"status", "player_id", "week"} <= set(rosters_df.columns)
    ):
        side = rosters_df[rosters_df["status"].isin(["RES", "INA"])]
        rosters_part = ",".join(
            sorted((side["player_id"].astype(str) + ":" + side["week"].astype(str)).unique())
        )
    else:
        rosters_part = ""
    blob = (
        f"{season}|{week}|{model_fp}|{slate_part}|{roster_part}|"
        f"{depth_part}|{inj_part}|{prac_part}|{contract_part}|{rosters_part}|{expert_digest}"
    )
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------
# Cache write + read
# --------------------------------------------------------------------------
def _write_artifact(payload: dict) -> None:
    path = _artifact_path()
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    os.makedirs(core._PREDICTIONS_CACHE_DIR, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def _write_unavailable(reason: str) -> None:
    _write_artifact(
        {"generated_at": datetime.now(UTC).isoformat(), "available": False, "reason": reason}
    )


def read_cached_artifact() -> dict | None:
    """Return the on-disk artifact, or ``None`` if absent/unreadable."""
    path = _artifact_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def refresh_upcoming_week_cache(force: bool = False) -> dict | None:
    """Detect the next week, build + infer, and write + publish the artifact.

    Change-gated on ``_input_signature`` (models + lines + roster) unless
    ``force``. Returns the artifact dict (or the unavailable/None result).

    Fails loudly instead of degrading in two builder-fatal cases: it propagates
    :class:`espn_live.EspnUnreachableError` from week detection (an ESPN outage
    must not be recorded as "offseason" — the 2026-08 silent-stale incident;
    raised before any write, so the last-good local artifact stays untouched),
    and ``_publish_artifact`` raises when a configured S3 upload fails (the
    fresh artifact is on disk but S3 — what serving reads — is still stale).
    """
    detected = espn_live.next_unplayed_week(SEASONS[-1])
    if detected is None:
        _write_unavailable("offseason")
        return read_cached_artifact()
    season, week = detected

    slate, sched_rows = espn_live.fetch_slate(season, week)
    if slate.empty:
        _write_unavailable("no_slate")
        return read_cached_artifact()
    team_id_to_code = dict(zip(slate["team_id"], slate["recent_team"], strict=False))
    roster = espn_live.fetch_active_rosters(team_id_to_code)
    if roster.empty:
        _write_unavailable("no_roster")
        return read_cached_artifact()

    injuries_df = espn_live.fetch_injuries_df(season, week)
    # Weekly rosters (RES/INA reserve/inactive) size the inheritance vacancy
    # out-set the same way training's refresh-splits does (which passes both
    # injuries_df AND rosters_df). Must be rosters_weekly — the seasonal frame is
    # ~1 row/player and registers no vacancies (#1277). Single season keeps the
    # fetch light; only the current (season, week) out-set feeds the served rows.
    # Optional enrichment: on a fetch failure degrade to None (the inheritance
    # feature handles it — pre-#1277 behavior) rather than crash the whole
    # refresh, matching the sibling live_sources.fetch_* network boundaries.
    try:
        rosters_df = nfl_source.rosters_weekly([season])
    except Exception as e:  # noqa: BLE001 - network/data boundary
        print(f"[upcoming_week] rosters_weekly({season}) failed: {e!r}; out-set -> injuries-only")
        rosters_df = None
    # Don't surface projections for players ruled OUT (they won't play); keep
    # them in injuries_df so the role-inheritance feature still sizes vacancies.
    out_ids = set(injuries_df.loc[injuries_df["report_status"] == "Out", "gsis_id"])
    if out_ids:
        roster = roster[~roster["player_id"].isin(out_ids)].copy()

    # Live role + health + contract signals the synthetic rows otherwise lack:
    # ESPN sets depth_chart_rank + game_status; nflverse (Sleeper fallback) sets
    # practice_status; nflverse OTC sets the current-season contract_* values.
    depth_chart_ranks = espn_live.fetch_depth_chart_ranks(season, team_id_to_code)
    game_status_map = espn_live.fetch_injury_status_map(season, week)
    practice_status_map = live_sources.fetch_practice_status_map(season, week)
    contract_features = live_sources.fetch_contract_features(season)

    # Expert projections for the slate (NFL.com + RotoWire + ESPN), fetched up front so
    # a projection update alone re-triggers the rebuild via the signature.
    raw_nflcom, raw_rotowire, raw_espn = _fetch_upcoming_expert_frames(season, week)

    sig = _input_signature(
        season,
        week,
        slate,
        roster,
        depth_chart_ranks,
        game_status_map,
        practice_status_map,
        contract_features,
        rosters_df=rosters_df,
        expert_digest=_expert_digest(raw_nflcom, raw_rotowire, raw_espn),
    )
    with _state_lock:
        if not force and sig == _last_signature and read_cached_artifact() is not None:
            return read_cached_artifact()

    _augment_schedules_cache(sched_rows)
    featurized = build_upcoming_week_frame(
        season,
        week,
        slate,
        roster,
        injuries_df=injuries_df,
        rosters_df=rosters_df,
        depth_chart_ranks=depth_chart_ranks,
        game_status_map=game_status_map,
        practice_status_map=practice_status_map,
        contract_features=contract_features,
    )
    results = run_upcoming_inference(featurized, roster, slate, season, week)
    # Age-at-kickoff + rookie flag for the frontend's Age/Rookies filters
    # (same best-effort semantics as the season-leaders path).
    results = roster_meta.attach_age_and_rookie(results)
    # Expert columns for the homepage (best-effort; nulls when a feed is down).
    _apply_upcoming_experts(results, raw_nflcom, raw_rotowire, raw_espn)
    payload = _build_artifact(season, week, results)
    _publish_artifact(payload, sig)
    print(f"[upcoming_week] refreshed {season} W{week}: {len(results)} players")
    return payload


def _publish_artifact(payload: dict, sig: str) -> None:
    """Write the artifact locally, publish it to S3, then commit ``sig``.

    When a bucket IS configured (the CI builder), a failed upload raises: the
    builder's one job is a fresh artifact ON S3, and a green exit with a stale
    S3 object is exactly the silent failure mode of the 2026-08 incident.
    ``_last_signature`` is committed only after a successful publish so the
    next run rebuilds + re-uploads instead of signature-skipping. No bucket
    (local/dev/tests) keeps the old best-effort behavior.
    """
    global _last_signature
    _write_artifact(payload)
    if not upload_artifact_to_s3() and os.environ.get(_ENV_BUCKET, "").strip():
        raise RuntimeError(
            "[upcoming_week] artifact built but the S3 upload failed; "
            "left _last_signature uncommitted so the next run re-publishes"
        )
    with _state_lock:
        _last_signature = sig


# --------------------------------------------------------------------------
# S3 transfer (CI builds + uploads; serving downloads)
# --------------------------------------------------------------------------
def _s3_artifact_key() -> str:
    prefix = os.environ.get(_ENV_PREFIX, "models").strip("/")
    return f"{prefix}/predictions_cache/{_ARTIFACT_NAME}"


def upload_artifact_to_s3() -> bool:
    """Upload the local artifact to S3 (the CI builder). Best-effort; ``False`` on
    unset bucket / missing file / any S3 error (never raises)."""
    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    path = _artifact_path()
    if not bucket or not os.path.isfile(path):
        return False
    try:
        import boto3

        boto3.client("s3").upload_file(
            path, bucket, _s3_artifact_key(), ExtraArgs={"ContentType": "application/json"}
        )
        print(f"[upcoming_week] uploaded artifact -> s3://{bucket}/{_s3_artifact_key()}")
        return True
    except Exception as e:  # noqa: BLE001 - best-effort
        print(f"[upcoming_week] S3 artifact upload failed: {e!r}")
        return False


def sync_artifact_from_s3() -> bool:
    """Download the CI-built artifact from S3 into data/serving_cache (serving).
    Best-effort: unset bucket / missing object / error → ``False`` and the route
    serves 503 ``warming`` until the next sync."""
    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    if not bucket:
        return False
    path = _artifact_path()
    tmp = f"{path}.s3.{os.getpid()}.tmp"
    try:
        import boto3

        os.makedirs(core._PREDICTIONS_CACHE_DIR, exist_ok=True)
        boto3.client("s3").download_file(bucket, _s3_artifact_key(), tmp)
        os.replace(tmp, path)
        return True
    except Exception as e:  # noqa: BLE001 - best-effort
        print(f"[upcoming_week] S3 artifact sync skipped: {e!r}")
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        return False


# --------------------------------------------------------------------------
# Serving-side download poller (lightweight — no build, just an S3 GET)
# --------------------------------------------------------------------------
def start_artifact_download_poller(
    interval_s: int | None = None, stop_event: threading.Event | None = None
) -> threading.Thread | None:
    """Daemon thread that re-pulls the CI-built artifact from S3 so a fresh build
    appears without a redeploy.

    Interval from ``FF_UPCOMING_SYNC_INTERVAL_S`` (default 10 min); ``0`` disables.
    Cheap (a single S3 GET) — none of the build/PBP/OOM cost that the in-serving
    build had. Broad try/except so a transient S3 error never kills the loop.
    """
    interval = _DOWNLOAD_INTERVAL_S if interval_s is None else interval_s
    if interval <= 0:
        return None
    ev = stop_event or threading.Event()

    def _loop():
        while not ev.is_set():
            try:
                sync_artifact_from_s3()
            except Exception as e:  # noqa: BLE001 - poller must never die
                print(f"[upcoming_week] artifact sync failed: {e!r}")
            ev.wait(interval)

    t = threading.Thread(target=_loop, name="upcoming-week-sync", daemon=True)
    t.start()
    return t


def main() -> None:
    """CLI entry for the scheduled CI builder: build the upcoming-week artifact
    from the live ESPN slate + the models/data on disk, write it, and upload to
    S3. Run by .github/workflows/refresh-upcoming-week.yml.

    Exits non-zero (red CI) whenever a fresh artifact was NOT published: an
    unreachable ESPN raises out of ``refresh_upcoming_week_cache``; an empty
    slate/roster after a successful week detection (inherently anomalous — the
    detection just saw a scheduled game) surfaces here as an ``available:
    False`` artifact with a non-offseason reason. Only a VERIFIED offseason
    (ESPN answered every probe: nothing scheduled) exits green without an
    upload. Guards the 2026-08 incident: a scoreboard 403 block read as
    "offseason", ~160 straight runs stayed green, and serving kept a 20-day-old
    artifact.
    """
    artifact = refresh_upcoming_week_cache(force=True)
    if artifact is None:
        raise SystemExit("[upcoming_week] FATAL: builder produced no artifact")
    if artifact.get("available") is False:
        reason = artifact.get("reason", "unknown")
        if reason == "offseason":
            print("[upcoming_week] verified offseason (nothing scheduled); S3 artifact left as-is")
            return
        raise SystemExit(
            f"[upcoming_week] FATAL: artifact unavailable (reason={reason}); nothing uploaded"
        )


if __name__ == "__main__":
    main()
