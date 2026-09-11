"""Offline historical prediction, metadata and cache generation.

The builder owns data loading and inference and uses HTTP-independent snapshot
state. Full local installs retain src.serving.core as a compatibility import;
production HTTP workers use the separate artifact reader instead.
"""

import hashlib
import io
import json
import os
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch

import src.data.roster_meta as roster_meta
import src.dst.data as dst_data
import src.dst.features as dst_features
import src.k.data as k_data
import src.k.features as k_features
from src.artifacts import serving_snapshot
from src.artifacts import snapshot_state as app_pkg
from src.artifacts.position_metadata import _ALL_POSITIONS, _ALL_TARGETS, _APPENDED_POSITIONS
from src.artifacts.serving_snapshot import CACHE_SCHEMA_VERSION as _PREDICTIONS_CACHE_SCHEMA_VERSION
from src.config import (
    CACHE_DIR,
    SCORING_HALF_PPR,
    SCORING_STANDARD,
    SEASONS,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
)
from src.contracts.serialization import (
    _EXPERT_PRED_PREFIXES,
    _MODEL_PRED_COLUMNS,
    _MODEL_PRED_PREFIXES,
    _VALID_SCORING,
    _actual_col,
    _pred_col,
    _records_to_player_rows,
    _safe_num,
)
from src.data.espn_projections import load_espn_with_gsis_id, project_espn_to_fantasy
from src.data.expert_sources import (
    load_sleeper_with_gsis_id,
    project_expert_comparison,
    project_nflcom_to_fantasy,
    score_offensive_projections,
)
from src.data.external_sources import _seasons_cache_signature
from src.data.loader import compute_fantasy_points
from src.data.nflcom_loader import load_nflcom_with_gsis_id
from src.data.providers.snapshot import live_provider_sources
from src.data.release import SEAL_NAME as DATA_RELEASE_SEAL_NAME
from src.shared.aggregate_targets import (
    DST_TARGETS,
    POSITION_TARGET_MAP,
    predictions_to_fantasy_points,
)
from src.shared.comparison_scoring import score_actual_components
from src.shared.evaluation import compute_metrics
from src.shared.model_sync import (
    refresh_sentinel_mtime,
    upload_predictions_cache_to_s3,  # noqa: F401 — legacy import compatibility; remote publication is offline
)
from src.shared.registry import INFERENCE_REGISTRY as POSITION_REGISTRY


def _compute_scoring_formats(df):
    if "fantasy_points_standard" not in df.columns:
        df["fantasy_points_standard"] = compute_fantasy_points(df, SCORING_STANDARD)
    if "fantasy_points_half_ppr" not in df.columns:
        df["fantasy_points_half_ppr"] = compute_fantasy_points(df, SCORING_HALF_PPR)


def _load_k_splits():
    """Load kicker data with features pre-computed on full dataset.

    K uses its own data pipeline because kicking stats (FG/PAT) are only
    available from 2025 onward via nflverse's weekly API, so they are
    reconstructed from play-by-play (1999+) for 2015-2025; it uses a
    cross-season split (Train 2015-2023, Val 2024, Test 2025).
    Also returns the per-kick records dataframe needed by the attention NN's
    nested kick-history builder at inference time.
    """
    k_df = k_data.load_data()
    k_df = POSITION_REGISTRY["K"]["compute_targets_fn"](k_df)
    k_features.compute_features(k_df)
    kicks_df = k_data.load_kicks(k_df)
    train, val, test = k_data.season_split(k_df)
    return train, val, test, kicks_df


def _load_dst_splits():
    """Load D/ST data with features pre-computed on full dataset.

    D/ST operates at team level (not player level), built from schedule
    scores and opponent offensive stats.
    """
    dst_df = dst_data.build_data(allow_scoring_fetch=False)
    dst_df = POSITION_REGISTRY["DST"]["compute_targets_fn"](dst_df)
    dst_features.compute_features(dst_df)
    train = dst_df[dst_df["season"].isin(TRAIN_SEASONS)].copy()
    val = dst_df[dst_df["season"].isin(VAL_SEASONS)].copy()
    test = dst_df[dst_df["season"].isin(TEST_SEASONS)].copy()
    return train, val, test


_EXPERT_KEY_COLS = ["player_id", "season", "week"]


def _empty_expert_frame(value_col: str) -> pd.DataFrame:
    return pd.DataFrame(columns=[*_EXPERT_KEY_COLS, value_col])


def _historical_expert_seasons(results: pd.DataFrame) -> list[int]:
    """Seasons with played rows that can be joined to historical expert feeds."""
    if results.empty or "season" not in results.columns:
        return []
    mask = pd.Series(True, index=results.index)
    if "fantasy_points" in results.columns:
        mask &= pd.to_numeric(results["fantasy_points"], errors="coerce").notna()
    seasons = pd.to_numeric(results.loc[mask, "season"], errors="coerce").dropna()
    allowed = {int(s) for s in TEST_SEASONS}
    if allowed:
        seasons = seasons[seasons.astype(int).isin(allowed)]
    # Return native Python ints, NOT numpy.int64. _apply_expert_predictions passes
    # this straight to load_nflcom_with_gsis_id -> nfl_source.rosters ->
    # nflreadpy.load_rosters, which validates seasons with a strict
    # ``not isinstance(season, int)`` check that REJECTS numpy integers (raising a
    # misleading "Season must be between 1920 and <year>" even for an in-range
    # year). A bare ``sorted(seasons.astype(int).unique())`` yields numpy.int64 and
    # silently killed the ENTIRE NFL.com expert column in serving (all-null ->
    # "undefined" in the Season Leaders tab). RotoWire was immune: sleeper's
    # _validate_sleeper_seasons coerces to int and player_ids() takes no season.
    return sorted(int(s) for s in seasons.astype(int).unique())


def _project_rotowire_to_fantasy(
    raw_df: pd.DataFrame | None, pos: str, scoring_format: str
) -> pd.DataFrame:
    """Project Sleeper/RotoWire raw-stat rows onto this app's fantasy-point scale."""
    value_col = "rotowire_pred_total"
    if raw_df is None or raw_df.empty or pos == "K":
        return _empty_expert_frame(value_col)
    if "position" not in raw_df.columns or "player_id" not in raw_df.columns:
        return _empty_expert_frame(value_col)

    pos_df = raw_df[(raw_df["position"] == pos) & raw_df["player_id"].notna()].copy()
    if pos_df.empty:
        return _empty_expert_frame(value_col)

    if pos in POSITION_TARGET_MAP:
        out = pos_df[_EXPERT_KEY_COLS].copy()
        out[value_col] = score_offensive_projections(pos_df, scoring_format)
        return out
    targets = list(DST_TARGETS) if pos == "DST" else []
    if not targets:
        return _empty_expert_frame(value_col)
    pred_dict = {}
    for target in targets:
        if target in pos_df.columns:
            pred_dict[target] = (
                pd.to_numeric(pos_df[target], errors="coerce").fillna(0.0).to_numpy()
            )
        else:
            pred_dict[target] = np.zeros(len(pos_df), dtype=float)

    out = pos_df[_EXPERT_KEY_COLS].copy()
    out[value_col] = predictions_to_fantasy_points(pos, pred_dict, scoring_format)
    return out


def _normalize_expert_frame(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Normalize expert projection keys for a stable left-join onto ``results``."""
    if df is None or df.empty or value_col not in df.columns:
        return _empty_expert_frame(value_col)
    cols = [*_EXPERT_KEY_COLS, value_col]
    if any(c not in df.columns for c in cols):
        return _empty_expert_frame(value_col)
    out = df[cols].copy()
    out["player_id"] = out["player_id"].astype(str)
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=["player_id", "season", "week"])
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out.drop_duplicates(_EXPERT_KEY_COLS, keep="last")


def _assign_expert_totals(
    results: pd.DataFrame, source: str, scoring_format: str, df: pd.DataFrame, value_col: str
) -> None:
    projection = _normalize_expert_frame(df, value_col)
    if projection.empty:
        return

    keys = results[_EXPERT_KEY_COLS].copy()
    keys["player_id"] = keys["player_id"].astype(str)
    keys["season"] = pd.to_numeric(keys["season"], errors="coerce")
    keys["week"] = pd.to_numeric(keys["week"], errors="coerce")
    joined = keys.merge(projection, on=_EXPERT_KEY_COLS, how="left", sort=False)
    values = pd.to_numeric(joined[value_col], errors="coerce")
    mask = values.notna().to_numpy()
    if not mask.any():
        return
    results.loc[mask, _pred_col(source, scoring_format)] = np.round(
        values.loc[mask].to_numpy(dtype=np.float64), 2
    ).astype(np.float32)


@live_provider_sources()
def _apply_expert_predictions(
    results: pd.DataFrame,
    *,
    nflcom_loader=None,
    rotowire_loader=None,
    espn_loader=None,
) -> None:
    """Add optional per-player expert projections to the serving results frame.

    Expert feeds are an auxiliary UI comparison surface. Loader/projection failures
    leave stable NaN columns instead of breaking model serving.
    """
    for source in _EXPERT_PRED_PREFIXES:
        results.attrs[f"{source}_complete"] = True
    for source in _EXPERT_PRED_PREFIXES:
        for fmt in _VALID_SCORING:
            results[_pred_col(source, fmt)] = np.nan
            results[_pred_col(f"{source}_comparison", fmt)] = np.nan
        results[f"{source}_pred"] = np.nan

    seasons = _historical_expert_seasons(results)
    if not seasons:
        return
    if nflcom_loader is None:
        nflcom_loader = load_nflcom_with_gsis_id
    if rotowire_loader is None:
        rotowire_loader = load_sleeper_with_gsis_id
    if espn_loader is None:
        espn_loader = load_espn_with_gsis_id

    raw_nflcom = None
    try:
        raw_nflcom = nflcom_loader(seasons=seasons)
    except Exception as e:  # noqa: BLE001 - expert data is optional in serving
        print(f"[experts] NFL.com projections unavailable: {e!r}")
    if raw_nflcom is not None and (raw_nflcom.empty or "position" not in raw_nflcom.columns):
        raw_nflcom = None

    raw_rotowire = None
    try:
        raw_rotowire = rotowire_loader(seasons)
    except Exception as e:  # noqa: BLE001 - expert data is optional in serving
        print(f"[experts] RotoWire projections unavailable: {e!r}")
    if raw_rotowire is not None and (raw_rotowire.empty or "position" not in raw_rotowire.columns):
        raw_rotowire = None

    raw_espn = None
    try:
        raw_espn = espn_loader(seasons)
    except Exception as e:  # noqa: BLE001 - expert data is optional in serving
        print(f"[experts] ESPN projections unavailable: {e!r}")
    if raw_espn is not None and (raw_espn.empty or "position" not in raw_espn.columns):
        raw_espn = None
    results.attrs["espn_complete"] = raw_espn is not None
    results.attrs["nflcom_complete"] = (
        raw_nflcom is not None and raw_nflcom.attrs.get("nflcom_fetch_complete_v1") is not False
    )
    results.attrs["rotowire_complete"] = (
        raw_rotowire is not None
        and raw_rotowire.attrs.get("sleeper_fetch_complete_v1") is not False
    )

    for source, raw in (("rotowire", raw_rotowire), ("espn", raw_espn)):
        results[_pred_col(source, "comparison")] = np.nan
        if raw is not None and {*_EXPERT_KEY_COLS, "position"}.issubset(raw.columns):
            dst = raw.loc[raw["position"].eq("DST") & raw["player_id"].notna()]
            scored = dst[_EXPERT_KEY_COLS].copy()
            scored["comparison_total"] = score_actual_components(dst, "DST")
            _assign_expert_totals(results, source, "comparison", scored, "comparison_total")

    for fmt in _VALID_SCORING:
        for pos in _ALL_POSITIONS:
            for source, raw in (
                ("nflcom", raw_nflcom),
                ("rotowire", raw_rotowire),
                ("espn", raw_espn),
            ):
                if raw is not None:
                    try:
                        shared = project_expert_comparison(raw, pos, fmt, source=source)
                        _assign_expert_totals(
                            results, f"{source}_comparison", fmt, shared, "expert_pred_total"
                        )
                    except Exception as e:  # noqa: BLE001 - optional source boundary
                        print(f"[experts] {source} {pos}/{fmt} comparison unavailable: {e!r}")
                        results.attrs[f"{source}_complete"] = False
            if raw_espn is not None:
                try:
                    espn = project_espn_to_fantasy(raw_espn, pos, fmt)
                    _assign_expert_totals(results, "espn", fmt, espn, "espn_pred_total")
                except Exception as e:  # noqa: BLE001 - one source/position can degrade
                    print(f"[experts] ESPN {pos}/{fmt} projection failed: {e!r}")
                    results.attrs["espn_complete"] = False
            if raw_nflcom is not None and pos != "DST":
                try:
                    nfl = project_nflcom_to_fantasy(raw_nflcom, pos, fmt)
                    _assign_expert_totals(results, "nflcom", fmt, nfl, "nflcom_pred_total")
                except Exception as e:  # noqa: BLE001 - one source/position can degrade
                    print(f"[experts] NFL.com {pos}/{fmt} projection failed: {e!r}")
                    results.attrs["nflcom_complete"] = False
            if raw_rotowire is not None and pos != "K":
                try:
                    rw = _project_rotowire_to_fantasy(raw_rotowire, pos, fmt)
                    _assign_expert_totals(results, "rotowire", fmt, rw, "rotowire_pred_total")
                except Exception as e:  # noqa: BLE001 - one source/position can degrade
                    print(f"[experts] RotoWire {pos}/{fmt} projection failed: {e!r}")
                    results.attrs["rotowire_complete"] = False

    for source in _EXPERT_PRED_PREFIXES:
        results[f"{source}_pred"] = results[_pred_col(source, "ppr")]


def _apply_position_models(
    train, val, test, pos, results, *, kick_history=None, opponent_weekly=None
):
    """Compatibility writer around the shared, HTTP-independent predictor."""
    from src.prediction.frames import predict_position
    from src.shared.comparison_scoring import ACTUAL_BASIS, EXCLUDED_SOURCES, comparison_actuals

    # Select a registry-owned name before deriving any artifact filename.
    # A request may choose a supported position; its string must never become
    # the filesystem path supplied to a pickle-based model/scaler loader.
    for configured_position in _ALL_POSITIONS:
        if pos == configured_position:
            canonical_position = configured_position
            break
    else:
        raise ValueError(f"Unknown model position: {pos!r}")
    pos = canonical_position
    reg = POSITION_REGISTRY[pos]
    if kick_history is None and pos == "K":
        kick_history = app_pkg._cache.get("k_kicks_df")
    if (
        opponent_weekly is None
        and reg.get("opp_attn_kind") == "offense"
        and reg.get("opp_attn_history_stats")
    ):
        opponent_weekly = pd.read_parquet(f"{CACHE_DIR}/weekly_{SEASONS[0]}_{SEASONS[-1]}.parquet")
        if "season_type" in opponent_weekly:
            opponent_weekly = opponent_weekly[opponent_weekly["season_type"].eq("REG")].copy()
    prediction = predict_position(
        pos,
        train,
        val,
        test,
        reg,
        kicks=kick_history,
        opponent_weekly=opponent_weekly,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    # Metadata must describe the exact descriptor used for these predictions.
    # A refresh can advance the mutable model directory after inference; never
    # substitute that newer generation's architecture under the recorded ID.
    from src.prediction.bundle import read_bundle

    model_metadata = {}
    for family, bundle_id in prediction.bundle_ids.items():
        try:
            descriptor = read_bundle(reg["model_dir"], family, verify=False)
            if descriptor is None or descriptor.bundle_id != bundle_id:
                raise ValueError("Prediction descriptor generation is no longer available")
            document = descriptor.to_dict()
            model_metadata[family] = {
                "bundle_id": bundle_id,
                "status": "available",
                **{
                    key: document.get(key)
                    for key in ("inputs", "architecture", "training_options", "provenance")
                },
            }
        except (OSError, ValueError, KeyError) as error:
            model_metadata[family] = {
                "bundle_id": bundle_id,
                "status": "unavailable",
                "reason": str(error),
            }
    index = prediction.frame.index
    with app_pkg._results_write_lock:
        errors = app_pkg._cache.setdefault("position_load_errors", {})
        for key in list(errors):
            if key == pos or key.startswith(f"{pos}_"):
                errors.pop(key)
        errors.update(prediction.errors)
        for scoring in _VALID_SCORING:
            results.loc[index, f"comparison_actual_{scoring}"] = comparison_actuals(
                prediction.frame, pos, scoring
            ).to_numpy(dtype=float)
        results.loc[index, "comparison_actual_basis"] = ACTUAL_BASIS
        results.loc[index, "comparison_excluded_sources"] = ",".join(EXCLUDED_SOURCES.get(pos, {}))
        for family in _MODEL_PRED_PREFIXES:
            totals = prediction.totals.get(family)
            for scoring in _VALID_SCORING:
                results.loc[index, _pred_col(family, scoring)] = (
                    np.round(totals[scoring], 2).astype(np.float32)
                    if totals is not None
                    else np.nan
                )
            results.loc[index, f"{family}_pred"] = (
                np.round(totals["ppr"], 2).astype(np.float32) if totals is not None else np.nan
            )
            raw = prediction.raw.get(family, {})
            for scoring in _VALID_SCORING:
                results.loc[index, _pred_col(f"{family}_comparison", scoring)] = (
                    score_actual_components(pd.DataFrame(raw), pos, scoring).to_numpy()
                    if raw
                    else np.nan
                )
            if pos == "DST":
                results.loc[index, _pred_col(family, "comparison")] = (
                    np.round(score_actual_components(pd.DataFrame(raw), "DST").to_numpy(), 2)
                    if raw
                    else np.nan
                )
            for target in reg["targets"]:
                results.loc[index, f"pred_{family}_{target}"] = (
                    np.round(np.asarray(raw[target], dtype=np.float64), 2).astype(np.float32)
                    if target in raw
                    else np.nan
                )
        for target in reg["targets"]:
            if target in prediction.frame:
                results.loc[index, f"actual_{target}"] = prediction.frame[target].to_numpy(
                    dtype=np.float32
                )
        app_pkg._cache.setdefault("position_details", {})[pos] = prediction.details
        app_pkg._cache.setdefault("model_bundle_ids", {})[pos] = prediction.bundle_ids
        app_pkg._cache.setdefault("model_metadata", {})[pos] = model_metadata
    for key, error in prediction.errors.items():
        print(f"[prediction] {key}: {error}")


def _artifact_only():
    return not app_pkg.allows_runtime_inference()


def _ensure_base_data():
    """Load splits + build empty results frame. Idempotent. No model loads."""
    _discard_invalidated_generation()
    if app_pkg.current_snapshot() is not None:
        return
    if _artifact_only():
        _ensure_metrics()
        return
    if app_pkg._cache.get("base_loaded"):
        return
    with app_pkg._cache_lock:
        # Re-check under lock: another thread may have populated between our
        # fast-path check and lock acquisition.
        if app_pkg._cache.get("base_loaded") or "results" in app_pkg._cache:
            return
        try:
            app_pkg._cache["prediction_inputs_fingerprint"] = _compute_models_fingerprint()[0]
            _load_base_data_locked()
        except Exception:
            # Shared split/K/DST initialization precedes per-position error
            # bookkeeping. Preserve the failure state while the original
            # exception propagates to request/prewarm diagnostics.
            app_pkg._cache["base_load_error"] = "Shared data initialization failed"
            raise
        app_pkg._cache.pop("base_load_error", None)


def _load_reg(path):
    """Read a split parquet, filtering to REG-season rows. Module-level so the
    boot path (``_load_base_data_locked``) and the hydrated-container refresh
    path (``_load_splits_locked``) share one definition of a loaded split."""
    df = pd.read_parquet(path)
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()
    return df


def _load_base_splits():
    """Load the skill train/val/test splits (REG-filtered + scoring formats).
    Shared by the boot path and the hydrated-container refresh path so the two
    can't drift (training/inference-path parity, AGENTS.md)."""
    train = _load_reg("data/splits/train.parquet")
    val = _load_reg("data/splits/val.parquet")
    test = _load_reg("data/splits/test.parquet")
    for df in [train, val, test]:
        _compute_scoring_formats(df)
    return train, val, test


def _build_splits_dict(train, val, test, k_split, dst_split):
    """Assemble the per-position cache ``splits`` dict. QB/RB/WR/TE share the
    skill (train, val, test); K and DST pass their own authoritative tuples."""
    return {
        "QB": (train, val, test),
        "RB": (train, val, test),
        "WR": (train, val, test),
        "TE": (train, val, test),
        "K": k_split,
        "DST": dst_split,
    }


def _load_base_data_locked():
    print("Loading data...")

    train, val, test = _load_base_splits()

    print("Loading kicker data...")
    k_train, k_val, k_test, k_kicks_df = _load_k_splits()
    print("Loading D/ST data...")
    dst_train, dst_val, dst_test = _load_dst_splits()

    keep_cols = [
        "player_id",
        "player_display_name",
        "position",
        "recent_team",
        "season",
        "week",
        "headshot_url",
        "fantasy_points",
        "fantasy_points_half_ppr",
        "fantasy_points_standard",
    ]
    keep_cols = [c for c in keep_cols if c in test.columns]
    # K/DST arrive via their authoritative splits in the append loop below; the
    # skill ``test.parquet`` ALSO carries kicker player-weeks (~0 offensive
    # fantasy_points), so copying them here would double every kicker once the K
    # split is appended — a phantom twin with actual≈0 and null preds (the model
    # writes preds only to the appended rows' index). Exclude the separately-
    # appended positions from the base copy. (DST players aren't in test.parquet
    # today; that half of the guard is defensive.)
    results = test.loc[~test["position"].isin(_APPENDED_POSITIONS), keep_cols].copy()

    # K/DST test frames need their index aligned to ``results``' offset so the
    # per-position writes in ``_apply_position_models`` land on the right rows.
    # ``.copy()`` first — mutating ``.index`` in place would persist into the
    # cached splits dict in a way that surprises any future caller expecting
    # the original index; explicit reindexed copies make the contract local.
    k_test_reindexed = None
    dst_test_reindexed = None
    for pos_label, pos_test_df in (("k", k_test), ("dst", dst_test)):
        # ``results.index.max()`` is NaN on an empty frame → ``range(nan, ...)``
        # raises TypeError; guard the cold-boot / empty-test-parquet case (#351 F19).
        offset = (results.index.max() + 1) if len(results) else 0
        pos_rows = pd.DataFrame(index=range(offset, offset + len(pos_test_df)))
        for col in keep_cols:
            if col in pos_test_df.columns:
                pos_rows[col] = pos_test_df[col].values
            elif col in ("fantasy_points_half_ppr", "fantasy_points_standard"):
                # NOT a fabricated value: K and DST scoring is format-invariant.
                # The three scoring dicts (SCORING_STANDARD/HALF_PPR/PPR) differ
                # ONLY in the ``receptions`` weight (0.0 / 0.5 / 1.0), and neither
                # K (sign-vector FG/PAT sum) nor DST (linear stats + tier-mapped
                # PA/YA bonuses) has a reception term — so their standard,
                # half-PPR, and PPR totals are identically equal to ``fantasy_points``
                # (PPR). The K/DST splits carry only the unsuffixed ``fantasy_points``
                # column, so we mirror it into the suffixed columns here. This keeps
                # ``/api/predictions?scoring=half_ppr|standard`` showing the correct
                # ``actual`` for K/DST players (via _records_to_player_rows ->
                # _actual_col) instead of null. (For QB/RB/WR/TE the suffixed
                # columns come straight from the split via the first branch.)
                pos_rows[col] = pos_test_df["fantasy_points"].values
            elif col == "headshot_url":
                pos_rows[col] = ""
            else:
                pos_rows[col] = np.nan
        pos_test_df = pos_test_df.copy()
        pos_test_df.index = pos_rows.index
        if pos_label == "k":
            k_test_reindexed = pos_test_df
        else:
            dst_test_reindexed = pos_test_df
        results = pd.concat([results, pos_rows])
    # Replace local refs so cached splits below carry the reindexed copies, not
    # the original frames (whose index would no longer match the results rows).
    k_test = k_test_reindexed
    dst_test = dst_test_reindexed

    # Age at kickoff + rookie flag from the synced rosters/schedules caches
    # (best-effort: unavailable caches leave the columns NaN and the frontend
    # hides the Age/Rookies filters). Feeds /api/predictions rows AND the
    # snapshot.json artifact via serialization._records_to_player_rows.
    results = roster_meta.attach_age_and_rookie(results)

    # Initialize ALL per-model, per-format prediction columns to NaN. A failed
    # or never-loaded model must leave its pred column NaN so the row is
    # excluded from overall MAE in _compute_metrics_locked and the frontend
    # renders "--". Previously ridge/nn defaulted to 0.0 while attn_nn/lgbm
    # defaulted to NaN — inconsistent failure semantics that let a failed ridge
    # or nn load silently serve 0.0 as if it were a real prediction (a 0.0 ridge
    # pred is indistinguishable from a genuine low projection and skews MAE).
    # On success _apply_position_models overwrites every row for the position;
    # on a per-model failure it NaN's that model's column explicitly — so NaN is
    # the correct "no result" sentinel for all four models uniformly.
    for fmt in _VALID_SCORING:
        results[_pred_col("ridge", fmt)] = np.nan
        results[_pred_col("nn", fmt)] = np.nan
        results[_pred_col("attn_nn", fmt)] = np.nan
        results[_pred_col("lgbm", fmt)] = np.nan
    # Legacy unsuffixed columns kept as PPR-only compatibility aliases. New
    # endpoint code must read the scoring-suffixed columns via _pred_col().
    results["ridge_pred"] = np.nan
    results["nn_pred"] = np.nan
    results["attn_nn_pred"] = np.nan
    results["lgbm_pred"] = np.nan

    # Per-target raw-stat columns for the predictions-tab breakdown drill-down
    # (/api/predictions/breakdown). One actual_{t} per target plus pred_{model}_{t}
    # per model. Raw stats are scoring-format-invariant, so a single set suffices
    # (not one per format). Sparse — each row only fills its own position's
    # targets, the rest stay NaN — but the schema is uniform across rows so the
    # parquet persist/hydrate round-trip is stable. Populated per position in
    # _apply_position_models; absent columns are tolerated by the endpoint (a
    # stale on-disk snapshot may predate this schema). Added in one concat block
    # (~95 columns) to avoid the BlockManager fragmentation a per-column insert
    # loop would cause at this width.
    per_target_cols = [f"actual_{t}" for t in _ALL_TARGETS] + [
        f"pred_{prefix}_{t}" for t in _ALL_TARGETS for prefix in _MODEL_PRED_PREFIXES
    ]
    results = pd.concat(
        [
            results,
            pd.DataFrame(np.nan, index=results.index, columns=per_target_cols),
        ],
        axis=1,
    )

    _apply_expert_predictions(results)

    app_pkg._cache["splits"] = _build_splits_dict(
        train, val, test, (k_train, k_val, k_test), (dst_train, dst_val, dst_test)
    )
    # K's attention NN needs the raw per-kick records to build nested history
    # at inference — stash here so _apply_position_models can reach it.
    app_pkg._cache["k_kicks_df"] = k_kicks_df
    app_pkg._cache["results"] = results
    app_pkg._cache["positions_loaded"] = set()
    app_pkg._cache["base_loaded"] = True


def _refresh_k_data_locked():
    """Re-derive K's split + per-kick records from current on-disk data.

    Called from ``_ensure_position_loaded``'s in-flight-refresh branch when K's
    sentinel advances. Caller must hold ``_cache_lock``. Idempotent and
    best-effort: a reload failure leaves the boot-cached K data in place (better
    a slightly-stale tensor than a crashed refresh) and surfaces via the normal
    ``_apply_position_models`` error path on the upcoming load.

    The fresh ``k_test`` is reindexed onto the EXISTING K-row index in
    ``_cache["results"]`` so ``_apply_position_models``' ``pos_index`` write still
    lands on the right rows. If the fresh test frame's row count diverges from
    the cached K rows (e.g. a mid-season PBP sync added kicker-weeks — rare; the
    model-tarball poller and the splits refresh are separate paths), we refresh
    only ``k_kicks_df`` (the nested-history source the finding is about) and keep
    the existing split frame, because rebuilding the results row layout for one
    position mid-flight would misalign every other position's cached rows.
    """
    if "results" not in app_pkg._cache or "splits" not in app_pkg._cache:
        return
    try:
        k_train, k_val, k_test, k_kicks_df = _load_k_splits()
    except Exception as e:  # noqa: BLE001 — refresh is best-effort
        print(f"[K] data refresh failed: {e!r} — reusing boot-cached k_kicks_df/splits")
        return
    # Refreshing the per-kick records is the core of the fix: the new model's
    # nested-history tensor must be built from the data it was trained against.
    app_pkg._cache["k_kicks_df"] = k_kicks_df
    results = app_pkg._cache["results"]
    existing_k_index = results.index[results["position"] == "K"]
    if len(existing_k_index) == len(k_test):
        k_test = k_test.copy()
        k_test.index = existing_k_index
        app_pkg._cache["splits"]["K"] = (k_train, k_val, k_test)
    else:
        # Row-count drift — keep the existing (correctly-indexed) split frame so
        # the per-position write stays aligned; only the kicks_df is refreshed.
        print(
            f"[K] refreshed kicks_df but split row count changed "
            f"({len(existing_k_index)} cached K rows vs {len(k_test)} fresh) — "
            f"keeping existing split index to preserve results-row alignment"
        )


def _load_splits_locked(results):
    """Populate ``_cache["splits"]`` + ``_cache["k_kicks_df"]`` WITHOUT rebuilding
    ``_cache["results"]``. Caller must hold ``_cache_lock``.

    ``_try_hydrate_from_disk`` restores ``results`` + metrics from the on-disk
    cache and sets ``base_loaded=True`` but deliberately skips the heavy
    ``_load_base_data_locked`` — so a hydrated container has NO ``splits``. The
    first in-flight refresh on such a container then needs the per-position
    splits to re-apply a model; without them ``_ensure_position_loaded`` marked
    every position failed (#550) and ``_ensure_all_positions_loaded`` silently
    recomputed metrics over the stale hydrated preds and re-persisted them under
    a fresh fingerprint (#789). Derive splits here from current on-disk data and
    reindex the appended K/DST test frames onto the EXISTING K/DST rows of the
    hydrated ``results`` (QB/RB/WR/TE rows keep their parquet index, which
    already matches ``results``) — the same alignment contract as
    ``_refresh_k_data_locked``. Best-effort: a load failure leaves ``splits``
    unset and the caller marks the position failed.

    Mirrors ``_load_base_data_locked``'s split-loading + ``_refresh_k_data_locked``'s
    reindex; keep them in sync (training/inference-path drift, see AGENTS.md).
    """
    try:
        train, val, test = _load_base_splits()
        k_train, k_val, k_test, k_kicks_df = _load_k_splits()
        dst_train, dst_val, dst_test = _load_dst_splits()
    except Exception as e:  # noqa: BLE001 — best-effort; caller marks failed
        print(f"[app] _load_splits_locked failed: {e!r} — splits unavailable")
        return
    reindexed = {"K": k_test, "DST": dst_test}
    for pos_label in ("K", "DST"):
        pos_test = reindexed[pos_label]
        existing_index = results.index[results["position"] == pos_label]
        if len(existing_index) == len(pos_test):
            pos_test = pos_test.copy()
            pos_test.index = existing_index
            reindexed[pos_label] = pos_test
        else:
            # Row counts diverged, so the fresh frame's index can't be aligned
            # to the shared results frame; writing preds against its range index
            # would CREATE ghost rows in results (a ``.loc`` insert), not skip
            # them. Mark the position failed instead so _apply_position_models
            # never runs for it and /health surfaces the degradation (#1000).
            print(
                f"[{pos_label}] _load_splits_locked: row-count mismatch "
                f"({len(existing_index)} cached rows vs {len(pos_test)} fresh) — "
                f"marking {pos_label} failed (cannot align pred writes)"
            )
            app_pkg._cache.setdefault("positions_failed", set()).add(pos_label)
            app_pkg._cache.setdefault("positions_failed_mtime", {})[pos_label] = (
                refresh_sentinel_mtime(pos_label)
            )
            app_pkg._cache.setdefault("position_load_errors", {})[pos_label] = (
                f"{pos_label} split row-count mismatch on hydrated container "
                f"({len(existing_index)} cached vs {len(pos_test)} fresh)"
            )
    app_pkg._cache["splits"] = _build_splits_dict(
        train,
        val,
        test,
        (k_train, k_val, reindexed["K"]),
        (dst_train, dst_val, reindexed["DST"]),
    )
    app_pkg._cache["k_kicks_df"] = k_kicks_df


def _refresh_dst_data_locked():
    """Re-derive DST's split from current on-disk data. Mirror of
    ``_refresh_k_data_locked`` (#441).

    DST, like K, builds its split from LIVE raw data (``dst_data.build_data`` via
    ``_load_dst_splits``) rather than the static ``data/splits/*.parquet`` — so
    its boot-cached split goes stale when a refresh swaps in a DST model trained
    on freshly-synced data. (QB/RB/WR/TE read the static parquets, so a model
    swap alone leaves their cached splits valid.) Caller holds ``_cache_lock``.
    Best-effort: a reload failure keeps the boot-cached split.

    DST has no separate per-kick records, so only the split frame is refreshed;
    the fresh ``dst_test`` is reindexed onto the EXISTING DST rows in
    ``_cache["results"]`` so ``_apply_position_models``' ``pos_index`` write
    stays aligned. A row-count change keeps the existing split index.
    """
    if app_pkg._cache.get("results") is None or "splits" not in app_pkg._cache:
        return
    try:
        dst_train, dst_val, dst_test = _load_dst_splits()
    except Exception as e:  # noqa: BLE001 — refresh is best-effort
        print(f"[DST] data refresh failed: {e!r} — reusing boot-cached splits")
        return
    results = app_pkg._cache["results"]
    existing_dst_index = results.index[results["position"] == "DST"]
    if len(existing_dst_index) == len(dst_test):
        dst_test = dst_test.copy()
        dst_test.index = existing_dst_index
        app_pkg._cache["splits"]["DST"] = (dst_train, dst_val, dst_test)
    else:
        print(
            f"[DST] refreshed split but row count changed "
            f"({len(existing_dst_index)} cached DST rows vs {len(dst_test)} fresh) — "
            f"keeping existing split index to preserve results-row alignment"
        )


def _ensure_position_loaded(pos):
    """Apply position-specific model. Idempotent, thread-safe, degrade-aware.

    ``_apply_position_models`` records per-model failures internally and
    NaN's the affected pred columns — the position still counts as "loaded"
    in that case (the DataFrame rows are there, just with some NaN preds).
    An outer ``try/except`` here catches unrecoverable setup failures
    (feature build, parquet read) and marks the position as fully failed
    so the other five still serve.

    In-flight refresh: the gunicorn refresh poller (see
    ``src.shared.model_sync.start_refresh_poller``) touches
    ``src/{pos.lower()}/outputs/.refreshed_at`` after atomically swapping in a
    new model tarball. We stat that sentinel on every call and compare its
    mtime to the value we recorded at the last successful load — when the
    sentinel advances, we invalidate this position's cache state and re-load
    from disk on the next request. Inert when the sentinel doesn't exist
    (dev, CI, before the first refresh).
    """
    _discard_invalidated_generation()
    if _artifact_only():
        _ensure_metrics()
        return
    if app_pkg.current_snapshot() is not None:
        return
    _ensure_base_data()
    sentinel_mtime = refresh_sentinel_mtime(pos)
    loaded_mtime = app_pkg._cache.get("positions_mtime", {}).get(pos, -1.0)
    failed_at = app_pkg._cache.get("positions_failed_mtime", {}).get(pos, -1.0)
    # Fast path: take the no-lock early return only when the sentinel hasn't
    # advanced beyond what we already loaded (or beyond what failed last time —
    # ``positions_failed_mtime`` records the sentinel value at the failure so
    # the next sentinel touch can retry. See the failure path below for where
    # the mtime gets stamped).
    if sentinel_mtime <= loaded_mtime and pos in app_pkg._cache.get("positions_loaded", ()):
        return
    if pos in app_pkg._cache.get("positions_failed", ()) and sentinel_mtime <= failed_at:
        # Cached hard-failure at this sentinel value — don't retry every
        # request. A subsequent sentinel advance breaks out of this branch
        # and the slow path below invalidates the failed state.
        return
    with app_pkg._cache_lock:
        if "splits" not in app_pkg._cache and app_pkg._cache.get("results") is not None:
            # Hydrated container (#550/#789): ``_try_hydrate_from_disk`` restored
            # ``results`` + metrics from the on-disk cache and set
            # ``base_loaded=True`` but skipped ``_load_base_data_locked``, so
            # there are no ``splits``. A refresh now needs them to re-apply this
            # position — derive them on demand (reindexed onto the hydrated
            # results) instead of marking the position failed.
            print(
                f"[{pos}] splits absent on hydrated container — loading splits "
                f"for in-flight refresh",
                flush=True,
            )
            _load_splits_locked(app_pkg._cache["results"])
        if "splits" not in app_pkg._cache:
            # Genuine load failure (results never built, or the on-demand split
            # load above raised). Log loudly so the failure surfaces in container
            # logs + register an error so /health and /api/predictions
            # degraded_positions reflect the state.
            err_msg = (
                f"_ensure_position_loaded({pos}) called but _cache has no "
                f"'splits' entry — base data did not populate the per-position "
                f"split index. Marking {pos} as failed."
            )
            print(f"[app] {err_msg}", flush=True)
            app_pkg._cache.setdefault("positions_failed", set()).add(pos)
            app_pkg._cache.setdefault("positions_failed_mtime", {})[pos] = sentinel_mtime
            app_pkg._cache.setdefault("position_load_errors", {})[pos] = err_msg
            return
        # Re-stat under the lock so we make the refresh decision against the
        # current filesystem state, not a possibly-stale snapshot from the
        # fast-path check.
        sentinel_mtime = refresh_sentinel_mtime(pos)
        loaded_mtime = app_pkg._cache.get("positions_mtime", {}).get(pos, -1.0)
        failed_at = app_pkg._cache.get("positions_failed_mtime", {}).get(pos, -1.0)
        loaded_advance = sentinel_mtime > loaded_mtime and loaded_mtime != -1.0
        failed_advance = (
            pos in app_pkg._cache.get("positions_failed", set()) and sentinel_mtime > failed_at
        )
        if loaded_advance or failed_advance:
            # In-flight refresh detected: drop cached state for this position
            # so the upcoming _apply_position_models writes against the new
            # on-disk model. metrics_by_format aggregates across positions so
            # it must be invalidated whenever ANY position re-loads — and the
            # persisted disk cache (predictions.parquet / metrics.json /
            # fingerprint.json) must go too, else _try_hydrate_from_disk would
            # restore the stale aggregate on the next container boot.
            app_pkg._cache.get("positions_loaded", set()).discard(pos)
            app_pkg._cache.get("positions_failed", set()).discard(pos)
            app_pkg._cache.get("positions_failed_mtime", {}).pop(pos, None)
            app_pkg._cache.get("position_load_errors", {}).pop(pos, None)
            # Also drop the stale per-target MAEs for this position. Without this,
            # if the upcoming _apply_position_models fails (slow-path exception
            # below), /api/position_details would keep serving the PREVIOUS
            # load's MAEs while /health reports the position failed and its preds
            # are NaN — a confusing inconsistency. _apply_position_models only
            # overwrites position_details[pos] on success, so we must clear it on
            # invalidation to avoid the stale-survives-failed-reload case.
            app_pkg._cache.get("position_details", {}).pop(pos, None)
            # K's nested attention builds its [N,G,K,kick_dim] history tensor from
            # the per-kick records in ``_cache["k_kicks_df"]`` (and reads its split
            # from ``_cache["splits"]["K"]``). Both were populated once at boot in
            # _load_base_data_locked and never refreshed. If a refresh swaps in a
            # K model trained on a freshly-synced PBP week, reusing the stale
            # kicks_df feeds the new model an inference tensor from the OLD data —
            # silent divergence. Re-derive both from the current on-disk data so
            # the new model sees the data distribution it was trained against.
            # DST is the same shape — its split comes from live
            # ``dst_data.build_data`` (not the static parquets) — so it needs the
            # same re-derivation (#441).
            if pos == "K":
                _refresh_k_data_locked()
            elif pos == "DST":
                _refresh_dst_data_locked()
            _invalidate_metrics_cache(reason=f"in-flight-refresh:{pos}")
            print(f"[{pos}] in-flight refresh detected (sentinel mtime advanced) — re-loading")
        if pos in app_pkg._cache.get("positions_loaded", set()):
            return
        if pos in app_pkg._cache.get("positions_failed", set()):
            return
        train, val, test = app_pkg._cache["splits"][pos]
        print(f"Applying {pos}-specific model...")
        try:
            _apply_position_models(train, val, test, pos, app_pkg._cache["results"])
        except Exception as e:
            # Anything that slips past the inner per-model try/excepts —
            # typically data-loading or feature-building failures that affect
            # the whole position. Record the sentinel value at failure under
            # ``positions_failed_mtime`` so we don't spam-retry every request,
            # but a later sentinel advance (a newly synced model) still
            # triggers a retry.
            traceback.print_exc()
            app_pkg._cache.setdefault("positions_failed", set()).add(pos)
            app_pkg._cache.setdefault("positions_failed_mtime", {})[pos] = sentinel_mtime
            app_pkg._cache.setdefault("position_load_errors", {})[pos] = repr(e)
            print(f"[app] {pos} fully failed: {e!r} — serving degraded")
            return
        # Stamp positions_mtime AFTER _apply succeeds so a transient failure
        # doesn't leave a misleading "loaded at sentinel X" entry. Successful
        # loads record the sentinel value they were taken against — the
        # invalidation check above uses this to detect refresh advances.
        app_pkg._cache.setdefault("positions_mtime", {})[pos] = sentinel_mtime
        app_pkg._cache["positions_loaded"].add(pos)


def _ensure_all_positions_loaded():
    """Load every position, best-effort, in parallel. A per-position failure
    records in ``positions_failed`` but does not re-raise — the remaining
    positions still get loaded. If EVERY position fails, raise a top-level
    error so gunicorn ``--preload`` aborts at boot and ECS blocks the broken
    rollout (preserves the existing fail-loud contract for the all-broken case).

    The 6 positions are loaded via a ``ThreadPoolExecutor`` because each one is
    independent: ``_apply_position_models`` writes to disjoint row-indices in
    ``_cache["results"]`` (filtered by ``pos_index``) and to per-position keys
    in ``position_load_errors``/``position_details``. Most of the per-position
    work is joblib unpickling + ``torch.load`` + numpy/BLAS, which all release
    the GIL — so threads give real wall-clock parallelism on CPU-only serving.

    Bypasses the per-position branch in ``_ensure_position_loaded`` (which
    re-acquires ``_cache_lock``) because the caller (``_ensure_metrics``)
    already holds the RLock — worker threads are different threads and would
    deadlock waiting for it. The caller's lock provides the "no concurrent
    request races on _cache" guarantee that the per-position lock provided
    on the request path; we update ``positions_loaded``/``positions_failed``
    here in the calling thread after each future completes.
    """
    _discard_invalidated_generation()
    if _artifact_only():
        _ensure_metrics()
        return
    if app_pkg.current_snapshot() is not None:
        return
    _ensure_base_data()
    if "splits" not in app_pkg._cache and app_pkg._cache.get("results") is not None:
        # Hydrated container: derive splits on demand so the aggregate rebuild
        # can re-apply advanced positions instead of silently recomputing
        # metrics over the stale hydrated preds and re-persisting them under a
        # fresh fingerprint (#789).
        _load_splits_locked(app_pkg._cache["results"])
    if "splits" not in app_pkg._cache:
        return
    splits = app_pkg._cache["splits"]
    loaded = app_pkg._cache.setdefault("positions_loaded", set())
    failed = app_pkg._cache.setdefault("positions_failed", set())
    errors = app_pkg._cache.setdefault("position_load_errors", {})
    mtimes = app_pkg._cache.setdefault("positions_mtime", {})
    failed_mtimes = app_pkg._cache.setdefault("positions_failed_mtime", {})
    details = app_pkg._cache.setdefault("position_details", {})
    # Re-check per-position sentinels against the recorded load mtimes. After a
    # successful pre-warm every position sits in ``loaded``, so the bare
    # ``p not in loaded`` filter would leave ``pending`` empty and this aggregate
    # rebuild (driven by _ensure_metrics' sentinel-advance branch) would recompute
    # metrics over STALE per-position preds — the per-position slow path in
    # _ensure_position_loaded re-applies an advanced position, but this all-
    # positions orchestrator never did. Evict any position whose on-disk sentinel
    # advanced beyond its stored value so it gets re-applied against the new model
    # below. Mirrors the per-position cleanup (drop loaded/failed/mtime/errors/
    # details; refresh K's per-kick data).
    for pos in _ALL_POSITIONS:
        # Source the stored mtime from either map: a failed position has only
        # ``failed_mtimes[pos]`` set (not ``mtimes[pos]``), so reading mtimes
        # alone left ``stored == -1.0`` and this aggregate orchestrator never
        # re-applied it after a redeploy fixed the model (the per-position path
        # already handles it via separate loaded/failed advance checks).
        stored = mtimes.get(pos, failed_mtimes.get(pos, -1.0))
        if stored != -1.0 and refresh_sentinel_mtime(pos) > stored:
            loaded.discard(pos)
            failed.discard(pos)
            mtimes.pop(pos, None)
            failed_mtimes.pop(pos, None)
            errors.pop(pos, None)
            for stale_key in [k for k in errors if k.startswith(f"{pos}_")]:
                errors.pop(stale_key, None)
            details.pop(pos, None)
            if pos == "K":
                _refresh_k_data_locked()
            elif pos == "DST":
                _refresh_dst_data_locked()
            print(f"[{pos}] sentinel advanced post-prewarm — re-applying in aggregate rebuild")
    pending = [p for p in _ALL_POSITIONS if p not in loaded and p not in failed]

    def _load_one(pos):
        # Snapshot the sentinel mtime BEFORE loading so a poller refresh that
        # races with us produces sentinel > stored on the next request, and
        # _ensure_position_loaded re-loads against the new on-disk model.
        sentinel_mtime = refresh_sentinel_mtime(pos)
        try:
            train, val, test = splits[pos]
            print(f"Applying {pos}-specific model...")
            _apply_position_models(train, val, test, pos, app_pkg._cache["results"])
            return pos, sentinel_mtime, None
        except Exception as e:
            return pos, sentinel_mtime, e

    if pending:
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            for future in [pool.submit(copy_context().run, _load_one, pos) for pos in pending]:
                pos, sentinel_mtime, err = future.result()
                if err is None:
                    loaded.add(pos)
                    mtimes[pos] = sentinel_mtime
                else:
                    traceback.print_exception(type(err), err, err.__traceback__)
                    failed.add(pos)
                    # Stamp failed_mtimes so a subsequent sentinel touch
                    # (refresh poller swapping in a new model) is detected as
                    # an advance and triggers a retry via
                    # ``_ensure_position_loaded``'s slow path. Without this,
                    # pre-warm failures stayed at ``loaded_mtime=-1.0`` forever
                    # and never retried on a refresh.
                    failed_mtimes[pos] = sentinel_mtime
                    errors[pos] = repr(err)
                    print(f"[app] {pos} fully failed: {err!r} — serving degraded")

    if failed and len(failed) == len(_ALL_POSITIONS):
        raise RuntimeError(
            f"All positions failed to load — see position_load_errors: "
            f"{list(app_pkg._cache.get('position_load_errors', {}).keys())}"
        )


def _degraded_positions() -> list[str]:
    """Positions with any recorded model-load error (fully failed OR partial
    per-model failure). Returned sorted so the frontend banner has a stable
    ordering across requests.
    """
    errs = app_pkg._cache.get("position_load_errors", {})
    if not errs:
        return []
    degraded: set[str] = set()
    # Snapshot the keys: pre-warm workers insert ``{pos}_{model}`` error keys
    # into the live dict without a lock, so iterating it directly can raise
    # "dictionary changed size during iteration" during the cold-start window.
    for key in list(errs):
        for p in _ALL_POSITIONS:
            if key == p or key.startswith(f"{p}_"):
                degraded.add(p)
                break
    return sorted(degraded)


# ---------------------------------------------------------------------------
# Predictions disk cache
# ---------------------------------------------------------------------------
#
# Offline/local inference persists complete immutable generations. The offline
# builder alone publishes the remote release after checking captured model heads.
# Artifact-only workers verify downloaded bytes without reading local model/data
# files. Explicit local inference also checks the pre/post input fingerprint.

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PREDICTIONS_CACHE_DIR = os.path.join(_REPO_ROOT, "data", "serving_cache")
_PREDICTIONS_PARQUET = "predictions.parquet"
_METRICS_JSON = "metrics.json"
_FINGERPRINT_JSON = "fingerprint.json"


# Bumped 5 -> 6 to force a one-time recompute on deploy: the prior cache was built
# with the numpy.int64 season bug above, which left every nflcom_pred* value null.
# The fix lives in serving code only, so the model fingerprint is unchanged and a
# fresh container would otherwise re-hydrate the stale null-NFL.com cache. The
# schema bump invalidates it so the corrected expert join repopulates the column.
# v8 adds ESPN historical projections to every per-row scoring format. Old
# snapshots must recompute or ESPN would remain null despite the new column.
def _cache_read_directory():
    from src.artifacts.serving_snapshot import active_directory

    return active_directory(_PREDICTIONS_CACHE_DIR)


# Browser-ready snapshot is a required member of each schema-11 generation, built
# from the same captured results as the API and verified before serving.
_SNAPSHOT_JSON = "snapshot.json"
_EXPERT_SOURCE_CACHE_PREFIXES = (
    "nflcom_projections_",
    "nflcom_projections_joined_",
    "sleeper_projections_",
    "sleeper_projections_joined_",
)
_K_PBP_SEASONS = tuple(s for s in k_data.SEASONS if s <= 2024)
_FULL_RAW_FINGERPRINT_FILES = frozenset(
    {
        f"depth_charts_v3_{_seasons_cache_signature(SEASONS)}.parquet",
        f"dst_scoring_pbp_v1_{_seasons_cache_signature(SEASONS)}.parquet",
        *(
            f"kicker_backfill_pbp_v1_{season}.parquet"
            for season in k_data.SEASONS
            if season >= 2025
        ),
    }
)
_SERVING_RAW_FINGERPRINT_FILES = frozenset(
    {
        ".release.json",
        *_FULL_RAW_FINGERPRINT_FILES,
        f"injuries_{SEASONS[0]}_{SEASONS[-1]}.parquet",
        *(
            {f"kicker_pbp_{_K_PBP_SEASONS[0]}_{_K_PBP_SEASONS[-1]}.parquet"}
            if _K_PBP_SEASONS
            else set()
        ),
        f"kicker_kicks_pbp_{k_data.SEASONS[0]}_{k_data.SEASONS[-1]}.parquet",
        f"rosters_{SEASONS[0]}_{SEASONS[-1]}.parquet",
        f"schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet",
        f"snap_counts_{SEASONS[0]}_{SEASONS[-1]}.parquet",
        f"team_stats_{SEASONS[0]}_{SEASONS[-1]}.parquet",
        f"weekly_{SEASONS[0]}_{SEASONS[-1]}.parquet",
    }
)


def _iter_fingerprint_paths():
    """Yield absolute paths whose (size, mtime) define cache validity.

    Walks each position's model dir, the base data splits, and the production
    serving raw inputs. Any change to a trained model, a split, or one of those
    raw inputs invalidates the predictions cache automatically.

    Do not fingerprint every local ``data/raw/*.parquet``. Developer checkouts
    often contain analysis-only or loader-side-effect caches (contracts,
    redzone, expert-source joins, older kicker PBP, generated team_stats, etc.)
    that the production container does not boot-sync before trying to hydrate
    the S3 prediction cache. Including them makes locally seeded caches miss in
    ECS even when the actual serving inputs are identical.
    """
    for pos in _ALL_POSITIONS:
        for name in (".manifest-etag", ".synced_model_key"):
            identity = os.path.join(_REPO_ROOT, "src", pos.lower(), "outputs", name)
            if os.path.isfile(identity):
                yield identity
        model_dir = os.path.join(_REPO_ROOT, "src", pos.lower(), "outputs", "models")
        if not os.path.isdir(model_dir):
            continue
        for dirpath, _, filenames in os.walk(model_dir):
            for fname in filenames:
                yield os.path.join(dirpath, fname)
    splits_dir = os.path.join(_REPO_ROOT, "data", "splits")
    for name in ("train.parquet", "val.parquet", "test.parquet", DATA_RELEASE_SEAL_NAME):
        path = os.path.join(splits_dir, name)
        if os.path.isfile(path):
            yield path
    raw_dir = os.path.join(_REPO_ROOT, "data", "raw")
    if os.path.isdir(raw_dir):
        for fname in sorted(os.listdir(raw_dir)):
            if fname.startswith(_EXPERT_SOURCE_CACHE_PREFIXES):
                continue
            if fname in _SERVING_RAW_FINGERPRINT_FILES:
                yield os.path.join(raw_dir, fname)


_FINGERPRINT_CONTENT_BYTES = 64 * 1024  # 64 KB head-sample per file
_FULL_CONTENT_FINGERPRINT_FILES = _FULL_RAW_FINGERPRINT_FILES | {
    ".release.json",
    DATA_RELEASE_SEAL_NAME,
}


def _compute_models_fingerprint():
    """Return (sha256_hex, files_list) over the fingerprint paths.

    Previously this combined ``(size, mtime_ns)`` per file. Mtime was the
    sensitive bit: ECS task replacement re-syncs the model dir from S3, and
    boto3's ``download_file`` stamps the local file with the *download* time,
    not the upload time on S3 — so every fresh container saw a different
    fingerprint and missed the cache on boot even when the content was
    byte-identical. Model files retain ``(size, head-bytes)`` hashing, where
    head-bytes is the first 64 KB of each file's content (every retrain rewrites
    weights at the start of the joblib pickle / torch state dict; even tiny
    config changes shift those leading bytes), and reading 64 KB per file
    keeps the boot-time fingerprint compute fast (~50ish files in the
    aggregate). Release metadata, depth charts, and historical K/DST scoring
    sources are hashed completely: later games can change without altering the
    first parquet row group or file size. In a coherent release, the release ID
    additionally identifies the complete content of every input dependency.
    """
    files = []
    paths = sorted(_iter_fingerprint_paths())
    h = hashlib.sha256()
    for path in paths:
        try:
            st = os.stat(path)
        except OSError:
            continue
        try:
            content = hashlib.sha256()
            with open(path, "rb") as f:
                if os.path.basename(path) in _FULL_CONTENT_FINGERPRINT_FILES:
                    for chunk in iter(lambda: f.read(_FINGERPRINT_CONTENT_BYTES), b""):
                        content.update(chunk)
                else:
                    content.update(f.read(_FINGERPRINT_CONTENT_BYTES))
        except OSError:
            # File disappeared between stat and open — same handling as the
            # stat OSError above (skip this entry; differing fingerprint will
            # naturally invalidate the cache).
            continue
        content_hash = content.hexdigest()
        rel = os.path.relpath(path, _REPO_ROOT)
        entry = {"path": rel, "size": st.st_size, "content_hash": content_hash}
        files.append(entry)
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(st.st_size).encode("ascii"))
        h.update(b"\x00")
        h.update(content_hash.encode("ascii"))
        h.update(b"\x00")
    return h.hexdigest(), files


def _snapshot_bytes():
    """Serialize browser data from the same completed results as API responses."""
    results = app_pkg._cache.get("results")
    if results is None:
        return None
    try:
        return json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "weeks": sorted(int(w) for w in results["week"].unique()),
                "degraded_positions": _degraded_positions(),
                "scoring": {
                    fmt: _records_to_player_rows(results, scoring=fmt) for fmt in _VALID_SCORING
                },
            }
        ).encode()
    except Exception as exc:
        print(f"[snapshot] serialization failed: {exc!r}")
        return None


def _verified_cache_bytes(generation=None):
    """Read one generation; only explicit runtime inference checks local inputs."""
    directory, files = serving_snapshot.read_generation(
        _PREDICTIONS_CACHE_DIR,
        generation,
        expected_dataset_id=os.environ.get("FF_DATA_RELEASE") if _artifact_only() else None,
    )
    fingerprint = json.loads(files[_FINGERPRINT_JSON])
    if (
        not isinstance(fingerprint, dict)
        or fingerprint.get("schema_version") != _PREDICTIONS_CACHE_SCHEMA_VERSION
    ):
        raise ValueError("Serving prediction cache schema mismatch")
    if not _artifact_only() and fingerprint.get("sha256") != _compute_models_fingerprint()[0]:
        raise ValueError("Serving prediction cache fingerprint mismatch")
    return directory, files


def _snapshot_response_bytes():
    """Return verified captured bytes, never a path reopened after verification."""
    captured = app_pkg.current_snapshot()
    generation = captured.cache.get("snapshot_generation") if captured is not None else None
    if captured is not None and generation is None:
        # A local in-memory publication need not have a disk representation.
        return _snapshot_bytes(), None
    try:
        directory, files = _verified_cache_bytes(generation)
        if serving_snapshot.is_invalidated(_PREDICTIONS_CACHE_DIR, directory.name):
            return None, None
        return files[_SNAPSHOT_JSON], directory.name
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


def _try_hydrate_from_disk():
    """Parse verified generation bytes and publish one complete owned snapshot."""
    try:
        mtimes = (
            {} if _artifact_only() else {pos: refresh_sentinel_mtime(pos) for pos in _ALL_POSITIONS}
        )
        directory, files = _verified_cache_bytes()
        stored = json.loads(files[_FINGERPRINT_JSON])
        results = pd.read_parquet(io.BytesIO(files[_PREDICTIONS_PARQUET]))
        metrics_payload = json.loads(files[_METRICS_JSON])
        metrics_by_format = metrics_payload["metrics_by_format"]
        position_details = metrics_payload.get("position_details") or {}
        position_load_errors = metrics_payload.get("position_load_errors") or {}
        if not _artifact_only() and _compute_models_fingerprint()[0] != stored.get("sha256"):
            return False
        if serving_snapshot.is_invalidated(_PREDICTIONS_CACHE_DIR, directory.name):
            return False
    except Exception as exc:
        print(f"[predcache] generation unavailable: {exc!r}")
        return False
    cache = app_pkg.current_state().cache
    cache.update(
        {
            "results": results,
            "metrics_by_format": metrics_by_format,
            "metrics": metrics_by_format.get("ppr", {}),
            "snapshot_generation": directory.name,
            "model_bundle_ids": metrics_payload.get("model_bundle_ids", {}),
            "model_metadata": metrics_payload.get("model_metadata", {}),
            "comparison_snapshot": metrics_payload.get("comparison_snapshot"),
            "positions_loaded": set(_ALL_POSITIONS)
            - {key.split("_", 1)[0] for key in position_load_errors},
            "positions_failed": set(),
            "positions_failed_mtime": {},
            "positions_mtime": mtimes,
            "base_loaded": True,
            "position_details": position_details,
            "position_load_errors": position_load_errors,
            "prediction_inputs_fingerprint": stored.get("sha256"),
        }
    )
    cache.pop("base_load_error", None)
    print(f"[predcache] hydrated generation {directory.name[:12]} (rows={len(results)})")
    if _artifact_only() or not _positions_pending():
        app_pkg.current_state().publish()
    return True


def _persist_cache_to_disk(*, required=False):
    """Publish a complete local generation only from unchanged inference inputs.

    Remote release publication belongs to the offline builder, which captured
    model heads before inference and performs the final pointer CAS.
    Required offline writes fail closed; runtime persistence remains best-effort.
    """

    def unavailable(reason):
        if required:
            raise RuntimeError(f"Required serving cache persistence failed: {reason}")
        return None

    if _artifact_only():
        return unavailable("artifact-only workers cannot construct snapshots")
    if "results" not in app_pkg._cache or "metrics_by_format" not in app_pkg._cache:
        return unavailable("results or metrics are missing")
    incomplete = [
        source
        for source in _EXPERT_PRED_PREFIXES
        if app_pkg._cache["results"].attrs.get(f"{source}_complete") is False
    ]
    if incomplete:
        reason = f"Expert results are incomplete: {', '.join(incomplete)}"
        print(f"[predcache] {reason} — not publishing incomplete results")
        return unavailable(reason)
    sha, input_files = _compute_models_fingerprint()
    expected = app_pkg._cache.get("prediction_inputs_fingerprint")
    if expected != sha or _any_position_sentinel_advanced():
        print("[predcache] inputs changed or not recorded before inference — skipping publication")
        return unavailable("inputs changed or were not recorded before inference")
    metrics_payload = {
        "metrics_by_format": app_pkg._cache["metrics_by_format"],
        "position_details": app_pkg._cache.get("position_details", {}),
        "position_load_errors": app_pkg._cache.get("position_load_errors", {}),
        "model_bundle_ids": app_pkg._cache.get("model_bundle_ids", {}),
        "model_metadata": app_pkg._cache.get("model_metadata", {}),
        "comparison_snapshot": app_pkg._cache.get("comparison_snapshot"),
    }
    try:
        parquet = io.BytesIO()
        app_pkg._cache["results"].to_parquet(parquet, index=True)
        snapshot = _snapshot_bytes()
        if snapshot is None:
            return unavailable("browser snapshot serialization failed")
        files = {
            _PREDICTIONS_PARQUET: parquet.getvalue(),
            _METRICS_JSON: json.dumps(metrics_payload).encode(),
            _FINGERPRINT_JSON: json.dumps(
                {
                    "schema_version": _PREDICTIONS_CACHE_SCHEMA_VERSION,
                    "computation_id": uuid.uuid4().hex,
                    "sha256": sha,
                    "files": input_files,
                }
            ).encode(),
            _SNAPSHOT_JSON: snapshot,
        }
        if _compute_models_fingerprint()[0] != expected or _any_position_sentinel_advanced():
            print("[predcache] inputs changed during serialization — skipping publication")
            return unavailable("inputs changed during serialization")
        directory = serving_snapshot.publish_local(_PREDICTIONS_CACHE_DIR, files)
        app_pkg.current_state().cache["snapshot_generation"] = directory.name
        print(f"[predcache] committed generation {directory.name[:12]}")
        return directory
    except Exception as exc:
        if required:
            raise RuntimeError("Required serving cache persistence failed") from exc
        print(f"[predcache] publication failed: {exc!r} — serving in-memory results")
        return None


def _ensure_metrics():
    _discard_invalidated_generation()
    if app_pkg.current_snapshot() is not None:
        return
    if _artifact_only():
        from werkzeug.exceptions import ServiceUnavailable

        with app_pkg._cache_lock:
            try:
                read_directory = _cache_read_directory()
                if (
                    app_pkg._cache.get("snapshot_generation") == read_directory.name
                    and "metrics_by_format" in app_pkg._cache
                ):
                    return
                if _try_hydrate_from_disk():
                    return
            except (OSError, ValueError, KeyError, TypeError):
                pass
        raise ServiceUnavailable("Serving snapshot is warming")
    if (
        "metrics_by_format" in app_pkg._cache
        and not _any_position_sentinel_advanced()
        and not _positions_pending()
    ):
        return
    with app_pkg._cache_lock:
        if (
            "metrics_by_format" in app_pkg._cache
            and not _any_position_sentinel_advanced()
            and not _positions_pending()
        ):
            return
        # A sentinel advanced under us — drop the cached aggregate so it
        # rebuilds against the freshly-loaded per-position predictions. The
        # per-position invalidation in ``_ensure_position_loaded`` already
        # discards ``metrics_by_format`` when one position re-loads, but
        # ``_ensure_metrics`` can also be hit *before* a per-position re-load
        # (e.g. /api/metrics with all positions still marked loaded against
        # stale mtimes), so the sentinel sweep here is the second line of
        # defense.
        sentinel_advanced = "metrics_by_format" in app_pkg._cache
        if sentinel_advanced:
            _invalidate_metrics_cache(reason="sentinel-advance")
        # Only re-hydrate from disk on a genuine cold start (no in-memory
        # aggregate to begin with). On a sentinel advance we MUST NOT hydrate:
        # A sentinel touch need not change model bytes. The revoked generation
        # remains on disk for readers that already captured it; a new complete
        # computation must get its own identity before it can be published.
        # A disk-hydrated cache can restore ``position_load_errors`` and drop the
        # errored positions from ``positions_loaded`` so they retry (#834), but
        # it never seeds ``positions_failed`` — so those positions are "pending"
        # (see ``_positions_pending``). Returning on the hydrate hit alone would
        # serve the degraded/NaN aggregate forever, because every aggregate entry
        # point funnels through here and the fast path above can never see the
        # excluded positions (``_any_position_sentinel_advanced`` iterates only
        # ``positions_loaded``). Only short-circuit when nothing is pending;
        # otherwise fall through to retry the pending positions once at hydrate
        # time and recompute the aggregate. A genuine re-failure lands the
        # position in ``positions_failed`` (stamped by
        # ``_ensure_all_positions_loaded``), so the fast path holds thereafter —
        # no retry storm. (#1442)
        if not sentinel_advanced and _try_hydrate_from_disk() and not _positions_pending():
            return
        if _artifact_only():
            from werkzeug.exceptions import ServiceUnavailable

            raise ServiceUnavailable("Serving snapshot is warming")
        app_pkg._cache["prediction_inputs_fingerprint"] = _compute_models_fingerprint()[0]
        _ensure_all_positions_loaded()
        _compute_metrics_locked()


def _any_position_sentinel_advanced() -> bool:
    """True iff any loaded position's on-disk sentinel mtime is greater than
    the value recorded when we last loaded that position. Used as the second
    line of defense for ``_ensure_metrics`` — see comment there.
    """
    stored = app_pkg._cache.get("positions_mtime", {})
    # Snapshot to a tuple: another thread can add/discard positions_loaded under
    # _cache_lock while this lock-free fast path iterates it, which would raise
    # "Set changed size during iteration". (#1014)
    loaded = tuple(app_pkg._cache.get("positions_loaded", ()))
    return any(refresh_sentinel_mtime(pos) > stored.get(pos, -1.0) for pos in loaded)


def _positions_pending() -> bool:
    """True iff some position is in neither ``positions_loaded`` nor
    ``positions_failed`` — i.e. it still needs a (re)load attempt.

    This is the second miss the ``_ensure_metrics`` fast path had to close for
    disk-hydrated degraded caches (#1442). ``_try_hydrate_from_disk`` restores
    ``position_load_errors`` and removes the errored positions from
    ``positions_loaded`` so they can retry (#834), but it does NOT seed
    ``positions_failed`` — so a hydrated errored position reads as *pending*
    here, which lets ``_ensure_metrics`` fall through to
    ``_ensure_all_positions_loaded`` and retry it once at hydrate time instead
    of serving the degraded aggregate indefinitely. After a genuine re-failure
    the position lands in ``positions_failed`` (stamped in
    ``_ensure_all_positions_loaded``), so it stops being pending and the fast
    path holds — bounded retry, no storm.

    A fully-loaded healthy container has every position in ``positions_loaded``
    and nothing pending, so this stays cheap and never forces a needless
    rebuild. When ``positions_loaded`` is absent entirely (a true cold start
    before any load), return False: the fast path can't fire anyway because
    ``metrics_by_format`` is absent, and treating that as pending would only add
    noise.
    """
    loaded = app_pkg._cache.get("positions_loaded")
    if loaded is None:
        return False
    # Snapshot to frozensets: a concurrent load under _cache_lock can mutate
    # these while this lock-free fast path reads them (mirrors the
    # "Set changed size during iteration" guard in _any_position_sentinel_advanced).
    loaded = frozenset(loaded)
    failed = frozenset(app_pkg._cache.get("positions_failed", ()))
    return any(pos not in loaded and pos not in failed for pos in _ALL_POSITIONS)


def _discard_invalidated_generation():
    """Stop using a generation another worker revoked, regardless of local mtimes."""
    owner = app_pkg.current_state()
    generation = owner.cache.get("snapshot_generation")
    if generation is None or not serving_snapshot.is_invalidated(
        _PREDICTIONS_CACHE_DIR, generation
    ):
        return
    with owner.cache_lock:
        generation = owner.cache.get("snapshot_generation")
        if generation is None or not serving_snapshot.is_invalidated(
            _PREDICTIONS_CACHE_DIR, generation
        ):
            return
        _invalidate_metrics_cache(reason="shared-generation-invalidation")
        owner.cache.pop("snapshot_generation", None)
        owner.cache.pop("prediction_inputs_fingerprint", None)
        for key in ("positions_loaded", "positions_failed"):
            owner.cache[key] = set()
        for key in (
            "positions_mtime",
            "positions_failed_mtime",
            "position_load_errors",
            "position_details",
        ):
            owner.cache[key] = {}


def _invalidate_metrics_cache(*, reason: str) -> None:
    """Revoke this exact consumed generation without deleting another reader's files."""
    owner = app_pkg.current_state()
    owner.cache.pop("metrics_by_format", None)
    owner.cache.pop("metrics", None)
    generation = owner.cache.get("snapshot_generation")
    if generation is not None:
        serving_snapshot.invalidate_generation(_PREDICTIONS_CACHE_DIR, generation)
    owner.snapshots.discard(generation)
    print(f"[predcache] invalidated generation ({reason})")


def _compute_metrics_locked():
    """Compute overall + per-position MAE/RMSE/R² for every model under each
    of the three scoring formats, caching the results under
    ``_cache["metrics_by_format"][fmt]``. Keeps ``_cache["metrics"]`` as a PPR
    alias for callers that haven't migrated.
    """
    results = app_pkg._cache["results"]
    metrics_by_format = {}
    for fmt in _VALID_SCORING:
        actual_col = _actual_col(fmt)
        if actual_col not in results.columns:
            continue
        actual_values = results[actual_col].values
        per_format = {}
        for name, prefix in _MODEL_PRED_COLUMNS:
            pred_col = _pred_col(prefix, fmt)
            if pred_col not in results.columns:
                per_format[name] = {"overall": None, "by_position": []}
                continue
            pred_series = results[pred_col]
            # Skip rows where this model has no prediction (a position whose
            # model failed to load). LightGBM and attn_nn are both trained for
            # all six positions, so K/DST rows are real, not skipped.
            available_mask = pred_series.notna().values
            if not available_mask.any():
                per_format[name] = {"overall": None, "by_position": []}
                continue
            y_avail = actual_values[available_mask]
            preds_avail = pred_series.values[available_mask]
            overall = compute_metrics(y_avail, preds_avail)
            positions_avail = results.loc[available_mask, "position"].values
            by_position = []
            for pos in _ALL_POSITIONS:
                pos_mask = positions_avail == pos
                if not pos_mask.any():
                    continue
                pm = compute_metrics(y_avail[pos_mask], preds_avail[pos_mask])
                # Round per-position metrics to 4 decimals to match the overall
                # row below — without this, by_position dicts ship full
                # double-precision floats while overall is pre-rounded.
                # _safe_num maps NaN/inf to None: compute_metrics returns
                # r2=NaN for <2-sample slices (e.g. a thin position/week), and a
                # literal NaN in the jsonified /api/metrics payload makes the
                # browser's JSON.parse reject the whole response (#363 F7).
                pm = {k: _safe_num(round(v, 4)) for k, v in pm.items()}
                pm["position"] = pos
                pm["n_samples"] = int(pos_mask.sum())
                by_position.append(pm)
            per_format[name] = {
                "overall": {k: _safe_num(round(v, 4)) for k, v in overall.items()},
                "by_position": by_position,
            }
        metrics_by_format[fmt] = per_format
    app_pkg._cache["metrics_by_format"] = metrics_by_format
    app_pkg._cache["metrics"] = metrics_by_format.get("ppr", {})
    _persist_cache_to_disk()
    app_pkg.current_state().publish()
    print("Ready!")


def _get_data(scoring="ppr"):
    """Full load: all positions + metrics for the requested scoring format."""
    _ensure_metrics()
    metrics_by_format = app_pkg._cache["metrics_by_format"]
    metrics = metrics_by_format.get(scoring) or metrics_by_format.get("ppr", {})
    return app_pkg._cache["results"], metrics


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
