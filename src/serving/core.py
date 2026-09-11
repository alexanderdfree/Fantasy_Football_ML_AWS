"""Model-loading, data-assembly, disk-cache and metrics engine for the serving app.

Extracted from ``app.py`` during the serving decomposition. Holds the heavy lifting:
per-position model load + inference (``_apply_position_models``), base/split data
loading + in-flight refresh, the predictions disk-cache (fingerprint/snapshot/
hydrate/persist), and metrics assembly (``_ensure_metrics`` / ``_get_data``).

Shared mutable state (``_cache`` + locks) stays in ``app.py`` and is reached via
``import src.serving.app as app_pkg`` (call-time attribute access, cycle-safe). The
route handlers in ``app.py`` call these functions as ``core.<fn>``.
"""

import hashlib
import io
import json
import os
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import torch

import src.dst.data as dst_data
import src.dst.features as dst_features
import src.k.data as k_data
import src.k.features as k_features
import src.serving.app as app_pkg
import src.serving.roster_meta as roster_meta
from src.config import (
    CACHE_DIR,
    MIN_GAMES_PER_SEASON,
    SCORING_HALF_PPR,
    SCORING_STANDARD,
    SEASONS,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
)
from src.data.external_sources import _seasons_cache_signature
from src.data.loader import compute_fantasy_points
from src.data.nflcom_loader import load_nflcom_with_gsis_id
from src.data.release import SEAL_NAME as DATA_RELEASE_SEAL_NAME
from src.features.engineer import (
    OPP_ATTN_PER_GAME_BUILDERS,
    build_game_history_arrays,
    build_opp_defense_history_arrays,
    get_attn_static_columns,
)
from src.serving.espn_projections import load_espn_with_gsis_id, project_espn_to_fantasy
from src.serving.expert_sources import (
    load_sleeper_with_gsis_id,
    project_expert_comparison,
    project_nflcom_to_fantasy,
    score_offensive_projections,
)
from src.serving.metadata import _ALL_POSITIONS, _ALL_TARGETS, _APPENDED_POSITIONS
from src.serving.serialization import (
    _EXPERT_PRED_PREFIXES,
    _MODEL_PRED_COLUMNS,
    _MODEL_PRED_PREFIXES,
    _VALID_SCORING,
    _actual_col,
    _pred_col,
    _records_to_player_rows,
    _safe_num,
)
from src.shared import prediction_cache
from src.shared.aggregate_targets import (
    DST_TARGETS,
    POSITION_TARGET_MAP,
    TARGET_UNITS,
    predictions_to_fantasy_points,
)
from src.shared.artifact_integrity import (
    assert_scaler_matches,
    read_scaler_meta,
    unwrap_state_dict,
)
from src.shared.comparison_scoring import score_actual_components
from src.shared.evaluation import compute_metrics
from src.shared.feature_build import build_position_features, scale_and_clip
from src.shared.model_sync import (
    refresh_sentinel_mtime,
    upload_predictions_cache_to_s3,
)
from src.shared.models import LightGBMMultiTarget, RidgeMultiTarget
from src.shared.neural_net import (
    MultiHeadNet,
    MultiHeadNetWithHistory,
    MultiHeadNetWithNestedHistory,
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
    results.attrs["espn_complete"] = True
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
            if raw_rotowire is not None and pos != "K":
                try:
                    rw = _project_rotowire_to_fantasy(raw_rotowire, pos, fmt)
                    _assign_expert_totals(results, "rotowire", fmt, rw, "rotowire_pred_total")
                except Exception as e:  # noqa: BLE001 - one source/position can degrade
                    print(f"[experts] RotoWire {pos}/{fmt} projection failed: {e!r}")

    for source in _EXPERT_PRED_PREFIXES:
        results[f"{source}_pred"] = results[_pred_col(source, "ppr")]


def _apply_position_models(
    train, val, test, pos, results, *, kick_history=None, opponent_weekly=None
):
    """Load pre-trained position-specific models and write predictions into
    results. Graceful per-model degradation: a single model's load failure is
    recorded in ``_cache["position_load_errors"]`` and the corresponding
    pred column is NaN'd for this position's rows, but other models still
    load and the caller continues with the remaining five positions.

    Setup failures (feature build, data filter) are unrecoverable and
    propagate — they usually mean base data is missing, which would break
    every position. ``_ensure_position_loaded`` catches these and records
    the position as fully failed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg = POSITION_REGISTRY[pos]

    targets = reg["targets"]
    model_dir = reg["model_dir"]

    # Prepare position data
    pos_train = reg["filter_fn"](train)
    pos_val = reg["filter_fn"](val)
    pos_test = reg["filter_fn"](test)

    if pos not in ("K", "DST"):
        pos_train = reg["compute_targets_fn"](pos_train)
        pos_val = reg["compute_targets_fn"](pos_val)
        pos_test = reg["compute_targets_fn"](pos_test)

    # Mirror the training-time min-games filter
    # (``src/shared/pipeline.py::_prepare_position_data_uncached``): training
    # drops low-volume player-seasons from ``pos_train`` BEFORE computing the
    # ``fill_nans`` train-means and fitting the StandardScaler. Serving must
    # replicate the exact same train frame or those imputation means + scaler
    # stats drift from what the loaded models were trained on (audit #569).
    # ``val``/``test`` stay unfiltered, exactly as in training.
    min_games = reg.get("min_games_per_season")
    if min_games is None:
        min_games = MIN_GAMES_PER_SEASON
    # Capture the pre-filter train: RB/WR compute their per-game team-total /
    # share / HHI / career features over the FULL player set (so dropped
    # low-volume teammates don't undercount the denominators) and return only
    # the filtered rows. fill_nans + the StandardScaler still fit on the
    # filtered train inside build_position_features (#569). Mirrors training
    # (src/shared/pipeline.py) — keep all three paths identical (#574/#531).
    full_train = pos_train
    games_per_season = pos_train.groupby(["player_id", "season"])["week"].transform("count")
    pos_train = pos_train[games_per_season >= min_games].copy()

    # Live reports can be unavailable for teams that have not practiced yet.
    # Neutralize those rows using the same filtered train population as the
    # scaler, rather than turning missing information into a full-practice flag.
    if "_practice_status_missing" in pos_test:
        unknown = pos_test["_practice_status_missing"].eq(True)
        if unknown.any():
            mean_practice = pos_train["practice_status"].mean()
            if pd.isna(mean_practice):
                raise ValueError("Training data has no practice status for live imputation")
            pos_test.loc[unknown, "practice_status"] = mean_practice

    feature_cols = reg["get_feature_columns_fn"]()
    pos_train, pos_val, pos_test = build_position_features(
        pos_train, pos_val, pos_test, reg, feature_cols, full_train=full_train
    )

    # No position currently uses a post-hoc adjustment: K encodes miss penalties
    # as signed raw-value heads (see ``target_signs``) and QB/RB/WR/TE/DST all
    # aggregate raw-stat preds via ``predictions_to_fantasy_points``. The
    # ``compute_adjustment_fn`` slot is plumbed through the registry (set
    # explicitly to ``None`` for K + DST in ``src/shared/registry.py``) so a
    # future position that needs one can opt in without touching this code.
    adj_values = None
    if reg.get("compute_adjustment_fn") is not None:
        adj = reg["compute_adjustment_fn"](pos_test)
        adj_values = adj.values
    # ``target_signs`` is set only for K — it acts as the dispatch discriminator
    # for ``_combine_total`` below. (The registry's ``aggregate_fn`` slot is not
    # consumed here; it is intentionally retained as a ``None`` compatibility
    # stub in registry.py — no separate cleanup is pending.)
    target_signs = reg.get("target_signs")

    X_test_pos = pos_test[feature_cols].values.astype(np.float32)
    pos_index = pos_test.index
    # ``position_load_errors`` is shared across the parallel pre-warm workers
    # (one thread per position in ``_ensure_all_positions_loaded``). The
    # get-or-create (``setdefault``) and the iterate-then-pop clear loop below
    # are read-modify-write sequences that can interleave between workers, so
    # hold ``_results_write_lock`` for them — the same lock that already guards
    # the shared ``results`` DataFrame writes lower down. (Per-model
    # ``errors[f"{pos}_..."] = ...`` assignments to distinct ``{pos}_*`` keys are
    # individually atomic under the GIL and only the worker owning ``pos`` ever
    # writes them, so those don't need the lock — only the dict-creation and the
    # multi-step clear do.)
    with app_pkg._results_write_lock:
        errors = app_pkg._cache.setdefault("position_load_errors", {})
        # Drop stale entries for this position before re-trying. Per-model
        # failures below write ``{pos}_{model_type}`` keys (e.g. ``"QB_ridge"``);
        # without this clear, a previous attempt's keys would linger after a
        # successful refresh and keep ``/health`` reporting "degraded"
        # indefinitely — which used to latch the ALB target to 503 and trigger
        # ECS replacement (see alexfree.me, 2026-05-21 12:16 UTC). Match the
        # key-parsing rule in ``_degraded_positions``: bare ``pos`` OR ``{pos}_*``.
        for stale_key in [k for k in errors if k == pos or k.startswith(f"{pos}_")]:
            errors.pop(stale_key, None)

    def _combine_total(preds: dict, fmt: str = "ppr") -> np.ndarray:
        # K — sign-vectored sum, no reception target, format-invariant.
        if target_signs is not None:
            total = sum(preds[t] * target_signs.get(t, 1.0) for t in targets)
            if adj_values is not None:
                total = total + adj_values
            return total
        # If a future position registers ``compute_adjustment_fn`` but doesn't
        # plug into ``POSITION_TARGET_MAP``/``predictions_to_fantasy_points``
        # yet, the adjustment slot keeps working via a plain raw-stat sum.
        # No prod position hits this branch today.
        if adj_values is not None:
            total = sum(preds[t] for t in targets)
            return total + adj_values
        # QB/RB/WR/TE/DST go through predictions_to_fantasy_points which knows
        # the per-format reception weight; QB/DST values happen to be format-
        # invariant by construction (no reception target / DST tier-mapped
        # aggregator) so the three calls return identical numbers there.
        return predictions_to_fantasy_points(pos, preds, scoring_format=fmt)

    def _per_format_totals(preds):
        """Return {fmt: total_array} for a single model's preds dict."""
        if preds is None:
            return None
        return {fmt: _combine_total(preds, fmt) for fmt in _VALID_SCORING}

    # Each of the four model blocks below sets its ``*_preds`` / ``*_total``
    # locals on success. On failure, the error is recorded under
    # ``{pos}_{model_type}`` in ``position_load_errors`` and the locals stay
    # None — the results-write block at the bottom NaN's the pred column for
    # this position's rows when that happens.

    ridge_preds = None
    ridge_totals = None
    try:
        ridge = RidgeMultiTarget(target_names=targets)
        ridge.load(model_dir)
        ridge_preds = ridge.predict(X_test_pos)
        ridge_totals = _per_format_totals(ridge_preds)
    except Exception as e:
        errors[f"{pos}_ridge"] = repr(e)
        print(f"[app] {pos} ridge load failed: {e!r} — NaN'ing ridge_pred")

    # NN predictions — integrity-check scaler+weights before running inference.
    nn_preds = None
    nn_totals = None
    try:
        nn_scaler = joblib.load(f"{model_dir}/nn_scaler.pkl")
        nn_meta = read_scaler_meta(f"{model_dir}/nn_scaler_meta.json")
        nn_checkpoint = torch.load(
            f"{model_dir}/{reg['nn_file']}", map_location=device, weights_only=True
        )
        nn_state_dict, nn_hash = unwrap_state_dict(nn_checkpoint)
        assert_scaler_matches(
            pos,
            nn_scaler,
            nn_hash,
            nn_meta,
            feature_cols,
            targets,
            scaler_label="nn_scaler",
        )

        X_test_scaled = scale_and_clip(nn_scaler, X_test_pos)
        nn_model = MultiHeadNet(
            input_dim=len(feature_cols), target_names=targets, **reg["nn_kwargs"]
        ).to(device)
        nn_model.load_state_dict(nn_state_dict)
        nn_preds = nn_model.predict_numpy(X_test_scaled, device)
        nn_totals = _per_format_totals(nn_preds)
    except Exception as e:
        errors[f"{pos}_nn"] = repr(e)
        print(f"[app] {pos} nn load failed: {e!r} — NaN'ing nn_pred")

    # Attention NN — gated per-position via ``reg["train_attention_nn"]``.
    # ALL SIX positions train AND serve an attention NN today (DST landed via
    # cc0c627, K via 801b61a — see CLAUDE.md "six-position symmetry"): flat-
    # history variant for QB/RB/WR/TE/DST, nested per-kick variant for K. The
    # per-position guard is a forward-compatibility fallback — a future position
    # that left ``train_attention_nn`` False would leave the column NaN so the
    # frontend renders "--"; no prod position hits that path now.
    attn_nn_preds = None
    attn_nn_totals = None
    if reg.get("train_attention_nn", False) and reg.get("attn_nn_file"):
        try:
            # K resolves its attention static columns directly from the
            # DataFrame (they live outside the Ridge/base-NN feature list, per
            # src/shared/pipeline.py::attn_static_from_df). Others use the filtered
            # whitelist over the base feature matrix.
            if reg.get("attn_static_from_df", False):
                attn_static_cols = list(reg.get("attn_static_features", []))
                X_test_attn = pos_test[attn_static_cols].to_numpy(dtype=np.float32)
            else:
                attn_static_cols = get_attn_static_columns(
                    feature_cols, reg.get("attn_static_features", [])
                )
                attn_static_col_set = set(attn_static_cols)
                attn_col_idx = [i for i, c in enumerate(feature_cols) if c in attn_static_col_set]
                X_test_attn = X_test_pos[:, attn_col_idx]

            attn_scaler = joblib.load(f"{model_dir}/attention_nn_scaler.pkl")
            attn_meta = read_scaler_meta(f"{model_dir}/attention_nn_scaler_meta.json")
            attn_checkpoint = torch.load(
                f"{model_dir}/{reg['attn_nn_file']}",
                map_location=device,
                weights_only=True,
            )
            attn_state_dict, attn_hash = unwrap_state_dict(attn_checkpoint)
            assert_scaler_matches(
                pos,
                attn_scaler,
                attn_hash,
                attn_meta,
                attn_static_cols,
                targets,
                scaler_label="attention_nn_scaler",
            )

            X_test_attn_scaled = scale_and_clip(attn_scaler, X_test_attn)

            structure = reg.get("attn_history_structure", "flat")
            if structure == "nested":
                # K: build 4-D [N, G, K, kick_dim] history from per-kick records.
                kicks_df = (
                    kick_history if kick_history is not None else app_pkg._cache.get("k_kicks_df")
                )
                if kicks_df is None:
                    raise RuntimeError(
                        "K nested attention requires kicks_df cached by _load_k_splits"
                    )
                hist_test, outer_test, inner_test = k_features.build_nested_kick_history(
                    pos_test,
                    kicks_df=kicks_df,
                    kick_stats=reg["attn_kick_stats"],
                    max_games=reg["attn_max_games"],
                    max_kicks_per_game=reg["attn_max_kicks_per_game"],
                )
                # Optional per-game aggregate branch: keep the outer sequence
                # length aligned with the kick tensor (max_games) so the
                # downstream concat matches train-time shape.
                game_history_stats = reg.get("attn_history_stats")
                game_hist_test = None
                if game_history_stats:
                    game_hist_test, _ = build_game_history_arrays(
                        pos_test,
                        history_stats=game_history_stats,
                        max_seq_len=reg["attn_max_games"],
                    )
                attn_model = MultiHeadNetWithNestedHistory(
                    static_dim=len(attn_static_cols),
                    kick_dim=hist_test.shape[-1],
                    target_names=targets,
                    **reg["attn_nn_kwargs_static"],
                ).to(device)
                attn_model.load_state_dict(attn_state_dict)
                attn_nn_preds = attn_model.predict_numpy(
                    X_test_attn_scaled,
                    hist_test,
                    outer_test,
                    inner_test,
                    device,
                    X_game_history=game_hist_test,
                )
            else:
                # Pass the FULL attn_history_stats list — identical to the
                # training path (src.shared.pipeline builds the train/val/test
                # history with cfg["attn_history_stats"] unfiltered). A previous
                # pre-filter (``[s for s in ... if s in pos_test.columns]``) was
                # added here to dodge the KeyError that build_game_history_arrays
                # raises on a missing column (PR #328), but that filter created a
                # silent train/inference drift: dropping a column the saved model
                # was trained on shrinks game_dim, mismatching the model's
                # first-layer weight shape (best case a state_dict load error,
                # worst case columns mapped to the wrong slots → silently wrong
                # preds). CLAUDE.md "Always diff training vs inference paths".
                # The KeyError is the correct fail-loud signal that serving's
                # feature build didn't produce a column the model needs; it's
                # caught by the enclosing ``except Exception`` which records a
                # ``{pos}_attn_nn`` error and NaN's the attn_nn pred (frontend
                # renders "--") — graceful degradation, not a silent wrong number.
                hist_stats = list(reg.get("attn_history_stats", []))
                max_seq_len = reg.get("attn_max_seq_len", 17)
                hist_test, mask_test = build_game_history_arrays(
                    pos_test, history_stats=hist_stats, max_seq_len=max_seq_len
                )

                # Optional opponent-side attention branch — kind-based
                # dispatch mirrors the pipeline (src.shared.pipeline). "defense"
                # (QB/RB/WR/TE) aggregates over the all-position concat;
                # "offense" (DST) loads the raw player-week cache because
                # DST's train/val/test frames are team-level and lack the
                # offensive columns the offense aggregation needs.
                # CLAUDE.md rule: keep training and inference feature paths
                # byte-for-byte consistent.
                opp_history_stats = reg.get("opp_attn_history_stats") or []
                opp_hist_test = opp_mask_test = None
                opp_game_dim = None
                if opp_history_stats:
                    opp_max_seq_len = reg.get("opp_attn_max_seq_len", max_seq_len)
                    opp_attn_kind = reg.get("opp_attn_kind", "defense")
                    builder = OPP_ATTN_PER_GAME_BUILDERS[opp_attn_kind]
                    if opp_attn_kind == "offense":
                        weekly_cache_path = f"{CACHE_DIR}/weekly_{SEASONS[0]}_{SEASONS[-1]}.parquet"
                        opp_source_df = (
                            pd.read_parquet(weekly_cache_path)
                            if opponent_weekly is None
                            else opponent_weekly
                        )
                        # Match training (src/shared/pipeline.py): the raw weekly
                        # cache carries postseason rows; the "defense" concat below
                        # is already REG-only (built from REG splits). Drop playoff
                        # rows so the opp-offense per-game aggregates align with the
                        # REG-only training path (#424). Guarded for frames missing
                        # the column (synthetic test caches).
                        if "season_type" in opp_source_df.columns:
                            opp_source_df = opp_source_df[
                                opp_source_df["season_type"] == "REG"
                            ].copy()
                    else:
                        opp_source_df = pd.concat([train, val, test], ignore_index=True)
                    opp_per_game = builder(opp_source_df)
                    opp_hist_test, opp_mask_test = build_opp_defense_history_arrays(
                        pos_test, opp_per_game, opp_history_stats, opp_max_seq_len
                    )
                    opp_game_dim = opp_hist_test.shape[2]

                attn_model = MultiHeadNetWithHistory(
                    static_dim=len(attn_static_cols),
                    game_dim=hist_test.shape[2],
                    target_names=targets,
                    opp_game_dim=opp_game_dim,
                    **reg["attn_nn_kwargs_static"],
                ).to(device)
                attn_model.load_state_dict(attn_state_dict)
                if opp_game_dim is not None:
                    attn_nn_preds = attn_model.predict_numpy(
                        X_test_attn_scaled,
                        hist_test,
                        mask_test,
                        device,
                        X_opp_history=opp_hist_test,
                        opp_history_mask=opp_mask_test,
                    )
                else:
                    attn_nn_preds = attn_model.predict_numpy(
                        X_test_attn_scaled, hist_test, mask_test, device
                    )
            attn_nn_totals = _per_format_totals(attn_nn_preds)
        except Exception as e:
            errors[f"{pos}_attn_nn"] = repr(e)
            print(f"[app] {pos} attn_nn load failed: {e!r} — leaving attn_nn_pred NaN")
            attn_nn_preds = None
            attn_nn_totals = None

    # LightGBM — gated by ``reg["train_lightgbm"]``. In production all six
    # positions train LightGBM (K/DST included), so lgbm_pred is populated for
    # every position; the gate stays defensive so that if a config ever sets
    # train_lightgbm=False, that position's lgbm_pred is left NaN and the
    # frontend renders "--". (Attention NN is likewise trained for all six —
    # see the attn_nn block above.)
    lgbm_preds = None
    lgbm_totals = None
    if reg.get("train_lightgbm", False):
        try:
            lgbm_model = LightGBMMultiTarget(target_names=targets)
            lgbm_model.load(model_dir)
            lgbm_preds = lgbm_model.predict(X_test_pos)
            lgbm_totals = _per_format_totals(lgbm_preds)
        except Exception as e:
            errors[f"{pos}_lgbm"] = repr(e)
            print(f"[app] {pos} lgbm load failed: {e!r} — leaving lgbm_pred NaN")
            lgbm_preds = None
            lgbm_totals = None

    # Write into results — NaN the pred column when its model failed so the
    # frontend renders "--" instead of a misleading 0.0 (the DataFrame
    # initializes every per-format column to NaN in _load_base_data_locked).
    # We write three format-specific columns per model AND the legacy unsuffixed
    # column. The unsuffixed columns are intentionally PPR-only compatibility
    # aliases; endpoints must use _pred_col(prefix, scoring) for scoring-aware
    # reads.
    #
    # Local-then-merge semantics: the per-model totals dicts above
    # (``ridge_totals``, ``nn_totals``, ``attn_nn_totals``, ``lgbm_totals``)
    # are computed independently into local variables by the inference branches
    # above. Here we merge them into the shared ``results`` DataFrame under
    # ``_results_write_lock`` — even though the parallel pre-warm path writes
    # disjoint row indices per position, pandas' BlockManager is not
    # thread-safe for concurrent ``.loc[]`` writes (see lock comment near the
    # module top). The lock acquisition is contended only in pre-warm; the
    # per-request path through ``_ensure_position_loaded`` already holds
    # ``_cache_lock`` so this is uncontended there.
    model_totals_pairs = (
        ("ridge", ridge_totals),
        ("nn", nn_totals),
        ("attn_nn", attn_nn_totals),
        ("lgbm", lgbm_totals),
    )
    with app_pkg._results_write_lock:
        for prefix, totals in model_totals_pairs:
            legacy_col = f"{prefix}_pred"
            if totals is not None:
                for fmt, arr in totals.items():
                    results.loc[pos_index, _pred_col(prefix, fmt)] = np.round(arr, 2).astype(
                        np.float32
                    )
                results.loc[pos_index, legacy_col] = np.round(totals["ppr"], 2).astype(np.float32)
            else:
                for fmt in _VALID_SCORING:
                    results.loc[pos_index, _pred_col(prefix, fmt)] = np.nan
                results.loc[pos_index, legacy_col] = np.nan

        # Per-target raw-stat predictions + actuals for the breakdown drill-down
        # (/api/predictions/breakdown). Raw stats are scoring-format-invariant, so
        # one column per (model, target) and one per target for the actual. These
        # are the same per-target arrays already consumed by the per-target MAE
        # block below; we persist them so the breakdown survives the parquet
        # hydrate path (a hydrated container never re-runs this function). Columns
        # are pre-declared in _load_base_data_locked. A missing model (e.g. lgbm
        # for K/DST) NaN's its target columns for this position's rows.
        per_target_preds = (
            ("ridge", ridge_preds),
            ("nn", nn_preds),
            ("attn_nn", attn_nn_preds),
            ("lgbm", lgbm_preds),
        )
        for prefix, preds in per_target_preds:
            if pos == "DST":
                col = _pred_col(prefix, "comparison")
                results.loc[pos_index, col] = (
                    np.round(score_actual_components(pd.DataFrame(preds), "DST").to_numpy(), 2)
                    if preds is not None
                    else np.nan
                )
            for t in targets:
                col = f"pred_{prefix}_{t}"
                if preds is not None and t in preds:
                    results.loc[pos_index, col] = np.round(
                        np.asarray(preds[t], dtype=np.float64), 2
                    ).astype(np.float32)
                else:
                    results.loc[pos_index, col] = np.nan
        for t in targets:
            if t in pos_test.columns:
                results.loc[pos_index, f"actual_{t}"] = pos_test[t].to_numpy(dtype=np.float32)

    # Cache per-target metrics for /api/position_details. Per-target MAEs are
    # raw-stat (yards / TDs / receptions count) and so are format-invariant —
    # only the aggregated "total" row depends on scoring format. We cache the
    # total row three times under target_metrics["total_by_format"][fmt] and
    # let the API endpoint pick the right one.
    target_metrics = {}
    for t in targets:
        if t in pos_test.columns:
            actual_t = pos_test[t].values
            tm = {}
            if ridge_preds is not None and t in ridge_preds:
                tm["ridge_mae"] = round(float(np.mean(np.abs(ridge_preds[t] - actual_t))), 3)
            if nn_preds is not None and t in nn_preds:
                tm["nn_mae"] = round(float(np.mean(np.abs(nn_preds[t] - actual_t))), 3)
            if attn_nn_preds is not None and t in attn_nn_preds:
                tm["attn_nn_mae"] = round(float(np.mean(np.abs(attn_nn_preds[t] - actual_t))), 3)
            if lgbm_preds is not None and t in lgbm_preds:
                tm["lgbm_mae"] = round(float(np.mean(np.abs(lgbm_preds[t] - actual_t))), 3)
            # Per-target unit (yds / TDs / ...) so the Model Performance tab can
            # suffix each MAE. The per-row breakdown already sets this; the
            # position-details path was the gap. "" → bare decimal (unchanged).
            tm["unit"] = TARGET_UNITS.get(t, "")
            target_metrics[t] = tm
    total_by_format = {}
    for fmt in _VALID_SCORING:
        actual_col = _actual_col(fmt)
        total_actual = pos_test[actual_col].values if actual_col in pos_test.columns else None
        # K/DST splits don't carry the format-suffixed actual columns, but their
        # scoring is format-invariant so fantasy_points is the same value for
        # all three; fall back to it if the suffixed column is missing.
        if total_actual is None and "fantasy_points" in pos_test.columns:
            total_actual = pos_test["fantasy_points"].values
        if total_actual is None:
            continue
        total_tm = {}
        if ridge_totals is not None:
            total_tm["ridge_mae"] = round(
                float(np.mean(np.abs(ridge_totals[fmt] - total_actual))), 3
            )
        if nn_totals is not None:
            total_tm["nn_mae"] = round(float(np.mean(np.abs(nn_totals[fmt] - total_actual))), 3)
        if attn_nn_totals is not None:
            total_tm["attn_nn_mae"] = round(
                float(np.mean(np.abs(attn_nn_totals[fmt] - total_actual))), 3
            )
        if lgbm_totals is not None:
            total_tm["lgbm_mae"] = round(float(np.mean(np.abs(lgbm_totals[fmt] - total_actual))), 3)
        total_by_format[fmt] = total_tm
    # Default "total" key keeps the PPR view for callers that haven't migrated.
    target_metrics["total"] = total_by_format.get("ppr", {})
    target_metrics["total_by_format"] = total_by_format
    app_pkg._cache.setdefault("position_details", {})[pos] = {
        "n_features": len(feature_cols),
        "n_samples_test": len(pos_test),
        "target_metrics": target_metrics,
    }


def _ensure_base_data():
    """Load splits + build empty results frame. Idempotent. No model loads."""
    if app_pkg._cache.get("base_loaded"):
        return
    with app_pkg._cache_lock:
        # Re-check under lock: another thread may have populated between our
        # fast-path check and lock acquisition.
        if app_pkg._cache.get("base_loaded") or "results" in app_pkg._cache:
            return
        try:
            _load_base_data_locked()
        except Exception:
            # This shared loader runs before per-position error bookkeeping.
            # Retain an affirmative failure for /health while the exception
            # propagates to the request/prewarm logger; never expose its text.
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
            for pos, sentinel_mtime, err in pool.map(_load_one, pending):
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
# After the first _ensure_metrics() compute, the assembled results DataFrame
# + metrics_by_format are persisted under data/serving_cache/ and uploaded
# to S3 (best-effort). On a subsequent boot — typically a fresh ECS task
# replacement — sync_predictions_cache_from_s3() pulls the files, and
# _try_hydrate_from_disk() short-circuits the whole model-load + inference
# path when the live model fingerprint matches the cached one. Fingerprint
# mismatch (e.g. a fresh model retrain) falls back to recompute + re-upload.
# See model_sync.py::sync_predictions_cache_from_s3 / upload_predictions_cache_to_s3.

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
# v9 includes historical K/DST scoring sources and the coherent data release.
# v10 stores dedicated DST comparison totals, excluding non-shared PA semantics.
# Old native totals and rounded drill-down heads cannot substitute for these.
# v11 separates full offensive expert forecasts from shared-component totals.
_PREDICTIONS_CACHE_SCHEMA_VERSION = 11
# Optional browser snapshot, committed with its prediction/metric generation.
# Its absence permits hydration and local regeneration from those same bytes.
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
    """Yield absolute paths whose content defines cache validity.

    Walks each position's model dir, the base data splits, and the production
    serving raw inputs and release identity. The release marker/seal covers
    source changes anywhere in a coherent snapshot, including beyond a file's
    sampled head bytes. Current depth charts and historical K/DST scoring inputs
    are tracked for directories without a hydrated release marker as well.

    Do not fingerprint every local ``data/raw/*.parquet``. Developer checkouts
    often contain analysis-only or loader-side-effect caches (contracts,
    redzone, expert-source joins, older kicker PBP, generated team_stats, etc.)
    that the production container does not boot-sync before trying to hydrate
    the S3 prediction cache. Including them makes locally seeded caches miss in
    ECS even when the actual serving inputs are identical.
    """
    for pos in _ALL_POSITIONS:
        # Boot and refresh record the exact manifest GET consumed. Its content
        # is stable across containers, unlike each local refresh sentinel mtime.
        manifest_etag = os.path.join(_REPO_ROOT, "src", pos.lower(), "outputs", ".manifest-etag")
        if os.path.isfile(manifest_etag):
            yield manifest_etag
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
    """Serialize the same locked results as the cache, without a separate commit."""
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


def _snapshot_path():
    """Resolve one verified generation without loading models or running inference."""
    try:
        directory, files = prediction_cache.read_generation(_PREDICTIONS_CACHE_DIR)
        if directory.name == app_pkg._cache.get("invalidated_generation"):
            return None
        fingerprint = json.loads(files[_FINGERPRINT_JSON])
        if not isinstance(fingerprint, dict):
            return None
        if fingerprint.get("schema_version") != _PREDICTIONS_CACHE_SCHEMA_VERSION:
            return None
        if fingerprint.get("sha256") != _compute_models_fingerprint()[0]:
            return None
        if prediction_cache.is_invalidated(_PREDICTIONS_CACHE_DIR, directory.name):
            return None
        return str(directory / _SNAPSHOT_JSON) if _SNAPSHOT_JSON in files else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _try_hydrate_from_disk():
    """Hydrate a single verified immutable generation matching the current inputs."""
    try:
        directory, files = prediction_cache.read_generation(_PREDICTIONS_CACHE_DIR)
        if directory.name == app_pkg._cache.get("invalidated_generation"):
            return False
        stored = json.loads(files[_FINGERPRINT_JSON])
        mtimes = {pos: refresh_sentinel_mtime(pos) for pos in _ALL_POSITIONS}
        live_sha, _ = _compute_models_fingerprint()
        if stored.get("schema_version") != _PREDICTIONS_CACHE_SCHEMA_VERSION:
            print("[predcache] schema mismatch — will recompute")
            return False
        if stored.get("sha256") != live_sha:
            print("[predcache] fingerprint mismatch — will recompute")
            return False
        results = pd.read_parquet(io.BytesIO(files[_PREDICTIONS_PARQUET]))
        metrics_payload = json.loads(files[_METRICS_JSON])
        metrics_by_format = metrics_payload["metrics_by_format"]
        position_details = metrics_payload.get("position_details") or {}
        position_load_errors = metrics_payload.get("position_load_errors") or {}
        if _compute_models_fingerprint()[0] != live_sha:
            return False
        if prediction_cache.is_invalidated(_PREDICTIONS_CACHE_DIR, directory.name):
            return False
    except Exception as exc:
        print(f"[predcache] generation unavailable: {exc!r} — will recompute")
        return False
    app_pkg._cache["results"] = results
    app_pkg._cache["metrics_by_format"] = metrics_by_format
    app_pkg._cache["metrics"] = metrics_by_format.get("ppr", {})
    errored_positions = {key.split("_", 1)[0] for key in position_load_errors}
    app_pkg._cache["positions_loaded"] = set(_ALL_POSITIONS) - errored_positions
    app_pkg._cache["positions_failed"] = set()
    app_pkg._cache["positions_failed_mtime"] = {}
    app_pkg._cache["positions_mtime"] = mtimes
    app_pkg._cache["base_loaded"] = True
    app_pkg._cache.pop("base_load_error", None)
    app_pkg._cache["position_details"] = position_details
    app_pkg._cache["position_load_errors"] = position_load_errors
    app_pkg._cache["prediction_inputs_fingerprint"] = live_sha
    app_pkg._cache["prediction_cache_generation"] = directory.name
    print(f"[predcache] hydrated generation (sha={live_sha[:8]}, rows={len(results)})")
    if _SNAPSHOT_JSON not in files:
        snapshot = _snapshot_bytes()
        if snapshot is not None and not prediction_cache.is_invalidated(
            _PREDICTIONS_CACHE_DIR, directory.name
        ):
            try:
                # Use captured bytes, never re-resolve a pointer another worker changed.
                regenerated = prediction_cache.publish_generation(
                    _PREDICTIONS_CACHE_DIR, {**files, _SNAPSHOT_JSON: snapshot}
                )
                app_pkg._cache["prediction_cache_generation"] = regenerated.name
            except (OSError, ValueError) as exc:
                print(f"[snapshot] generation publish failed: {exc!r}")
    return True


def _persist_cache_to_disk():
    """Commit predictions, metrics and snapshot together, then upload one bundle.

    The caller holds _cache_lock for its in-memory results. Immutable generations
    and one atomic pointer keep concurrent workers' publications separate.
    """
    if "results" not in app_pkg._cache or "metrics_by_format" not in app_pkg._cache:
        return
    if app_pkg._cache["results"].attrs.get("espn_complete") is False:
        print("[predcache] ESPN unavailable — not publishing incomplete results")
        return
    sha, input_files = _compute_models_fingerprint()
    expected = app_pkg._cache.get("prediction_inputs_fingerprint")
    if expected != sha or _any_position_sentinel_advanced():
        print("[predcache] inputs changed or not recorded before inference — skipping publication")
        return
    metrics_payload = {
        "metrics_by_format": app_pkg._cache["metrics_by_format"],
        "position_details": app_pkg._cache.get("position_details", {}),
        "position_load_errors": app_pkg._cache.get("position_load_errors", {}),
    }
    try:
        parquet = io.BytesIO()
        app_pkg._cache["results"].to_parquet(parquet, index=True)
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
        }
        snapshot = _snapshot_bytes()
        if snapshot is not None:
            files[_SNAPSHOT_JSON] = snapshot
        if _compute_models_fingerprint()[0] != expected or _any_position_sentinel_advanced():
            print("[predcache] inputs changed during serialization — skipping publication")
            return
        directory = prediction_cache.publish_generation(_PREDICTIONS_CACHE_DIR, files)
        app_pkg._cache["prediction_cache_generation"] = directory.name
        app_pkg._cache.pop("invalidated_generation", None)
        print(f"[predcache] committed generation {directory.name[:12]}")
        upload_predictions_cache_to_s3()
    except Exception as exc:
        print(f"[predcache] publication failed: {exc!r} — serving in-memory results")


def _ensure_metrics():
    _discard_invalidated_generation()
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
        # Per-position invalidation in ``_ensure_position_loaded`` already
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
        # _invalidate_metrics_cache's unlink is best-effort (contextlib.suppress
        # on OSError — NFS lag, a lost race with another writer, a perms blip can
        # all leave the stale predictions.parquet/metrics.json/fingerprint.json
        # in place), and a sentinel touch doesn't change any model file's
        # (size, head-bytes) so the fingerprint still matches — _try_hydrate_from_disk
        # would re-load the exact stale aggregate we just invalidated, silently
        # un-invalidating the refresh. Recompute from the (now re-applied)
        # per-position preds instead and let _compute_metrics_locked overwrite
        # the disk cache with fresh content.
        # A disk-hydrated cache can restore ``position_load_errors`` and drop the
        # errored positions from ``positions_loaded`` so they retry (#834), but
        # it never seeds ``positions_failed`` — so those positions are "pending"
        # (see ``_positions_pending``). Returning on the hydrate hit alone would
        # serve the degraded/NaN aggregate forever, because every aggregate entry
        # point funnels through here and the fast path above can never see the
        # excluded positions (they have no recorded attempt mtime yet). Only
        # short-circuit when nothing is pending;
        # otherwise fall through to retry the pending positions once at hydrate
        # time and recompute the aggregate. A genuine re-failure lands the
        # position in ``positions_failed`` (stamped by
        # ``_ensure_all_positions_loaded``), so the fast path holds thereafter —
        # no retry storm. (#1442)
        if not sentinel_advanced and _try_hydrate_from_disk() and not _positions_pending():
            return
        _ensure_base_data()
        app_pkg._cache["prediction_inputs_fingerprint"] = _compute_models_fingerprint()[0]
        _ensure_all_positions_loaded()
        _compute_metrics_locked()


def _any_position_sentinel_advanced() -> bool:
    """True iff a loaded or failed position has newer on-disk artifacts.

    Failed positions use the sentinel recorded at their last failed attempt so
    aggregate requests retry them once after a refresh, never on every request.
    """
    stored = app_pkg._cache.get("positions_mtime", {})
    failed_at = app_pkg._cache.get("positions_failed_mtime", {})
    # Snapshot to tuples: another thread can add/discard positions under
    # _cache_lock while this lock-free fast path iterates it, which would raise
    # "Set changed size during iteration". (#1014)
    loaded = tuple(app_pkg._cache.get("positions_loaded", ()))
    failed = tuple(app_pkg._cache.get("positions_failed", ()))
    return any(refresh_sentinel_mtime(pos) > stored.get(pos, -1.0) for pos in loaded) or any(
        refresh_sentinel_mtime(pos) > failed_at.get(pos, -1.0) for pos in failed
    )


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


def _discard_invalidated_generation() -> None:
    """Stop using a generation another worker revoked after this worker hydrated.

    Hydration can record an already-advanced sentinel just before a peer marks
    the old generation invalid. That worker's matching mtimes cannot establish
    freshness: discard its model-load bookkeeping so the next request either
    hydrates a newer generation or actually reapplies the models.
    """
    generation = app_pkg._cache.get("prediction_cache_generation")
    if generation is None or not prediction_cache.is_invalidated(
        _PREDICTIONS_CACHE_DIR, generation
    ):
        return
    with app_pkg._cache_lock:
        generation = app_pkg._cache.get("prediction_cache_generation")
        if generation is None or not prediction_cache.is_invalidated(
            _PREDICTIONS_CACHE_DIR, generation
        ):
            return
        _invalidate_metrics_cache(reason="shared-generation-invalidation")
        app_pkg._cache.pop("prediction_cache_generation", None)
        app_pkg._cache.pop("prediction_inputs_fingerprint", None)
        for key in ("positions_loaded", "positions_failed"):
            app_pkg._cache[key] = set()
        for key in (
            "positions_mtime",
            "positions_failed_mtime",
            "position_load_errors",
            "position_details",
        ):
            app_pkg._cache[key] = {}


def _invalidate_metrics_cache(*, reason: str) -> None:
    """Invalidate this worker's consumed generation across all local workers.

    Fingerprint mismatches exclude old snapshots. Sentinel refresh skips hydrate
    and commits a new generation after inference; in-progress readers keep theirs.
    """
    app_pkg._cache.pop("metrics_by_format", None)
    app_pkg._cache.pop("metrics", None)
    generation = app_pkg._cache.get("prediction_cache_generation")
    if generation is not None:
        app_pkg._cache["invalidated_generation"] = generation
        prediction_cache.invalidate_generation(_PREDICTIONS_CACHE_DIR, generation)
    print(f"[predcache] invalidated in-memory cache ({reason})")


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
