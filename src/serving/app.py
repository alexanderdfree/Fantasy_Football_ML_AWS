"""Flask web application for the Fantasy Football Points Predictor.

All predictions come from position-specific models (QB, RB, WR, TE, K, DST).
No general cross-position model is used.
"""

import contextlib
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib

matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request, send_file

import src.dst.config as dst_cfg
import src.dst.data as dst_data
import src.dst.features as dst_features
import src.k.config as k_cfg
import src.k.data as k_data
import src.k.features as k_features
import src.qb.config as qb_cfg
import src.rb.config as rb_cfg
import src.te.config as te_cfg
import src.wr.config as wr_cfg
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
from src.data.loader import compute_fantasy_points
from src.features.engineer import (
    OPP_ATTN_PER_GAME_BUILDERS,
    build_game_history_arrays,
    build_opp_defense_history_arrays,
    get_attn_static_columns,
)
from src.serving import benchmark_history, comparison
from src.serving.metadata import _ALL_TARGETS, POSITION_INFO

# Pure serialization / scoring-column helpers live in serialization.py; imported
# here (and thus re-exported as ``src.serving.app.<name>``) so route handlers and
# existing ``from src.serving.app import _safe_num`` call sites keep working.
from src.serving.serialization import (
    _MODEL_PRED_COLUMNS,
    _MODEL_PRED_PREFIXES,
    _VALID_SCORING,
    _actual_col,
    _pred_col,
    _records_to_player_rows,
    _round_or_none,
    _safe_num,
    _safe_str,
    _validate_scoring,
)
from src.serving.wiki import (
    _WIKI_GITHUB_BLOB_BASE as _WIKI_GITHUB_BLOB_BASE,
)
from src.serving.wiki import (
    WIKI_DOCS,
    _render_wiki_doc,
)
from src.serving.wiki import (
    _wiki_rewrite_href as _wiki_rewrite_href,
)
from src.shared.aggregate_targets import TARGET_UNITS, predictions_to_fantasy_points
from src.shared.artifact_integrity import (
    assert_scaler_matches,
    read_scaler_meta,
    unwrap_state_dict,
)
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
from src.shared.weather_features import WEATHER_FEATURES_ALL

# Boot-time S3 sync lives in gunicorn.conf.py::on_starting (master-level,
# before --preload import) so this module has no import-time side effects.
# See that hook for the rationale; cross-link kept here so future readers
# don't reach for the simpler-looking module-level call.

app = Flask(__name__)

_cache = {}
# Serializes lazy model/data loads — Flask dispatches requests on multiple
# threads, so two concurrent first-hit requests would otherwise both see
# _cache as empty and race on duplicate I/O plus .loc-writes into the shared
# results DataFrame. Reentrant because _ensure_metrics nests into
# _ensure_position_loaded.
_cache_lock = threading.RLock()
# ``_apply_position_models`` writes per-position prediction columns into the
# shared ``_cache["results"]`` DataFrame. Even when row indices are disjoint
# across positions (QB rows vs RB rows, etc.), pandas' BlockManager is not
# thread-safe for concurrent ``.loc[]`` writes — the internal column-block
# representation is shared across columns, and two writers can corrupt block
# state mid-update. The parallel pre-warm path in ``_ensure_all_positions_loaded``
# spawns one worker per position; without this lock those workers race on the
# DataFrame's internals. Plain ``threading.Lock`` is correct here (no
# reentrancy needed — the write block doesn't call back into itself).
_results_write_lock = threading.Lock()
# Wiki page caching uses its own lock so a slow first-hit ``_ensure_metrics``
# (model loads, feature build) doesn't serialize wiki-tab GETs behind it.
# Wiki entries live in the SAME ``_cache`` dict (keyed by ("wiki", slug))
# because the existing module-global cache structure is shared; only the
# locking discipline diverges. Plain ``threading.Lock`` is sufficient — the
# wiki cache path doesn't nest into other cache helpers, so the RLock
# reentrancy that ``_cache_lock`` requires is overkill here.
#
# Originally split out under code-review finding L-SS4 (one RLock serializing
# two unrelated cache disciplines).
_wiki_cache_lock = threading.Lock()
# Benchmark-history cache (see ``_load_benchmark_history_rows``) is a third
# discipline because its invalidation is mtime-driven rather than write-driven
# and the cache structure is a tuple, not a dict slot. Documented at
# ``_BENCHMARK_HISTORY_LOCK`` near the rendering helpers.


@app.errorhandler(Exception)
def handle_api_error(e):
    """Return JSON errors for /api/ routes, default HTML for others."""
    if request.path.startswith("/api/"):
        # Log the full traceback server-side; never echo exception text to the
        # client. str(e) on a Python exception can leak filesystem paths,
        # config values, or library internals (CodeQL py/stack-trace-exposure).
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500
    raise e


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
    dst_df = dst_data.build_data()
    dst_df = POSITION_REGISTRY["DST"]["compute_targets_fn"](dst_df)
    dst_features.compute_features(dst_df)
    train = dst_df[dst_df["season"].isin(TRAIN_SEASONS)].copy()
    val = dst_df[dst_df["season"].isin(VAL_SEASONS)].copy()
    test = dst_df[dst_df["season"].isin(TEST_SEASONS)].copy()
    return train, val, test


def _apply_position_models(train, val, test, pos, results):
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
    games_per_season = pos_train.groupby(["player_id", "season"])["week"].transform("count")
    pos_train = pos_train[games_per_season >= min_games].copy()

    feature_cols = reg["get_feature_columns_fn"]()
    pos_train, pos_val, pos_test = build_position_features(
        pos_train, pos_val, pos_test, reg, feature_cols
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
    # for ``_combine_total`` below. (The registry's ``aggregate_fn`` slot is no
    # longer consumed here; see W.SHARED-PIPE for the registry-side cleanup.)
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
    with _results_write_lock:
        errors = _cache.setdefault("position_load_errors", {})
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
                kicks_df = _cache.get("k_kicks_df")
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
                        opp_source_df = pd.read_parquet(weekly_cache_path)
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
    with _results_write_lock:
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
    _cache.setdefault("position_details", {})[pos] = {
        "n_features": len(feature_cols),
        "n_samples_test": len(pos_test),
        "target_metrics": target_metrics,
    }


_ALL_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]

# Positions sourced from their own dedicated splits (``_load_k_splits`` /
# ``_load_dst_splits``) and appended to ``results`` separately in
# ``_load_base_data_locked``. They MUST be excluded from the skill-position
# ``test.parquet`` base copy: kickers ALSO appear in the offensive player table
# (with ~0 offensive fantasy_points), so copying them there would double every
# kicker row once the K split is appended — a phantom twin with actual≈0 and
# null preds. See TODO.md Fixed archive ("Kicker rows duplicated in serving").
_APPENDED_POSITIONS = ("K", "DST")


def _ensure_base_data():
    """Load splits + build empty results frame. Idempotent. No model loads."""
    if _cache.get("base_loaded"):
        return
    with _cache_lock:
        # Re-check under lock: another thread may have populated between our
        # fast-path check and lock acquisition.
        if _cache.get("base_loaded") or "results" in _cache:
            return
        _load_base_data_locked()


def _load_reg(path):
    """Read a split parquet, filtering to REG-season rows. Module-level so the
    boot path (``_load_base_data_locked``) and the hydrated-container refresh
    path (``_load_splits_locked``) share one definition of a loaded split."""
    df = pd.read_parquet(path)
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()
    return df


def _load_base_data_locked():
    print("Loading data...")

    train = _load_reg("data/splits/train.parquet")
    val = _load_reg("data/splits/val.parquet")
    test = _load_reg("data/splits/test.parquet")

    for df in [train, val, test]:
        _compute_scoring_formats(df)

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
        offset = results.index.max() + 1
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

    _cache["splits"] = {
        "QB": (train, val, test),
        "RB": (train, val, test),
        "WR": (train, val, test),
        "TE": (train, val, test),
        "K": (k_train, k_val, k_test),
        "DST": (dst_train, dst_val, dst_test),
    }
    # K's attention NN needs the raw per-kick records to build nested history
    # at inference — stash here so _apply_position_models can reach it.
    _cache["k_kicks_df"] = k_kicks_df
    _cache["results"] = results
    _cache["positions_loaded"] = set()
    _cache["base_loaded"] = True


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
    if "results" not in _cache or "splits" not in _cache:
        return
    try:
        k_train, k_val, k_test, k_kicks_df = _load_k_splits()
    except Exception as e:  # noqa: BLE001 — refresh is best-effort
        print(f"[K] data refresh failed: {e!r} — reusing boot-cached k_kicks_df/splits")
        return
    # Refreshing the per-kick records is the core of the fix: the new model's
    # nested-history tensor must be built from the data it was trained against.
    _cache["k_kicks_df"] = k_kicks_df
    results = _cache["results"]
    existing_k_index = results.index[results["position"] == "K"]
    if len(existing_k_index) == len(k_test):
        k_test = k_test.copy()
        k_test.index = existing_k_index
        _cache["splits"]["K"] = (k_train, k_val, k_test)
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
        train = _load_reg("data/splits/train.parquet")
        val = _load_reg("data/splits/val.parquet")
        test = _load_reg("data/splits/test.parquet")
        for df in [train, val, test]:
            _compute_scoring_formats(df)
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
            print(
                f"[{pos_label}] _load_splits_locked: row-count mismatch "
                f"({len(existing_index)} cached rows vs {len(pos_test)} fresh) — "
                f"keeping fresh index (unaligned pred writes may be skipped)"
            )
    _cache["splits"] = {
        "QB": (train, val, test),
        "RB": (train, val, test),
        "WR": (train, val, test),
        "TE": (train, val, test),
        "K": (k_train, k_val, reindexed["K"]),
        "DST": (dst_train, dst_val, reindexed["DST"]),
    }
    _cache["k_kicks_df"] = k_kicks_df


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
    if _cache.get("results") is None or "splits" not in _cache:
        return
    try:
        dst_train, dst_val, dst_test = _load_dst_splits()
    except Exception as e:  # noqa: BLE001 — refresh is best-effort
        print(f"[DST] data refresh failed: {e!r} — reusing boot-cached splits")
        return
    results = _cache["results"]
    existing_dst_index = results.index[results["position"] == "DST"]
    if len(existing_dst_index) == len(dst_test):
        dst_test = dst_test.copy()
        dst_test.index = existing_dst_index
        _cache["splits"]["DST"] = (dst_train, dst_val, dst_test)
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
    _ensure_base_data()
    sentinel_mtime = refresh_sentinel_mtime(pos)
    loaded_mtime = _cache.get("positions_mtime", {}).get(pos, -1.0)
    failed_at = _cache.get("positions_failed_mtime", {}).get(pos, -1.0)
    # Fast path: take the no-lock early return only when the sentinel hasn't
    # advanced beyond what we already loaded (or beyond what failed last time —
    # ``positions_failed_mtime`` records the sentinel value at the failure so
    # the next sentinel touch can retry. See the failure path below for where
    # the mtime gets stamped).
    if sentinel_mtime <= loaded_mtime and pos in _cache.get("positions_loaded", ()):
        return
    if pos in _cache.get("positions_failed", ()) and sentinel_mtime <= failed_at:
        # Cached hard-failure at this sentinel value — don't retry every
        # request. A subsequent sentinel advance breaks out of this branch
        # and the slow path below invalidates the failed state.
        return
    with _cache_lock:
        if "splits" not in _cache and _cache.get("results") is not None:
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
            _load_splits_locked(_cache["results"])
        if "splits" not in _cache:
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
            _cache.setdefault("positions_failed", set()).add(pos)
            _cache.setdefault("positions_failed_mtime", {})[pos] = sentinel_mtime
            _cache.setdefault("position_load_errors", {})[pos] = err_msg
            return
        # Re-stat under the lock so we make the refresh decision against the
        # current filesystem state, not a possibly-stale snapshot from the
        # fast-path check.
        sentinel_mtime = refresh_sentinel_mtime(pos)
        loaded_mtime = _cache.get("positions_mtime", {}).get(pos, -1.0)
        failed_at = _cache.get("positions_failed_mtime", {}).get(pos, -1.0)
        loaded_advance = sentinel_mtime > loaded_mtime and loaded_mtime != -1.0
        failed_advance = pos in _cache.get("positions_failed", set()) and sentinel_mtime > failed_at
        if loaded_advance or failed_advance:
            # In-flight refresh detected: drop cached state for this position
            # so the upcoming _apply_position_models writes against the new
            # on-disk model. metrics_by_format aggregates across positions so
            # it must be invalidated whenever ANY position re-loads — and the
            # persisted disk cache (predictions.parquet / metrics.json /
            # fingerprint.json) must go too, else _try_hydrate_from_disk would
            # restore the stale aggregate on the next container boot.
            _cache.get("positions_loaded", set()).discard(pos)
            _cache.get("positions_failed", set()).discard(pos)
            _cache.get("positions_failed_mtime", {}).pop(pos, None)
            _cache.get("position_load_errors", {}).pop(pos, None)
            # Also drop the stale per-target MAEs for this position. Without this,
            # if the upcoming _apply_position_models fails (slow-path exception
            # below), /api/position_details would keep serving the PREVIOUS
            # load's MAEs while /health reports the position failed and its preds
            # are NaN — a confusing inconsistency. _apply_position_models only
            # overwrites position_details[pos] on success, so we must clear it on
            # invalidation to avoid the stale-survives-failed-reload case.
            _cache.get("position_details", {}).pop(pos, None)
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
        if pos in _cache.get("positions_loaded", set()):
            return
        if pos in _cache.get("positions_failed", set()):
            return
        train, val, test = _cache["splits"][pos]
        print(f"Applying {pos}-specific model...")
        try:
            _apply_position_models(train, val, test, pos, _cache["results"])
        except Exception as e:
            # Anything that slips past the inner per-model try/excepts —
            # typically data-loading or feature-building failures that affect
            # the whole position. Record the sentinel value at failure under
            # ``positions_failed_mtime`` so we don't spam-retry every request,
            # but a later sentinel advance (a newly synced model) still
            # triggers a retry.
            traceback.print_exc()
            _cache.setdefault("positions_failed", set()).add(pos)
            _cache.setdefault("positions_failed_mtime", {})[pos] = sentinel_mtime
            _cache.setdefault("position_load_errors", {})[pos] = repr(e)
            print(f"[app] {pos} fully failed: {e!r} — serving degraded")
            return
        # Stamp positions_mtime AFTER _apply succeeds so a transient failure
        # doesn't leave a misleading "loaded at sentinel X" entry. Successful
        # loads record the sentinel value they were taken against — the
        # invalidation check above uses this to detect refresh advances.
        _cache.setdefault("positions_mtime", {})[pos] = sentinel_mtime
        _cache["positions_loaded"].add(pos)


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
    if "splits" not in _cache and _cache.get("results") is not None:
        # Hydrated container: derive splits on demand so the aggregate rebuild
        # can re-apply advanced positions instead of silently recomputing
        # metrics over the stale hydrated preds and re-persisting them under a
        # fresh fingerprint (#789).
        _load_splits_locked(_cache["results"])
    if "splits" not in _cache:
        return
    splits = _cache["splits"]
    loaded = _cache.setdefault("positions_loaded", set())
    failed = _cache.setdefault("positions_failed", set())
    errors = _cache.setdefault("position_load_errors", {})
    mtimes = _cache.setdefault("positions_mtime", {})
    failed_mtimes = _cache.setdefault("positions_failed_mtime", {})
    details = _cache.setdefault("position_details", {})
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
        stored = mtimes.get(pos, -1.0)
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
            _apply_position_models(train, val, test, pos, _cache["results"])
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
            f"{list(_cache.get('position_load_errors', {}).keys())}"
        )


def _degraded_positions() -> list[str]:
    """Positions with any recorded model-load error (fully failed OR partial
    per-model failure). Returned sorted so the frontend banner has a stable
    ordering across requests.
    """
    errs = _cache.get("position_load_errors", {})
    if not errs:
        return []
    degraded: set[str] = set()
    for key in errs:
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
# Browser-ready snapshot the frontend hydrates its first paint from (see
# /api/snapshot + static/js/app.js). Auxiliary to the cache triple above —
# its absence is non-fatal (frontend falls back to /api/predictions), so it is
# deliberately NOT part of model_sync._PREDICTIONS_CACHE_FILES (that tuple is
# all-or-nothing; a missing member forces a 30-60s recompute). It rides the
# best-effort model_sync._PREDICTIONS_CACHE_OPTIONAL path instead.
_SNAPSHOT_JSON = "snapshot.json"


def _iter_fingerprint_paths():
    """Yield absolute paths whose (size, mtime) define cache validity.

    Walks each position's model dir, the base data splits, and every
    ``data/raw/*.parquet`` file. Any change to a trained model, a split, or
    a raw cache invalidates the predictions cache automatically.

    Why widen to all of ``data/raw/`` rather than just the kicker PBP cache:
    ``sync_data_from_s3`` pulls every ``data/raw/*.parquet`` at boot, and
    serving inference reads many of them at request time
    (``weekly_*.parquet``, ``schedules_*.parquet``,
    ``team_stats_*.parquet``, ``injuries_*.parquet``,
    ``depth_charts_*.parquet``, ``snap_counts_*.parquet``,
    ``rosters_*.parquet`` — via ``build_position_features`` for weather /
    Vegas / depth-chart features). A refreshed weekly cache without a
    fingerprint bump would let a new container hydrate stale predictions
    paired against newer raw data. The kicker PBP file is the only one
    that affects K's own feature build directly, but every position picks
    up weather/Vegas/etc. from the shared raws.
    """
    for pos in _ALL_POSITIONS:
        model_dir = os.path.join(_REPO_ROOT, "src", pos.lower(), "outputs", "models")
        if not os.path.isdir(model_dir):
            continue
        for dirpath, _, filenames in os.walk(model_dir):
            for fname in filenames:
                yield os.path.join(dirpath, fname)
    splits_dir = os.path.join(_REPO_ROOT, "data", "splits")
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        path = os.path.join(splits_dir, name)
        if os.path.isfile(path):
            yield path
    raw_dir = os.path.join(_REPO_ROOT, "data", "raw")
    if os.path.isdir(raw_dir):
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".parquet"):
                yield os.path.join(raw_dir, fname)


_FINGERPRINT_CONTENT_BYTES = 64 * 1024  # 64 KB head-sample per file


def _compute_models_fingerprint():
    """Return (sha256_hex, files_list) over the fingerprint paths.

    Previously this combined ``(size, mtime_ns)`` per file. Mtime was the
    sensitive bit: ECS task replacement re-syncs the model dir from S3, and
    boto3's ``download_file`` stamps the local file with the *download* time,
    not the upload time on S3 — so every fresh container saw a different
    fingerprint and missed the cache on boot even when the content was
    byte-identical. We now hash ``(size, head-bytes)`` instead, where
    head-bytes is the first 64 KB of each file's content. That's enough
    collision resistance for our use case (every model retrain rewrites
    weights at the start of the joblib pickle / torch state dict; even tiny
    config changes shift those leading bytes), and reading 64 KB per file
    keeps the boot-time fingerprint compute fast (~50ish files in the
    aggregate, <5 ms on local SSD).
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
            with open(path, "rb") as f:
                head_bytes = f.read(_FINGERPRINT_CONTENT_BYTES)
        except OSError:
            # File disappeared between stat and open — same handling as the
            # stat OSError above (skip this entry; differing fingerprint will
            # naturally invalidate the cache).
            continue
        content_hash = hashlib.sha256(head_bytes).hexdigest()
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


def _write_snapshot_json():
    """Write the browser-ready predictions snapshot to ``snapshot.json``.

    Built from the same ``_records_to_player_rows`` serializer ``/api/predictions``
    uses, so the static snapshot can never drift from the live API shape. Carries
    every player-week row for all three scoring formats plus the week list and
    degraded-position set, so the frontend can filter/sort/scoring-switch entirely
    client-side off a single fetch.

    Best-effort and auxiliary: a write failure logs and returns (serving
    continues; the frontend falls back to ``/api/predictions``). Atomic
    ``os.replace`` so a concurrent reader never sees a half-written file. Caller
    must hold ``_cache_lock`` and have ``_cache["results"]`` populated.
    """
    results = _cache.get("results")
    if results is None:
        return
    path = os.path.join(_PREDICTIONS_CACHE_DIR, _SNAPSHOT_JSON)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    # Payload construction (results["week"], _records_to_player_rows) is inside
    # the try so a malformed/partial results frame can't break the caller — this
    # writer is auxiliary and must never abort _persist_cache_to_disk or the S3
    # upload that follows it. path/tmp stay above the try (pure string ops) so
    # the except's os.unlink(tmp) cleanup always has a bound name.
    try:
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "weeks": sorted(int(w) for w in results["week"].unique()),
            "degraded_positions": _degraded_positions(),
            "scoring": {
                fmt: _records_to_player_rows(results, scoring=fmt) for fmt in _VALID_SCORING
            },
        }
        os.makedirs(_PREDICTIONS_CACHE_DIR, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001 — snapshot is best-effort, must not break serving
        print(f"[snapshot] write failed: {e!r} — frontend will fall back to /api/predictions")
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        return
    print(
        f"[snapshot] wrote {path} (weeks={len(payload['weeks'])}, rows/format={len(payload['scoring']['ppr'])})"
    )


def _try_hydrate_from_disk():
    """Populate ``_cache`` directly from data/serving_cache/ when the stored
    fingerprint matches the live one. Caller must hold ``_cache_lock``.
    Returns ``True`` on hit (caller can skip the heavy compute path).

    Restores ``position_details`` (per-target MAE for ``/api/position_details``)
    and ``position_load_errors`` (for ``/health`` + degraded_positions) when
    they're present in the cache. Older caches written before M16 lacked these
    keys; we fall through gracefully — the endpoints already handle missing
    keys defensively, the next recompute populates them, and the next persist
    writes the full payload.
    """
    parquet_path = os.path.join(_PREDICTIONS_CACHE_DIR, _PREDICTIONS_PARQUET)
    metrics_path = os.path.join(_PREDICTIONS_CACHE_DIR, _METRICS_JSON)
    fingerprint_path = os.path.join(_PREDICTIONS_CACHE_DIR, _FINGERPRINT_JSON)
    if not (
        os.path.isfile(parquet_path)
        and os.path.isfile(metrics_path)
        and os.path.isfile(fingerprint_path)
    ):
        return False
    try:
        with open(fingerprint_path) as f:
            stored = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[predcache] fingerprint read failed: {e!r} — will recompute")
        return False
    live_sha, _ = _compute_models_fingerprint()
    if stored.get("sha256") != live_sha:
        print(
            f"[predcache] fingerprint mismatch "
            f"(cache={(stored.get('sha256') or '<none>')[:8]}, live={live_sha[:8]}) "
            f"— will recompute"
        )
        return False
    try:
        results = pd.read_parquet(parquet_path)
        with open(metrics_path) as f:
            metrics_payload = json.load(f)
    except Exception as e:  # noqa: BLE001 — corrupt cache must not crash boot
        print(f"[predcache] cache read failed: {e!r} — will recompute")
        return False
    # metrics.json schema:
    #   Old (pre-M16): the bare metrics_by_format dict ({"ppr": {...}, ...}).
    #   New (M16+): a wrapper {"metrics_by_format": {...},
    #                          "position_details": {...},
    #                          "position_load_errors": {...}}.
    # Detect by presence of the wrapper key to stay compatible with caches
    # written by older containers in the S3 bucket.
    if isinstance(metrics_payload, dict) and "metrics_by_format" in metrics_payload:
        metrics_by_format = metrics_payload["metrics_by_format"]
        position_details = metrics_payload.get("position_details") or {}
        position_load_errors = metrics_payload.get("position_load_errors") or {}
    else:
        # Legacy bare-dict format — no position_details / position_load_errors
        # to restore. Endpoints already tolerate missing keys.
        metrics_by_format = metrics_payload
        position_details = {}
        position_load_errors = {}
    _cache["results"] = results
    _cache["metrics_by_format"] = metrics_by_format
    _cache["metrics"] = metrics_by_format.get("ppr", {})
    _cache["positions_loaded"] = set(_ALL_POSITIONS)
    # Seed positions_mtime from the current on-disk sentinel for each position.
    # Without this seed, ``_ensure_position_loaded``'s in-flight refresh path
    # is dead on hydrated containers: ``loaded_mtime`` defaults to -1.0, the
    # invalidation condition ``sentinel > loaded_mtime AND loaded_mtime != -1.0``
    # never fires, and a post-hydration model refresh wouldn't trigger a reload.
    # Reading the sentinel here records "we hydrated against whatever model is
    # currently on disk" — any subsequent poller-driven advance will be detected.
    hydrated_mtimes = {pos: refresh_sentinel_mtime(pos) for pos in _ALL_POSITIONS}
    _cache["positions_mtime"] = hydrated_mtimes
    _cache["base_loaded"] = True
    if position_details:
        _cache["position_details"] = position_details
    if position_load_errors:
        _cache["position_load_errors"] = position_load_errors
    print(
        f"[predcache] hydrated from disk "
        f"(sha={live_sha[:8]}, rows={len(results)}, "
        f"position_details={len(position_details)}, "
        f"errors={len(position_load_errors)})"
    )
    # The browser snapshot may be absent on caches written by pre-snapshot
    # containers (the core triple synced from S3, snapshot.json did not exist
    # there yet). Regenerate it locally from the just-hydrated results so
    # /api/snapshot serves on this container without waiting for the next
    # retrain. Local-only (no S3 upload): regeneration is a cheap in-memory
    # serialize, and the canonical snapshot lands in S3 via the full-compute
    # _persist_cache_to_disk path. Off the request path (hydrate runs in the
    # post_fork warm thread).
    if not os.path.isfile(os.path.join(_PREDICTIONS_CACHE_DIR, _SNAPSHOT_JSON)):
        _write_snapshot_json()
    return True


def _persist_cache_to_disk():
    """Atomic write of predictions.parquet + metrics.json + fingerprint.json,
    followed by a best-effort S3 upload. Caller must hold ``_cache_lock``.

    Two concurrent workers' pre-warm threads can race on cold-container
    first-boot: each computes, each writes its own temp file, ``os.replace``
    is atomic per-file so the final state is one consistent triple (whichever
    worker finished last for each file). Both workers then populate their
    own in-memory ``_cache`` from the compute they already finished — no
    cross-process re-read needed.
    """
    if "results" not in _cache or "metrics_by_format" not in _cache:
        return
    os.makedirs(_PREDICTIONS_CACHE_DIR, exist_ok=True)
    sha, files = _compute_models_fingerprint()
    parquet_path = os.path.join(_PREDICTIONS_CACHE_DIR, _PREDICTIONS_PARQUET)
    metrics_path = os.path.join(_PREDICTIONS_CACHE_DIR, _METRICS_JSON)
    fingerprint_path = os.path.join(_PREDICTIONS_CACHE_DIR, _FINGERPRINT_JSON)
    # PID alone isn't enough — two pre-warm threads in the same worker would
    # collide on the temp name; thread id makes the suffix unique per writer.
    suffix = f"{os.getpid()}.{threading.get_ident()}.tmp"
    parquet_tmp = f"{parquet_path}.{suffix}"
    metrics_tmp = f"{metrics_path}.{suffix}"
    fingerprint_tmp = f"{fingerprint_path}.{suffix}"
    # M16: persist position_details + position_load_errors so hydrate restores
    # them on the next boot. Folded into metrics.json (rather than a separate
    # file) to keep the artifact triple {parquet, json, json} unchanged — the
    # S3 sync helper iterates _PREDICTIONS_CACHE_FILES, so adding a new file
    # would require touching model_sync.py. _try_hydrate_from_disk reads
    # either schema transparently.
    metrics_payload = {
        "metrics_by_format": _cache["metrics_by_format"],
        "position_details": _cache.get("position_details", {}),
        "position_load_errors": _cache.get("position_load_errors", {}),
    }
    try:
        _cache["results"].to_parquet(parquet_tmp, index=True)
        with open(metrics_tmp, "w") as f:
            json.dump(metrics_payload, f)
        with open(fingerprint_tmp, "w") as f:
            json.dump({"sha256": sha, "files": files}, f)
        os.replace(parquet_tmp, parquet_path)
        os.replace(metrics_tmp, metrics_path)
        os.replace(fingerprint_tmp, fingerprint_path)
    except Exception as e:  # noqa: BLE001 — persist must not break serving
        print(f"[predcache] persist failed: {e!r} — serving continues from in-memory cache")
        for tmp in (parquet_tmp, metrics_tmp, fingerprint_tmp):
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        return
    print(f"[predcache] wrote cache to {_PREDICTIONS_CACHE_DIR} (sha={sha[:8]})")
    # Write the browser snapshot before uploading so the upload (which also
    # pushes the auxiliary snapshot.json) finds it on disk.
    _write_snapshot_json()
    try:
        upload_predictions_cache_to_s3()
    except Exception as e:  # noqa: BLE001 — upload best-effort
        print(f"[predcache] upload failed: {e!r}")


def _ensure_metrics():
    if "metrics_by_format" in _cache and not _any_position_sentinel_advanced():
        return
    with _cache_lock:
        if "metrics_by_format" in _cache and not _any_position_sentinel_advanced():
            return
        # A sentinel advanced under us — drop the cached aggregate so it
        # rebuilds against the freshly-loaded per-position predictions. The
        # per-position invalidation in ``_ensure_position_loaded`` already
        # discards ``metrics_by_format`` when one position re-loads, but
        # ``_ensure_metrics`` can also be hit *before* a per-position re-load
        # (e.g. /api/metrics with all positions still marked loaded against
        # stale mtimes), so the sentinel sweep here is the second line of
        # defense.
        sentinel_advanced = "metrics_by_format" in _cache
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
        if not sentinel_advanced and _try_hydrate_from_disk():
            return
        _ensure_all_positions_loaded()
        _compute_metrics_locked()


def _any_position_sentinel_advanced() -> bool:
    """True iff any loaded position's on-disk sentinel mtime is greater than
    the value recorded when we last loaded that position. Used as the second
    line of defense for ``_ensure_metrics`` — see comment there.
    """
    stored = _cache.get("positions_mtime", {})
    loaded = _cache.get("positions_loaded", set())
    return any(refresh_sentinel_mtime(pos) > stored.get(pos, -1.0) for pos in loaded)


def _invalidate_metrics_cache(*, reason: str) -> None:
    """Pop the in-memory ``metrics_by_format`` and delete the persisted
    artifacts on disk. The disk files would otherwise survive an in-memory
    invalidation — and ``_try_hydrate_from_disk`` could then re-hydrate them
    on the next boot, restoring the same stale metrics we just invalidated.
    (S3 cleanup is out of scope here; the next ``_persist_cache_to_disk``
    call after recompute will overwrite the S3 object with fresh content.)
    """
    _cache.pop("metrics_by_format", None)
    _cache.pop("metrics", None)
    for name in (_PREDICTIONS_PARQUET, _METRICS_JSON, _FINGERPRINT_JSON):
        path = os.path.join(_PREDICTIONS_CACHE_DIR, name)
        with contextlib.suppress(OSError):
            os.unlink(path)
    print(f"[predcache] invalidated in-memory + on-disk cache ({reason})")


def _compute_metrics_locked():
    """Compute overall + per-position MAE/RMSE/R² for every model under each
    of the three scoring formats, caching the results under
    ``_cache["metrics_by_format"][fmt]``. Keeps ``_cache["metrics"]`` as a PPR
    alias for callers that haven't migrated.
    """
    results = _cache["results"]
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
            # Skip rows where this model has no prediction (K/DST for lgbm, or
            # any position whose model failed to load). attn_nn IS trained for
            # all six positions, so K/DST attn_nn rows are real, not skipped.
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
                pm = {k: round(v, 4) for k, v in pm.items()}
                pm["position"] = pos
                pm["n_samples"] = int(pos_mask.sum())
                by_position.append(pm)
            per_format[name] = {
                "overall": {k: round(v, 4) for k, v in overall.items()},
                "by_position": by_position,
            }
        metrics_by_format[fmt] = per_format
    _cache["metrics_by_format"] = metrics_by_format
    _cache["metrics"] = metrics_by_format.get("ppr", {})
    _persist_cache_to_disk()
    print("Ready!")


def _get_data(scoring="ppr"):
    """Full load: all positions + metrics for the requested scoring format."""
    _ensure_metrics()
    metrics_by_format = _cache["metrics_by_format"]
    metrics = metrics_by_format.get(scoring) or metrics_by_format.get("ppr", {})
    return _cache["results"], metrics


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


def _results_for_position(position, scoring):
    """Results rows for ``position`` ("ALL" → all positions).

    A single position triggers its lazy per-position load and slices the shared
    ``_cache["results"]``; "ALL" returns the full multi-position frame. Shared by
    /api/predictions and /api/top_players.
    """
    if position != "ALL":
        _ensure_position_loaded(position)
        df = _cache["results"]
        return df[df["position"] == position]
    results, _ = _get_data(scoring)
    return results


@app.route("/api/predictions")
def api_predictions():
    position = request.args.get("position", "ALL")
    week = request.args.get("week", "ALL")
    search = request.args.get("search", "").strip().lower()
    sort_by = request.args.get("sort", "fantasy_points")
    order = request.args.get("order", "desc")
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))

    df = _results_for_position(position, scoring)
    if week != "ALL":
        try:
            df = df[df["week"] == int(week)]
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid week: {week}"}), 400
    if search:
        df = df[df["player_display_name"].str.lower().str.contains(search, na=False, regex=False)]

    rows = _records_to_player_rows(df, scoring=scoring)

    reverse = order == "desc"
    if sort_by in ("actual", "ridge_pred", "nn_pred", "attn_nn_pred", "lgbm_pred", "week"):
        rows.sort(key=lambda x: x.get(sort_by) or 0, reverse=reverse)
    else:
        rows.sort(key=lambda x: x.get("actual") or 0, reverse=reverse)

    return jsonify(
        {
            "players": rows,
            "total": len(rows),
            "scoring": scoring,
            # Surfaced so the frontend can render a banner. See
            # static/js/app.js::loadPredictions and _degraded_positions above.
            "degraded_positions": _degraded_positions(),
        }
    )


@app.route("/api/snapshot")
def api_snapshot():
    """Serve the precomputed predictions snapshot straight off disk.

    The frontend hydrates its first paint from this (static/js/app.js::init), so
    this route MUST NOT call ``_ensure_metrics`` / load models — that is the whole
    point: instant first paint with zero compute on the request path. Missing
    file -> 404, and the frontend falls back to ``/api/predictions``. The file is
    produced off the request path by ``_write_snapshot_json`` (the compute +
    hydrate paths, both driven by the post_fork warm thread) and synced from S3
    at boot as an auxiliary cache file.
    """
    path = os.path.join(_PREDICTIONS_CACHE_DIR, _SNAPSHOT_JSON)
    if not os.path.isfile(path):
        return jsonify({"error": "snapshot not available"}), 404
    resp = send_file(path, mimetype="application/json", conditional=True)
    # Snapshot content changes on retrain; send_file stamps a size/mtime ETag,
    # so "no-cache" makes browsers revalidate cheaply (304 when unchanged)
    # rather than serving a stale snapshot after a model refresh.
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/api/metrics")
def api_metrics():
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))
    _, metrics = _get_data(scoring)
    return jsonify(metrics)


@app.route("/api/weeks")
def api_weeks():
    _ensure_base_data()
    results = _cache["results"]
    weeks = sorted(results["week"].unique().tolist())
    return jsonify({"weeks": [int(w) for w in weeks]})


@app.route("/api/teams")
def api_teams():
    _ensure_base_data()
    results = _cache["results"]
    teams = sorted(t for t in results["recent_team"].dropna().unique().tolist() if t)
    return jsonify({"teams": [str(t) for t in teams]})


@app.route("/api/player/<player_id>")
def api_player(player_id):
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))
    _ensure_base_data()
    results = _cache["results"]
    match = results[results["player_id"] == player_id]
    if match.empty:
        return jsonify({"error": "Player not found"}), 404
    _ensure_position_loaded(match.iloc[0]["position"])
    results = _cache["results"]
    df = results[results["player_id"] == player_id].sort_values("week")

    info = df.iloc[0]
    actual_col = _actual_col(scoring)
    pred_cols = {prefix: _pred_col(prefix, scoring) for prefix in _MODEL_PRED_PREFIXES}
    weekly_cols = ["week", actual_col, *pred_cols.values()]
    weekly_cols = [c for c in weekly_cols if c in df.columns]
    weekly = [
        {
            "week": int(r["week"]),
            "actual": _round_or_none(r.get(actual_col)),
            "ridge_pred": _safe_num(r.get(pred_cols["ridge"])),
            "nn_pred": _safe_num(r.get(pred_cols["nn"])),
            "attn_nn_pred": _safe_num(r.get(pred_cols["attn_nn"])),
            "lgbm_pred": _safe_num(r.get(pred_cols["lgbm"])),
        }
        for r in df[weekly_cols].to_dict(orient="records")
    ]

    return jsonify(
        {
            "player_id": player_id,
            "name": _safe_str(info["player_display_name"]),
            "position": _safe_str(info["position"]),
            "team": _safe_str(info["recent_team"]),
            "headshot": _safe_str(info.get("headshot_url", "")),
            "weekly": weekly,
            "season_avg": _safe_num(round(df[actual_col].mean(), 2)),
            "season_total": _safe_num(round(df[actual_col].sum(), 2)),
            "scoring": scoring,
        }
    )


@app.route("/api/predictions/breakdown")
def api_predictions_breakdown():
    """Per-stat breakdown for one player-week: each model's predicted raw stats
    plus the actual, for the expandable row in the predictions tab.

    Raw stats (yards / TDs / receptions / PA / YA / ...) are
    scoring-format-invariant, so this endpoint takes no ``scoring`` param and the
    sub-table is identical across PPR / half / standard. Columns are persisted by
    ``_apply_position_models`` (``pred_{model}_{t}`` / ``actual_{t}``); a model
    with no value for any target (e.g. lgbm for K/DST) is reported in
    ``unavailable_models``. A stale on-disk snapshot predating these columns
    degrades to ``{"components": [], "unavailable": true}`` rather than erroring.
    """
    player_id = request.args.get("player_id", "")
    week_arg = request.args.get("week", "")
    if not player_id:
        return jsonify({"error": "player_id required"}), 400
    try:
        week = int(week_arg)
    except (ValueError, TypeError):
        return jsonify({"error": f"Invalid week: {week_arg}"}), 400

    _ensure_base_data()
    df = _cache["results"]
    match = df[(df["player_id"] == player_id) & (df["week"] == week)]
    if match.empty:
        return jsonify({"error": "Player/week not found"}), 404
    position = _safe_str(match.iloc[0]["position"])
    if position not in POSITION_INFO:
        return jsonify({"error": f"Unknown position: {position}"}), 400

    # Ensure the position's models have run so the per-target columns are
    # populated (mirrors api_player). No-op on a disk-hydrated container.
    _ensure_position_loaded(position)
    df = _cache["results"]
    match = df[(df["player_id"] == player_id) & (df["week"] == week)]
    if match.empty:
        return jsonify({"error": "Player/week not found"}), 404
    r = match.iloc[0]
    cols = df.columns
    target_infos = POSITION_INFO[position]["targets"]

    # Stale-snapshot guard: if not one per-target column exists, the cache
    # predates this feature — report degraded rather than all-empty cells.
    expected = [f"actual_{t['key']}" for t in target_infos]
    for t in target_infos:
        expected.extend(f"pred_{prefix}_{t['key']}" for prefix in _MODEL_PRED_PREFIXES)
    if not any(c in cols for c in expected):
        return jsonify(
            {
                "player_id": player_id,
                "week": week,
                "position": position,
                "components": [],
                "unavailable_models": list(_MODEL_PRED_PREFIXES),
                "unavailable": True,
            }
        )

    components = []
    for t in target_infos:
        tkey = t["key"]
        actual_col = f"actual_{tkey}"
        comp = {
            "key": tkey,
            "label": t["label"],
            "unit": TARGET_UNITS.get(tkey, ""),
            "actual": _safe_num(r.get(actual_col)) if actual_col in cols else None,
        }
        for prefix in _MODEL_PRED_PREFIXES:
            pcol = f"pred_{prefix}_{tkey}"
            comp[prefix] = _safe_num(r.get(pcol)) if pcol in cols else None
        components.append(comp)

    # A model is unavailable if it has no value for any target (lgbm on K/DST, or
    # a position whose model failed to load).
    unavailable = [
        prefix for prefix in _MODEL_PRED_PREFIXES if all(c.get(prefix) is None for c in components)
    ]

    return jsonify(
        {
            "player_id": player_id,
            "week": week,
            "position": position,
            "components": components,
            "unavailable_models": unavailable,
            "unavailable": False,
        }
    )


@app.route("/api/top_players")
def api_top_players():
    position = request.args.get("position", "ALL")
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))

    df = _results_for_position(position, scoring)

    agg_dict = {
        "avg_actual": (_actual_col(scoring), "mean"),
        "avg_ridge": (_pred_col("ridge", scoring), "mean"),
        "avg_nn": (_pred_col("nn", scoring), "mean"),
        "avg_attn_nn": (_pred_col("attn_nn", scoring), "mean"),
        "avg_lgbm": (_pred_col("lgbm", scoring), "mean"),
        "games": ("week", "count"),
    }
    avg = (
        df.groupby(["player_id", "player_display_name", "position", "recent_team"])
        .agg(
            **agg_dict,
        )
        .reset_index()
    )
    avg = avg[avg["games"] >= 6]
    avg = avg.sort_values("avg_actual", ascending=False).head(25)

    rows = [
        {
            "player_id": _safe_str(r["player_id"]),
            "name": _safe_str(r["player_display_name"]),
            "position": _safe_str(r["position"]),
            "team": _safe_str(r["recent_team"]),
            "avg_actual": _round_or_none(r["avg_actual"]),
            "avg_ridge": _round_or_none(r["avg_ridge"]),
            "avg_nn": _round_or_none(r["avg_nn"]),
            "avg_attn_nn": _round_or_none(r["avg_attn_nn"]),
            "avg_lgbm": _round_or_none(r["avg_lgbm"]),
            "games": int(r["games"]),
        }
        for r in avg.to_dict(orient="records")
    ]

    return jsonify({"players": rows, "scoring": scoring})


@app.route("/api/weekly_accuracy")
def api_weekly_accuracy():
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))
    results, _ = _get_data(scoring)
    actual = results[_actual_col(scoring)].values
    # Per-row abs error; NaN where a model has no prediction (K/DST for lgbm,
    # plus any position whose model failed to load) so groupby.mean() excludes
    # those rows from that model's weekly MAE. attn_nn is trained for all six
    # positions, so K/DST attn_nn rows carry real preds and are not NaN.
    err_df = results.assign(
        _ridge_err=np.abs(actual - results[_pred_col("ridge", scoring)].values),
        _nn_err=np.abs(actual - results[_pred_col("nn", scoring)].values),
        _attn_nn_err=np.abs(actual - results[_pred_col("attn_nn", scoring)].values),
        _lgbm_err=np.abs(actual - results[_pred_col("lgbm", scoring)].values),
    )
    weekly = (
        err_df.groupby("week")
        .agg(
            ridge_mae=("_ridge_err", "mean"),
            nn_mae=("_nn_err", "mean"),
            attn_nn_mae=("_attn_nn_err", "mean"),
            lgbm_mae=("_lgbm_err", "mean"),
        )
        .round(3)
        .sort_index()
    )

    def _series_to_list(s):
        # Convert pandas Series with NaN -> list with None so jsonify works.
        return [None if pd.isna(v) else float(v) for v in s]

    return jsonify(
        {
            "weeks": [int(w) for w in weekly.index],
            "ridge_mae": _series_to_list(weekly["ridge_mae"]),
            "nn_mae": _series_to_list(weekly["nn_mae"]),
            "attn_nn_mae": _series_to_list(weekly["attn_nn_mae"]),
            "lgbm_mae": _series_to_list(weekly["lgbm_mae"]),
            "scoring": scoring,
        }
    )


@app.route("/api/position_details")
def api_position_details():
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))
    _get_data(scoring)  # ensure cache is populated
    details = _cache.get("position_details", {})
    result = {}
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        info = dict(POSITION_INFO[pos])
        pos_details = dict(details.get(pos, {}))
        # Swap target_metrics["total"] for the format-specific cached row.
        # Per-target raw-stat MAE is format-invariant and is left untouched.
        target_metrics = dict(pos_details.get("target_metrics", {}))
        total_by_format = target_metrics.pop("total_by_format", {}) or {}
        if total_by_format:
            target_metrics["total"] = total_by_format.get(scoring, target_metrics.get("total", {}))
        if target_metrics:
            pos_details["target_metrics"] = target_metrics
        info.update(pos_details)
        result[pos] = info
    return jsonify(result)


def _categorize_features(features):
    """Bucket feature names into human-readable categories by prefix."""
    weather_set = set(WEATHER_FEATURES_ALL) | {"game_wind", "game_temp"}
    categories = {
        "rolling": [],
        "prior_season": [],
        "ewma": [],
        "trend": [],
        "share": [],
        "matchup": [],
        "defense": [],
        "weather_vegas": [],
        "contextual": [],
        "other": [],
    }
    contextual_set = {
        "is_home",
        "week",
        "is_returning_from_absence",
        "days_rest",
        "practice_status",
        "game_status",
        "depth_chart_rank",
        "rest_days",
        "div_game",
        "spread_line",
    }
    for f in features:
        if f in weather_set:
            categories["weather_vegas"].append(f)
        elif f.startswith("rolling_"):
            categories["rolling"].append(f)
        elif f.startswith("prior_season_"):
            categories["prior_season"].append(f)
        elif f.startswith("ewma_"):
            categories["ewma"].append(f)
        elif f.startswith("trend_") or f.endswith("_trend"):
            categories["trend"].append(f)
        elif "share" in f or "hhi" in f:
            categories["share"].append(f)
        elif f.startswith("opp_def_") or (f.startswith("opp_") and "def" in f):
            categories["defense"].append(f)
        elif f.startswith("opp_") or f.endswith("_rank_vs_pos"):
            categories["matchup"].append(f)
        elif f in contextual_set:
            categories["contextual"].append(f)
        else:
            categories["other"].append(f)
    return {k: v for k, v in categories.items() if v}


def _position_arch_payload(pc, include_features, attn_history=None):
    """Build the per-position JSON payload for /api/model_architecture.

    ``pc`` is the position's :class:`~src.shared.position_config.PositionConfig`;
    ``include_features`` may be a categorized dict (QB/RB/WR/TE) or a flat list
    (K/DST contextual); either shape is normalized to categorized groups.
    """
    scheduler = pc.scheduler_type or "unknown"
    if scheduler == "cosine_warm_restarts":
        scheduler_str = (
            f"CosineAnnealingWarmRestarts(T0={pc.cosine_t0}, T_mult={pc.cosine_t_mult}, "
            f"eta_min={pc.cosine_eta_min})"
        )
    elif scheduler == "onecycle":
        scheduler_str = (
            f"OneCycleLR(max_lr={pc.onecycle_max_lr}, pct_start={pc.onecycle_pct_start})"
        )
    elif scheduler == "plateau":
        scheduler_str = "ReduceLROnPlateau"
    else:
        scheduler_str = str(scheduler)

    specific = pc.specific_features

    # Normalize feature groupings
    if isinstance(include_features, dict):
        grouped = {k: list(v) for k, v in include_features.items() if v}
        # Ensure position-specific features surface even if not keyed in dict
        if "specific" not in grouped and specific:
            grouped["specific"] = list(specific)
        flat_features = [f for group in grouped.values() for f in group]
    else:
        flat = list(include_features or [])
        grouped = {"specific": list(specific or [])}
        grouped.update(_categorize_features(flat))
        flat_features = list(specific or []) + flat

    payload = {
        "targets": list(pc.targets),
        "backbone_layers": list(pc.nn_backbone_layers),
        "head_hidden": pc.nn_head_hidden,
        "head_hidden_overrides": dict(pc.nn_head_hidden_overrides or {}),
        "dropout": pc.nn_dropout,
        "lr": pc.nn_lr,
        "weight_decay": pc.nn_weight_decay,
        "batch_size": pc.nn_batch_size,
        "epochs": pc.nn_epochs,
        "patience": pc.nn_patience,
        "scheduler": scheduler_str,
        "attention_enabled": bool(pc.train_attention_nn),
        "lightgbm_enabled": bool(pc.train_lightgbm),
        "feature_count": len(flat_features),
        "features": grouped,
    }
    if attn_history:
        payload["features"]["attention_history"] = list(attn_history)
    return payload


@app.route("/api/model_architecture")
def api_model_architecture():
    try:
        cfg_modules = {
            "QB": qb_cfg,
            "RB": rb_cfg,
            "WR": wr_cfg,
            "TE": te_cfg,
            "K": k_cfg,
            "DST": dst_cfg,
        }
        positions = {}
        for pos in _ALL_POSITIONS:
            pc = cfg_modules[pos].POSITION_CONFIG
            if pos in ("K", "DST"):
                positions[pos] = _position_arch_payload(pc, pc.contextual_features)
            else:
                positions[pos] = _position_arch_payload(
                    pc, pc.include_features, pc.attn_history_stats
                )
        return jsonify(
            {
                "overview": {
                    "framework": "PyTorch 2.11 + CUDA 12.6 (AWS Batch)",
                    "device": "CUDA if available, else CPU",
                    "data_splits": (
                        f"Train {min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)}, "
                        f"Val {', '.join(map(str, VAL_SEASONS))}, "
                        f"Test {', '.join(map(str, TEST_SEASONS))} "
                        "(K uses 2015+)"
                    ),
                    "ensemble": [
                        "Season-average baseline",
                        "Ridge multi-target",
                        "MultiHeadNet (dense)",
                        "MultiHeadNetWithHistory (attention)",
                        "LightGBM",
                    ],
                },
                "training_loop": {
                    "optimizer": "AdamW",
                    "loss": "MultiTargetLoss: per-target Huber or Poisson NLL + optional BCE on TD gate",
                    "gradient_clip": "clip_grad_norm_(max_norm=1.0)",
                    "feature_scaling": "StandardScaler, clipped to [-4, 4]",
                    "early_stopping": "Best loss-weighted val MAE restored on patience",
                    "checkpoint": "Best state_dict kept in memory, saved as .pt",
                },
                "positions": positions,
            }
        )
    except Exception:
        # Log server-side; don't leak str(e) to caller (py/stack-trace-exposure).
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/wiki/index")
def api_wiki_index():
    return jsonify(
        [
            {"slug": slug, "name": meta["name"], "group": meta["group"]}
            for slug, meta in WIKI_DOCS.items()
        ]
    )


@app.route("/api/wiki/<slug>")
def api_wiki_page(slug):
    if slug not in WIKI_DOCS:
        return jsonify({"error": "Unknown doc"}), 404
    meta = WIKI_DOCS[slug]
    html = _render_wiki_doc(slug)
    return jsonify({"slug": slug, "name": meta["name"], "group": meta["group"], "html": html})


@app.route("/api/comparison")
def api_comparison():
    """Our model (live) vs NFL.com / RotoWire (static), by position, for two
    subsets (all rostered players + top-30 per position). MAE/RMSE/R² each.
    """
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))
    experts = comparison._load_comparison_experts()
    if experts is None:
        return jsonify({"error": "Comparison data unavailable"}), 500

    # Live model metrics — best-effort. If models can't load (e.g. a cold box
    # with no artifacts), the experts still render and the model column shows —.
    model_source = "live"
    results = None
    try:
        _ensure_metrics()
        with _cache_lock:
            results = _cache.get("results")
    except Exception:
        traceback.print_exc()
        model_source = "unavailable"
    if results is None or getattr(results, "empty", True):
        model_source = "unavailable"

    top30_ids = experts.get("top30_ids", {})
    expert_subsets = experts.get("subsets", {})
    out_subsets = {}
    for subset in ("all", "top30"):
        out_subsets[subset] = {}
        pos_experts = expert_subsets.get(subset, {})
        for pos in _ALL_POSITIONS:
            cell = pos_experts.get(pos) or {}
            id_filter = set(map(str, top30_ids.get(pos, []))) if subset == "top30" else None
            model_block = (
                comparison._model_block_from_results(results, scoring, pos, id_filter)
                if model_source == "live"
                else None
            )
            out_subsets[subset][pos] = {
                "model": model_block,
                "nflcom": cell.get("nflcom"),
                "rotowire": cell.get("rotowire"),
            }

    # Live model residual σ on the 2025 test rows — the model-side counterpart to the
    # static expert_reliability block. Computed fresh each request (auto-updates on
    # retrain); None per position when models aren't loaded, so the tab degrades to
    # the expert columns alone.
    model_reliability = {
        pos: (
            comparison._model_reliability_from_results(results, scoring, pos)
            if model_source == "live"
            else None
        )
        for pos in _ALL_POSITIONS
    }
    # Per-projection prediction intervals (static, optional) ride along on the
    # same payload so the Comparison tab needs only one fetch.
    intervals = comparison._load_expert_intervals()

    return jsonify(
        {
            "scoring": scoring,
            "model_source": model_source,
            "generated_at": experts.get("generated_at"),
            "experts_meta": experts.get("experts_meta", {}),
            "top_n": experts.get("top_n"),
            "subsets": out_subsets,
            # Per-source residual σ: experts are static multi-season (2018–2025) from the
            # committed JSON; the model side is computed live on the 2025 test season
            # (its only leakage-free season). The frontend renders both on the 2025 basis
            # for an apples-to-apples table and shows each expert's full-archive σ on hover.
            "expert_reliability": experts.get("expert_reliability"),
            "model_reliability": model_reliability,
            "intervals": intervals,
        }
    )


@app.route("/api/benchmark_history")
def api_benchmark_history():
    """Return per-run summary rows for the History tab, newest first.

    Reads every top-level ``*.json`` under ``benchmark_history/``. Filesystem
    is the source of truth — the container is kept fresh by
    ``sync_benchmark_history_from_s3()`` at boot. On the active Batch path
    (``train-batch.yml``, ``BATCH_ACTIVE=true``) the forced ECS refresh was
    removed in PR #330, so a fresh sync happens on the next natural task roll /
    in-flight model-refresh; the rollback EC2 path (``train-ec2.yml``) still
    forces a redeploy via ``aws ecs update-service --force-new-deployment``.

    ``target_labels`` / ``target_units`` are static lookup maps the History
    tab's detailed mode uses to render per-target MAE rows ("Passing Yards …
    yds") from the raw target keys carried on each pill's ``per_target``.
    """
    return jsonify(
        {
            "repo_slug": benchmark_history._BENCHMARK_REPO_SLUG,
            "rows": benchmark_history._load_benchmark_history_rows(),
            "target_labels": benchmark_history._TARGET_LABELS,
            "target_units": TARGET_UNITS,
        }
    )


@app.route("/health")
def health():
    """Liveness probe for ALB + ECS.

    Three return shapes, matched on the joint state of ``positions_loaded``
    and ``position_load_errors``:

    - **200 ``{"status": "ok"}``** — happy path. Either steady state (every
      position loaded, no errors) OR cold-start before any load attempt
      (both maps empty). Cold start is treated as "alive but not yet
      ready"; the `post_fork` pre-warm thread populates ``positions_loaded``
      within ~30 s and ``/api/predictions`` would lazy-load on first hit
      either way.
    - **200 ``{"status": "degraded", ...}``** — at least one position is
      loaded AND ``position_load_errors`` is non-empty. The frontend
      renders a ``degraded_positions`` banner and the loaded positions'
      predictions are real, so the task is still useful. ``/api/predictions``
      already returns 200 + ``degraded_positions`` for this exact state —
      ``/health`` must agree, otherwise ALB recycles a still-serving task.
    - **503 ``{"status": "unhealthy", ...}``** — we have affirmatively
      failed: ``position_load_errors`` is non-empty AND no position is
      loaded (every attempt failed). ALB rotates us out; ECS replaces the
      task.

    Why no "503 when empty everything": that would 503 the ~30 s cold-start
    window before pre-warm completes (interval=10s × unhealthy_threshold=3
    ⇒ ALB deregisters before warmup finishes), killing every fresh task.
    The combined "errors AND nothing loaded" predicate distinguishes "we
    haven't tried yet" from "we tried and failed."

    Why 200 in the degraded case: a single transient S3 model-refresh
    failure used to poison ``position_load_errors`` and latch ``/health``
    to 503, recycling a task that was still serving five of six positions
    cleanly (alexfree.me, 2026-05-21 12:16 UTC, ~60 s ALB 5xx window).
    """
    loaded = _cache.get("positions_loaded") or set()
    errors = _cache.get("position_load_errors") or {}
    if errors and not loaded:
        return jsonify(
            {
                "status": "unhealthy",
                "position_load_errors": errors,
            }
        ), 503
    if errors:
        return jsonify(
            {
                "status": "degraded",
                "positions_loaded": sorted(loaded),
                "position_load_errors": errors,
            }
        ), 200
    return jsonify({"status": "ok"})


@app.route("/warm")
def warm():
    """Explicit, observable cache-warm trigger for CI / operators.

    Runs the same ``_ensure_metrics()`` the first ``/api/predictions?position=ALL``
    call would — so the heavy model-load + inference (or a disk/S3 hydrate)
    happens on THIS synthetic request instead of a real user's. After a
    retrain the in-flight refresh poller swaps new models into the running
    task; the next ``_ensure_metrics()`` recomputes and re-uploads the S3
    prediction cache (``_persist_cache_to_disk`` -> ``upload_predictions_cache_to_s3``).
    Hitting ``/warm`` from the post-train workflow moves that 30-60 s recompute
    off the user path AND guarantees the fresh cache lands in S3 before real
    traffic, so subsequent containers hydrate instantly (D14).

    Idempotent and cheap once warm: ``_ensure_metrics`` early-returns when the
    aggregate is cached and no position sentinel advanced (see app.py around
    its first line), so repeated probes are dict reads. Strictly lighter than
    the already-public ``/api/predictions?position=ALL`` — no new abuse surface.
    Mirrors that endpoint's fail-loud contract: if every position fails to
    load, ``_ensure_metrics`` raises and this returns 500, same as a real
    predictions request would.

    The reported ``fingerprint`` is content-based (``_compute_models_fingerprint``)
    so it is stable across containers for byte-identical models — CI can poll
    until it advances past a pre-retrain baseline to confirm the new models are
    live and the cache was rebuilt.
    """
    t0 = time.time()
    _ensure_metrics()
    elapsed = round(time.time() - t0, 3)
    fingerprint, _ = _compute_models_fingerprint()
    return jsonify(
        {
            "status": "ok",
            "fingerprint": fingerprint[:12],
            # ``_ensure_metrics`` populates ``results`` on every success path
            # (hydrate or compute), same guarantee ``/api/predictions`` relies
            # on. ``positions_loaded`` uses the defensive read ``/health`` uses.
            "rows": int(len(_cache["results"])),
            "positions_loaded": sorted(_cache.get("positions_loaded") or set()),
            "degraded_positions": _degraded_positions(),
            "elapsed_s": elapsed,
        }
    )


if __name__ == "__main__":
    # Production runs under gunicorn (see Dockerfile CMD); this branch is the
    # local dev entrypoint. Debug defaults off — set FLASK_DEBUG=1 for the
    # Werkzeug debugger locally. Bound to 127.0.0.1 so the debugger console is
    # never reachable off-box even when enabled.
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(debug=debug, host="127.0.0.1", port=5050, use_reloader=False)
