"""HTTP routes for the serving app — the 19 Flask handlers + their helpers.

Extracted from ``app.py`` during the serving decomposition (increment 5). Uses the
canonical Flask pattern: ``from src.serving.app import app`` for the ``@app.route``
decorators, and ``app_pkg`` for the shared mutable cache/locks (call-time attribute
access). ``app.py`` imports this module at the bottom, which registers the handlers.
"""

import os
import time
import traceback

import numpy as np
import pandas as pd
from flask import jsonify, render_template, request, send_file

import src.dst.config as dst_cfg
import src.k.config as k_cfg
import src.qb.config as qb_cfg
import src.rb.config as rb_cfg
import src.serving.app as app_pkg
import src.serving.core as core
import src.te.config as te_cfg
import src.wr.config as wr_cfg
from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
from src.serving import benchmark_history, comparison
from src.serving.app import app
from src.serving.metadata import _ALL_POSITIONS, POSITION_INFO
from src.serving.serialization import (
    _MODEL_PRED_PREFIXES,
    _actual_col,
    _pred_col,
    _records_to_player_rows,
    _round_or_none,
    _safe_num,
    _safe_str,
    _validate_scoring,
)
from src.serving.wiki import WIKI_DOCS, _render_wiki_doc
from src.shared.aggregate_targets import TARGET_UNITS
from src.shared.weather_features import WEATHER_FEATURES_ALL


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
        core._ensure_position_loaded(position)
        df = app_pkg._cache["results"]
        return df[df["position"] == position]
    results, _ = core._get_data(scoring)
    return results


@app.route("/api/predictions")
def api_predictions():
    position = request.args.get("position", "ALL")
    week = request.args.get("week", "ALL")
    search = request.args.get("search", "").strip().lower()
    sort_by = request.args.get("sort", "fantasy_points")
    order = request.args.get("order", "desc")
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))
    if position != "ALL" and position not in POSITION_INFO:
        return jsonify({"error": f"Unknown position: {position}"}), 400

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
    # "fantasy_points" (the default sort) maps to "actual" — the per-player rows
    # expose the realized fantasy total as "actual", not a separate
    # "fantasy_points" key. Without this alias the default silently fell through
    # to the else-branch (which also sorts by "actual", so same result, but the
    # whitelist no longer hides the default). (#498)
    if sort_by == "fantasy_points":
        sort_by = "actual"
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
            "degraded_positions": core._degraded_positions(),
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
    path = os.path.join(core._PREDICTIONS_CACHE_DIR, core._SNAPSHOT_JSON)
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
    _, metrics = core._get_data(scoring)
    return jsonify(metrics)


@app.route("/api/weeks")
def api_weeks():
    core._ensure_base_data()
    results = app_pkg._cache["results"]
    weeks = sorted(results["week"].unique().tolist())
    return jsonify({"weeks": [int(w) for w in weeks]})


@app.route("/api/teams")
def api_teams():
    core._ensure_base_data()
    results = app_pkg._cache["results"]
    teams = sorted(t for t in results["recent_team"].dropna().unique().tolist() if t)
    return jsonify({"teams": [str(t) for t in teams]})


@app.route("/api/player/<player_id>")
def api_player(player_id):
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))
    core._ensure_base_data()
    results = app_pkg._cache["results"]
    match = results[results["player_id"] == player_id]
    if match.empty:
        return jsonify({"error": "Player not found"}), 404
    core._ensure_position_loaded(match.iloc[0]["position"])
    results = app_pkg._cache["results"]
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

    core._ensure_base_data()
    df = app_pkg._cache["results"]
    match = df[(df["player_id"] == player_id) & (df["week"] == week)]
    if match.empty:
        return jsonify({"error": "Player/week not found"}), 404
    position = _safe_str(match.iloc[0]["position"])
    if position not in POSITION_INFO:
        return jsonify({"error": f"Unknown position: {position}"}), 400

    # Ensure the position's models have run so the per-target columns are
    # populated (mirrors api_player). No-op on a disk-hydrated container.
    core._ensure_position_loaded(position)
    df = app_pkg._cache["results"]
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
    if position != "ALL" and position not in POSITION_INFO:
        return jsonify({"error": f"Unknown position: {position}"}), 400

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
    results, _ = core._get_data(scoring)
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
    core._get_data(scoring)  # ensure cache is populated
    details = app_pkg._cache.get("position_details", {})
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
        # The attention NN can train at a distinct max_lr (e.g. K) via the
        # ``attn_`` scheduler prefix; the base string above only shows the dense
        # NN's, so surface the attention override when it differs (#845).
        attn_max_lr = getattr(pc, "attn_onecycle_max_lr", None)
        if pc.train_attention_nn and attn_max_lr is not None and attn_max_lr != pc.onecycle_max_lr:
            scheduler_str += f" · attention NN: OneCycleLR(max_lr={attn_max_lr})"
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
                positions[pos] = _position_arch_payload(
                    pc, pc.contextual_features, pc.attn_history_stats
                )
            else:
                positions[pos] = _position_arch_payload(
                    pc, pc.include_features, pc.attn_history_stats
                )
        return jsonify(
            {
                "overview": {
                    "framework": "PyTorch 2.12 + CUDA 12.6 (AWS Batch)",
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
        core._ensure_metrics()
        with app_pkg._cache_lock:
            results = app_pkg._cache.get("results")
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
            # One block per model (ridge/nn/attn_nn/lgbm), each None when that model
            # has no predictions for the slice; spread alongside the static experts.
            blocks = (
                comparison._model_blocks_from_results(results, scoring, pos, id_filter)
                if model_source == "live"
                else {}
            )
            out_subsets[subset][pos] = {
                **blocks,
                "nflcom": cell.get("nflcom"),
                "rotowire": cell.get("rotowire"),
            }

    # Live per-model residual σ on the 2025 test rows — the model-side counterpart to
    # the static expert_reliability block. Computed fresh each request (auto-updates on
    # retrain); each position maps to a per-model dict (or None when models aren't
    # loaded), so the tab degrades to the expert columns alone.
    model_reliability = {
        pos: (
            comparison._model_reliabilities_from_results(results, scoring, pos)
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
    loaded = app_pkg._cache.get("positions_loaded") or set()
    errors = app_pkg._cache.get("position_load_errors") or {}
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
    core._ensure_metrics()
    elapsed = round(time.time() - t0, 3)
    fingerprint, _ = core._compute_models_fingerprint()
    return jsonify(
        {
            "status": "ok",
            "fingerprint": fingerprint[:12],
            # ``_ensure_metrics`` populates ``results`` on every success path
            # (hydrate or compute), same guarantee ``/api/predictions`` relies
            # on. ``positions_loaded`` uses the defensive read ``/health`` uses.
            "rows": int(len(app_pkg._cache["results"])),
            "positions_loaded": sorted(app_pkg._cache.get("positions_loaded") or set()),
            "degraded_positions": core._degraded_positions(),
            "elapsed_s": elapsed,
        }
    )
