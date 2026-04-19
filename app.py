"""Flask web application for the Fantasy Football Points Predictor.

All predictions come from position-specific models (QB, RB, WR, TE, K, DST).
No general cross-position model is used.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")

import traceback

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request

import DST.dst_config as dst_cfg
import K.k_config as k_cfg
import QB.qb_config as qb_cfg
import RB.rb_config as rb_cfg
import TE.te_config as te_cfg
import WR.wr_config as wr_cfg
from DST.dst_config import DST_SPECIFIC_FEATURES, DST_TARGETS
from DST.dst_data import build_dst_data
from DST.dst_features import compute_dst_features
from K.k_config import K_SPECIFIC_FEATURES, K_TARGETS

# Per-position imports needed only for /api/model_architecture metadata and
# for K/DST data loaders (these have their own data pipelines, not in registry).
from K.k_data import kicker_season_split, load_kicker_data
from K.k_features import compute_k_features
from QB.qb_config import QB_SPECIFIC_FEATURES, QB_TARGETS
from RB.rb_config import RB_SPECIFIC_FEATURES, RB_TARGETS
from shared.model_sync import sync_models_from_s3
from shared.models import RidgeMultiTarget
from shared.neural_net import MultiHeadNet
from shared.registry import INFERENCE_REGISTRY as POSITION_REGISTRY
from shared.weather_features import WEATHER_FEATURES_ALL, merge_schedule_features
from src.config import SCORING_HALF_PPR, SCORING_STANDARD, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
from src.data.loader import compute_fantasy_points
from src.evaluation.metrics import compute_metrics, compute_positional_metrics
from TE.te_config import TE_SPECIFIC_FEATURES, TE_TARGETS
from WR.wr_config import WR_SPECIFIC_FEATURES, WR_TARGETS

sync_models_from_s3()

app = Flask(__name__)

_cache = {}


def _safe_num(v):
    """Convert NaN/inf to None so jsonify produces valid JSON (browsers reject NaN)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _safe_str(v, default=""):
    """Return default for NaN/None/non-string values."""
    if v is None:
        return default
    if isinstance(v, float) and not np.isfinite(v):
        return default
    return str(v)


_PLAYER_ROW_COLS = [
    "player_id",
    "player_display_name",
    "position",
    "recent_team",
    "week",
    "fantasy_points",
    "ridge_pred",
    "nn_pred",
    "headshot_url",
]


def _records_to_player_rows(df):
    cols = [c for c in _PLAYER_ROW_COLS if c in df.columns]
    return [
        {
            "player_id": _safe_str(r.get("player_id")),
            "name": _safe_str(r.get("player_display_name")),
            "position": _safe_str(r.get("position")),
            "team": _safe_str(r.get("recent_team")),
            "week": int(r["week"]),
            "actual": _safe_num(round(r["fantasy_points"], 2)),
            "ridge_pred": _safe_num(r.get("ridge_pred")),
            "nn_pred": _safe_num(r.get("nn_pred")),
            "headshot": _safe_str(r.get("headshot_url", "")),
        }
        for r in df[cols].to_dict(orient="records")
    ]


@app.errorhandler(Exception)
def handle_api_error(e):
    """Return JSON errors for /api/ routes, default HTML for others."""
    if request.path.startswith("/api/"):
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    raise e


# ---------------------------------------------------------------------------
# Position model metadata (static, for the UI)
# ---------------------------------------------------------------------------
POSITION_INFO = {
    "QB": {
        "label": "Quarterback",
        "targets": [
            {"key": "passing_floor", "label": "Passing Floor", "formula": "passing_yards x 0.04"},
            {"key": "rushing_floor", "label": "Rushing Floor", "formula": "rushing_yards x 0.1"},
            {"key": "td_points", "label": "TD Points", "formula": "pass_TD x 4 + rush_TD x 6"},
        ],
        "adjustments": "Interception penalty + fumble rate + receiving component (historical L8 rolling avg)",
        "specific_features": QB_SPECIFIC_FEATURES,
        "architecture": {
            "backbone": list(qb_cfg.QB_NN_BACKBONE_LAYERS),
            "head_hidden": qb_cfg.QB_NN_HEAD_HIDDEN,
        },
    },
    "RB": {
        "label": "Running Back",
        "targets": [
            {"key": "rushing_floor", "label": "Rushing Floor", "formula": "rushing_yards x 0.1"},
            {
                "key": "receiving_floor",
                "label": "Receiving Floor",
                "formula": "receptions x PPR + recv_yards x 0.1",
            },
            {"key": "td_points", "label": "TD Points", "formula": "rush_TD x 6 + recv_TD x 6"},
        ],
        "adjustments": "Fumble rate (historical L8 rolling avg)",
        "specific_features": list(RB_SPECIFIC_FEATURES),
        "architecture": {
            "backbone": list(rb_cfg.RB_NN_BACKBONE_LAYERS),
            "head_hidden": rb_cfg.RB_NN_HEAD_HIDDEN,
        },
    },
    "WR": {
        "label": "Wide Receiver",
        "targets": [
            {
                "key": "receiving_floor",
                "label": "Receiving Floor",
                "formula": "receptions x PPR + recv_yards x 0.1",
            },
            {"key": "rushing_floor", "label": "Rushing Floor", "formula": "rushing_yards x 0.1"},
            {"key": "td_points", "label": "TD Points", "formula": "recv_TD x 6 + rush_TD x 6"},
        ],
        "adjustments": "Fumble rate (historical L8 rolling avg)",
        "specific_features": list(WR_SPECIFIC_FEATURES),
        "architecture": {
            "backbone": list(wr_cfg.WR_NN_BACKBONE_LAYERS),
            "head_hidden": wr_cfg.WR_NN_HEAD_HIDDEN,
        },
    },
    "TE": {
        "label": "Tight End",
        "targets": [
            {
                "key": "receiving_floor",
                "label": "Receiving Floor",
                "formula": "receptions x PPR + recv_yards x 0.1",
            },
            {"key": "rushing_floor", "label": "Rushing Floor", "formula": "rushing_yards x 0.1"},
            {"key": "td_points", "label": "TD Points", "formula": "recv_TD x 6 + rush_TD x 6"},
        ],
        "adjustments": "Fumble rate (historical L8 rolling avg)",
        "specific_features": list(TE_SPECIFIC_FEATURES),
        "architecture": {
            "backbone": list(te_cfg.TE_NN_BACKBONE_LAYERS),
            "head_hidden": te_cfg.TE_NN_HEAD_HIDDEN,
        },
    },
    "K": {
        "label": "Kicker",
        "targets": [
            {
                "key": "fg_points",
                "label": "FG Points",
                "formula": "FG 0-39yd x 3 + FG 40-49yd x 4 + FG 50+yd x 5",
            },
            {"key": "pat_points", "label": "PAT Points", "formula": "PAT_made x 1"},
        ],
        "adjustments": "Miss penalty (rolling L8 FG/PAT miss rate)",
        "specific_features": list(K_SPECIFIC_FEATURES),
        "architecture": {
            "backbone": list(k_cfg.K_NN_BACKBONE_LAYERS),
            "head_hidden": k_cfg.K_NN_HEAD_HIDDEN,
        },
    },
    "DST": {
        "label": "Defense/Special Teams",
        "targets": [
            {
                "key": "defensive_scoring",
                "label": "Defensive Scoring",
                "formula": "sacks x 1 + INT x 2 + fum_rec x 2",
            },
            {"key": "td_points", "label": "TD Points", "formula": "ST_TD x 6"},
            {
                "key": "pts_allowed_bonus",
                "label": "Pts Allowed Bonus",
                "formula": "tiered: 0pts=+10 ... 35+=−4",
            },
        ],
        "adjustments": "Defensive TDs + safeties (nflreadr 2025-only; excluded from targets)",
        "specific_features": list(DST_SPECIFIC_FEATURES),
        "architecture": {
            "backbone": list(dst_cfg.DST_NN_BACKBONE_LAYERS),
            "head_hidden": dst_cfg.DST_NN_HEAD_HIDDEN,
        },
    },
}


def _compute_scoring_formats(df):
    if "fantasy_points_standard" not in df.columns:
        df["fantasy_points_standard"] = compute_fantasy_points(df, SCORING_STANDARD)
    if "fantasy_points_half_ppr" not in df.columns:
        df["fantasy_points_half_ppr"] = compute_fantasy_points(df, SCORING_HALF_PPR)


def _load_k_splits():
    """Load kicker data with features pre-computed on full dataset.

    K uses its own data pipeline because kicking stats (FG/PAT) are only
    available for 2025 in nflverse, and uses within-season temporal splits.
    """
    k_df = load_kicker_data()
    k_df = POSITION_REGISTRY["K"]["compute_targets_fn"](k_df)
    compute_k_features(k_df)
    return kicker_season_split(k_df)


def _load_dst_splits():
    """Load D/ST data with features pre-computed on full dataset.

    D/ST operates at team level (not player level), built from schedule
    scores and opponent offensive stats.
    """
    dst_df = build_dst_data()
    dst_df = POSITION_REGISTRY["DST"]["compute_targets_fn"](dst_df)
    compute_dst_features(dst_df)
    train = dst_df[dst_df["season"].isin(TRAIN_SEASONS)].copy()
    val = dst_df[dst_df["season"].isin(VAL_SEASONS)].copy()
    test = dst_df[dst_df["season"].isin(TEST_SEASONS)].copy()
    return train, val, test


def _apply_position_models(train, val, test, pos, results):
    """Load pre-trained position-specific models and write predictions into results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg = POSITION_REGISTRY[pos]

    targets = reg["targets"]
    model_dir = reg["model_dir"]

    # Prepare position data
    pos_train = reg["filter_fn"](train)
    pos_val = reg["filter_fn"](val)
    pos_test = reg["filter_fn"](test)

    pos_train = reg["compute_targets_fn"](pos_train)
    pos_val = reg["compute_targets_fn"](pos_val)
    pos_test = reg["compute_targets_fn"](pos_test)

    for _df in [pos_train, pos_val, pos_test]:
        merge_schedule_features(_df)

    pos_train, pos_val, pos_test = reg["add_features_fn"](pos_train, pos_val, pos_test)
    pos_train, pos_val, pos_test = reg["fill_nans_fn"](
        pos_train, pos_val, pos_test, reg["specific_features"]
    )

    feature_cols = reg["get_feature_columns_fn"]()
    adj = reg["compute_adjustment_fn"](pos_test)

    # Prepare features — fill missing columns with 0 (must match training dimension)
    missing_cols = [c for c in feature_cols if c not in pos_train.columns]
    if missing_cols:
        print(f"  WARNING: {pos} filling {len(missing_cols)} missing feature cols with 0")
        for col in missing_cols:
            for df in [pos_train, pos_val, pos_test]:
                df[col] = 0.0
    for df in [pos_train, pos_val, pos_test]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_test_pos = pos_test[feature_cols].values.astype(np.float32)

    # Ridge predictions — load failures propagate; global handler returns JSON 500.
    try:
        ridge = RidgeMultiTarget(target_names=targets)
        ridge.load(model_dir)
        ridge_preds = ridge.predict(X_test_pos)
        ridge_total = sum(ridge_preds[t] for t in targets) + adj.values
    except Exception as e:
        _cache.setdefault("position_load_errors", {})[f"{pos}_ridge"] = str(e)
        raise

    # NN predictions
    try:
        nn_scaler = joblib.load(f"{model_dir}/nn_scaler.pkl")
        X_test_scaled = np.clip(nn_scaler.transform(X_test_pos), -4, 4)

        nn_model = MultiHeadNet(
            input_dim=len(feature_cols), target_names=targets, **reg["nn_kwargs"]
        ).to(device)
        nn_model.load_state_dict(
            torch.load(f"{model_dir}/{reg['nn_file']}", map_location=device, weights_only=True)
        )
        nn_preds = nn_model.predict_numpy(X_test_scaled, device)
        nn_total = sum(nn_preds[t] for t in targets) + adj.values
    except Exception as e:
        _cache.setdefault("position_load_errors", {})[f"{pos}_nn"] = str(e)
        raise

    # Write into results
    pos_index = pos_test.index
    results.loc[pos_index, "ridge_pred"] = np.round(ridge_total, 2).astype(np.float32)
    results.loc[pos_index, "nn_pred"] = np.round(nn_total, 2).astype(np.float32)

    # Cache per-target metrics for /api/position_details
    target_metrics = {}
    for t in targets:
        if t in pos_test.columns:
            actual_t = pos_test[t].values
            tm = {
                "ridge_mae": round(float(np.mean(np.abs(ridge_preds[t] - actual_t))), 3),
                "nn_mae": round(float(np.mean(np.abs(nn_preds[t] - actual_t))), 3),
            }
            target_metrics[t] = tm
    total_actual = pos_test["fantasy_points"].values
    total_tm = {
        "ridge_mae": round(float(np.mean(np.abs(ridge_total - total_actual))), 3),
        "nn_mae": round(float(np.mean(np.abs(nn_total - total_actual))), 3),
    }
    target_metrics["total"] = total_tm
    _cache.setdefault("position_details", {})[pos] = {
        "n_features": len(feature_cols),
        "n_samples_test": len(pos_test),
        "target_metrics": target_metrics,
    }


_ALL_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]


def _ensure_base_data():
    """Load splits + build empty results frame. Idempotent. No model loads."""
    if _cache.get("base_loaded") or "results" in _cache:
        return

    print("Loading data...")
    train = pd.read_parquet("data/splits/train.parquet")
    val = pd.read_parquet("data/splits/val.parquet")
    test = pd.read_parquet("data/splits/test.parquet")

    for df in [train, val, test]:
        _compute_scoring_formats(df)

    print("Loading kicker data...")
    k_train, k_val, k_test = _load_k_splits()
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
        "fantasy_points_floor",
    ]
    keep_cols = [c for c in keep_cols if c in test.columns]
    results = test[keep_cols].copy()

    for pos_test_df in [k_test, dst_test]:
        offset = results.index.max() + 1
        pos_rows = pd.DataFrame(index=range(offset, offset + len(pos_test_df)))
        for col in keep_cols:
            if col in pos_test_df.columns:
                pos_rows[col] = pos_test_df[col].values
            elif col in ("fantasy_points_half_ppr", "fantasy_points_standard"):
                pos_rows[col] = pos_test_df["fantasy_points"].values
            elif col == "headshot_url":
                pos_rows[col] = ""
            else:
                pos_rows[col] = np.nan
        pos_test_df.index = pos_rows.index
        results = pd.concat([results, pos_rows])

    results["ridge_pred"] = 0.0
    results["nn_pred"] = 0.0

    _cache["splits"] = {
        "QB": (train, val, test),
        "RB": (train, val, test),
        "WR": (train, val, test),
        "TE": (train, val, test),
        "K": (k_train, k_val, k_test),
        "DST": (dst_train, dst_val, dst_test),
    }
    _cache["results"] = results
    _cache["positions_loaded"] = set()
    _cache["base_loaded"] = True


def _ensure_position_loaded(pos):
    """Apply position-specific model. Idempotent."""
    _ensure_base_data()
    if "splits" not in _cache:
        return
    if pos in _cache["positions_loaded"]:
        return
    train, val, test = _cache["splits"][pos]
    print(f"Applying {pos}-specific model...")
    _apply_position_models(train, val, test, pos, _cache["results"])
    _cache["positions_loaded"].add(pos)


def _ensure_all_positions_loaded():
    for pos in _ALL_POSITIONS:
        _ensure_position_loaded(pos)


def _ensure_metrics():
    if "metrics" in _cache:
        return
    _ensure_all_positions_loaded()
    results = _cache["results"]
    y_all = results["fantasy_points"].values
    metrics = {}
    for name, pred_col in [("Ridge Regression", "ridge_pred"), ("Neural Network", "nn_pred")]:
        preds_full = results[pred_col].fillna(0).values
        overall = compute_metrics(y_all, preds_full)
        pos_df = results[["position"]].copy()
        pos_df["pred"] = preds_full
        pos_df["actual"] = y_all
        pos_metrics = compute_positional_metrics(pos_df, "pred", "actual")
        metrics[name] = {
            "overall": {k: round(v, 4) for k, v in overall.items()},
            "by_position": pos_metrics.to_dict(orient="records"),
        }
    _cache["metrics"] = metrics
    print("Ready!")


def _get_data():
    """Full load: all positions + metrics. Backward-compatible."""
    _ensure_metrics()
    return _cache["results"], _cache["metrics"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predictions")
def api_predictions():
    position = request.args.get("position", "ALL")
    week = request.args.get("week", "ALL")
    search = request.args.get("search", "").strip().lower()
    sort_by = request.args.get("sort", "fantasy_points")
    order = request.args.get("order", "desc")

    if position != "ALL":
        _ensure_position_loaded(position)
        df = _cache["results"]
        df = df[df["position"] == position]
    else:
        results, _ = _get_data()
        df = results
    if week != "ALL":
        try:
            df = df[df["week"] == int(week)]
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid week: {week}"}), 400
    if search:
        df = df[df["player_display_name"].str.lower().str.contains(search, na=False, regex=False)]

    rows = _records_to_player_rows(df)

    reverse = order == "desc"
    if sort_by in ("actual", "ridge_pred", "nn_pred", "week"):
        rows.sort(key=lambda x: x.get(sort_by) or 0, reverse=reverse)
    else:
        rows.sort(key=lambda x: x.get("actual") or 0, reverse=reverse)

    return jsonify({"players": rows, "total": len(rows)})


@app.route("/api/metrics")
def api_metrics():
    _, metrics = _get_data()
    return jsonify(metrics)


@app.route("/api/weeks")
def api_weeks():
    _ensure_base_data()
    results = _cache["results"]
    weeks = sorted(results["week"].unique().tolist())
    return jsonify({"weeks": [int(w) for w in weeks]})


@app.route("/api/player/<player_id>")
def api_player(player_id):
    _ensure_base_data()
    results = _cache["results"]
    match = results[results["player_id"] == player_id]
    if match.empty:
        return jsonify({"error": "Player not found"}), 404
    _ensure_position_loaded(match.iloc[0]["position"])
    results = _cache["results"]
    df = results[results["player_id"] == player_id].sort_values("week")

    info = df.iloc[0]
    weekly = [
        {
            "week": int(r["week"]),
            "actual": _safe_num(round(r["fantasy_points"], 2)),
            "ridge_pred": _safe_num(r["ridge_pred"]),
            "nn_pred": _safe_num(r["nn_pred"]),
        }
        for r in df[["week", "fantasy_points", "ridge_pred", "nn_pred"]].to_dict(orient="records")
    ]

    return jsonify(
        {
            "player_id": player_id,
            "name": _safe_str(info["player_display_name"]),
            "position": _safe_str(info["position"]),
            "team": _safe_str(info["recent_team"]),
            "headshot": _safe_str(info.get("headshot_url", "")),
            "weekly": weekly,
            "season_avg": _safe_num(round(df["fantasy_points"].mean(), 2)),
            "season_total": _safe_num(round(df["fantasy_points"].sum(), 2)),
        }
    )


@app.route("/api/top_players")
def api_top_players():
    position = request.args.get("position", "ALL")

    if position != "ALL":
        _ensure_position_loaded(position)
        df = _cache["results"]
        df = df[df["position"] == position]
    else:
        results, _ = _get_data()
        df = results

    agg_dict = {
        "avg_actual": ("fantasy_points", "mean"),
        "avg_ridge": ("ridge_pred", "mean"),
        "avg_nn": ("nn_pred", "mean"),
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
            "avg_actual": _safe_num(round(r["avg_actual"], 2)),
            "avg_ridge": _safe_num(round(r["avg_ridge"], 2)),
            "avg_nn": _safe_num(round(r["avg_nn"], 2)),
            "games": int(r["games"]),
        }
        for r in avg.to_dict(orient="records")
    ]

    return jsonify({"players": rows})


@app.route("/api/weekly_accuracy")
def api_weekly_accuracy():
    results, _ = _get_data()
    actual = results["fantasy_points"].values
    weekly = (
        results.assign(
            _ridge_err=np.abs(actual - results["ridge_pred"].values),
            _nn_err=np.abs(actual - results["nn_pred"].values),
        )
        .groupby("week")
        .agg(ridge_mae=("_ridge_err", "mean"), nn_mae=("_nn_err", "mean"))
        .round(3)
        .sort_index()
    )
    return jsonify(
        {
            "weeks": [int(w) for w in weekly.index],
            "ridge_mae": weekly["ridge_mae"].tolist(),
            "nn_mae": weekly["nn_mae"].tolist(),
        }
    )


@app.route("/api/position_details")
def api_position_details():
    _get_data()  # ensure cache is populated
    details = _cache.get("position_details", {})
    result = {}
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        info = dict(POSITION_INFO[pos])
        info.update(details.get(pos, {}))
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


def _position_arch_payload(pos, cfg, specific, targets, include_features, attn_history=None):
    """Build the per-position JSON payload for /api/model_architecture.

    `include_features` may be a categorized dict (QB/RB/WR/TE) or a flat list
    (K/DST contextual); either shape is normalized to categorized groups.
    """

    def get(name, default=None):
        return getattr(cfg, name, default)

    prefix = pos.upper()

    # Scheduler summary string
    scheduler = get(f"{prefix}_SCHEDULER_TYPE", "unknown")
    if scheduler == "cosine_warm_restarts":
        t0 = get(f"{prefix}_COSINE_T0", "?")
        tm = get(f"{prefix}_COSINE_T_MULT", "?")
        em = get(f"{prefix}_COSINE_ETA_MIN", "?")
        scheduler_str = f"CosineAnnealingWarmRestarts(T0={t0}, T_mult={tm}, eta_min={em})"
    elif scheduler == "onecycle":
        mx = get(f"{prefix}_ONECYCLE_MAX_LR", "?")
        ps = get(f"{prefix}_ONECYCLE_PCT_START", "?")
        scheduler_str = f"OneCycleLR(max_lr={mx}, pct_start={ps})"
    elif scheduler == "plateau":
        scheduler_str = "ReduceLROnPlateau"
    else:
        scheduler_str = str(scheduler)

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
        "targets": list(targets),
        "backbone_layers": list(get(f"{prefix}_NN_BACKBONE_LAYERS", [])),
        "head_hidden": get(f"{prefix}_NN_HEAD_HIDDEN"),
        "head_hidden_overrides": dict(get(f"{prefix}_NN_HEAD_HIDDEN_OVERRIDES", {}) or {}),
        "dropout": get(f"{prefix}_NN_DROPOUT"),
        "lr": get(f"{prefix}_NN_LR"),
        "weight_decay": get(f"{prefix}_NN_WEIGHT_DECAY"),
        "batch_size": get(f"{prefix}_NN_BATCH_SIZE"),
        "epochs": get(f"{prefix}_NN_EPOCHS"),
        "patience": get(f"{prefix}_NN_PATIENCE"),
        "scheduler": scheduler_str,
        "huber_delta_total": (get(f"{prefix}_HUBER_DELTAS", {}) or {}).get("total"),
        "attention_enabled": bool(get(f"{prefix}_TRAIN_ATTENTION_NN", False)),
        "lightgbm_enabled": bool(get(f"{prefix}_TRAIN_LIGHTGBM", False)),
        "feature_count": len(flat_features),
        "features": grouped,
    }
    if attn_history:
        payload["features"]["attention_history"] = list(attn_history)
    return payload


@app.route("/api/model_architecture")
def api_model_architecture():
    try:
        positions = {
            "QB": _position_arch_payload(
                "QB",
                qb_cfg,
                QB_SPECIFIC_FEATURES,
                QB_TARGETS,
                getattr(qb_cfg, "QB_INCLUDE_FEATURES", []),
                getattr(qb_cfg, "QB_ATTN_HISTORY_STATS", None),
            ),
            "RB": _position_arch_payload(
                "RB",
                rb_cfg,
                RB_SPECIFIC_FEATURES,
                RB_TARGETS,
                getattr(rb_cfg, "RB_INCLUDE_FEATURES", []),
                getattr(rb_cfg, "RB_ATTN_HISTORY_STATS", None),
            ),
            "WR": _position_arch_payload(
                "WR",
                wr_cfg,
                WR_SPECIFIC_FEATURES,
                WR_TARGETS,
                getattr(wr_cfg, "WR_INCLUDE_FEATURES", []),
                getattr(wr_cfg, "WR_ATTN_HISTORY_STATS", None),
            ),
            "TE": _position_arch_payload(
                "TE",
                te_cfg,
                TE_SPECIFIC_FEATURES,
                TE_TARGETS,
                getattr(te_cfg, "TE_INCLUDE_FEATURES", []),
                getattr(te_cfg, "TE_ATTN_HISTORY_STATS", None),
            ),
            "K": _position_arch_payload(
                "K",
                k_cfg,
                K_SPECIFIC_FEATURES,
                K_TARGETS,
                getattr(k_cfg, "K_CONTEXTUAL_FEATURES", []),
            ),
            "DST": _position_arch_payload(
                "DST",
                dst_cfg,
                DST_SPECIFIC_FEATURES,
                DST_TARGETS,
                getattr(dst_cfg, "DST_CONTEXTUAL_FEATURES", []),
            ),
        }
        return jsonify(
            {
                "overview": {
                    "framework": "PyTorch 2.11 + CUDA 12.6 (AWS Batch)",
                    "device": "CUDA if available, else CPU",
                    "data_splits": "Train 2012-2023, Val 2024, Test 2025 (K uses 2015+)",
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
                    "loss": "MultiTargetLoss: sum of per-target Huber + w_total * Huber(total) + optional BCE on TD gate",
                    "gradient_clip": "clip_grad_norm_(max_norm=1.0)",
                    "feature_scaling": "StandardScaler, clipped to [-4, 4]",
                    "early_stopping": "Best val_mae_total restored on patience",
                    "checkpoint": "Best state_dict kept in memory, saved as .pt",
                },
                "positions": positions,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    errors = _cache.get("position_load_errors")
    if errors:
        return jsonify({"status": "degraded", "position_load_errors": errors}), 503
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5050, use_reloader=False)
