"""Flask web application for the Fantasy Football Points Predictor.

All predictions come from position-specific models (QB, RB, WR, TE, K, DST).
No general cross-position model is used.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")

import traceback
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import torch
import joblib

from src.config import SCORING_STANDARD, SCORING_HALF_PPR, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS
from src.features.engineer import get_feature_columns, fill_nans_safe
from src.data.loader import compute_fantasy_points
from src.evaluation.metrics import compute_metrics, compute_positional_metrics

from shared.models import RidgeMultiTarget
from shared.neural_net import MultiHeadNet
from shared.weather_features import merge_schedule_features

# --- Position-specific imports (data, targets, features, config) ---
from QB.qb_data import filter_to_qb
from QB.qb_targets import compute_qb_targets, compute_qb_adjustment
from QB.qb_features import add_qb_specific_features, get_qb_feature_columns, fill_qb_nans
from QB.qb_config import (
    QB_TARGETS, QB_SPECIFIC_FEATURES,
    QB_NN_BACKBONE_LAYERS, QB_NN_HEAD_HIDDEN, QB_NN_DROPOUT,
)

from RB.rb_data import filter_to_rb
from RB.rb_targets import compute_rb_targets, compute_fumble_adjustment
from RB.rb_features import add_rb_specific_features, get_rb_feature_columns, fill_rb_nans
from RB.rb_config import (
    RB_TARGETS, RB_SPECIFIC_FEATURES,
    RB_NN_BACKBONE_LAYERS, RB_NN_HEAD_HIDDEN, RB_NN_DROPOUT,
)

from WR.wr_data import filter_to_wr
from WR.wr_targets import compute_wr_targets, compute_wr_fumble_adjustment
from WR.wr_features import add_wr_specific_features, get_wr_feature_columns, fill_wr_nans
from WR.wr_config import (
    WR_TARGETS, WR_SPECIFIC_FEATURES,
    WR_NN_BACKBONE_LAYERS, WR_NN_HEAD_HIDDEN, WR_NN_DROPOUT,
)

from TE.te_data import filter_to_te
from TE.te_targets import compute_te_targets, compute_te_fumble_adjustment
from TE.te_features import add_te_specific_features, get_te_feature_columns, fill_te_nans
from TE.te_config import (
    TE_TARGETS, TE_SPECIFIC_FEATURES,
    TE_NN_BACKBONE_LAYERS, TE_NN_HEAD_HIDDEN, TE_NN_HEAD_HIDDEN_OVERRIDES,
    TE_NN_DROPOUT,
)

from K.k_data import filter_to_k, load_kicker_data, kicker_season_split
from K.k_targets import compute_k_targets, compute_k_miss_adjustment
from K.k_features import compute_k_features, add_k_specific_features, get_k_feature_columns, fill_k_nans
from K.k_config import (
    K_TARGETS, K_SPECIFIC_FEATURES,
    K_NN_BACKBONE_LAYERS, K_NN_HEAD_HIDDEN, K_NN_DROPOUT,
)

from DST.dst_data import filter_to_dst, build_dst_data
from DST.dst_targets import compute_dst_targets, compute_dst_adjustment
from DST.dst_features import compute_dst_features, add_dst_specific_features, get_dst_feature_columns, fill_dst_nans
from DST.dst_config import (
    DST_TARGETS, DST_SPECIFIC_FEATURES,
    DST_NN_BACKBONE_LAYERS, DST_NN_HEAD_HIDDEN, DST_NN_HEAD_HIDDEN_OVERRIDES,
    DST_NN_DROPOUT, DST_NN_NON_NEGATIVE_TARGETS,
)

# Full config modules (used by /api/model_architecture to surface per-position
# hyperparameters, feature lists, and training flags to the frontend).
import QB.qb_config as qb_cfg
import RB.rb_config as rb_cfg
import WR.wr_config as wr_cfg
import TE.te_config as te_cfg
import K.k_config as k_cfg
import DST.dst_config as dst_cfg
from shared.weather_features import WEATHER_FEATURES_ALL

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


@app.errorhandler(Exception)
def handle_api_error(e):
    """Return JSON errors for /api/ routes, default HTML for others."""
    if request.path.startswith("/api/"):
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    raise e


# ---------------------------------------------------------------------------
# Position registry — replaces per-position if/elif chains
# ---------------------------------------------------------------------------
POSITION_REGISTRY = {
    "QB": {
        "targets": QB_TARGETS,
        "specific_features": QB_SPECIFIC_FEATURES,
        "filter_fn": filter_to_qb,
        "compute_targets_fn": compute_qb_targets,
        "add_features_fn": add_qb_specific_features,
        "fill_nans_fn": fill_qb_nans,
        "get_feature_columns_fn": get_qb_feature_columns,
        "compute_adjustment_fn": compute_qb_adjustment,
        "model_dir": "QB/outputs/models",
        "nn_file": "qb_multihead_nn.pt",
        "nn_kwargs": dict(backbone_layers=QB_NN_BACKBONE_LAYERS, head_hidden=QB_NN_HEAD_HIDDEN, dropout=QB_NN_DROPOUT),
    },
    "RB": {
        "targets": RB_TARGETS,
        "specific_features": RB_SPECIFIC_FEATURES,
        "filter_fn": filter_to_rb,
        "compute_targets_fn": compute_rb_targets,
        "add_features_fn": add_rb_specific_features,
        "fill_nans_fn": fill_rb_nans,
        "get_feature_columns_fn": get_rb_feature_columns,
        "compute_adjustment_fn": compute_fumble_adjustment,
        "model_dir": "RB/outputs/models",
        "nn_file": "rb_multihead_nn.pt",
        "nn_kwargs": dict(backbone_layers=RB_NN_BACKBONE_LAYERS, head_hidden=RB_NN_HEAD_HIDDEN, dropout=RB_NN_DROPOUT),
    },
    "WR": {
        "targets": WR_TARGETS,
        "specific_features": WR_SPECIFIC_FEATURES,
        "filter_fn": filter_to_wr,
        "compute_targets_fn": compute_wr_targets,
        "add_features_fn": add_wr_specific_features,
        "fill_nans_fn": fill_wr_nans,
        "get_feature_columns_fn": get_wr_feature_columns,
        "compute_adjustment_fn": compute_wr_fumble_adjustment,
        "model_dir": "WR/outputs/models",
        "nn_file": "wr_multihead_nn.pt",
        "nn_kwargs": dict(backbone_layers=WR_NN_BACKBONE_LAYERS, head_hidden=WR_NN_HEAD_HIDDEN, dropout=WR_NN_DROPOUT),
    },
    "TE": {
        "targets": TE_TARGETS,
        "specific_features": TE_SPECIFIC_FEATURES,
        "filter_fn": filter_to_te,
        "compute_targets_fn": compute_te_targets,
        "add_features_fn": add_te_specific_features,
        "fill_nans_fn": fill_te_nans,
        "get_feature_columns_fn": get_te_feature_columns,
        "compute_adjustment_fn": compute_te_fumble_adjustment,
        "model_dir": "TE/outputs/models",
        "nn_file": "te_multihead_nn.pt",
        "nn_kwargs": dict(
            backbone_layers=TE_NN_BACKBONE_LAYERS, head_hidden=TE_NN_HEAD_HIDDEN,
            dropout=TE_NN_DROPOUT, head_hidden_overrides=TE_NN_HEAD_HIDDEN_OVERRIDES,
        ),
    },
    "K": {
        "targets": K_TARGETS,
        "specific_features": K_SPECIFIC_FEATURES,
        "filter_fn": filter_to_k,
        "compute_targets_fn": compute_k_targets,
        "add_features_fn": add_k_specific_features,
        "fill_nans_fn": fill_k_nans,
        "get_feature_columns_fn": get_k_feature_columns,
        "compute_adjustment_fn": compute_k_miss_adjustment,
        "model_dir": "K/outputs/models",
        "nn_file": "k_multihead_nn.pt",
        "nn_kwargs": dict(backbone_layers=K_NN_BACKBONE_LAYERS, head_hidden=K_NN_HEAD_HIDDEN, dropout=K_NN_DROPOUT),
    },
    "DST": {
        "targets": DST_TARGETS,
        "specific_features": DST_SPECIFIC_FEATURES,
        "filter_fn": filter_to_dst,
        "compute_targets_fn": compute_dst_targets,
        "add_features_fn": add_dst_specific_features,
        "fill_nans_fn": fill_dst_nans,
        "get_feature_columns_fn": get_dst_feature_columns,
        "compute_adjustment_fn": compute_dst_adjustment,
        "model_dir": "DST/outputs/models",
        "nn_file": "dst_multihead_nn.pt",
        "nn_kwargs": dict(
            backbone_layers=DST_NN_BACKBONE_LAYERS, head_hidden=DST_NN_HEAD_HIDDEN,
            dropout=DST_NN_DROPOUT, head_hidden_overrides=DST_NN_HEAD_HIDDEN_OVERRIDES,
            non_negative_targets=DST_NN_NON_NEGATIVE_TARGETS,
        ),
    },
}

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
        "architecture": {"backbone": list(QB_NN_BACKBONE_LAYERS), "head_hidden": QB_NN_HEAD_HIDDEN},
    },
    "RB": {
        "label": "Running Back",
        "targets": [
            {"key": "rushing_floor", "label": "Rushing Floor", "formula": "rushing_yards x 0.1"},
            {"key": "receiving_floor", "label": "Receiving Floor", "formula": "receptions x PPR + recv_yards x 0.1"},
            {"key": "td_points", "label": "TD Points", "formula": "rush_TD x 6 + recv_TD x 6"},
        ],
        "adjustments": "Fumble rate (historical L8 rolling avg)",
        "specific_features": list(RB_SPECIFIC_FEATURES),
        "architecture": {"backbone": list(RB_NN_BACKBONE_LAYERS), "head_hidden": RB_NN_HEAD_HIDDEN},
    },
    "WR": {
        "label": "Wide Receiver",
        "targets": [
            {"key": "receiving_floor", "label": "Receiving Floor", "formula": "receptions x PPR + recv_yards x 0.1"},
            {"key": "rushing_floor", "label": "Rushing Floor", "formula": "rushing_yards x 0.1"},
            {"key": "td_points", "label": "TD Points", "formula": "recv_TD x 6 + rush_TD x 6"},
        ],
        "adjustments": "Fumble rate (historical L8 rolling avg)",
        "specific_features": list(WR_SPECIFIC_FEATURES),
        "architecture": {"backbone": list(WR_NN_BACKBONE_LAYERS), "head_hidden": WR_NN_HEAD_HIDDEN},
    },
    "TE": {
        "label": "Tight End",
        "targets": [
            {"key": "receiving_floor", "label": "Receiving Floor", "formula": "receptions x PPR + recv_yards x 0.1"},
            {"key": "rushing_floor", "label": "Rushing Floor", "formula": "rushing_yards x 0.1"},
            {"key": "td_points", "label": "TD Points", "formula": "recv_TD x 6 + rush_TD x 6"},
        ],
        "adjustments": "Fumble rate (historical L8 rolling avg)",
        "specific_features": list(TE_SPECIFIC_FEATURES),
        "architecture": {"backbone": list(TE_NN_BACKBONE_LAYERS), "head_hidden": TE_NN_HEAD_HIDDEN},
    },
    "K": {
        "label": "Kicker",
        "targets": [
            {"key": "fg_points", "label": "FG Points", "formula": "FG 0-39yd x 3 + FG 40-49yd x 4 + FG 50+yd x 5"},
            {"key": "pat_points", "label": "PAT Points", "formula": "PAT_made x 1"},
        ],
        "adjustments": "Miss penalty (rolling L8 FG/PAT miss rate)",
        "specific_features": list(K_SPECIFIC_FEATURES),
        "architecture": {"backbone": list(K_NN_BACKBONE_LAYERS), "head_hidden": K_NN_HEAD_HIDDEN},
    },
    "DST": {
        "label": "Defense/Special Teams",
        "targets": [
            {"key": "defensive_scoring", "label": "Defensive Scoring", "formula": "sacks x 1 + INT x 2 + fum_rec x 2"},
            {"key": "td_points", "label": "TD Points", "formula": "ST_TD x 6"},
            {"key": "pts_allowed_bonus", "label": "Pts Allowed Bonus", "formula": "tiered: 0pts=+10 ... 35+=−4"},
        ],
        "adjustments": "Defensive TDs + safeties (nflreadr 2025-only; excluded from targets)",
        "specific_features": list(DST_SPECIFIC_FEATURES),
        "architecture": {"backbone": list(DST_NN_BACKBONE_LAYERS), "head_hidden": DST_NN_HEAD_HIDDEN},
    },
}


def _compute_scoring_formats(df):
    if "fantasy_points_standard" not in df.columns:
        df["fantasy_points_standard"] = compute_fantasy_points(df, SCORING_STANDARD)
    if "fantasy_points_half_ppr" not in df.columns:
        df["fantasy_points_half_ppr"] = compute_fantasy_points(df, SCORING_HALF_PPR)
    return df


def _load_k_splits():
    """Load kicker data with features pre-computed on full dataset.

    K uses its own data pipeline because kicking stats (FG/PAT) are only
    available for 2025 in nflverse, and uses within-season temporal splits.
    """
    k_df = load_kicker_data()
    k_df = compute_k_targets(k_df)
    compute_k_features(k_df)
    return kicker_season_split(k_df)


def _load_dst_splits():
    """Load D/ST data with features pre-computed on full dataset.

    D/ST operates at team level (not player level), built from schedule
    scores and opponent offensive stats.
    """
    dst_df = build_dst_data()
    dst_df = compute_dst_targets(dst_df)
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

    # Ridge predictions
    try:
        ridge = RidgeMultiTarget(target_names=targets)
        ridge.load(model_dir)
        ridge_preds = ridge.predict(X_test_pos)
        ridge_total = sum(ridge_preds[t] for t in targets) + adj.values
    except Exception as e:
        print(f"  ERROR: Failed to load Ridge model for {pos}: {e}")
        return

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
        print(f"  ERROR: Failed to load NN model for {pos}: {e}")
        return

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


def _get_data():
    """Load data and generate all position-specific predictions."""
    if "results" in _cache:
        return _cache["results"], _cache["metrics"]

    print("Loading data...")
    train = pd.read_parquet("data/splits/train.parquet")
    val = pd.read_parquet("data/splits/val.parquet")
    test = pd.read_parquet("data/splits/test.parquet")

    for df in [train, val, test]:
        _compute_scoring_formats(df)

    # K and DST use their own data pipelines (not the general splits)
    print("Loading kicker data...")
    k_train, k_val, k_test = _load_k_splits()
    print("Loading D/ST data...")
    dst_train, dst_val, dst_test = _load_dst_splits()

    # Build results frame from general test + K/DST test data
    keep_cols = [
        "player_id", "player_display_name", "position", "recent_team",
        "season", "week", "headshot_url",
        "fantasy_points", "fantasy_points_half_ppr", "fantasy_points_standard",
        "fantasy_points_floor",
    ]
    keep_cols = [c for c in keep_cols if c in test.columns]
    results = test[keep_cols].copy()

    # Append K/DST test rows with non-overlapping indices
    for pos_test_df in [k_test, dst_test]:
        offset = results.index.max() + 1
        pos_rows = pd.DataFrame(index=range(offset, offset + len(pos_test_df)))
        for col in keep_cols:
            if col in pos_test_df.columns:
                pos_rows[col] = pos_test_df[col].values
            elif col in ("fantasy_points_half_ppr", "fantasy_points_standard"):
                # K/DST scoring doesn't vary by format (no receptions)
                pos_rows[col] = pos_test_df["fantasy_points"].values
            elif col == "headshot_url":
                pos_rows[col] = ""
            else:
                pos_rows[col] = np.nan
        # Sync position test data index so _apply_position_models can
        # map predictions back to the correct results rows
        pos_test_df.index = pos_rows.index
        results = pd.concat([results, pos_rows])

    results["ridge_pred"] = 0.0
    results["nn_pred"] = 0.0

    # Apply position-specific models (general positions use general splits)
    for pos in ["QB", "RB", "WR", "TE"]:
        print(f"Applying {pos}-specific model...")
        _apply_position_models(train, val, test, pos, results)

    # K and DST use their own pre-processed splits
    print("Applying K-specific model...")
    _apply_position_models(k_train, k_val, k_test, "K", results)
    print("Applying DST-specific model...")
    _apply_position_models(dst_train, dst_val, dst_test, "DST", results)

    # Compute metrics from combined results (all positions)
    y_all = results["fantasy_points"].values
    metrics = {}
    model_cols = [("Ridge Regression", "ridge_pred"), ("Neural Network", "nn_pred")]
    for name, pred_col in model_cols:
        mask = results[pred_col].notna()
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

    _cache["results"] = results
    _cache["metrics"] = metrics
    print("Ready!")
    return results, metrics


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predictions")
def api_predictions():
    results, _ = _get_data()
    df = results.copy()

    position = request.args.get("position", "ALL")
    week = request.args.get("week", "ALL")
    search = request.args.get("search", "").strip().lower()
    sort_by = request.args.get("sort", "fantasy_points")
    order = request.args.get("order", "desc")

    if position != "ALL":
        df = df[df["position"] == position]
    if week != "ALL":
        try:
            df = df[df["week"] == int(week)]
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid week: {week}"}), 400
    if search:
        df = df[df["player_display_name"].str.lower().str.contains(search, na=False, regex=False)]

    rows = []
    for _, r in df.iterrows():
        row = {
            "player_id": _safe_str(r["player_id"]),
            "name": _safe_str(r["player_display_name"]),
            "position": _safe_str(r["position"]),
            "team": _safe_str(r["recent_team"]),
            "week": int(r["week"]),
            "actual": _safe_num(round(r["fantasy_points"], 2)),
            "ridge_pred": _safe_num(r["ridge_pred"]),
            "nn_pred": _safe_num(r["nn_pred"]),
            "headshot": _safe_str(r.get("headshot_url", "")),
        }
        rows.append(row)

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
    results, _ = _get_data()
    weeks = sorted(results["week"].unique().tolist())
    return jsonify({"weeks": [int(w) for w in weeks]})


@app.route("/api/player/<player_id>")
def api_player(player_id):
    results, _ = _get_data()
    df = results[results["player_id"] == player_id].sort_values("week")
    if df.empty:
        return jsonify({"error": "Player not found"}), 404

    info = df.iloc[0]
    weekly = []
    for _, r in df.iterrows():
        entry = {
            "week": int(r["week"]),
            "actual": _safe_num(round(r["fantasy_points"], 2)),
            "ridge_pred": _safe_num(r["ridge_pred"]),
            "nn_pred": _safe_num(r["nn_pred"]),
        }
        weekly.append(entry)

    return jsonify({
        "player_id": player_id,
        "name": _safe_str(info["player_display_name"]),
        "position": _safe_str(info["position"]),
        "team": _safe_str(info["recent_team"]),
        "headshot": _safe_str(info.get("headshot_url", "")),
        "weekly": weekly,
        "season_avg": _safe_num(round(df["fantasy_points"].mean(), 2)),
        "season_total": _safe_num(round(df["fantasy_points"].sum(), 2)),
    })


@app.route("/api/top_players")
def api_top_players():
    results, _ = _get_data()
    position = request.args.get("position", "ALL")

    df = results.copy()
    if position != "ALL":
        df = df[df["position"] == position]

    agg_dict = {
        "avg_actual": ("fantasy_points", "mean"),
        "avg_ridge": ("ridge_pred", "mean"),
        "avg_nn": ("nn_pred", "mean"),
        "games": ("week", "count"),
    }
    avg = df.groupby(["player_id", "player_display_name", "position", "recent_team"]).agg(
        **agg_dict,
    ).reset_index()
    avg = avg[avg["games"] >= 6]
    avg = avg.sort_values("avg_actual", ascending=False).head(25)

    rows = []
    for _, r in avg.iterrows():
        row = {
            "player_id": _safe_str(r["player_id"]),
            "name": _safe_str(r["player_display_name"]),
            "position": _safe_str(r["position"]),
            "team": _safe_str(r["recent_team"]),
            "avg_actual": _safe_num(round(r["avg_actual"], 2)),
            "avg_ridge": _safe_num(round(r["avg_ridge"], 2)),
            "avg_nn": _safe_num(round(r["avg_nn"], 2)),
            "games": int(r["games"]),
        }
        rows.append(row)

    return jsonify({"players": rows})


@app.route("/api/weekly_accuracy")
def api_weekly_accuracy():
    results, _ = _get_data()
    weeks = sorted(results["week"].unique())
    data = {"weeks": [], "ridge_mae": [], "nn_mae": []}

    for w in weeks:
        wdf = results[results["week"] == w]
        actual = wdf["fantasy_points"].values
        data["weeks"].append(int(w))
        data["ridge_mae"].append(round(np.mean(np.abs(actual - wdf["ridge_pred"].values)), 3))
        data["nn_mae"].append(round(np.mean(np.abs(actual - wdf["nn_pred"].values)), 3))

    return jsonify(data)


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
        "rolling": [], "prior_season": [], "ewma": [], "trend": [],
        "share": [], "matchup": [], "defense": [], "weather_vegas": [],
        "contextual": [], "other": [],
    }
    contextual_set = {
        "is_home", "week", "is_returning_from_absence", "days_rest",
        "practice_status", "game_status", "depth_chart_rank", "rest_days",
        "div_game", "spread_line",
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


def _position_arch_payload(pos, cfg, specific, targets, include_features,
                           attn_history=None):
    """Build the per-position JSON payload for /api/model_architecture.

    `include_features` may be a categorized dict (QB/RB/WR/TE) or a flat list
    (K/DST contextual); either shape is normalized to categorized groups.
    """
    get = lambda name, default=None: getattr(cfg, name, default)
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
                "QB", qb_cfg, QB_SPECIFIC_FEATURES, QB_TARGETS,
                getattr(qb_cfg, "QB_INCLUDE_FEATURES", []),
                getattr(qb_cfg, "QB_ATTN_HISTORY_STATS", None),
            ),
            "RB": _position_arch_payload(
                "RB", rb_cfg, RB_SPECIFIC_FEATURES, RB_TARGETS,
                getattr(rb_cfg, "RB_INCLUDE_FEATURES", []),
                getattr(rb_cfg, "RB_ATTN_HISTORY_STATS", None),
            ),
            "WR": _position_arch_payload(
                "WR", wr_cfg, WR_SPECIFIC_FEATURES, WR_TARGETS,
                getattr(wr_cfg, "WR_INCLUDE_FEATURES", []),
                getattr(wr_cfg, "WR_ATTN_HISTORY_STATS", None),
            ),
            "TE": _position_arch_payload(
                "TE", te_cfg, TE_SPECIFIC_FEATURES, TE_TARGETS,
                getattr(te_cfg, "TE_INCLUDE_FEATURES", []),
                getattr(te_cfg, "TE_ATTN_HISTORY_STATS", None),
            ),
            "K": _position_arch_payload(
                "K", k_cfg, K_SPECIFIC_FEATURES, K_TARGETS,
                getattr(k_cfg, "K_CONTEXTUAL_FEATURES", []),
            ),
            "DST": _position_arch_payload(
                "DST", dst_cfg, DST_SPECIFIC_FEATURES, DST_TARGETS,
                getattr(dst_cfg, "DST_CONTEXTUAL_FEATURES", []),
            ),
        }
        return jsonify({
            "overview": {
                "framework": "PyTorch 2.11 + CUDA 12.6 (AWS Batch)",
                "device": "CUDA if available, else CPU",
                "data_splits": "Train 2012-2023, Val 2024, Test 2025 (K uses 2015+)",
                "ensemble": ["Season-average baseline", "Ridge multi-target",
                             "MultiHeadNet (dense)",
                             "MultiHeadNetWithHistory (attention)", "LightGBM"],
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
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5050, use_reloader=False)
