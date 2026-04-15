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
from shared.weather_features import merge_schedule_features, get_weather_feature_columns

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

app = Flask(__name__)

_cache = {}


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

    pos_train, pos_val, pos_test = reg["add_features_fn"](pos_train, pos_val, pos_test)
    pos_train, pos_val, pos_test = reg["fill_nans_fn"](
        pos_train, pos_val, pos_test, reg["specific_features"]
    )

    feature_cols = reg["get_feature_columns_fn"]()
    adj = reg["compute_adjustment_fn"](pos_test)

    # Prepare features
    expected_count = len(feature_cols)
    feature_cols = [c for c in feature_cols if c in pos_train.columns]
    if len(feature_cols) < expected_count:
        missing = expected_count - len(feature_cols)
        print(f"  WARNING: {pos} dropped {missing} feature cols not found in data")
    for df in [pos_train, pos_val, pos_test]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_test_pos = pos_test[feature_cols].values.astype(np.float32)

    # Ridge predictions
    ridge = RidgeMultiTarget(target_names=targets)
    ridge.load(model_dir)
    ridge_preds = ridge.predict(X_test_pos)
    ridge_total = sum(ridge_preds[t] for t in targets) + adj.values

    # NN predictions
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

    # Weather NN predictions (only if model was trained, QB/RB/WR/TE only)
    weather_nn_total = None
    weather_nn_preds = None
    weather_nn_file = f"{model_dir}/{pos.lower()}_weather_nn.pt"
    if os.path.exists(weather_nn_file) and pos in ("QB", "RB", "WR", "TE"):
        for df in [pos_train, pos_val, pos_test]:
            merge_schedule_features(df)

        weather_cols = get_weather_feature_columns(pos, feature_cols)
        weather_cols = [c for c in weather_cols if c in pos_test.columns]

        for df in [pos_train, pos_val, pos_test]:
            df[weather_cols] = df[weather_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        X_test_weather = pos_test[weather_cols].values.astype(np.float32)
        weather_scaler = joblib.load(f"{model_dir}/weather_nn_scaler.pkl")
        X_test_weather_scaled = np.clip(weather_scaler.transform(X_test_weather), -4, 4)

        weather_model = MultiHeadNet(
            input_dim=len(weather_cols), target_names=targets, **reg["nn_kwargs"]
        ).to(device)
        weather_model.load_state_dict(
            torch.load(weather_nn_file, map_location=device, weights_only=True)
        )
        weather_nn_preds = weather_model.predict_numpy(X_test_weather_scaled, device)
        weather_nn_total = sum(weather_nn_preds[t] for t in targets) + adj.values

    # Write into results
    pos_index = pos_test.index
    results.loc[pos_index, "ridge_pred"] = np.round(ridge_total, 2).astype(np.float32)
    results.loc[pos_index, "nn_pred"] = np.round(nn_total, 2).astype(np.float32)
    if weather_nn_total is not None:
        results.loc[pos_index, "weather_nn_pred"] = np.round(weather_nn_total, 2).astype(np.float32)

    # Cache per-target metrics for /api/position_details
    target_metrics = {}
    for t in targets:
        if t in pos_test.columns:
            actual_t = pos_test[t].values
            tm = {
                "ridge_mae": round(float(np.mean(np.abs(ridge_preds[t] - actual_t))), 3),
                "nn_mae": round(float(np.mean(np.abs(nn_preds[t] - actual_t))), 3),
            }
            if weather_nn_preds is not None:
                tm["weather_nn_mae"] = round(float(np.mean(np.abs(weather_nn_preds[t] - actual_t))), 3)
            target_metrics[t] = tm
    total_actual = pos_test["fantasy_points"].values
    total_tm = {
        "ridge_mae": round(float(np.mean(np.abs(ridge_total - total_actual))), 3),
        "nn_mae": round(float(np.mean(np.abs(nn_total - total_actual))), 3),
    }
    if weather_nn_total is not None:
        total_tm["weather_nn_mae"] = round(float(np.mean(np.abs(weather_nn_total - total_actual))), 3)
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
    results["weather_nn_pred"] = np.nan

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
    if results["weather_nn_pred"].notna().any():
        model_cols.append(("Weather NN", "weather_nn_pred"))
    for name, pred_col in model_cols:
        mask = results[pred_col].notna()
        preds_full = results[pred_col].fillna(0).values
        overall = compute_metrics(y_all, preds_full)
        pos_df = results[["position"]].copy()
        pos_df["pred"] = preds_full
        pos_df["actual"] = y_all
        pos_metrics = compute_positional_metrics(pos_df, "pred", "actual")
        # For Weather NN, also compute metrics only on positions that have it
        if pred_col == "weather_nn_pred":
            wnn_mask = mask
            wnn_y = y_all[wnn_mask]
            wnn_preds = results.loc[wnn_mask, pred_col].values
            overall = compute_metrics(wnn_y, wnn_preds)
            pos_df_w = results.loc[wnn_mask, ["position"]].copy()
            pos_df_w["pred"] = wnn_preds
            pos_df_w["actual"] = wnn_y
            pos_metrics = compute_positional_metrics(pos_df_w, "pred", "actual")
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
    scoring = request.args.get("scoring", "ppr")
    sort_by = request.args.get("sort", "fantasy_points")
    order = request.args.get("order", "desc")

    if position != "ALL":
        df = df[df["position"] == position]
    if week != "ALL":
        df = df[df["week"] == int(week)]
    if search:
        df = df[df["player_display_name"].str.lower().str.contains(search, na=False)]

    scoring_col = {
        "ppr": "fantasy_points",
        "half_ppr": "fantasy_points_half_ppr",
        "standard": "fantasy_points_standard",
    }.get(scoring, "fantasy_points")

    rows = []
    for _, r in df.iterrows():
        actual = round(r[scoring_col], 2) if scoring_col in r.index else round(r["fantasy_points"], 2)
        row = {
            "player_id": r["player_id"],
            "name": r["player_display_name"],
            "position": r["position"],
            "team": r["recent_team"],
            "week": int(r["week"]),
            "actual": actual,
            "ridge_pred": r["ridge_pred"],
            "nn_pred": r["nn_pred"],
            "headshot": r.get("headshot_url", ""),
        }
        wnn = r.get("weather_nn_pred")
        row["weather_nn_pred"] = round(float(wnn), 2) if pd.notna(wnn) else None
        rows.append(row)

    reverse = order == "desc"
    if sort_by in ("actual", "ridge_pred", "nn_pred", "weather_nn_pred", "week"):
        rows.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse)
    else:
        rows.sort(key=lambda x: x.get("actual", 0), reverse=reverse)

    scoring_note = None
    if scoring != "ppr":
        scoring_note = "Predictions are PPR-trained; actuals reflect selected scoring format"

    resp = {"players": rows, "total": len(rows)}
    if scoring_note:
        resp["scoring_note"] = scoring_note
    return jsonify(resp)


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
            "actual": round(r["fantasy_points"], 2),
            "ridge_pred": r["ridge_pred"],
            "nn_pred": r["nn_pred"],
        }
        wnn = r.get("weather_nn_pred")
        entry["weather_nn_pred"] = round(float(wnn), 2) if pd.notna(wnn) else None
        weekly.append(entry)

    return jsonify({
        "player_id": player_id,
        "name": info["player_display_name"],
        "position": info["position"],
        "team": info["recent_team"],
        "headshot": info.get("headshot_url", ""),
        "weekly": weekly,
        "season_avg": round(df["fantasy_points"].mean(), 2),
        "season_total": round(df["fantasy_points"].sum(), 2),
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
    if "weather_nn_pred" in df.columns and df["weather_nn_pred"].notna().any():
        agg_dict["avg_weather_nn"] = ("weather_nn_pred", "mean")
    avg = df.groupby(["player_id", "player_display_name", "position", "recent_team"]).agg(
        **agg_dict,
    ).reset_index()
    avg = avg[avg["games"] >= 6]
    avg = avg.sort_values("avg_actual", ascending=False).head(25)

    rows = []
    for _, r in avg.iterrows():
        row = {
            "player_id": r["player_id"],
            "name": r["player_display_name"],
            "position": r["position"],
            "team": r["recent_team"],
            "avg_actual": round(r["avg_actual"], 2),
            "avg_ridge": round(r["avg_ridge"], 2),
            "avg_nn": round(r["avg_nn"], 2),
            "games": int(r["games"]),
        }
        if "avg_weather_nn" in r.index and pd.notna(r["avg_weather_nn"]):
            row["avg_weather_nn"] = round(r["avg_weather_nn"], 2)
        rows.append(row)

    return jsonify({"players": rows})


@app.route("/api/weekly_accuracy")
def api_weekly_accuracy():
    results, _ = _get_data()
    weeks = sorted(results["week"].unique())
    has_weather = results["weather_nn_pred"].notna().any()
    data = {"weeks": [], "ridge_mae": [], "nn_mae": []}
    if has_weather:
        data["weather_nn_mae"] = []

    for w in weeks:
        wdf = results[results["week"] == w]
        actual = wdf["fantasy_points"].values
        data["weeks"].append(int(w))
        data["ridge_mae"].append(round(np.mean(np.abs(actual - wdf["ridge_pred"].values)), 3))
        data["nn_mae"].append(round(np.mean(np.abs(actual - wdf["nn_pred"].values)), 3))
        if has_weather:
            wnn_mask = wdf["weather_nn_pred"].notna()
            if wnn_mask.any():
                data["weather_nn_mae"].append(
                    round(np.mean(np.abs(
                        wdf.loc[wnn_mask, "fantasy_points"].values -
                        wdf.loc[wnn_mask, "weather_nn_pred"].values
                    )), 3)
                )
            else:
                data["weather_nn_mae"].append(None)

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


if __name__ == "__main__":
    app.run(debug=True, port=5050, use_reloader=False)
