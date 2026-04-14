"""Flask web application for the Fantasy Football Points Predictor.

All predictions come from position-specific models (QB, RB, WR, TE).
No general cross-position model is used.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import torch
import joblib

from src.config import SCORING_STANDARD, SCORING_HALF_PPR
from src.features.engineer import get_feature_columns, fill_nans_safe
from src.data.loader import compute_fantasy_points
from src.evaluation.metrics import compute_metrics, compute_positional_metrics

# --- QB imports ---
from QB.qb_data import filter_to_qb
from QB.qb_targets import compute_qb_targets, compute_qb_adjustment
from QB.qb_features import add_qb_specific_features, get_qb_feature_columns, fill_qb_nans
from QB.qb_config import (
    QB_TARGETS, QB_SPECIFIC_FEATURES,
    QB_NN_BACKBONE_LAYERS, QB_NN_HEAD_HIDDEN, QB_NN_DROPOUT,
)
# --- RB imports ---
from RB.rb_data import filter_to_rb
from RB.rb_targets import compute_rb_targets, compute_fumble_adjustment
from RB.rb_features import add_rb_specific_features, get_rb_feature_columns, fill_rb_nans
from RB.rb_config import (
    RB_SPECIFIC_FEATURES, RB_NN_BACKBONE_LAYERS, RB_NN_HEAD_HIDDEN, RB_NN_DROPOUT,
)
from RB.rb_models import RBRidgeMultiTarget
from RB.rb_neural_net import RBMultiHeadNet
# --- WR imports ---
from WR.wr_data import filter_to_wr
from WR.wr_targets import compute_wr_targets, compute_wr_fumble_adjustment
from WR.wr_features import add_wr_specific_features, get_wr_feature_columns, fill_wr_nans
from WR.wr_config import (
    WR_TARGETS, WR_SPECIFIC_FEATURES,
    WR_NN_BACKBONE_LAYERS, WR_NN_HEAD_HIDDEN, WR_NN_DROPOUT,
)
# --- TE imports ---
from TE.te_data import filter_to_te
from TE.te_targets import compute_te_targets, compute_te_fumble_adjustment
from TE.te_features import add_te_specific_features, get_te_feature_columns, fill_te_nans
from TE.te_config import (
    TE_TARGETS, TE_SPECIFIC_FEATURES,
    TE_NN_BACKBONE_LAYERS, TE_NN_HEAD_HIDDEN, TE_NN_HEAD_HIDDEN_OVERRIDES,
    TE_NN_DROPOUT,
)
# --- Position-specific models ---
from QB.qb_models import QBRidgeMultiTarget
from QB.qb_neural_net import QBMultiHeadNet
from WR.wr_models import WRRidgeMultiTarget
from WR.wr_neural_net import WRMultiHeadNet
from TE.te_models import TERidgeMultiTarget
from TE.te_neural_net import TEMultiHeadNet

app = Flask(__name__)

_cache = {}

# ---------------------------------------------------------------------------
# Position model metadata (static)
# ---------------------------------------------------------------------------
POSITION_INFO = {
    "QB": {
        "label": "Quarterback",
        "targets": [
            {"key": "passing_floor", "label": "Passing Floor", "formula": "passing_yards x 0.04"},
            {"key": "rushing_floor", "label": "Rushing Floor", "formula": "rushing_yards x 0.1"},
            {"key": "td_points", "label": "TD Points", "formula": "pass_TD x 4 + rush_TD x 6 + 2pt x 2"},
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
            {"key": "td_points", "label": "TD Points", "formula": "rush_TD x 6 + recv_TD x 6 + 2pt x 2"},
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
            {"key": "td_points", "label": "TD Points", "formula": "recv_TD x 6 + rush_TD x 6 + 2pt x 2"},
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
            {"key": "td_points", "label": "TD Points", "formula": "recv_TD x 6 + rush_TD x 6 + 2pt x 2"},
        ],
        "adjustments": "Fumble rate (historical L8 rolling avg)",
        "specific_features": list(TE_SPECIFIC_FEATURES),
        "architecture": {"backbone": list(TE_NN_BACKBONE_LAYERS), "head_hidden": TE_NN_HEAD_HIDDEN},
    },
}


def _compute_scoring_formats(df):
    if "fantasy_points_standard" not in df.columns:
        df["fantasy_points_standard"] = compute_fantasy_points(df, SCORING_STANDARD)
    if "fantasy_points_half_ppr" not in df.columns:
        df["fantasy_points_half_ppr"] = compute_fantasy_points(df, SCORING_HALF_PPR)
    return df


def _apply_position_models(train, val, test, pos, results):
    """Load pre-trained position-specific models and write predictions into results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pos == "QB":
        model_dir = "QB/outputs/models"
        pos_train, pos_val, pos_test = filter_to_qb(train), filter_to_qb(val), filter_to_qb(test)
        pos_train = compute_qb_targets(pos_train)
        pos_val = compute_qb_targets(pos_val)
        pos_test = compute_qb_targets(pos_test)
        pos_train, pos_val, pos_test = add_qb_specific_features(pos_train, pos_val, pos_test)
        pos_train, pos_val, pos_test = fill_qb_nans(pos_train, pos_val, pos_test, QB_SPECIFIC_FEATURES)
        feature_cols = get_qb_feature_columns()
        targets = QB_TARGETS
        adj = compute_qb_adjustment(pos_test)
        nn_file = "qb_multihead_nn.pt"
        nn_cls_args = dict(
            backbone_layers=QB_NN_BACKBONE_LAYERS,
            head_hidden=QB_NN_HEAD_HIDDEN,
            dropout=QB_NN_DROPOUT,
        )
        ridge_cls = QBRidgeMultiTarget
        ridge_args = {}
        nn_cls = QBMultiHeadNet

    elif pos == "RB":
        model_dir = "RB/outputs/models"
        pos_train, pos_val, pos_test = filter_to_rb(train), filter_to_rb(val), filter_to_rb(test)
        pos_train = compute_rb_targets(pos_train)
        pos_val = compute_rb_targets(pos_val)
        pos_test = compute_rb_targets(pos_test)
        pos_train, pos_val, pos_test = add_rb_specific_features(pos_train, pos_val, pos_test)
        pos_train, pos_val, pos_test = fill_rb_nans(pos_train, pos_val, pos_test, RB_SPECIFIC_FEATURES)
        feature_cols = get_rb_feature_columns()
        targets = ["rushing_floor", "receiving_floor", "td_points"]
        adj = compute_fumble_adjustment(pos_test)
        nn_file = "rb_multihead_nn.pt"
        nn_cls_args = dict(
            backbone_layers=RB_NN_BACKBONE_LAYERS,
            head_hidden=RB_NN_HEAD_HIDDEN,
            dropout=RB_NN_DROPOUT,
        )
        ridge_cls = RBRidgeMultiTarget
        ridge_args = {}
        nn_cls = RBMultiHeadNet

    elif pos == "WR":
        model_dir = "WR/outputs/models"
        pos_train, pos_val, pos_test = filter_to_wr(train), filter_to_wr(val), filter_to_wr(test)
        pos_train = compute_wr_targets(pos_train)
        pos_val = compute_wr_targets(pos_val)
        pos_test = compute_wr_targets(pos_test)
        pos_train, pos_val, pos_test = add_wr_specific_features(pos_train, pos_val, pos_test)
        pos_train, pos_val, pos_test = fill_wr_nans(pos_train, pos_val, pos_test, WR_SPECIFIC_FEATURES)
        feature_cols = get_wr_feature_columns()
        targets = WR_TARGETS
        adj = compute_wr_fumble_adjustment(pos_test)
        nn_file = "wr_multihead_nn.pt"
        nn_cls_args = dict(
            backbone_layers=WR_NN_BACKBONE_LAYERS,
            head_hidden=WR_NN_HEAD_HIDDEN,
            dropout=WR_NN_DROPOUT,
        )
        ridge_cls = WRRidgeMultiTarget
        ridge_args = {}
        nn_cls = WRMultiHeadNet

    elif pos == "TE":
        model_dir = "TE/outputs/models"
        pos_train, pos_val, pos_test = filter_to_te(train), filter_to_te(val), filter_to_te(test)
        pos_train = compute_te_targets(pos_train)
        pos_val = compute_te_targets(pos_val)
        pos_test = compute_te_targets(pos_test)
        pos_train, pos_val, pos_test = add_te_specific_features(pos_train, pos_val, pos_test)
        pos_train, pos_val, pos_test = fill_te_nans(pos_train, pos_val, pos_test, TE_SPECIFIC_FEATURES)
        feature_cols = get_te_feature_columns()
        targets = TE_TARGETS
        adj = compute_te_fumble_adjustment(pos_test)
        nn_file = "te_multihead_nn.pt"
        nn_cls_args = dict(
            backbone_layers=TE_NN_BACKBONE_LAYERS,
            head_hidden=TE_NN_HEAD_HIDDEN,
            td_head_hidden=TE_NN_HEAD_HIDDEN_OVERRIDES.get("td_points"),
            dropout=TE_NN_DROPOUT,
        )
        ridge_cls = TERidgeMultiTarget
        ridge_args = {}
        nn_cls = TEMultiHeadNet

    # Prepare features
    feature_cols = [c for c in feature_cols if c in pos_train.columns]
    for df in [pos_train, pos_val, pos_test]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_test_pos = pos_test[feature_cols].values.astype(np.float32)

    # Ridge predictions
    ridge = ridge_cls(**ridge_args)
    ridge.load(model_dir)
    ridge_preds = ridge.predict(X_test_pos)
    ridge_total = sum(ridge_preds[t] for t in targets) + adj.values

    # NN predictions
    nn_scaler = joblib.load(f"{model_dir}/nn_scaler.pkl")
    X_test_scaled = nn_scaler.transform(X_test_pos)

    nn_model = nn_cls(input_dim=len(feature_cols), **nn_cls_args).to(device)
    nn_model.load_state_dict(
        torch.load(f"{model_dir}/{nn_file}", map_location=device, weights_only=True)
    )
    nn_preds = nn_model.predict_numpy(X_test_scaled, device)
    nn_total = sum(nn_preds[t] for t in targets) + adj.values

    # Write into results
    pos_index = pos_test.index
    results.loc[pos_index, "ridge_pred"] = np.round(ridge_total, 2).astype(np.float32)
    results.loc[pos_index, "nn_pred"] = np.round(nn_total, 2).astype(np.float32)

    # Cache per-target metrics for /api/position_details
    target_metrics = {}
    for t in targets:
        if t in pos_test.columns:
            actual_t = pos_test[t].values
            target_metrics[t] = {
                "ridge_mae": round(float(np.mean(np.abs(ridge_preds[t] - actual_t))), 3),
                "nn_mae": round(float(np.mean(np.abs(nn_preds[t] - actual_t))), 3),
            }
    total_actual = pos_test["fantasy_points"].values
    target_metrics["total"] = {
        "ridge_mae": round(float(np.mean(np.abs(ridge_total - total_actual))), 3),
        "nn_mae": round(float(np.mean(np.abs(nn_total - total_actual))), 3),
    }
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

    # Build results frame
    keep_cols = [
        "player_id", "player_display_name", "position", "recent_team",
        "season", "week", "headshot_url",
        "fantasy_points", "fantasy_points_half_ppr", "fantasy_points_standard",
        "fantasy_points_floor",
    ]
    keep_cols = [c for c in keep_cols if c in test.columns]
    results = test[keep_cols].copy()
    results["ridge_pred"] = 0.0
    results["nn_pred"] = 0.0

    # Apply position-specific models
    for pos in ["QB", "RB", "WR", "TE"]:
        print(f"Applying {pos}-specific model...")
        _apply_position_models(train, val, test, pos, results)

    # Compute metrics using position-specific predictions
    y_test = test["fantasy_points"].values
    metrics = {}
    for name, pred_col in [("Ridge Regression", "ridge_pred"), ("Neural Network", "nn_pred")]:
        preds = results[pred_col].values
        overall = compute_metrics(y_test, preds)
        pos_df = test[["position"]].copy()
        pos_df["pred"] = preds
        pos_df["actual"] = y_test
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
        rows.append({
            "player_id": r["player_id"],
            "name": r["player_display_name"],
            "position": r["position"],
            "team": r["recent_team"],
            "week": int(r["week"]),
            "actual": actual,
            "ridge_pred": r["ridge_pred"],
            "nn_pred": r["nn_pred"],
            "headshot": r.get("headshot_url", ""),
        })

    reverse = order == "desc"
    if sort_by in ("actual", "ridge_pred", "nn_pred", "week"):
        rows.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
    else:
        rows.sort(key=lambda x: x.get("actual", 0), reverse=reverse)

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
        weekly.append({
            "week": int(r["week"]),
            "actual": round(r["fantasy_points"], 2),
            "ridge_pred": r["ridge_pred"],
            "nn_pred": r["nn_pred"],
        })

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

    avg = df.groupby(["player_id", "player_display_name", "position", "recent_team"]).agg(
        avg_actual=("fantasy_points", "mean"),
        avg_ridge=("ridge_pred", "mean"),
        avg_nn=("nn_pred", "mean"),
        games=("week", "count"),
    ).reset_index()
    avg = avg[avg["games"] >= 6]
    avg = avg.sort_values("avg_actual", ascending=False).head(25)

    rows = []
    for _, r in avg.iterrows():
        rows.append({
            "player_id": r["player_id"],
            "name": r["player_display_name"],
            "position": r["position"],
            "team": r["recent_team"],
            "avg_actual": round(r["avg_actual"], 2),
            "avg_ridge": round(r["avg_ridge"], 2),
            "avg_nn": round(r["avg_nn"], 2),
            "games": int(r["games"]),
        })

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
    for pos in ["QB", "RB", "WR", "TE"]:
        info = dict(POSITION_INFO[pos])
        info.update(details.get(pos, {}))
        result[pos] = info
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
