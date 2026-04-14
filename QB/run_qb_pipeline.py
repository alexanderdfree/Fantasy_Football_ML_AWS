"""End-to-end QB position model pipeline."""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SPLITS_DIR
from src.evaluation.metrics import compute_metrics
from src.models.baseline import SeasonAverageBaseline

from QB.qb_config import (
    QB_TARGETS, QB_RIDGE_ALPHAS, QB_SPECIFIC_FEATURES,
    QB_NN_BACKBONE_LAYERS, QB_NN_HEAD_HIDDEN, QB_NN_DROPOUT,
    QB_NN_LR, QB_NN_WEIGHT_DECAY, QB_NN_EPOCHS, QB_NN_BATCH_SIZE,
    QB_NN_PATIENCE,
    QB_LOSS_WEIGHTS, QB_LOSS_W_TOTAL, QB_HUBER_DELTAS,
    QB_SCHEDULER_TYPE, QB_ONECYCLE_MAX_LR, QB_ONECYCLE_PCT_START,
)
from QB.qb_data import filter_to_qb
from QB.qb_targets import compute_qb_targets, compute_qb_adjustment
from QB.qb_features import add_qb_specific_features, get_qb_feature_columns, fill_qb_nans

from QB.qb_models import QBRidgeMultiTarget
from QB.qb_neural_net import QBMultiHeadNet
from QB.qb_training import MultiTargetLoss, QBMultiHeadTrainer, make_qb_dataloaders
from QB.qb_evaluation import (
    compute_qb_metrics, compute_qb_ranking_metrics,
    print_qb_comparison_table, plot_qb_pred_vs_actual,
)
from QB.qb_backtest import run_qb_weekly_simulation, plot_qb_weekly_accuracy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_qb_pipeline(train_df=None, val_df=None, test_df=None, seed=42):
    """Run the full QB position model pipeline."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if train_df is None:
        print("Loading general splits from disk...")
        train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
        val_df = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
        test_df = pd.read_parquet(f"{SPLITS_DIR}/test.parquet")

    # Filter to QB
    print("Filtering to QB...")
    qb_train = filter_to_qb(train_df)
    qb_val = filter_to_qb(val_df)
    qb_test = filter_to_qb(test_df)
    print(f"  QB splits: train={len(qb_train)}, val={len(qb_val)}, test={len(qb_test)}")

    # Compute targets
    print("Computing QB targets...")
    qb_train = compute_qb_targets(qb_train)
    qb_val = compute_qb_targets(qb_val)
    qb_test = compute_qb_targets(qb_test)

    # Add QB-specific features
    print("Adding QB-specific features...")
    qb_train, qb_val, qb_test = add_qb_specific_features(qb_train, qb_val, qb_test)
    qb_train, qb_val, qb_test = fill_qb_nans(qb_train, qb_val, qb_test, QB_SPECIFIC_FEATURES)

    # Prepare feature arrays
    feature_cols = get_qb_feature_columns()
    available_cols = [c for c in feature_cols if c in qb_train.columns]
    missing_cols = [c for c in feature_cols if c not in qb_train.columns]
    if missing_cols:
        print(f"  WARNING: {len(missing_cols)} feature columns missing, filling with 0")
        for col in missing_cols:
            for df in [qb_train, qb_val, qb_test]:
                df[col] = 0.0
    feature_cols_final = feature_cols

    for df in [qb_train, qb_val, qb_test]:
        df[feature_cols_final] = df[feature_cols_final].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train = qb_train[feature_cols_final].values.astype(np.float32)
    X_val = qb_val[feature_cols_final].values.astype(np.float32)
    X_test = qb_test[feature_cols_final].values.astype(np.float32)

    y_train_dict = {t: qb_train[t].values for t in QB_TARGETS}
    y_val_dict = {t: qb_val[t].values for t in QB_TARGETS}
    y_test_dict = {t: qb_test[t].values for t in QB_TARGETS}

    y_train_dict["total"] = sum(qb_train[t].values for t in QB_TARGETS)
    y_val_dict["total"] = sum(qb_val[t].values for t in QB_TARGETS)
    y_test_dict["total"] = qb_test["fantasy_points"].values

    print(f"  Feature matrix shape: {X_train.shape}")

    # Baseline
    print("\n=== QB Baseline ===")
    baseline = SeasonAverageBaseline()
    baseline_preds = baseline.predict(qb_test)
    baseline_metrics = {"total": compute_metrics(y_test_dict["total"], baseline_preds)}
    print(f"  Season Avg Baseline MAE: {baseline_metrics['total']['mae']:.3f}")

    # Ridge multi-target with alpha tuning
    print("\n=== QB Ridge Multi-Target ===")
    adj_val = compute_qb_adjustment(qb_val)
    adj_test = compute_qb_adjustment(qb_test)
    y_val_actual = qb_val["fantasy_points"].values

    best_alpha, best_val_mae = None, float("inf")
    for alpha in QB_RIDGE_ALPHAS:
        ridge = QBRidgeMultiTarget(alpha=alpha)
        ridge.fit(X_train, y_train_dict)
        val_preds = ridge.predict(X_val)
        val_total_adj = val_preds["total"] + adj_val.values
        val_mae = np.mean(np.abs(val_total_adj - y_val_actual))
        print(f"  Alpha={alpha:.2f}: Val MAE={val_mae:.3f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_alpha = alpha

    print(f"  Best alpha: {best_alpha}")
    ridge_model = QBRidgeMultiTarget(alpha=best_alpha)
    ridge_model.fit(X_train, y_train_dict)
    ridge_test_preds = ridge_model.predict(X_test)
    ridge_test_preds["total"] = (
        sum(ridge_test_preds[t] for t in QB_TARGETS) + adj_test.values
    )
    ridge_metrics = compute_qb_metrics(y_test_dict, ridge_test_preds)

    # Multi-head NN
    print("\n=== QB Multi-Head Neural Net ===")
    nn_scaler = StandardScaler()
    X_train_s = nn_scaler.fit_transform(X_train)
    X_val_s = nn_scaler.transform(X_val)
    X_test_s = nn_scaler.transform(X_test)

    train_loader, val_loader = make_qb_dataloaders(
        X_train_s, y_train_dict, X_val_s, y_val_dict, batch_size=QB_NN_BATCH_SIZE,
    )

    device = torch.device("cpu")
    model = QBMultiHeadNet(
        input_dim=X_train_s.shape[1],
        backbone_layers=QB_NN_BACKBONE_LAYERS, head_hidden=QB_NN_HEAD_HIDDEN,
        dropout=QB_NN_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=QB_NN_LR, weight_decay=QB_NN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=QB_ONECYCLE_MAX_LR,
        epochs=QB_NN_EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=QB_ONECYCLE_PCT_START,
    )
    criterion = MultiTargetLoss(
        w_passing=QB_LOSS_WEIGHTS["passing_floor"],
        w_rushing=QB_LOSS_WEIGHTS["rushing_floor"],
        w_td=QB_LOSS_WEIGHTS["td_points"],
        w_total=QB_LOSS_W_TOTAL,
        huber_deltas=QB_HUBER_DELTAS,
    )

    trainer = QBMultiHeadTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, device=device,
        patience=QB_NN_PATIENCE, scheduler_per_batch=True,
    )
    history = trainer.train(train_loader, val_loader, n_epochs=QB_NN_EPOCHS)

    nn_test_preds = model.predict_numpy(X_test_s, device)
    nn_test_preds["total"] = (
        sum(nn_test_preds[t] for t in QB_TARGETS) + adj_test.values
    )
    nn_metrics = compute_qb_metrics(y_test_dict, nn_test_preds)

    # Comparison
    print_qb_comparison_table({
        "Season Average Baseline": baseline_metrics,
        "QB Ridge Multi-Target": ridge_metrics,
        "QB Multi-Head NN": nn_metrics,
    })

    # Ranking metrics
    qb_test = qb_test.copy()
    qb_test["pred_ridge_total"] = ridge_test_preds["total"]
    qb_test["pred_nn_total"] = nn_test_preds["total"]
    qb_test["pred_baseline"] = baseline_preds

    ridge_ranking = compute_qb_ranking_metrics(qb_test, pred_col="pred_ridge_total")
    nn_ranking = compute_qb_ranking_metrics(qb_test, pred_col="pred_nn_total")
    print(f"\nRidge Top-12 Hit Rate: {ridge_ranking['season_avg_hit_rate']:.3f}")
    print(f"NN Top-12 Hit Rate:    {nn_ranking['season_avg_hit_rate']:.3f}")

    # Weekly backtest
    print("\n=== Weekly Backtest ===")
    sim_results = run_qb_weekly_simulation(
        qb_test,
        pred_columns={"Season Avg": "pred_baseline", "Ridge": "pred_ridge_total", "Neural Net": "pred_nn_total"},
    )
    for model_name, summary in sim_results["season_summary"].items():
        print(f"  {model_name}: MAE={summary['mae']:.3f}, R2={summary['r2']:.3f}")

    # Save outputs
    os.makedirs("QB/outputs/models", exist_ok=True)
    os.makedirs("QB/outputs/figures", exist_ok=True)

    ridge_model.save("QB/outputs/models")
    torch.save(model.state_dict(), "QB/outputs/models/qb_multihead_nn.pt")
    joblib.dump(nn_scaler, "QB/outputs/models/nn_scaler.pkl")

    trainer.plot_training_curves(history, "QB/outputs/figures/qb_training_curves.png")
    plot_qb_weekly_accuracy(sim_results, "QB/outputs/figures/qb_weekly_mae.png")
    plot_qb_pred_vs_actual(
        y_test_dict, nn_test_preds, "QB Multi-Head NN",
        "QB/outputs/figures/qb_pred_vs_actual_scatter.png",
    )

    feature_importance = ridge_model.get_feature_importance(feature_cols_final)
    fig, axes = plt.subplots(1, len(QB_TARGETS), figsize=(6 * len(QB_TARGETS), 8))
    for ax, (target, importance) in zip(axes, feature_importance.items()):
        importance.head(15).plot(kind="barh", ax=ax)
        ax.set_title(f"Ridge: {target} Top-15 Features")
        ax.set_xlabel("Absolute Coefficient")
    plt.tight_layout()
    plt.savefig("QB/outputs/figures/qb_ridge_feature_importance.png", dpi=150)
    plt.close()

    print("\nQB pipeline complete. Outputs saved to QB/outputs/")
    return {
        "ridge_metrics": ridge_metrics, "nn_metrics": nn_metrics,
        "ridge_ranking": ridge_ranking, "nn_ranking": nn_ranking,
        "history": history, "sim_results": sim_results,
    }


if __name__ == "__main__":
    run_qb_pipeline()
