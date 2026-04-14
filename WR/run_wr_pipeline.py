"""End-to-end WR position model pipeline."""

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

from WR.wr_config import (
    WR_TARGETS, WR_RIDGE_ALPHAS, WR_SPECIFIC_FEATURES,
    WR_NN_BACKBONE_LAYERS, WR_NN_HEAD_HIDDEN, WR_NN_DROPOUT,
    WR_NN_LR, WR_NN_WEIGHT_DECAY, WR_NN_EPOCHS, WR_NN_BATCH_SIZE,
    WR_NN_PATIENCE,
    WR_LOSS_WEIGHTS, WR_LOSS_W_TOTAL, WR_HUBER_DELTAS,
    WR_SCHEDULER_TYPE, WR_COSINE_T0, WR_COSINE_T_MULT, WR_COSINE_ETA_MIN,
)
from WR.wr_data import filter_to_wr
from WR.wr_targets import compute_wr_targets, compute_wr_fumble_adjustment
from WR.wr_features import add_wr_specific_features, get_wr_feature_columns, fill_wr_nans

from WR.wr_models import WRRidgeMultiTarget
from WR.wr_neural_net import WRMultiHeadNet
from WR.wr_training import MultiTargetLoss, WRMultiHeadTrainer, make_wr_dataloaders
from WR.wr_evaluation import (
    compute_wr_metrics, compute_wr_ranking_metrics,
    print_wr_comparison_table, plot_wr_pred_vs_actual,
)
from WR.wr_backtest import run_wr_weekly_simulation, plot_wr_weekly_accuracy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_wr_pipeline(train_df=None, val_df=None, test_df=None, seed=42):
    """Run the full WR position model pipeline."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if train_df is None:
        print("Loading general splits from disk...")
        train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
        val_df = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
        test_df = pd.read_parquet(f"{SPLITS_DIR}/test.parquet")

    print("Filtering to WR...")
    wr_train = filter_to_wr(train_df)
    wr_val = filter_to_wr(val_df)
    wr_test = filter_to_wr(test_df)
    print(f"  WR splits: train={len(wr_train)}, val={len(wr_val)}, test={len(wr_test)}")

    print("Computing WR targets...")
    wr_train = compute_wr_targets(wr_train)
    wr_val = compute_wr_targets(wr_val)
    wr_test = compute_wr_targets(wr_test)

    print("Adding WR-specific features...")
    wr_train, wr_val, wr_test = add_wr_specific_features(wr_train, wr_val, wr_test)
    wr_train, wr_val, wr_test = fill_wr_nans(wr_train, wr_val, wr_test, WR_SPECIFIC_FEATURES)

    feature_cols = get_wr_feature_columns()
    missing_cols = [c for c in feature_cols if c not in wr_train.columns]
    if missing_cols:
        print(f"  WARNING: {len(missing_cols)} feature columns missing, filling with 0")
        for col in missing_cols:
            for df in [wr_train, wr_val, wr_test]:
                df[col] = 0.0
    feature_cols_final = feature_cols

    for df in [wr_train, wr_val, wr_test]:
        df[feature_cols_final] = df[feature_cols_final].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train = wr_train[feature_cols_final].values.astype(np.float32)
    X_val = wr_val[feature_cols_final].values.astype(np.float32)
    X_test = wr_test[feature_cols_final].values.astype(np.float32)

    y_train_dict = {t: wr_train[t].values for t in WR_TARGETS}
    y_val_dict = {t: wr_val[t].values for t in WR_TARGETS}
    y_test_dict = {t: wr_test[t].values for t in WR_TARGETS}

    y_train_dict["total"] = sum(wr_train[t].values for t in WR_TARGETS)
    y_val_dict["total"] = sum(wr_val[t].values for t in WR_TARGETS)
    y_test_dict["total"] = wr_test["fantasy_points"].values

    print(f"  Feature matrix shape: {X_train.shape}")

    # Baseline
    print("\n=== WR Baseline ===")
    baseline = SeasonAverageBaseline()
    baseline_preds = baseline.predict(wr_test)
    baseline_metrics = {"total": compute_metrics(y_test_dict["total"], baseline_preds)}
    print(f"  Season Avg Baseline MAE: {baseline_metrics['total']['mae']:.3f}")

    # Ridge
    print("\n=== WR Ridge Multi-Target ===")
    adj_val = compute_wr_fumble_adjustment(wr_val)
    adj_test = compute_wr_fumble_adjustment(wr_test)
    y_val_actual = wr_val["fantasy_points"].values

    best_alpha, best_val_mae = None, float("inf")
    for alpha in WR_RIDGE_ALPHAS:
        ridge = WRRidgeMultiTarget(alpha=alpha)
        ridge.fit(X_train, y_train_dict)
        val_preds = ridge.predict(X_val)
        val_total_adj = val_preds["total"] + adj_val.values
        val_mae = np.mean(np.abs(val_total_adj - y_val_actual))
        print(f"  Alpha={alpha:.2f}: Val MAE={val_mae:.3f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_alpha = alpha

    print(f"  Best alpha: {best_alpha}")
    ridge_model = WRRidgeMultiTarget(alpha=best_alpha)
    ridge_model.fit(X_train, y_train_dict)
    ridge_test_preds = ridge_model.predict(X_test)
    ridge_test_preds["total"] = (
        sum(ridge_test_preds[t] for t in WR_TARGETS) + adj_test.values
    )
    ridge_metrics = compute_wr_metrics(y_test_dict, ridge_test_preds)

    # NN
    print("\n=== WR Multi-Head Neural Net ===")
    nn_scaler = StandardScaler()
    X_train_s = nn_scaler.fit_transform(X_train)
    X_val_s = nn_scaler.transform(X_val)
    X_test_s = nn_scaler.transform(X_test)

    train_loader, val_loader = make_wr_dataloaders(
        X_train_s, y_train_dict, X_val_s, y_val_dict, batch_size=WR_NN_BATCH_SIZE,
    )

    device = torch.device("cpu")
    model = WRMultiHeadNet(
        input_dim=X_train_s.shape[1],
        backbone_layers=WR_NN_BACKBONE_LAYERS, head_hidden=WR_NN_HEAD_HIDDEN,
        dropout=WR_NN_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=WR_NN_LR, weight_decay=WR_NN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=WR_COSINE_T0, T_mult=WR_COSINE_T_MULT, eta_min=WR_COSINE_ETA_MIN,
    )
    criterion = MultiTargetLoss(
        w_receiving=WR_LOSS_WEIGHTS["receiving_floor"],
        w_rushing=WR_LOSS_WEIGHTS["rushing_floor"],
        w_td=WR_LOSS_WEIGHTS["td_points"],
        w_total=WR_LOSS_W_TOTAL,
        huber_deltas=WR_HUBER_DELTAS,
    )

    trainer = WRMultiHeadTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, device=device,
        patience=WR_NN_PATIENCE,
    )
    history = trainer.train(train_loader, val_loader, n_epochs=WR_NN_EPOCHS)

    nn_test_preds = model.predict_numpy(X_test_s, device)
    nn_test_preds["total"] = (
        sum(nn_test_preds[t] for t in WR_TARGETS) + adj_test.values
    )
    nn_metrics = compute_wr_metrics(y_test_dict, nn_test_preds)

    print_wr_comparison_table({
        "Season Average Baseline": baseline_metrics,
        "WR Ridge Multi-Target": ridge_metrics,
        "WR Multi-Head NN": nn_metrics,
    })

    wr_test = wr_test.copy()
    wr_test["pred_ridge_total"] = ridge_test_preds["total"]
    wr_test["pred_nn_total"] = nn_test_preds["total"]
    wr_test["pred_baseline"] = baseline_preds

    ridge_ranking = compute_wr_ranking_metrics(wr_test, pred_col="pred_ridge_total")
    nn_ranking = compute_wr_ranking_metrics(wr_test, pred_col="pred_nn_total")
    print(f"\nRidge Top-12 Hit Rate: {ridge_ranking['season_avg_hit_rate']:.3f}")
    print(f"NN Top-12 Hit Rate:    {nn_ranking['season_avg_hit_rate']:.3f}")

    print("\n=== Weekly Backtest ===")
    sim_results = run_wr_weekly_simulation(
        wr_test,
        pred_columns={"Season Avg": "pred_baseline", "Ridge": "pred_ridge_total", "Neural Net": "pred_nn_total"},
    )
    for model_name, summary in sim_results["season_summary"].items():
        print(f"  {model_name}: MAE={summary['mae']:.3f}, R2={summary['r2']:.3f}")

    os.makedirs("WR/outputs/models", exist_ok=True)
    os.makedirs("WR/outputs/figures", exist_ok=True)

    ridge_model.save("WR/outputs/models")
    torch.save(model.state_dict(), "WR/outputs/models/wr_multihead_nn.pt")
    joblib.dump(nn_scaler, "WR/outputs/models/nn_scaler.pkl")

    trainer.plot_training_curves(history, "WR/outputs/figures/wr_training_curves.png")
    plot_wr_weekly_accuracy(sim_results, "WR/outputs/figures/wr_weekly_mae.png")
    plot_wr_pred_vs_actual(
        y_test_dict, nn_test_preds, "WR Multi-Head NN",
        "WR/outputs/figures/wr_pred_vs_actual_scatter.png",
    )

    feature_importance = ridge_model.get_feature_importance(feature_cols_final)
    fig, axes = plt.subplots(1, len(WR_TARGETS), figsize=(6 * len(WR_TARGETS), 8))
    for ax, (target, importance) in zip(axes, feature_importance.items()):
        importance.head(15).plot(kind="barh", ax=ax)
        ax.set_title(f"Ridge: {target} Top-15 Features")
        ax.set_xlabel("Absolute Coefficient")
    plt.tight_layout()
    plt.savefig("WR/outputs/figures/wr_ridge_feature_importance.png", dpi=150)
    plt.close()

    print("\nWR pipeline complete. Outputs saved to WR/outputs/")
    return {
        "ridge_metrics": ridge_metrics, "nn_metrics": nn_metrics,
        "ridge_ranking": ridge_ranking, "nn_ranking": nn_ranking,
        "history": history, "sim_results": sim_results,
    }


if __name__ == "__main__":
    run_wr_pipeline()
