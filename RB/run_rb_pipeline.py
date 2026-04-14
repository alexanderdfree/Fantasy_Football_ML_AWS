"""
End-to-end RB position model pipeline.

Can be run standalone (loads general splits from disk) or called from
the general run_pipeline.py with DataFrames passed directly.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SPLITS_DIR
from src.evaluation.metrics import compute_metrics
from src.models.baseline import SeasonAverageBaseline

from RB.rb_config import (
    RB_TARGETS, RB_RIDGE_ALPHAS, RB_SPECIFIC_FEATURES,
    RB_NN_BACKBONE_LAYERS, RB_NN_HEAD_HIDDEN, RB_NN_DROPOUT,
    RB_NN_LR, RB_NN_WEIGHT_DECAY, RB_NN_EPOCHS, RB_NN_BATCH_SIZE,
    RB_NN_PATIENCE, RB_SCHEDULER_PATIENCE, RB_SCHEDULER_FACTOR,
    RB_LOSS_W_RUSHING, RB_LOSS_W_RECEIVING, RB_LOSS_W_TD, RB_LOSS_W_TOTAL,
)
from RB.rb_data import filter_to_rb
from RB.rb_targets import compute_rb_targets, compute_fumble_adjustment
from RB.rb_features import add_rb_specific_features, get_rb_feature_columns, fill_rb_nans
from RB.rb_models import RBRidgeMultiTarget
from RB.rb_neural_net import RBMultiHeadNet
from RB.rb_training import MultiTargetLoss, RBMultiHeadTrainer, make_rb_dataloaders
from RB.rb_evaluation import (
    compute_rb_metrics, compute_rb_ranking_metrics,
    print_rb_comparison_table, plot_rb_pred_vs_actual,
)
from RB.rb_backtest import run_rb_weekly_simulation, plot_rb_weekly_accuracy


def run_rb_pipeline(train_df=None, val_df=None, test_df=None):
    """Run the full RB position model pipeline."""

    # --- Step 1: Load data ---
    if train_df is None:
        print("Loading general splits from disk...")
        train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
        val_df = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
        test_df = pd.read_parquet(f"{SPLITS_DIR}/test.parquet")

    # --- Step 2: Filter to RB ---
    print("Filtering to RB...")
    rb_train = filter_to_rb(train_df)
    rb_val = filter_to_rb(val_df)
    rb_test = filter_to_rb(test_df)
    print(f"  RB splits: train={len(rb_train)}, val={len(rb_val)}, test={len(rb_test)}")

    # --- Step 3: Compute targets ---
    print("Computing RB targets...")
    rb_train = compute_rb_targets(rb_train)
    rb_val = compute_rb_targets(rb_val)
    rb_test = compute_rb_targets(rb_test)

    # --- Step 4: Add RB-specific features ---
    print("Adding RB-specific features...")
    rb_train, rb_val, rb_test = add_rb_specific_features(rb_train, rb_val, rb_test)

    # --- Step 5: Fill NaNs for RB features ---
    rb_train, rb_val, rb_test = fill_rb_nans(rb_train, rb_val, rb_test, RB_SPECIFIC_FEATURES)

    # --- Step 6: Prepare feature arrays ---
    feature_cols = get_rb_feature_columns()
    # Only keep columns that actually exist in the data
    available_cols = [c for c in feature_cols if c in rb_train.columns]
    missing_cols = [c for c in feature_cols if c not in rb_train.columns]
    if missing_cols:
        print(f"  WARNING: {len(missing_cols)} feature columns missing, filling with 0")
        for col in missing_cols:
            for df in [rb_train, rb_val, rb_test]:
                df[col] = 0.0
    feature_cols_final = feature_cols

    # Replace any remaining NaN/inf
    for df in [rb_train, rb_val, rb_test]:
        df[feature_cols_final] = df[feature_cols_final].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train = rb_train[feature_cols_final].values.astype(np.float32)
    X_val = rb_val[feature_cols_final].values.astype(np.float32)
    X_test = rb_test[feature_cols_final].values.astype(np.float32)

    y_train_dict = {t: rb_train[t].values for t in RB_TARGETS}
    y_train_dict["total"] = rb_train["fantasy_points"].values
    y_val_dict = {t: rb_val[t].values for t in RB_TARGETS}
    y_val_dict["total"] = rb_val["fantasy_points"].values
    y_test_dict = {t: rb_test[t].values for t in RB_TARGETS}
    y_test_dict["total"] = rb_test["fantasy_points"].values

    print(f"  Feature matrix shape: {X_train.shape}")

    # --- Step 7: Baseline ---
    print("\n=== RB Baseline ===")
    baseline = SeasonAverageBaseline()
    baseline_preds = baseline.predict(rb_test)
    baseline_metrics = {"total": compute_metrics(y_test_dict["total"], baseline_preds)}
    print(f"  Season Avg Baseline MAE: {baseline_metrics['total']['mae']:.3f}")

    # --- Step 8: Ridge multi-target with alpha tuning ---
    print("\n=== RB Ridge Multi-Target ===")
    best_alpha = None
    best_val_mae = float("inf")
    for alpha in RB_RIDGE_ALPHAS:
        ridge = RBRidgeMultiTarget(alpha=alpha)
        ridge.fit(X_train, y_train_dict)
        val_preds = ridge.predict(X_val)
        val_mae = np.mean(np.abs(val_preds["total"] - y_val_dict["total"]))
        print(f"  Alpha={alpha:.2f}: Val MAE={val_mae:.3f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_alpha = alpha

    print(f"  Best alpha: {best_alpha}")
    ridge_model = RBRidgeMultiTarget(alpha=best_alpha)
    ridge_model.fit(X_train, y_train_dict)
    ridge_test_preds = ridge_model.predict(X_test)

    # Add fumble adjustment to Ridge predictions
    fumble_adj = compute_fumble_adjustment(rb_test)
    ridge_test_preds["total"] = (
        ridge_test_preds["rushing_floor"]
        + ridge_test_preds["receiving_floor"]
        + ridge_test_preds["td_points"]
        + fumble_adj.values
    )

    ridge_metrics = compute_rb_metrics(y_test_dict, ridge_test_preds)

    # --- Step 9: Multi-head NN ---
    print("\n=== RB Multi-Head Neural Net ===")
    # Scale features using the rushing Ridge model's scaler
    scaler = ridge_model.rushing_model.scaler
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create dataloaders
    train_loader, val_loader = make_rb_dataloaders(
        X_train_scaled, y_train_dict, X_val_scaled, y_val_dict,
        batch_size=RB_NN_BATCH_SIZE,
    )

    # Initialize model
    device = torch.device("cpu")
    input_dim = X_train_scaled.shape[1]
    model = RBMultiHeadNet(
        input_dim=input_dim,
        backbone_layers=RB_NN_BACKBONE_LAYERS,
        head_hidden=RB_NN_HEAD_HIDDEN,
        dropout=RB_NN_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=RB_NN_LR, weight_decay=RB_NN_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=RB_SCHEDULER_PATIENCE, factor=RB_SCHEDULER_FACTOR,
    )
    criterion = MultiTargetLoss(
        w_rushing=RB_LOSS_W_RUSHING, w_receiving=RB_LOSS_W_RECEIVING,
        w_td=RB_LOSS_W_TD, w_total=RB_LOSS_W_TOTAL,
    )

    trainer = RBMultiHeadTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, device=device, patience=RB_NN_PATIENCE,
    )

    history = trainer.train(train_loader, val_loader, n_epochs=RB_NN_EPOCHS)

    # Test predictions
    nn_test_preds = model.predict_numpy(X_test_scaled, device)
    # Add fumble adjustment
    nn_test_preds["total"] = (
        nn_test_preds["rushing_floor"]
        + nn_test_preds["receiving_floor"]
        + nn_test_preds["td_points"]
        + fumble_adj.values
    )
    nn_metrics = compute_rb_metrics(y_test_dict, nn_test_preds)

    # --- Step 10: Comparison ---
    print_rb_comparison_table({
        "Season Average Baseline": baseline_metrics,
        "RB Ridge Multi-Target": ridge_metrics,
        "RB Multi-Head NN": nn_metrics,
    })

    # --- Step 11: Ranking metrics ---
    rb_test = rb_test.copy()
    rb_test["pred_ridge_total"] = ridge_test_preds["total"]
    rb_test["pred_nn_total"] = nn_test_preds["total"]
    rb_test["pred_baseline"] = baseline_preds

    ridge_ranking = compute_rb_ranking_metrics(rb_test, pred_col="pred_ridge_total")
    nn_ranking = compute_rb_ranking_metrics(rb_test, pred_col="pred_nn_total")

    print(f"\nRidge Top-12 Hit Rate: {ridge_ranking['season_avg_hit_rate']:.3f}")
    print(f"NN Top-12 Hit Rate:    {nn_ranking['season_avg_hit_rate']:.3f}")
    print(f"Ridge Spearman:        {ridge_ranking['season_avg_spearman']:.3f}")
    print(f"NN Spearman:           {nn_ranking['season_avg_spearman']:.3f}")

    # --- Step 12: Weekly backtest ---
    print("\n=== Weekly Backtest ===")
    sim_results = run_rb_weekly_simulation(
        rb_test,
        pred_columns={
            "Season Avg": "pred_baseline",
            "Ridge": "pred_ridge_total",
            "Neural Net": "pred_nn_total",
        },
    )
    for model_name, summary in sim_results["season_summary"].items():
        print(f"  {model_name}: MAE={summary['mae']:.3f}, R2={summary['r2']:.3f}")

    # --- Step 13: Save outputs ---
    os.makedirs("RB/outputs/models", exist_ok=True)
    os.makedirs("RB/outputs/figures", exist_ok=True)

    ridge_model.save("RB/outputs/models")
    torch.save(model.state_dict(), "RB/outputs/models/rb_multihead_nn.pt")

    # Save figures
    trainer.plot_training_curves(history, "RB/outputs/figures/rb_training_curves.png")
    plot_rb_weekly_accuracy(sim_results, "RB/outputs/figures/rb_weekly_mae.png")
    plot_rb_pred_vs_actual(
        y_test_dict, nn_test_preds, "RB Multi-Head NN",
        "RB/outputs/figures/rb_pred_vs_actual_scatter.png",
    )

    # Ridge feature importance
    feature_importance = ridge_model.get_feature_importance(feature_cols_final)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, (target, importance) in zip(axes, feature_importance.items()):
        importance.head(15).plot(kind="barh", ax=ax)
        ax.set_title(f"Ridge: {target} Top-15 Features")
        ax.set_xlabel("Absolute Coefficient")
    plt.tight_layout()
    plt.savefig("RB/outputs/figures/rb_ridge_feature_importance.png", dpi=150)
    plt.close()

    print("\nRB pipeline complete. Outputs saved to RB/outputs/")

    return {
        "ridge_metrics": ridge_metrics,
        "nn_metrics": nn_metrics,
        "ridge_ranking": ridge_ranking,
        "nn_ranking": nn_ranking,
        "history": history,
        "sim_results": sim_results,
    }


if __name__ == "__main__":
    results = run_rb_pipeline()
