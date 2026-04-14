"""End-to-end TE position model pipeline."""

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

from TE.te_config import (
    TE_TARGETS, TE_RIDGE_ALPHAS, TE_SPECIFIC_FEATURES,
    TE_NN_BACKBONE_LAYERS, TE_NN_HEAD_HIDDEN, TE_NN_HEAD_HIDDEN_OVERRIDES,
    TE_NN_DROPOUT, TE_NN_LR, TE_NN_WEIGHT_DECAY, TE_NN_EPOCHS, TE_NN_BATCH_SIZE,
    TE_NN_PATIENCE,
    TE_LOSS_WEIGHTS, TE_LOSS_W_TOTAL, TE_HUBER_DELTAS,
    TE_SCHEDULER_TYPE, TE_ONECYCLE_MAX_LR, TE_ONECYCLE_PCT_START,
)
from TE.te_data import filter_to_te
from TE.te_targets import compute_te_targets, compute_te_fumble_adjustment
from TE.te_features import add_te_specific_features, get_te_feature_columns, fill_te_nans

from TE.te_models import TERidgeMultiTarget
from TE.te_neural_net import TEMultiHeadNet
from TE.te_training import MultiTargetLoss, TEMultiHeadTrainer, make_te_dataloaders
from TE.te_evaluation import (
    compute_te_metrics, compute_te_ranking_metrics,
    print_te_comparison_table, plot_te_pred_vs_actual,
)
from TE.te_backtest import run_te_weekly_simulation, plot_te_weekly_accuracy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_te_pipeline(train_df=None, val_df=None, test_df=None, seed=42):
    """Run the full TE position model pipeline."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if train_df is None:
        print("Loading general splits from disk...")
        train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
        val_df = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
        test_df = pd.read_parquet(f"{SPLITS_DIR}/test.parquet")

    print("Filtering to TE...")
    te_train = filter_to_te(train_df)
    te_val = filter_to_te(val_df)
    te_test = filter_to_te(test_df)
    print(f"  TE splits: train={len(te_train)}, val={len(te_val)}, test={len(te_test)}")

    print("Computing TE targets...")
    te_train = compute_te_targets(te_train)
    te_val = compute_te_targets(te_val)
    te_test = compute_te_targets(te_test)

    print("Adding TE-specific features...")
    te_train, te_val, te_test = add_te_specific_features(te_train, te_val, te_test)
    te_train, te_val, te_test = fill_te_nans(te_train, te_val, te_test, TE_SPECIFIC_FEATURES)

    feature_cols = get_te_feature_columns()
    missing_cols = [c for c in feature_cols if c not in te_train.columns]
    if missing_cols:
        print(f"  WARNING: {len(missing_cols)} feature columns missing, filling with 0")
        for col in missing_cols:
            for df in [te_train, te_val, te_test]:
                df[col] = 0.0
    feature_cols_final = feature_cols

    for df in [te_train, te_val, te_test]:
        df[feature_cols_final] = df[feature_cols_final].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train = te_train[feature_cols_final].values.astype(np.float32)
    X_val = te_val[feature_cols_final].values.astype(np.float32)
    X_test = te_test[feature_cols_final].values.astype(np.float32)

    y_train_dict = {t: te_train[t].values for t in TE_TARGETS}
    y_val_dict = {t: te_val[t].values for t in TE_TARGETS}
    y_test_dict = {t: te_test[t].values for t in TE_TARGETS}

    y_train_dict["total"] = sum(te_train[t].values for t in TE_TARGETS)
    y_val_dict["total"] = sum(te_val[t].values for t in TE_TARGETS)
    y_test_dict["total"] = te_test["fantasy_points"].values

    print(f"  Feature matrix shape: {X_train.shape}")

    # Baseline
    print("\n=== TE Baseline ===")
    baseline = SeasonAverageBaseline()
    baseline_preds = baseline.predict(te_test)
    baseline_metrics = {"total": compute_metrics(y_test_dict["total"], baseline_preds)}
    print(f"  Season Avg Baseline MAE: {baseline_metrics['total']['mae']:.3f}")

    # Ridge
    print("\n=== TE Ridge Multi-Target ===")
    adj_val = compute_te_fumble_adjustment(te_val)
    adj_test = compute_te_fumble_adjustment(te_test)
    y_val_actual = te_val["fantasy_points"].values

    best_alpha, best_val_mae = None, float("inf")
    for alpha in TE_RIDGE_ALPHAS:
        ridge = TERidgeMultiTarget(alpha=alpha)
        ridge.fit(X_train, y_train_dict)
        val_preds = ridge.predict(X_val)
        val_total_adj = val_preds["total"] + adj_val.values
        val_mae = np.mean(np.abs(val_total_adj - y_val_actual))
        print(f"  Alpha={alpha:.2f}: Val MAE={val_mae:.3f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_alpha = alpha

    print(f"  Best alpha: {best_alpha}")
    ridge_model = TERidgeMultiTarget(alpha=best_alpha)
    ridge_model.fit(X_train, y_train_dict)
    ridge_test_preds = ridge_model.predict(X_test)
    ridge_test_preds["total"] = (
        sum(ridge_test_preds[t] for t in TE_TARGETS) + adj_test.values
    )
    ridge_metrics = compute_te_metrics(y_test_dict, ridge_test_preds)

    # NN
    print("\n=== TE Multi-Head Neural Net ===")
    nn_scaler = StandardScaler()
    X_train_s = nn_scaler.fit_transform(X_train)
    X_val_s = nn_scaler.transform(X_val)
    X_test_s = nn_scaler.transform(X_test)

    train_loader, val_loader = make_te_dataloaders(
        X_train_s, y_train_dict, X_val_s, y_val_dict, batch_size=TE_NN_BATCH_SIZE,
    )

    device = torch.device("cpu")
    model = TEMultiHeadNet(
        input_dim=X_train_s.shape[1],
        backbone_layers=TE_NN_BACKBONE_LAYERS, head_hidden=TE_NN_HEAD_HIDDEN,
        td_head_hidden=TE_NN_HEAD_HIDDEN_OVERRIDES.get("td_points"),
        dropout=TE_NN_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TE_NN_LR, weight_decay=TE_NN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=TE_ONECYCLE_MAX_LR,
        epochs=TE_NN_EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=TE_ONECYCLE_PCT_START,
    )
    criterion = MultiTargetLoss(
        w_receiving=TE_LOSS_WEIGHTS["receiving_floor"],
        w_rushing=TE_LOSS_WEIGHTS["rushing_floor"],
        w_td=TE_LOSS_WEIGHTS["td_points"],
        w_total=TE_LOSS_W_TOTAL,
        huber_deltas=TE_HUBER_DELTAS,
    )

    trainer = TEMultiHeadTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, device=device,
        patience=TE_NN_PATIENCE, scheduler_per_batch=True,
    )
    history = trainer.train(train_loader, val_loader, n_epochs=TE_NN_EPOCHS)

    nn_test_preds = model.predict_numpy(X_test_s, device)
    nn_test_preds["total"] = (
        sum(nn_test_preds[t] for t in TE_TARGETS) + adj_test.values
    )
    nn_metrics = compute_te_metrics(y_test_dict, nn_test_preds)

    print_te_comparison_table({
        "Season Average Baseline": baseline_metrics,
        "TE Ridge Multi-Target": ridge_metrics,
        "TE Multi-Head NN": nn_metrics,
    })

    te_test = te_test.copy()
    te_test["pred_ridge_total"] = ridge_test_preds["total"]
    te_test["pred_nn_total"] = nn_test_preds["total"]
    te_test["pred_baseline"] = baseline_preds

    ridge_ranking = compute_te_ranking_metrics(te_test, pred_col="pred_ridge_total")
    nn_ranking = compute_te_ranking_metrics(te_test, pred_col="pred_nn_total")
    print(f"\nRidge Top-12 Hit Rate: {ridge_ranking['season_avg_hit_rate']:.3f}")
    print(f"NN Top-12 Hit Rate:    {nn_ranking['season_avg_hit_rate']:.3f}")

    print("\n=== Weekly Backtest ===")
    sim_results = run_te_weekly_simulation(
        te_test,
        pred_columns={"Season Avg": "pred_baseline", "Ridge": "pred_ridge_total", "Neural Net": "pred_nn_total"},
    )
    for model_name, summary in sim_results["season_summary"].items():
        print(f"  {model_name}: MAE={summary['mae']:.3f}, R2={summary['r2']:.3f}")

    os.makedirs("TE/outputs/models", exist_ok=True)
    os.makedirs("TE/outputs/figures", exist_ok=True)

    ridge_model.save("TE/outputs/models")
    torch.save(model.state_dict(), "TE/outputs/models/te_multihead_nn.pt")
    joblib.dump(nn_scaler, "TE/outputs/models/nn_scaler.pkl")

    trainer.plot_training_curves(history, "TE/outputs/figures/te_training_curves.png")
    plot_te_weekly_accuracy(sim_results, "TE/outputs/figures/te_weekly_mae.png")
    plot_te_pred_vs_actual(
        y_test_dict, nn_test_preds, "TE Multi-Head NN",
        "TE/outputs/figures/te_pred_vs_actual_scatter.png",
    )

    feature_importance = ridge_model.get_feature_importance(feature_cols_final)
    fig, axes = plt.subplots(1, len(TE_TARGETS), figsize=(6 * len(TE_TARGETS), 8))
    for ax, (target, importance) in zip(axes, feature_importance.items()):
        importance.head(15).plot(kind="barh", ax=ax)
        ax.set_title(f"Ridge: {target} Top-15 Features")
        ax.set_xlabel("Absolute Coefficient")
    plt.tight_layout()
    plt.savefig("TE/outputs/figures/te_ridge_feature_importance.png", dpi=150)
    plt.close()

    print("\nTE pipeline complete. Outputs saved to TE/outputs/")
    return {
        "ridge_metrics": ridge_metrics, "nn_metrics": nn_metrics,
        "ridge_ranking": ridge_ranking, "nn_ranking": nn_ranking,
        "history": history, "sim_results": sim_results,
    }


if __name__ == "__main__":
    run_te_pipeline()
