"""Generic position model pipeline.

Each position calls run_pipeline() with a config dict that bundles all
position-specific callables and hyperparameters.
"""

import os
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SPLITS_DIR
from src.evaluation.metrics import compute_metrics
from src.models.baseline import SeasonAverageBaseline

from shared.models import RidgeMultiTarget
from shared.neural_net import MultiHeadNet
from shared.training import MultiTargetLoss, MultiHeadTrainer, make_dataloaders, plot_training_curves
from shared.evaluation import (
    compute_target_metrics, compute_ranking_metrics,
    print_comparison_table, plot_pred_vs_actual,
)
from shared.backtest import run_weekly_simulation, plot_weekly_accuracy


def _build_scheduler(optimizer, cfg, train_loader):
    """Create the LR scheduler from config."""
    sched_type = cfg["scheduler_type"]
    if sched_type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["onecycle_max_lr"],
            epochs=cfg["nn_epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=cfg["onecycle_pct_start"],
        ), True  # scheduler_per_batch=True
    elif sched_type == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg["cosine_t0"],
            T_mult=cfg["cosine_t_mult"],
            eta_min=cfg["cosine_eta_min"],
        ), False  # scheduler_per_batch=False
    elif sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg["plateau_factor"],
            patience=cfg["plateau_patience"],
        ), False  # scheduler_per_batch=False
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


def run_pipeline(position, cfg, train_df=None, val_df=None, test_df=None, seed=42):
    """Run the full position model pipeline.

    Args:
        position: Position abbreviation (e.g. "QB", "RB", "WR", "TE")
        cfg: Dict with keys:
            # Targets & features
            targets: list[str]
            ridge_alphas: list[float]
            specific_features: list[str]
            # Position-specific callables
            filter_fn: callable(df) -> df
            compute_targets_fn: callable(df) -> df
            add_features_fn: callable(train, val, test) -> (train, val, test)
            fill_nans_fn: callable(train, val, test, features) -> (train, val, test)
            get_feature_columns_fn: callable() -> list[str]
            compute_adjustment_fn: callable(df) -> pd.Series
            # Neural net architecture
            nn_backbone_layers: list[int]
            nn_head_hidden: int
            nn_dropout: float
            nn_head_hidden_overrides: dict | None
            nn_lr: float
            nn_weight_decay: float
            nn_epochs: int
            nn_batch_size: int
            nn_patience: int
            # Loss
            loss_weights: dict[str, float]
            loss_w_total: float
            huber_deltas: dict[str, float]
            # Scheduler
            scheduler_type: str  ("onecycle" | "cosine_warm_restarts")
            + scheduler-specific keys (onecycle_max_lr, etc.)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    pos = position
    pos_lower = pos.lower()
    targets = cfg["targets"]
    output_dir = f"{pos}/outputs"

    # --- Load data ---
    if train_df is None:
        print("Loading general splits from disk...")
        train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
        val_df = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
        test_df = pd.read_parquet(f"{SPLITS_DIR}/test.parquet")

    # --- Filter to position ---
    print(f"Filtering to {pos}...")
    pos_train = cfg["filter_fn"](train_df)
    pos_val = cfg["filter_fn"](val_df)
    pos_test = cfg["filter_fn"](test_df)
    print(f"  {pos} splits: train={len(pos_train)}, val={len(pos_val)}, test={len(pos_test)}")

    # --- Compute targets ---
    print(f"Computing {pos} targets...")
    pos_train = cfg["compute_targets_fn"](pos_train)
    pos_val = cfg["compute_targets_fn"](pos_val)
    pos_test = cfg["compute_targets_fn"](pos_test)

    # --- Add position-specific features ---
    print(f"Adding {pos}-specific features...")
    pos_train, pos_val, pos_test = cfg["add_features_fn"](pos_train, pos_val, pos_test)
    pos_train, pos_val, pos_test = cfg["fill_nans_fn"](
        pos_train, pos_val, pos_test, cfg["specific_features"]
    )

    # --- Prepare feature arrays ---
    feature_cols = cfg["get_feature_columns_fn"]()
    missing_cols = [c for c in feature_cols if c not in pos_train.columns]
    if missing_cols:
        print(f"  WARNING: {len(missing_cols)} feature columns missing, filling with 0")
        for col in missing_cols:
            for df in [pos_train, pos_val, pos_test]:
                df[col] = 0.0

    for df in [pos_train, pos_val, pos_test]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train = pos_train[feature_cols].values.astype(np.float32)
    X_val = pos_val[feature_cols].values.astype(np.float32)
    X_test = pos_test[feature_cols].values.astype(np.float32)

    y_train_dict = {t: pos_train[t].values for t in targets}
    y_val_dict = {t: pos_val[t].values for t in targets}
    y_test_dict = {t: pos_test[t].values for t in targets}

    # Use actual fantasy_points as the total target across all splits.
    # sum(targets) omits fumble penalty and passing component, creating a
    # train/test mismatch.  Using fantasy_points gives the total-loss term
    # a consistent signal aligned with the evaluation metric.
    y_train_dict["total"] = pos_train["fantasy_points"].values
    y_val_dict["total"] = pos_val["fantasy_points"].values
    y_test_dict["total"] = pos_test["fantasy_points"].values

    print(f"  Feature matrix shape: {X_train.shape}")

    # --- Baseline ---
    print(f"\n=== {pos} Baseline ===")
    baseline = SeasonAverageBaseline()
    baseline_preds = baseline.predict(pos_test)
    baseline_metrics = {"total": compute_metrics(y_test_dict["total"], baseline_preds)}
    print(f"  Season Avg Baseline MAE: {baseline_metrics['total']['mae']:.3f}")

    # --- Ridge multi-target with alpha tuning ---
    print(f"\n=== {pos} Ridge Multi-Target ===")
    adj_val = cfg["compute_adjustment_fn"](pos_val)
    adj_test = cfg["compute_adjustment_fn"](pos_test)
    y_val_actual = pos_val["fantasy_points"].values

    best_alpha, best_val_mae = None, float("inf")
    for alpha in cfg["ridge_alphas"]:
        ridge = RidgeMultiTarget(target_names=targets, alpha=alpha)
        ridge.fit(X_train, y_train_dict)
        val_preds = ridge.predict(X_val)
        val_total_adj = val_preds["total"] + adj_val.values
        val_mae = np.mean(np.abs(val_total_adj - y_val_actual))
        print(f"  Alpha={alpha:.2f}: Val MAE={val_mae:.3f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_alpha = alpha

    print(f"  Best alpha: {best_alpha}")
    ridge_model = RidgeMultiTarget(target_names=targets, alpha=best_alpha)
    ridge_model.fit(X_train, y_train_dict)
    ridge_test_preds = ridge_model.predict(X_test)
    ridge_test_preds["total"] = sum(ridge_test_preds[t] for t in targets) + adj_test.values
    ridge_metrics = compute_target_metrics(y_test_dict, ridge_test_preds, targets)

    # --- Multi-head NN ---
    print(f"\n=== {pos} Multi-Head Neural Net ===")
    nn_scaler = StandardScaler()
    X_train_s = nn_scaler.fit_transform(X_train)
    X_val_s = nn_scaler.transform(X_val)
    X_test_s = nn_scaler.transform(X_test)

    train_loader, val_loader = make_dataloaders(
        X_train_s, y_train_dict, X_val_s, y_val_dict, batch_size=cfg["nn_batch_size"],
    )

    device = torch.device("cpu")
    model = MultiHeadNet(
        input_dim=X_train_s.shape[1],
        target_names=targets,
        backbone_layers=cfg["nn_backbone_layers"],
        head_hidden=cfg["nn_head_hidden"],
        dropout=cfg["nn_dropout"],
        head_hidden_overrides=cfg.get("nn_head_hidden_overrides"),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["nn_lr"], weight_decay=cfg["nn_weight_decay"],
    )
    scheduler, scheduler_per_batch = _build_scheduler(optimizer, cfg, train_loader)
    criterion = MultiTargetLoss(
        target_names=targets,
        loss_weights=cfg["loss_weights"],
        huber_deltas=cfg["huber_deltas"],
        w_total=cfg["loss_w_total"],
    )

    trainer = MultiHeadTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, device=device, target_names=targets,
        patience=cfg["nn_patience"], scheduler_per_batch=scheduler_per_batch,
    )
    history = trainer.train(train_loader, val_loader, n_epochs=cfg["nn_epochs"])

    nn_test_preds = model.predict_numpy(X_test_s, device)
    nn_test_preds["total"] = sum(nn_test_preds[t] for t in targets) + adj_test.values
    nn_metrics = compute_target_metrics(y_test_dict, nn_test_preds, targets)

    # --- Ensemble: average Ridge + NN ---
    # Ridge has good calibration (near-zero bias), NN has better per-target
    # precision.  Simple average combines both strengths.
    ensemble_preds = {}
    for t in targets:
        ensemble_preds[t] = 0.5 * ridge_test_preds[t] + 0.5 * nn_test_preds[t]
    ensemble_preds["total"] = 0.5 * ridge_test_preds["total"] + 0.5 * nn_test_preds["total"]
    ensemble_metrics = compute_target_metrics(y_test_dict, ensemble_preds, targets)

    # --- Comparison ---
    print_comparison_table({
        "Season Average Baseline": baseline_metrics,
        f"{pos} Ridge Multi-Target": ridge_metrics,
        f"{pos} Multi-Head NN": nn_metrics,
        f"{pos} Ensemble (Ridge+NN)": ensemble_metrics,
    }, position=pos, target_names=targets)

    # --- Ranking metrics ---
    pos_test = pos_test.copy()
    pos_test["pred_ridge_total"] = ridge_test_preds["total"]
    pos_test["pred_nn_total"] = nn_test_preds["total"]
    pos_test["pred_ensemble"] = ensemble_preds["total"]
    pos_test["pred_baseline"] = baseline_preds

    ridge_ranking = compute_ranking_metrics(pos_test, pred_col="pred_ridge_total")
    nn_ranking = compute_ranking_metrics(pos_test, pred_col="pred_nn_total")
    ensemble_ranking = compute_ranking_metrics(pos_test, pred_col="pred_ensemble")
    print(f"\nRidge Top-12 Hit Rate:    {ridge_ranking['season_avg_hit_rate']:.3f}")
    print(f"NN Top-12 Hit Rate:       {nn_ranking['season_avg_hit_rate']:.3f}")
    print(f"Ensemble Top-12 Hit Rate: {ensemble_ranking['season_avg_hit_rate']:.3f}")

    # --- Weekly backtest ---
    print("\n=== Weekly Backtest ===")
    sim_results = run_weekly_simulation(
        pos_test,
        pred_columns={
            "Season Avg": "pred_baseline",
            "Ridge": "pred_ridge_total",
            "Neural Net": "pred_nn_total",
            "Ensemble": "pred_ensemble",
        },
    )
    for model_name, summary in sim_results["season_summary"].items():
        print(f"  {model_name}: MAE={summary['mae']:.3f}, R2={summary['r2']:.3f}")

    # --- Save outputs ---
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    ridge_model.save(f"{output_dir}/models")
    torch.save(model.state_dict(), f"{output_dir}/models/{pos_lower}_multihead_nn.pt")
    joblib.dump(nn_scaler, f"{output_dir}/models/nn_scaler.pkl")

    plot_training_curves(history, targets, f"{output_dir}/figures/{pos_lower}_training_curves.png")
    plot_weekly_accuracy(sim_results, pos, f"{output_dir}/figures/{pos_lower}_weekly_mae.png")
    plot_pred_vs_actual(
        y_test_dict, nn_test_preds, targets, f"{pos} Multi-Head NN",
        f"{output_dir}/figures/{pos_lower}_pred_vs_actual_scatter.png",
    )

    feature_importance = ridge_model.get_feature_importance(feature_cols)
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 8))
    if len(targets) == 1:
        axes = [axes]
    for ax, (target, importance) in zip(axes, feature_importance.items()):
        importance.head(15).plot(kind="barh", ax=ax)
        ax.set_title(f"Ridge: {target} Top-15 Features")
        ax.set_xlabel("Absolute Coefficient")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/{pos_lower}_ridge_feature_importance.png", dpi=150)
    plt.close()

    print(f"\n{pos} pipeline complete. Outputs saved to {output_dir}/")
    return {
        "ridge_metrics": ridge_metrics,
        "nn_metrics": nn_metrics,
        "ridge_ranking": ridge_ranking,
        "nn_ranking": nn_ranking,
        "history": history,
        "sim_results": sim_results,
    }
