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

from src.config import SPLITS_DIR, MIN_GAMES_PER_SEASON, TRAIN_SEASONS, VAL_SEASONS
from src.evaluation.metrics import compute_metrics
from src.models.baseline import SeasonAverageBaseline
from src.models.linear import RidgeModel
from src.data.split import expanding_window_folds

from shared.models import RidgeMultiTarget
from shared.neural_net import MultiHeadNet
from shared.training import MultiTargetLoss, MultiHeadTrainer, make_dataloaders, plot_training_curves
from shared.evaluation import (
    compute_target_metrics, compute_ranking_metrics,
    print_comparison_table, plot_pred_vs_actual,
)
from shared.backtest import run_weekly_simulation, plot_weekly_accuracy


# ---------------------------------------------------------------------------
# Ridge hyperparameter tuning helpers
# ---------------------------------------------------------------------------

def _build_expanding_cv_folds(split_values, n_folds):
    """Build expanding-window cross-validation fold indices.

    Args:
        split_values: Array of season or week labels for each row.
        n_folds: Number of CV folds.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    unique_periods = sorted(np.unique(split_values))
    val_periods = unique_periods[-n_folds:]

    folds = []
    for val_period in val_periods:
        train_mask = split_values < val_period
        val_mask = split_values == val_period
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        folds.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return folds


def _eval_alpha_cv(X, y, folds, alpha):
    """Evaluate a single Ridge alpha across CV folds, returning mean MAE."""
    maes = []
    for train_idx, val_idx in folds:
        model = RidgeModel(alpha=alpha)
        model.fit(X[train_idx], y[train_idx])
        preds = np.maximum(model.predict(X[val_idx]), 0)
        maes.append(np.mean(np.abs(preds - y[val_idx])))
    return np.mean(maes)


def _tune_ridge_alphas_cv(X_train, y_train_dict, split_values, targets,
                          alpha_grids, n_cv_folds=4, refine_points=5):
    """Per-target Ridge alpha tuning with expanding-window CV.

    Pass 1: coarse grid search across CV folds.
    Pass 2: fine refinement around the best coarse alpha.

    Returns dict mapping each target name to its optimal alpha.
    """
    folds = _build_expanding_cv_folds(split_values, n_cv_folds)
    best_alphas = {}

    for target in targets:
        y = y_train_dict[target]
        grid = alpha_grids[target]

        # --- Pass 1: coarse search ---
        best_mae, best_alpha = float("inf"), grid[0]
        for alpha in grid:
            mae = _eval_alpha_cv(X_train, y, folds, alpha)
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha

        # --- Pass 2: fine refinement ---
        if refine_points > 0 and len(grid) >= 2:
            log_step = np.log10(grid[1]) - np.log10(grid[0])
            center = np.log10(best_alpha)
            fine_grid = np.logspace(center - log_step, center + log_step, refine_points)
            for alpha in fine_grid:
                mae = _eval_alpha_cv(X_train, y, folds, alpha)
                if mae < best_mae:
                    best_mae = mae
                    best_alpha = alpha

        best_alphas[target] = round(best_alpha, 6)
        print(f"  {target}: best alpha={best_alphas[target]:.4f} (CV MAE={best_mae:.3f})")

    return best_alphas


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


def _prepare_position_data(position, cfg, train_df, val_df, test_df=None):
    """Filter to position, compute targets, add features, build arrays.

    Returns:
        (X_train, X_val, X_test_or_None,
         y_train_dict, y_val_dict, y_test_dict_or_None,
         pos_train, pos_val, pos_test_or_None, feature_cols)
    """
    pos = position

    # Filter to position
    pos_train = cfg["filter_fn"](train_df)
    pos_val = cfg["filter_fn"](val_df)
    pos_test = cfg["filter_fn"](test_df) if test_df is not None else None

    # Min-games filter: training only
    games_per_season = pos_train.groupby(["player_id", "season"])["week"].transform("count")
    pos_train = pos_train[games_per_season >= MIN_GAMES_PER_SEASON].copy()

    dfs_for_features = [pos_train, pos_val] + ([pos_test] if pos_test is not None else [])
    sizes = ", ".join(
        f"{name}={len(df)}"
        for name, df in zip(["train", "val", "test"], dfs_for_features)
    )
    print(f"  {pos} splits: {sizes}")

    # Compute targets
    targets = cfg["targets"]
    pos_train = cfg["compute_targets_fn"](pos_train)
    pos_val = cfg["compute_targets_fn"](pos_val)
    if pos_test is not None:
        pos_test = cfg["compute_targets_fn"](pos_test)

    # Add position-specific features
    if pos_test is not None:
        pos_train, pos_val, pos_test = cfg["add_features_fn"](pos_train, pos_val, pos_test)
        pos_train, pos_val, pos_test = cfg["fill_nans_fn"](
            pos_train, pos_val, pos_test, cfg["specific_features"]
        )
    else:
        pos_train, pos_val, _dummy = cfg["add_features_fn"](pos_train, pos_val, pos_val)
        pos_train, pos_val, _dummy = cfg["fill_nans_fn"](
            pos_train, pos_val, pos_val, cfg["specific_features"]
        )

    # Build feature arrays
    feature_cols = cfg["get_feature_columns_fn"]()
    all_dfs = [pos_train, pos_val] + ([pos_test] if pos_test is not None else [])
    missing_cols = [c for c in feature_cols if c not in pos_train.columns]
    if missing_cols:
        print(f"  WARNING: {len(missing_cols)} feature columns missing, filling with 0")
        for col in missing_cols:
            for df in all_dfs:
                df[col] = 0.0

    for df in all_dfs:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train = pos_train[feature_cols].values.astype(np.float32)
    X_val = pos_val[feature_cols].values.astype(np.float32)
    X_test = pos_test[feature_cols].values.astype(np.float32) if pos_test is not None else None

    y_train_dict = {t: pos_train[t].values for t in targets}
    y_val_dict = {t: pos_val[t].values for t in targets}
    y_test_dict = {t: pos_test[t].values for t in targets} if pos_test is not None else None

    y_train_dict["total"] = pos_train["fantasy_points"].values
    y_val_dict["total"] = pos_val["fantasy_points"].values
    if y_test_dict is not None:
        y_test_dict["total"] = pos_test["fantasy_points"].values

    return (X_train, X_val, X_test,
            y_train_dict, y_val_dict, y_test_dict,
            pos_train, pos_val, pos_test, feature_cols)


def run_pipeline(position, cfg, train_df=None, val_df=None, test_df=None, seed=42):
    """Run the full position model pipeline.

    Args:
        position: Position abbreviation (e.g. "QB", "RB", "WR", "TE")
        cfg: Dict with keys:
            # Targets & features
            targets: list[str]
            ridge_alpha_grids: dict[str, list[float]]
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

    # --- Prepare position data ---
    print(f"Preparing {pos} data...")
    (X_train, X_val, X_test,
     y_train_dict, y_val_dict, y_test_dict,
     pos_train, pos_val, pos_test, feature_cols) = _prepare_position_data(
        position, cfg, train_df, val_df, test_df
    )

    print(f"  Feature matrix shape: {X_train.shape}")

    # --- Baseline ---
    print(f"\n=== {pos} Baseline ===")
    baseline = SeasonAverageBaseline()
    baseline_preds = baseline.predict(pos_test)
    baseline_metrics = {"total": compute_metrics(y_test_dict["total"], baseline_preds)}
    print(f"  Season Avg Baseline MAE: {baseline_metrics['total']['mae']:.3f}")

    # --- Ridge multi-target with per-target alpha tuning ---
    print(f"\n=== {pos} Ridge Multi-Target (Per-Target CV Tuning) ===")
    adj_test = cfg["compute_adjustment_fn"](pos_test)

    cv_col = cfg.get("cv_split_column", "season")
    best_alphas = _tune_ridge_alphas_cv(
        X_train, y_train_dict, pos_train[cv_col].values,
        targets=targets,
        alpha_grids=cfg["ridge_alpha_grids"],
        n_cv_folds=cfg.get("ridge_cv_folds", 4),
        refine_points=cfg.get("ridge_refine_points", 5),
    )

    ridge_model = RidgeMultiTarget(target_names=targets, alpha=best_alphas)
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


def run_cv_pipeline(position, cfg, full_df=None, test_df=None, seed=42):
    """Run expanding-window cross-validation, then final holdout evaluation.

    CV folds determine best Ridge alpha and report multi-season metrics for
    both Ridge and NN.  After CV, retrains on all pre-test data and evaluates
    on the holdout test set.

    Args:
        position: Position abbreviation (e.g. "QB")
        cfg: Position config dict (same as run_pipeline)
        full_df: Combined DataFrame containing all seasons (train + val).
                 If None, loads train + val from disk and concatenates.
        test_df: Holdout test DataFrame. If None, loads from disk.
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    pos = position
    pos_lower = pos.lower()
    targets = cfg["targets"]
    output_dir = f"{pos}/outputs"

    # --- Load data ---
    if full_df is None:
        print("Loading splits from disk and combining for CV...")
        train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
        val_df = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
        full_df = pd.concat([train_df, val_df], ignore_index=True)
    if test_df is None:
        test_df = pd.read_parquet(f"{SPLITS_DIR}/test.parquet")

    # --- Generate CV folds ---
    print(f"\n{'=' * 60}")
    print(f"  {pos} Expanding-Window Cross-Validation")
    print(f"{'=' * 60}")
    folds = expanding_window_folds(full_df)

    # --- Tune Ridge alphas once on full training data ---
    print(f"\nTuning Ridge alphas on full training data...")
    alpha_train_df = full_df[full_df["season"].isin(TRAIN_SEASONS)].copy()
    alpha_val_df = full_df[full_df["season"].isin(VAL_SEASONS)].copy()
    (X_alpha, _, _, y_alpha_dict, _, _, pos_alpha, _, _, _) = _prepare_position_data(
        position, cfg, alpha_train_df, alpha_val_df
    )
    cv_col = cfg.get("cv_split_column", "season")
    best_alphas = _tune_ridge_alphas_cv(
        X_alpha, y_alpha_dict, pos_alpha[cv_col].values,
        targets=targets, alpha_grids=cfg["ridge_alpha_grids"],
        n_cv_folds=cfg.get("ridge_cv_folds", 4),
        refine_points=cfg.get("ridge_refine_points", 5),
    )

    # --- Per-fold training ---
    fold_nn_metrics = []
    fold_ridge_metrics = []

    for fold_idx, fold_train_df, fold_val_df in folds:
        print(f"\n--- Fold {fold_idx + 1} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)

        (X_train, X_val, _,
         y_train_dict, y_val_dict, _,
         pos_train, pos_val, _, feature_cols) = _prepare_position_data(
            position, cfg, fold_train_df, fold_val_df
        )

        adj_val = cfg["compute_adjustment_fn"](pos_val)

        ridge_fold = RidgeMultiTarget(target_names=targets, alpha=best_alphas)
        ridge_fold.fit(X_train, y_train_dict)
        ridge_val_preds = ridge_fold.predict(X_val)
        ridge_val_preds["total"] = sum(ridge_val_preds[t] for t in targets) + adj_val.values
        fold_ridge_metrics.append(
            compute_target_metrics(y_val_dict, ridge_val_preds, targets)
        )

        # NN training for this fold
        nn_scaler = StandardScaler()
        X_train_s = nn_scaler.fit_transform(X_train)
        X_val_s = nn_scaler.transform(X_val)

        train_loader, val_loader = make_dataloaders(
            X_train_s, y_train_dict, X_val_s, y_val_dict,
            batch_size=cfg["nn_batch_size"],
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
        trainer.train(train_loader, val_loader, n_epochs=cfg["nn_epochs"])

        nn_val_preds = model.predict_numpy(X_val_s, device)
        nn_val_preds["total"] = sum(nn_val_preds[t] for t in targets) + adj_val.values
        fold_nn_metrics.append(
            compute_target_metrics(y_val_dict, nn_val_preds, targets)
        )

    # --- Aggregate CV results ---
    print(f"\n{'=' * 60}")
    print(f"  {pos} Cross-Validation Results ({len(folds)} folds)")
    print(f"{'=' * 60}")

    # Final per-target alphas: tune on full pre-test training data
    print(f"\nFinal per-target Ridge alpha tuning (full training data)...")

    # Aggregate per-fold metrics
    cv_metrics = {"ridge": {}, "nn": {}}
    for model_name, fold_metrics_list in [("ridge", fold_ridge_metrics), ("nn", fold_nn_metrics)]:
        for key in ["total"] + targets:
            maes = [fm[key]["mae"] for fm in fold_metrics_list]
            r2s = [fm[key]["r2"] for fm in fold_metrics_list]
            cv_metrics[model_name][key] = {
                "mae_mean": np.mean(maes),
                "mae_std": np.std(maes),
                "r2_mean": np.mean(r2s),
                "r2_std": np.std(r2s),
                "mae_per_fold": maes,
                "r2_per_fold": r2s,
            }

    print(f"\n{'Model':<12} {'MAE (mean +/- std)':>22} {'R2 (mean +/- std)':>22}")
    print("-" * 58)
    for model_name in ["ridge", "nn"]:
        m = cv_metrics[model_name]["total"]
        print(f"{model_name.upper():<12} {m['mae_mean']:>8.3f} +/- {m['mae_std']:<8.3f} "
              f"{m['r2_mean']:>8.3f} +/- {m['r2_std']:<8.3f}")

    # --- Final holdout evaluation ---
    print(f"\n{'=' * 60}")
    print(f"  {pos} Final Holdout Evaluation (test on 2025)")
    print(f"{'=' * 60}")

    # Build final train = all pre-test data, val = last CV fold (2024) for NN early stopping
    final_train_seasons = TRAIN_SEASONS
    final_val_seasons = VAL_SEASONS
    final_train_df = full_df[full_df["season"].isin(final_train_seasons)].copy()
    final_val_df = full_df[full_df["season"].isin(final_val_seasons)].copy()

    (X_train, X_val, X_test,
     y_train_dict, y_val_dict, y_test_dict,
     pos_train, pos_val, pos_test, feature_cols) = _prepare_position_data(
        position, cfg, final_train_df, final_val_df, test_df
    )

    # Baseline
    baseline = SeasonAverageBaseline()
    baseline_preds = baseline.predict(pos_test)
    baseline_metrics = {"total": compute_metrics(y_test_dict["total"], baseline_preds)}

    # Ridge with per-target CV alphas tuned on full training data
    adj_val = cfg["compute_adjustment_fn"](pos_val)
    adj_test = cfg["compute_adjustment_fn"](pos_test)

    best_cv_alphas = best_alphas

    ridge_model = RidgeMultiTarget(target_names=targets, alpha=best_cv_alphas)
    ridge_model.fit(X_train, y_train_dict)
    ridge_test_preds = ridge_model.predict(X_test)
    ridge_test_preds["total"] = sum(ridge_test_preds[t] for t in targets) + adj_test.values
    ridge_metrics = compute_target_metrics(y_test_dict, ridge_test_preds, targets)

    # NN
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    # Ensemble
    ensemble_preds = {}
    for t in targets:
        ensemble_preds[t] = 0.5 * ridge_test_preds[t] + 0.5 * nn_test_preds[t]
    ensemble_preds["total"] = 0.5 * ridge_test_preds["total"] + 0.5 * nn_test_preds["total"]
    ensemble_metrics = compute_target_metrics(y_test_dict, ensemble_preds, targets)

    # Comparison
    print_comparison_table({
        "Season Average Baseline": baseline_metrics,
        f"{pos} Ridge (per-target CV alphas)": ridge_metrics,
        f"{pos} Multi-Head NN": nn_metrics,
        f"{pos} Ensemble (Ridge+NN)": ensemble_metrics,
    }, position=pos, target_names=targets)

    # Ranking metrics
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

    # Weekly backtest
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

    # Save outputs
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

    print(f"\n{pos} CV pipeline complete. Outputs saved to {output_dir}/")
    return {
        "cv_metrics": cv_metrics,
        "best_cv_alphas": best_cv_alphas,
        "ridge_metrics": ridge_metrics,
        "nn_metrics": nn_metrics,
        "ridge_ranking": ridge_ranking,
        "nn_ranking": nn_ranking,
        "history": history,
        "sim_results": sim_results,
    }
