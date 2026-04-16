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

from shared.models import RidgeMultiTarget, LightGBMMultiTarget
from shared.neural_net import MultiHeadNet, MultiHeadNetWithHistory
from shared.training import (
    MultiTargetLoss, MultiHeadTrainer, MultiHeadHistoryTrainer,
    make_dataloaders, make_history_dataloaders, plot_training_curves,
)
from shared.evaluation import (
    compute_target_metrics, compute_ranking_metrics,
    print_comparison_table, plot_pred_vs_actual,
)
from shared.backtest import run_weekly_simulation, plot_weekly_accuracy
from shared.weather_features import merge_schedule_features
from src.features.engineer import build_game_history_arrays, get_attn_static_columns


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


def _eval_alpha_cv(X, y, folds, alpha, pca_n_components=None):
    """Evaluate a single Ridge alpha across CV folds, returning mean MAE."""
    maes = []
    for train_idx, val_idx in folds:
        model = RidgeModel(alpha=alpha, pca_n_components=pca_n_components)
        model.fit(X[train_idx], y[train_idx])
        preds = np.maximum(model.predict(X[val_idx]), 0)
        maes.append(np.mean(np.abs(preds - y[val_idx])))
    return np.mean(maes)


def _tune_ridge_alphas_cv(X_train, y_train_dict, split_values, targets,
                          alpha_grids, n_cv_folds=4, refine_points=5,
                          pca_n_components=None):
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
            mae = _eval_alpha_cv(X_train, y, folds, alpha, pca_n_components)
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha

        # --- Pass 2: fine refinement ---
        if refine_points > 0 and len(grid) >= 2:
            log_step = np.log10(grid[1]) - np.log10(grid[0])
            center = np.log10(best_alpha)
            fine_grid = np.logspace(center - log_step, center + log_step, refine_points)
            for alpha in fine_grid:
                mae = _eval_alpha_cv(X_train, y, folds, alpha, pca_n_components)
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

    # Merge schedule-derived weather/Vegas features before target & feature computation
    from shared.weather_features import merge_schedule_features
    for _df in dfs_for_features:
        merge_schedule_features(_df)

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
        empty = pos_val.iloc[:0].copy()
        pos_train, pos_val, _dummy = cfg["add_features_fn"](pos_train, pos_val, empty)
        pos_train, pos_val, _dummy = cfg["fill_nans_fn"](
            pos_train, pos_val, empty, cfg["specific_features"]
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

    # Total target = sum of decomposed targets (NOT fantasy_points, which
    # includes adjustments like INT/fumble penalties that are added post-hoc).
    # Using fantasy_points here causes the total aux loss to train heads to
    # absorb adjustments, which then get double-counted at inference.
    y_train_dict["total"] = sum(pos_train[t].values for t in targets)
    y_val_dict["total"] = sum(pos_val[t].values for t in targets)
    if y_test_dict is not None:
        y_test_dict["total"] = sum(pos_test[t].values for t in targets)

    return (X_train, X_val, X_test,
            y_train_dict, y_val_dict, y_test_dict,
            pos_train, pos_val, pos_test, feature_cols)


def _train_nn(X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict,
              cfg, targets, seed):
    """Train a MultiHeadNet and return (model, scaler, test_preds, metrics, history).

    Shared by the regular NN and Weather NN to guarantee identical training.
    The only thing that differs between them is the input feature matrix.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    nn_scaler = StandardScaler()
    X_train_s = np.clip(nn_scaler.fit_transform(X_train), -4, 4)
    X_val_s = np.clip(nn_scaler.transform(X_val), -4, 4)
    X_test_s = np.clip(nn_scaler.transform(X_test), -4, 4)

    train_loader, val_loader = make_dataloaders(
        X_train_s, y_train_dict, X_val_s, y_val_dict, batch_size=cfg["nn_batch_size"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadNet(
        input_dim=X_train_s.shape[1],
        target_names=targets,
        backbone_layers=cfg["nn_backbone_layers"],
        head_hidden=cfg["nn_head_hidden"],
        dropout=cfg["nn_dropout"],
        head_hidden_overrides=cfg.get("nn_head_hidden_overrides"),
        non_negative_targets=cfg.get("nn_non_negative_targets"),
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
        log_every=cfg.get("nn_log_every", 10),
    )
    history = trainer.train(train_loader, val_loader, n_epochs=cfg["nn_epochs"])

    val_preds = model.predict_numpy(X_val_s, device)
    val_preds["total"] = sum(val_preds[t] for t in targets)

    test_preds = model.predict_numpy(X_test_s, device)
    test_preds["total"] = sum(test_preds[t] for t in targets)
    metrics = compute_target_metrics(y_test_dict, test_preds, targets)

    return model, nn_scaler, val_preds, test_preds, metrics, history


def _train_attention_nn(X_train, X_val, X_test,
                        hist_train, mask_train, hist_val, mask_val,
                        hist_test, mask_test,
                        y_train_dict, y_val_dict, y_test_dict,
                        cfg, targets, seed, feature_cols=None):
    """Train a MultiHeadNetWithHistory and return (model, scaler, test_preds, metrics, history).

    Like _train_nn but feeds both static features and game history sequences.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Filter out rolling/EWMA/trend/share features that duplicate game history —
    # the attention branch learns its own temporal representation from raw game stats.
    if feature_cols is not None:
        static_cols = get_attn_static_columns(feature_cols)
        col_idx = [i for i, c in enumerate(feature_cols) if c in set(static_cols)]
        X_train = X_train[:, col_idx]
        X_val = X_val[:, col_idx]
        X_test = X_test[:, col_idx]
        print(f"  Attention static features: {len(col_idx)}/{len(feature_cols)} (filtered)")

    nn_scaler = StandardScaler()
    X_train_s = np.clip(nn_scaler.fit_transform(X_train), -4, 4)
    X_val_s = np.clip(nn_scaler.transform(X_val), -4, 4)
    X_test_s = np.clip(nn_scaler.transform(X_test), -4, 4)

    # Convert history arrays to lists-of-arrays for the variable-length dataset
    def _to_history_list(hist_arr, mask_arr):
        """Convert padded [n, max_len, dim] to list of [actual_len, dim]."""
        result = []
        for i in range(len(hist_arr)):
            seq_len = mask_arr[i].sum()
            result.append(hist_arr[i, :seq_len])
        return result

    attn_batch_size = cfg.get("attn_batch_size", cfg["nn_batch_size"])
    train_loader, val_loader = make_history_dataloaders(
        X_train_s, _to_history_list(hist_train, mask_train), y_train_dict,
        X_val_s, _to_history_list(hist_val, mask_val), y_val_dict,
        batch_size=attn_batch_size,
    )

    game_dim = hist_train.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadNetWithHistory(
        static_dim=X_train_s.shape[1],
        game_dim=game_dim,
        target_names=targets,
        backbone_layers=cfg["nn_backbone_layers"],
        d_model=cfg.get("attn_d_model", 32),
        n_attn_heads=cfg.get("attn_n_heads", 2),
        head_hidden=cfg["nn_head_hidden"],
        dropout=cfg["nn_dropout"],
        head_hidden_overrides=cfg.get("nn_head_hidden_overrides"),
        non_negative_targets=cfg.get("nn_non_negative_targets"),
        project_kv=cfg.get("attn_project_kv", False),
        use_positional_encoding=cfg.get("attn_positional_encoding", False),
        max_seq_len=cfg.get("attn_max_seq_len", 17),
        use_gated_fusion=cfg.get("attn_gated_fusion", False),
        attn_dropout=cfg.get("attn_dropout", 0.0),
        encoder_hidden_dim=cfg.get("attn_encoder_hidden_dim", 0),
        gated_td=cfg.get("attn_gated_td", False),
        td_gate_hidden=cfg.get("attn_td_gate_hidden", 16),
        gated_td_target=cfg.get("gated_td_target", "td_points"),
    ).to(device)

    attn_lr = cfg.get("attn_lr", cfg["nn_lr"])
    attn_wd = cfg.get("attn_weight_decay", cfg["nn_weight_decay"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=attn_lr, weight_decay=attn_wd,
    )
    scheduler, scheduler_per_batch = _build_scheduler(optimizer, cfg, train_loader)
    criterion = MultiTargetLoss(
        target_names=targets,
        loss_weights=cfg["loss_weights"],
        huber_deltas=cfg["huber_deltas"],
        w_total=cfg["loss_w_total"],
        td_gate_weight=cfg.get("attn_td_gate_weight", 1.0),
        gated_td_target=cfg.get("gated_td_target", "td_points"),
    )

    attn_patience = cfg.get("attn_patience", cfg["nn_patience"])
    trainer = MultiHeadHistoryTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, device=device, target_names=targets,
        patience=attn_patience, scheduler_per_batch=scheduler_per_batch,
        log_every=cfg.get("nn_log_every", 10),
    )
    history = trainer.train(train_loader, val_loader, n_epochs=cfg["nn_epochs"])

    val_preds = model.predict_numpy(X_val_s, hist_val, mask_val, device)
    val_preds["total"] = sum(val_preds[t] for t in targets)

    test_preds = model.predict_numpy(X_test_s, hist_test, mask_test, device)
    test_preds["total"] = sum(test_preds[t] for t in targets)
    metrics = compute_target_metrics(y_test_dict, test_preds, targets)

    return model, nn_scaler, val_preds, test_preds, metrics, history


def _train_lightgbm(X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict,
                    cfg, targets, feature_cols, seed):
    """Train a LightGBM multi-target model. Returns (model, val_preds, test_preds, metrics)."""
    model = LightGBMMultiTarget(
        target_names=targets,
        n_estimators=cfg.get("lgbm_n_estimators", 500),
        learning_rate=cfg.get("lgbm_learning_rate", 0.05),
        num_leaves=cfg.get("lgbm_num_leaves", 31),
        max_depth=cfg.get("lgbm_max_depth", -1),
        subsample=cfg.get("lgbm_subsample", 0.8),
        colsample_bytree=cfg.get("lgbm_colsample_bytree", 0.8),
        reg_lambda=cfg.get("lgbm_reg_lambda", 1.0),
        reg_alpha=cfg.get("lgbm_reg_alpha", 0.0),
        min_child_samples=cfg.get("lgbm_min_child_samples", 20),
        min_split_gain=cfg.get("lgbm_min_split_gain", 0.0),
        objective=cfg.get("lgbm_objective", "huber"),
        seed=seed,
    )
    model.fit(X_train, y_train_dict, X_val, y_val_dict, feature_names=feature_cols)

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)
    metrics = compute_target_metrics(y_test_dict, test_preds, targets)
    return model, val_preds, test_preds, metrics


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
    two_stage_targets = cfg.get("two_stage_targets", {})
    classification_targets = cfg.get("classification_targets", {})
    pca_n = cfg.get("ridge_pca_components")
    special_targets = set(two_stage_targets) | set(classification_targets)
    ridge_tune_targets = [t for t in targets if t not in special_targets]
    ridge_tune_grids = {t: cfg["ridge_alpha_grids"][t] for t in ridge_tune_targets}
    best_alphas = _tune_ridge_alphas_cv(
        X_train, y_train_dict, pos_train[cv_col].values,
        targets=ridge_tune_targets,
        alpha_grids=ridge_tune_grids,
        n_cv_folds=cfg.get("ridge_cv_folds", 4),
        refine_points=cfg.get("ridge_refine_points", 5),
        pca_n_components=pca_n,
    ) if ridge_tune_targets else {}
    if two_stage_targets:
        print(f"  Two-stage targets: {list(two_stage_targets.keys())}")
    if classification_targets:
        print(f"  Ordinal classification targets: {list(classification_targets.keys())}")
    if pca_n:
        print(f"  PCR: {pca_n} components")
    ridge_non_neg = cfg.get("nn_non_negative_targets")
    ridge_model = RidgeMultiTarget(target_names=targets, alpha=best_alphas,
                                   two_stage_targets=two_stage_targets,
                                   classification_targets=classification_targets,
                                   pca_n_components=pca_n,
                                   non_negative_targets=ridge_non_neg)
    ridge_model.fit(X_train, y_train_dict)
    ridge_test_preds = ridge_model.predict(X_test)
    # Evaluate total as sum(per-target preds) without adjustment — matches
    # the training total target (sum of clean targets, no penalties).
    # Adjustment is only applied at inference (app.py).
    ridge_test_preds["total"] = sum(ridge_test_preds[t] for t in targets)
    ridge_metrics = compute_target_metrics(y_test_dict, ridge_test_preds, targets)

    # --- Multi-head NN ---
    print(f"\n=== {pos} Multi-Head Neural Net ===")
    model, nn_scaler, nn_val_preds, nn_test_preds, nn_metrics, history = _train_nn(
        X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict,
        cfg, targets, seed,
    )

    # --- Attention NN (game history as variable-length sequences) ---
    attn_nn_val_preds = None
    attn_nn_test_preds = None
    attn_nn_metrics = None
    attn_model = None
    attn_nn_scaler = None
    attn_history = None
    if cfg.get("train_attention_nn", False):
        print(f"\n=== {pos} Attention Multi-Head Neural Net ===")
        history_stats = cfg.get("attn_history_stats", None)
        max_seq_len = cfg.get("attn_max_seq_len", 17)

        hist_train, mask_train = build_game_history_arrays(
            pos_train, history_stats=history_stats, max_seq_len=max_seq_len)
        hist_val, mask_val = build_game_history_arrays(
            pos_val, history_stats=history_stats, max_seq_len=max_seq_len)
        hist_test, mask_test = build_game_history_arrays(
            pos_test, history_stats=history_stats, max_seq_len=max_seq_len)
        print(f"  History shape: {hist_train.shape} (game_dim={hist_train.shape[2]})")

        attn_model, attn_nn_scaler, attn_nn_val_preds, attn_nn_test_preds, attn_nn_metrics, attn_history = (
            _train_attention_nn(
                X_train, X_val, X_test,
                hist_train, mask_train, hist_val, mask_val,
                hist_test, mask_test,
                y_train_dict, y_val_dict, y_test_dict,
                cfg, targets, seed,
                feature_cols=feature_cols,
            )
        )

    # --- LightGBM Multi-Target (conditional) ---
    lgbm_val_preds = None
    lgbm_test_preds = None
    lgbm_metrics = None
    lgbm_model = None
    if cfg.get("train_lightgbm", False):
        print(f"\n=== {pos} LightGBM Multi-Target ===")
        lgbm_model, lgbm_val_preds, lgbm_test_preds, lgbm_metrics = _train_lightgbm(
            X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict,
            cfg, targets, feature_cols, seed,
        )

    # --- Comparison ---
    comparison = {
        "Season Average Baseline": baseline_metrics,
        f"{pos} Ridge Multi-Target": ridge_metrics,
        f"{pos} Multi-Head NN": nn_metrics,
    }
    if attn_nn_metrics is not None:
        comparison[f"{pos} Attention NN"] = attn_nn_metrics
    if lgbm_metrics is not None:
        comparison[f"{pos} LightGBM"] = lgbm_metrics
    print_comparison_table(comparison, position=pos, target_names=targets)

    # --- Attach predictions to test DataFrame ---
    pos_test = pos_test.copy()
    pos_test["pred_ridge_total"] = ridge_test_preds["total"]
    pos_test["pred_nn_total"] = nn_test_preds["total"]
    pos_test["pred_baseline"] = baseline_preds
    for t in targets:
        pos_test[f"pred_ridge_{t}"] = ridge_test_preds[t]
        pos_test[f"pred_nn_{t}"] = nn_test_preds[t]

    backtest_pred_columns = {
        "Season Avg": "pred_baseline",
        "Ridge": "pred_ridge_total",
        "Neural Net": "pred_nn_total",
    }

    ridge_ranking = compute_ranking_metrics(pos_test, pred_col="pred_ridge_total")
    nn_ranking = compute_ranking_metrics(pos_test, pred_col="pred_nn_total")
    print(f"\nRidge Top-12 Hit Rate:    {ridge_ranking['season_avg_hit_rate']:.3f}")
    print(f"NN Top-12 Hit Rate:       {nn_ranking['season_avg_hit_rate']:.3f}")

    attn_nn_ranking = None
    if attn_nn_test_preds is not None:
        pos_test["pred_attn_nn_total"] = attn_nn_test_preds["total"]
        for t in targets:
            pos_test[f"pred_attn_nn_{t}"] = attn_nn_test_preds[t]
        backtest_pred_columns["Attention NN"] = "pred_attn_nn_total"
        attn_nn_ranking = compute_ranking_metrics(pos_test, pred_col="pred_attn_nn_total")
        print(f"Attention NN Top-12 Hit Rate: {attn_nn_ranking['season_avg_hit_rate']:.3f}")

    lgbm_ranking = None
    if lgbm_test_preds is not None:
        pos_test["pred_lgbm_total"] = lgbm_test_preds["total"]
        for t in targets:
            pos_test[f"pred_lgbm_{t}"] = lgbm_test_preds[t]
        backtest_pred_columns["LightGBM"] = "pred_lgbm_total"
        lgbm_ranking = compute_ranking_metrics(pos_test, pred_col="pred_lgbm_total")
        print(f"LightGBM Top-12 Hit Rate: {lgbm_ranking['season_avg_hit_rate']:.3f}")

    # --- Weekly backtest ---
    print("\n=== Weekly Backtest ===")
    sim_results = run_weekly_simulation(pos_test, pred_columns=backtest_pred_columns)
    for model_name, summary in sim_results["season_summary"].items():
        print(f"  {model_name}: MAE={summary['mae']:.3f}, R2={summary['r2']:.3f}")

    # --- Save outputs ---
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    ridge_model.save(f"{output_dir}/models")
    torch.save(model.state_dict(), f"{output_dir}/models/{pos_lower}_multihead_nn.pt")
    joblib.dump(nn_scaler, f"{output_dir}/models/nn_scaler.pkl")

    if attn_model is not None:
        torch.save(attn_model.state_dict(), f"{output_dir}/models/{pos_lower}_attention_nn.pt")
        joblib.dump(attn_nn_scaler, f"{output_dir}/models/attention_nn_scaler.pkl")

    if lgbm_model is not None:
        lgbm_model.save(f"{output_dir}/models")

    plot_training_curves(history, targets, f"{output_dir}/figures/{pos_lower}_training_curves.png")
    if attn_history is not None:
        plot_training_curves(attn_history, targets,
                             f"{output_dir}/figures/{pos_lower}_attention_training_curves.png")
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

    if lgbm_model is not None:
        lgbm_importance = lgbm_model.get_feature_importance(feature_cols)
        fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 8))
        if len(targets) == 1:
            axes = [axes]
        for ax, (target, importance) in zip(axes, lgbm_importance.items()):
            importance.head(15).plot(kind="barh", ax=ax)
            ax.set_title(f"LightGBM: {target} Top-15 Features")
            ax.set_xlabel("Gain")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/{pos_lower}_lgbm_feature_importance.png", dpi=150)
        plt.close()

    print(f"\n{pos} pipeline complete. Outputs saved to {output_dir}/")
    per_target_preds = {
        "ridge": ridge_test_preds,
        "nn": nn_test_preds,
    }
    if attn_nn_test_preds is not None:
        per_target_preds["attn_nn"] = attn_nn_test_preds
    if lgbm_test_preds is not None:
        per_target_preds["lgbm"] = lgbm_test_preds

    result = {
        "ridge_metrics": ridge_metrics,
        "nn_metrics": nn_metrics,
        "ridge_ranking": ridge_ranking,
        "nn_ranking": nn_ranking,
        "history": history,
        "sim_results": sim_results,
        "test_df": pos_test,
        "per_target_preds": per_target_preds,
    }
    if attn_nn_metrics is not None:
        result["attn_nn_metrics"] = attn_nn_metrics
        result["attn_nn_ranking"] = attn_nn_ranking
    if lgbm_metrics is not None:
        result["lgbm_metrics"] = lgbm_metrics
        result["lgbm_ranking"] = lgbm_ranking
    return result


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
    cv_special = set(cfg.get("two_stage_targets", {})) | set(cfg.get("classification_targets", {}))
    cv_ridge_targets = [t for t in targets if t not in cv_special]
    cv_ridge_grids = {t: cfg["ridge_alpha_grids"][t] for t in cv_ridge_targets}
    best_alphas = _tune_ridge_alphas_cv(
        X_alpha, y_alpha_dict, pos_alpha[cv_col].values,
        targets=cv_ridge_targets, alpha_grids=cv_ridge_grids,
        n_cv_folds=cfg.get("ridge_cv_folds", 4),
        refine_points=cfg.get("ridge_refine_points", 5),
    )

    # --- Per-fold training ---
    fold_nn_metrics = []
    fold_ridge_metrics = []
    fold_lgbm_metrics = []

    for fold_idx, fold_train_df, fold_val_df in folds:
        print(f"\n--- Fold {fold_idx + 1} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)

        (X_train, X_val, _,
         y_train_dict, y_val_dict, _,
         pos_train, pos_val, _, feature_cols) = _prepare_position_data(
            position, cfg, fold_train_df, fold_val_df
        )

        ridge_fold = RidgeMultiTarget(
            target_names=targets, alpha=best_alphas,
            two_stage_targets=cfg.get("two_stage_targets", {}),
            classification_targets=cfg.get("classification_targets", {}),
            pca_n_components=cfg.get("ridge_pca_components"),
            non_negative_targets=cfg.get("nn_non_negative_targets"),
        )
        ridge_fold.fit(X_train, y_train_dict)
        ridge_val_preds = ridge_fold.predict(X_val)
        ridge_val_preds["total"] = sum(ridge_val_preds[t] for t in targets)
        fold_ridge_metrics.append(
            compute_target_metrics(y_val_dict, ridge_val_preds, targets)
        )

        # NN training for this fold
        nn_scaler = StandardScaler()
        X_train_s = np.clip(nn_scaler.fit_transform(X_train), -4, 4)
        X_val_s = np.clip(nn_scaler.transform(X_val), -4, 4)

        train_loader, val_loader = make_dataloaders(
            X_train_s, y_train_dict, X_val_s, y_val_dict,
            batch_size=cfg["nn_batch_size"],
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiHeadNet(
            input_dim=X_train_s.shape[1],
            target_names=targets,
            backbone_layers=cfg["nn_backbone_layers"],
            head_hidden=cfg["nn_head_hidden"],
            dropout=cfg["nn_dropout"],
            head_hidden_overrides=cfg.get("nn_head_hidden_overrides"),
            non_negative_targets=cfg.get("nn_non_negative_targets"),
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
            log_every=cfg.get("nn_log_every", 10),
        )
        trainer.train(train_loader, val_loader, n_epochs=cfg["nn_epochs"])

        nn_val_preds = model.predict_numpy(X_val_s, device)
        nn_val_preds["total"] = sum(nn_val_preds[t] for t in targets)
        fold_nn_metrics.append(
            compute_target_metrics(y_val_dict, nn_val_preds, targets)
        )

        # LightGBM for this fold
        if cfg.get("train_lightgbm", False):
            lgbm_fold = LightGBMMultiTarget(
                target_names=targets,
                n_estimators=cfg.get("lgbm_n_estimators", 500),
                learning_rate=cfg.get("lgbm_learning_rate", 0.05),
                num_leaves=cfg.get("lgbm_num_leaves", 31),
                max_depth=cfg.get("lgbm_max_depth", -1),
                subsample=cfg.get("lgbm_subsample", 0.8),
                colsample_bytree=cfg.get("lgbm_colsample_bytree", 0.8),
                reg_lambda=cfg.get("lgbm_reg_lambda", 1.0),
                reg_alpha=cfg.get("lgbm_reg_alpha", 0.0),
                min_child_samples=cfg.get("lgbm_min_child_samples", 20),
                min_split_gain=cfg.get("lgbm_min_split_gain", 0.0),
                objective=cfg.get("lgbm_objective", "huber"),
                seed=seed,
            )
            lgbm_fold.fit(X_train, y_train_dict, X_val, y_val_dict,
                          feature_names=feature_cols)
            lgbm_val_preds = lgbm_fold.predict(X_val)
            lgbm_val_preds["total"] = sum(lgbm_val_preds[t] for t in targets)
            fold_lgbm_metrics.append(
                compute_target_metrics(y_val_dict, lgbm_val_preds, targets)
            )

    # --- Aggregate CV results ---
    print(f"\n{'=' * 60}")
    print(f"  {pos} Cross-Validation Results ({len(folds)} folds)")
    print(f"{'=' * 60}")

    # Final per-target alphas: tune on full pre-test training data
    print(f"\nFinal per-target Ridge alpha tuning (full training data)...")

    # Aggregate per-fold metrics
    cv_metrics = {"ridge": {}, "nn": {}}
    model_fold_pairs = [("ridge", fold_ridge_metrics), ("nn", fold_nn_metrics)]
    if fold_lgbm_metrics:
        cv_metrics["lgbm"] = {}
        model_fold_pairs.append(("lgbm", fold_lgbm_metrics))
    for model_name, fold_metrics_list in model_fold_pairs:
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

    cv_model_names = ["ridge", "nn"] + (["lgbm"] if fold_lgbm_metrics else [])
    print(f"\n{'Model':<12} {'MAE (mean +/- std)':>22} {'R2 (mean +/- std)':>22}")
    print("-" * 58)
    for model_name in cv_model_names:
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
    best_cv_alphas = best_alphas

    ridge_model = RidgeMultiTarget(
        target_names=targets, alpha=best_cv_alphas,
        two_stage_targets=cfg.get("two_stage_targets", {}),
        classification_targets=cfg.get("classification_targets", {}),
        pca_n_components=cfg.get("ridge_pca_components"),
        non_negative_targets=cfg.get("nn_non_negative_targets"),
    )
    ridge_model.fit(X_train, y_train_dict)
    ridge_test_preds = ridge_model.predict(X_test)
    ridge_test_preds["total"] = sum(ridge_test_preds[t] for t in targets)
    ridge_metrics = compute_target_metrics(y_test_dict, ridge_test_preds, targets)

    # NN
    print(f"\n=== {pos} Multi-Head NN (Final Holdout) ===")
    model, nn_scaler, nn_val_preds, nn_test_preds, nn_metrics, history = _train_nn(
        X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict,
        cfg, targets, seed,
    )

    # LightGBM
    lgbm_test_preds = None
    lgbm_metrics = None
    lgbm_model = None
    if cfg.get("train_lightgbm", False):
        print(f"\n=== {pos} LightGBM Multi-Target (Final Holdout) ===")
        lgbm_model, _, lgbm_test_preds, lgbm_metrics = _train_lightgbm(
            X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict,
            cfg, targets, feature_cols, seed,
        )

    # Comparison
    comparison = {
        "Season Average Baseline": baseline_metrics,
        f"{pos} Ridge (per-target CV alphas)": ridge_metrics,
        f"{pos} Multi-Head NN": nn_metrics,
    }
    if lgbm_metrics is not None:
        comparison[f"{pos} LightGBM"] = lgbm_metrics
    print_comparison_table(comparison, position=pos, target_names=targets)

    # Ranking metrics
    pos_test = pos_test.copy()
    pos_test["pred_ridge_total"] = ridge_test_preds["total"]
    pos_test["pred_nn_total"] = nn_test_preds["total"]
    pos_test["pred_baseline"] = baseline_preds

    backtest_pred_columns = {
        "Season Avg": "pred_baseline",
        "Ridge": "pred_ridge_total",
        "Neural Net": "pred_nn_total",
    }

    ridge_ranking = compute_ranking_metrics(pos_test, pred_col="pred_ridge_total")
    nn_ranking = compute_ranking_metrics(pos_test, pred_col="pred_nn_total")
    print(f"\nRidge Top-12 Hit Rate:    {ridge_ranking['season_avg_hit_rate']:.3f}")
    print(f"NN Top-12 Hit Rate:       {nn_ranking['season_avg_hit_rate']:.3f}")

    lgbm_ranking = None
    if lgbm_test_preds is not None:
        pos_test["pred_lgbm_total"] = lgbm_test_preds["total"]
        for t in targets:
            pos_test[f"pred_lgbm_{t}"] = lgbm_test_preds[t]
        backtest_pred_columns["LightGBM"] = "pred_lgbm_total"
        lgbm_ranking = compute_ranking_metrics(pos_test, pred_col="pred_lgbm_total")
        print(f"LightGBM Top-12 Hit Rate: {lgbm_ranking['season_avg_hit_rate']:.3f}")

    # Weekly backtest
    print("\n=== Weekly Backtest ===")
    sim_results = run_weekly_simulation(pos_test, pred_columns=backtest_pred_columns)
    for model_name, summary in sim_results["season_summary"].items():
        print(f"  {model_name}: MAE={summary['mae']:.3f}, R2={summary['r2']:.3f}")

    # Save outputs
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    ridge_model.save(f"{output_dir}/models")
    torch.save(model.state_dict(), f"{output_dir}/models/{pos_lower}_multihead_nn.pt")
    joblib.dump(nn_scaler, f"{output_dir}/models/nn_scaler.pkl")

    if lgbm_model is not None:
        lgbm_model.save(f"{output_dir}/models")

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

    if lgbm_model is not None:
        lgbm_importance = lgbm_model.get_feature_importance(feature_cols)
        fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 8))
        if len(targets) == 1:
            axes = [axes]
        for ax, (target, importance) in zip(axes, lgbm_importance.items()):
            importance.head(15).plot(kind="barh", ax=ax)
            ax.set_title(f"LightGBM: {target} Top-15 Features")
            ax.set_xlabel("Gain")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/{pos_lower}_lgbm_feature_importance.png", dpi=150)
        plt.close()

    print(f"\n{pos} CV pipeline complete. Outputs saved to {output_dir}/")
    result = {
        "cv_metrics": cv_metrics,
        "best_cv_alphas": best_cv_alphas,
        "ridge_metrics": ridge_metrics,
        "nn_metrics": nn_metrics,
        "ridge_ranking": ridge_ranking,
        "nn_ranking": nn_ranking,
        "history": history,
        "sim_results": sim_results,
    }
    if lgbm_metrics is not None:
        result["lgbm_metrics"] = lgbm_metrics
        result["lgbm_ranking"] = lgbm_ranking
    return result
