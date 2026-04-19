"""Optuna-based hyperparameter tuning for LightGBM across positions.

Usage:
    python tune_lgbm.py QB                     # tune one position
    python tune_lgbm.py QB RB WR TE            # tune all LGBM-enabled positions
    python tune_lgbm.py RB --n-trials 100      # more trials
    python tune_lgbm.py RB --timeout 3600      # time limit in seconds
    python tune_lgbm.py RB --print-best        # print best params from saved study
"""
import os
import sys
import json
import copy
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.config import SPLITS_DIR, TRAIN_SEASONS, VAL_SEASONS
from src.data.split import expanding_window_folds
from src.evaluation.metrics import compute_metrics
from shared.models import LightGBMMultiTarget
from shared.pipeline import _prepare_position_data
from shared.evaluation import compute_target_metrics, compute_ranking_metrics


# ---------------------------------------------------------------------------
# Position config loading
# ---------------------------------------------------------------------------

def _get_position_config(pos):
    """Import and return the CONFIG dict for a position."""
    from shared.registry import get_config
    return get_config(pos.upper())


# ---------------------------------------------------------------------------
# CV data preparation
# ---------------------------------------------------------------------------

def _prepare_cv_folds(pos, cfg):
    """Load data and build CV folds with prepared feature matrices.

    Returns:
        (folds_data, targets) where folds_data is a list of
        (X_train, X_val, y_train_dict, y_val_dict, feature_cols) tuples.
    """
    print(f"\nPreparing CV folds for {pos}...")
    train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
    val_df = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    folds = expanding_window_folds(full_df)
    targets = cfg["targets"]

    folds_data = []
    for fold_idx, fold_train_df, fold_val_df in folds:
        (X_train, X_val, _,
         y_train_dict, y_val_dict, _,
         _, _, _, feature_cols) = _prepare_position_data(
            pos, cfg, fold_train_df, fold_val_df
        )
        folds_data.append((X_train, X_val, y_train_dict, y_val_dict, feature_cols))

    print(f"  {len(folds_data)} folds prepared, {len(feature_cols)} features, "
          f"{len(targets)} targets: {targets}")
    return folds_data, targets


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _make_objective(folds_data, targets):
    """Return an Optuna objective function that evaluates LightGBM on CV folds."""

    def objective(trial):
        # --- Sample hyperparameters ---
        use_max_depth = trial.suggest_categorical("use_max_depth", [True, False])
        max_depth = trial.suggest_int("max_depth", 3, 10) if use_max_depth else -1

        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 8, 64),
            max_depth=max_depth,
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 300, 2000, step=100),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
            objective=trial.suggest_categorical("objective", ["huber", "fair", "regression"]),
        )

        # --- Evaluate across CV folds ---
        fold_maes = []
        for fold_i, (X_train, X_val, y_train_dict, y_val_dict, feature_cols) in enumerate(folds_data):
            model = LightGBMMultiTarget(target_names=targets, seed=42, **params)
            model.fit(X_train, y_train_dict, X_val, y_val_dict,
                      feature_names=feature_cols)

            preds = model.predict(X_val)
            total_mae = np.mean(np.abs(preds["total"] - y_val_dict["total"]))
            fold_maes.append(total_mae)

            # Report intermediate value for pruning
            trial.report(np.mean(fold_maes), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_maes)

    return objective


# ---------------------------------------------------------------------------
# Before/after comparison on holdout
# ---------------------------------------------------------------------------

def _run_comparison(pos, cfg, best_params):
    """Train with old and new params on final split, print comparison."""
    print(f"\n{'=' * 70}")
    print(f"  {pos} Before/After Comparison on Holdout Test (2025)")
    print(f"{'=' * 70}")

    train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
    val_df = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
    test_df = pd.read_parquet(f"{SPLITS_DIR}/test.parquet")

    targets = cfg["targets"]

    (X_train, X_val, X_test,
     y_train_dict, y_val_dict, y_test_dict,
     pos_train, pos_val, pos_test, feature_cols) = _prepare_position_data(
        pos, cfg, train_df, val_df, test_df
    )

    # --- Old params (from config) ---
    old_params = {
        k.replace("lgbm_", ""): v
        for k, v in cfg.items()
        if k.startswith("lgbm_") and k != "lgbm_objective"
    }
    old_params["objective"] = cfg.get("lgbm_objective", "huber")

    old_model = LightGBMMultiTarget(target_names=targets, seed=42, **old_params)
    old_model.fit(X_train, y_train_dict, X_val, y_val_dict, feature_names=feature_cols)
    old_preds = old_model.predict(X_test)
    old_metrics = compute_target_metrics(y_test_dict, old_preds, targets)

    pos_test_old = pos_test.copy()
    pos_test_old["pred_lgbm_total"] = old_preds["total"]
    old_ranking = compute_ranking_metrics(pos_test_old, pred_col="pred_lgbm_total")

    # --- New (tuned) params ---
    new_model = LightGBMMultiTarget(target_names=targets, seed=42, **best_params)
    new_model.fit(X_train, y_train_dict, X_val, y_val_dict, feature_names=feature_cols)
    new_preds = new_model.predict(X_test)
    new_metrics = compute_target_metrics(y_test_dict, new_preds, targets)

    pos_test_new = pos_test.copy()
    pos_test_new["pred_lgbm_total"] = new_preds["total"]
    new_ranking = compute_ranking_metrics(pos_test_new, pred_col="pred_lgbm_total")

    # --- Print comparison ---
    print(f"\n{'Metric':<25} {'Old':>10} {'Tuned':>10} {'Delta':>10}")
    print("-" * 57)

    for key in ["total"] + targets:
        label = key.replace("_", " ").title()
        old_mae = old_metrics[key]["mae"]
        new_mae = new_metrics[key]["mae"]
        delta = new_mae - old_mae
        sign = "+" if delta > 0 else ""
        print(f"  {label + ' MAE':<23} {old_mae:>10.3f} {new_mae:>10.3f} {sign}{delta:>9.3f}")

    old_r2 = old_metrics["total"]["r2"]
    new_r2 = new_metrics["total"]["r2"]
    delta_r2 = new_r2 - old_r2
    sign_r2 = "+" if delta_r2 > 0 else ""
    print(f"  {'Total R2':<23} {old_r2:>10.3f} {new_r2:>10.3f} {sign_r2}{delta_r2:>9.3f}")

    old_hit = old_ranking["season_avg_hit_rate"]
    new_hit = new_ranking["season_avg_hit_rate"]
    delta_hit = new_hit - old_hit
    sign_hit = "+" if delta_hit > 0 else ""
    print(f"  {'Top-12 Hit Rate':<23} {old_hit:>10.3f} {new_hit:>10.3f} {sign_hit}{delta_hit:>9.3f}")

    old_sp = old_ranking["season_avg_spearman"]
    new_sp = new_ranking["season_avg_spearman"]
    delta_sp = new_sp - old_sp
    sign_sp = "+" if delta_sp > 0 else ""
    print(f"  {'Spearman rho':<23} {old_sp:>10.3f} {new_sp:>10.3f} {sign_sp}{delta_sp:>9.3f}")

    return {
        "old_metrics": {k: v for k, v in old_metrics.items()},
        "new_metrics": {k: v for k, v in new_metrics.items()},
        "old_ranking": {"hit_rate": old_hit, "spearman": old_sp},
        "new_ranking": {"hit_rate": new_hit, "spearman": new_sp},
    }


# ---------------------------------------------------------------------------
# Config file output
# ---------------------------------------------------------------------------

def _format_config_lines(pos, best_params):
    """Format tuned params as config file constants."""
    prefix = pos.upper()
    param_map = {
        "n_estimators": "N_ESTIMATORS",
        "learning_rate": "LEARNING_RATE",
        "num_leaves": "NUM_LEAVES",
        "max_depth": "MAX_DEPTH",
        "subsample": "SUBSAMPLE",
        "colsample_bytree": "COLSAMPLE_BYTREE",
        "reg_lambda": "REG_LAMBDA",
        "reg_alpha": "REG_ALPHA",
        "min_child_samples": "MIN_CHILD_SAMPLES",
        "min_split_gain": "MIN_SPLIT_GAIN",
        "objective": "OBJECTIVE",
    }

    lines = [f"# Tuned LightGBM params for {pos} — paste into {pos}/{pos.lower()}_config.py:"]
    for param, const_suffix in param_map.items():
        if param in best_params:
            val = best_params[param]
            if isinstance(val, str):
                lines.append(f'{prefix}_LGBM_{const_suffix} = "{val}"')
            elif isinstance(val, float):
                lines.append(f'{prefix}_LGBM_{const_suffix} = {val:.6g}')
            else:
                lines.append(f'{prefix}_LGBM_{const_suffix} = {val}')
    return "\n".join(lines)


def _trial_to_params(trial):
    """Extract clean params dict from a completed Optuna trial."""
    p = trial.params.copy()
    if not p.pop("use_max_depth", True):
        p["max_depth"] = -1
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tune LightGBM hyperparameters per position")
    parser.add_argument("positions", nargs="+", help="Positions to tune (QB, RB, WR, TE)")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds per position")
    parser.add_argument("--print-best", action="store_true",
                        help="Print best params from saved study without running new trials")
    args = parser.parse_args()

    all_results = {}

    for pos in args.positions:
        pos = pos.upper()
        study_name = f"lgbm_{pos.lower()}"
        db_path = f"tune_lgbm_{pos.lower()}.db"

        if args.print_best:
            # Load existing study and print
            try:
                study = optuna.load_study(
                    study_name=study_name,
                    storage=f"sqlite:///{db_path}",
                )
                best = _trial_to_params(study.best_trial)
                print(f"\n{pos} best trial #{study.best_trial.number} "
                      f"(CV MAE = {study.best_value:.4f}):")
                print(_format_config_lines(pos, best))
            except Exception as e:
                print(f"No saved study for {pos}: {e}")
            continue

        cfg = _get_position_config(pos)
        t0 = time.time()

        # Prepare data once
        folds_data, targets = _prepare_cv_folds(pos, cfg)

        # Create or resume study
        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        )

        objective = _make_objective(folds_data, targets)

        print(f"\n{'=' * 70}")
        print(f"  Tuning {pos} LightGBM — {args.n_trials} trials")
        print(f"{'=' * 70}")

        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True,
        )

        elapsed = time.time() - t0
        best = _trial_to_params(study.best_trial)

        print(f"\n{pos} tuning complete in {elapsed:.0f}s")
        print(f"  Best trial #{study.best_trial.number}: CV MAE = {study.best_value:.4f}")
        print(f"\n{_format_config_lines(pos, best)}")

        # Before/after holdout comparison
        comparison = _run_comparison(pos, cfg, best)

        all_results[pos] = {
            "best_trial": study.best_trial.number,
            "best_cv_mae": study.best_value,
            "best_params": best,
            "n_trials": len(study.trials),
            "elapsed_seconds": round(elapsed, 1),
            "comparison": comparison,
        }

    # Save results
    if all_results:
        results_path = "tune_lgbm_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
