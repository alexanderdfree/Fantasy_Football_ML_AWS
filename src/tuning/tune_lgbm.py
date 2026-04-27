"""Optuna-based hyperparameter tuning for LightGBM across positions.

Usage:
    python tune_lgbm.py QB                     # tune one position
    python tune_lgbm.py QB RB WR TE K DST      # tune all LGBM-enabled positions
    python tune_lgbm.py RB --n-trials 100      # more trials
    python tune_lgbm.py RB --timeout 3600      # time limit in seconds
    python tune_lgbm.py RB --print-best        # print best params from saved study
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.config import SPLITS_DIR
from src.data.split import expanding_window_folds
from src.shared.evaluation import compute_ranking_metrics, compute_target_metrics
from src.shared.models import LightGBMMultiTarget
from src.shared.pipeline import _prepare_position_data


def _ensure_data_from_s3():
    """Download train/val/test splits and data/raw/ from S3 when ``S3_BUCKET``
    is set and the local files are missing.

    Lets tune_lgbm.py run inside the training container (which has no baked-in
    data) by reusing the same boto3 + ETag-cache helpers as ``src/batch/train.py``.
    No-op locally when ``S3_BUCKET`` isn't set or when the files already exist
    — callers with their own ``data/splits`` layout aren't affected.
    """
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return
    prefix = os.environ.get("S3_DATA_PREFIX", "data")

    # Local import so ``python tune_lgbm.py --print-best`` can run on machines
    # that don't have boto3 + full src/batch/train deps, e.g. offline inspection.
    from src.batch.train import download_data, sync_raw_data

    splits_needed = not all(
        os.path.exists(os.path.join(SPLITS_DIR, f))
        for f in ("train.parquet", "val.parquet", "test.parquet")
    )
    if splits_needed:
        print(f"[tune_lgbm] Downloading splits from s3://{bucket}/{prefix}/ to {SPLITS_DIR}/")
        download_data(bucket, prefix, SPLITS_DIR)
    else:
        print(f"[tune_lgbm] Splits already present at {SPLITS_DIR}/")

    if not os.path.isdir("data/raw") or not any(
        f.endswith(".parquet") for f in os.listdir("data/raw")
    ):
        print(f"[tune_lgbm] Syncing data/raw/ from s3://{bucket}/data/raw/")
        sync_raw_data(bucket)
    else:
        print("[tune_lgbm] data/raw/ already populated")


# ---------------------------------------------------------------------------
# Position config loading
# ---------------------------------------------------------------------------


def _get_position_config(pos):
    """Import and return the CONFIG dict for a position."""
    from src.shared.registry import get_config

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
    if pos == "K":
        # Kickers use a PBP-reconstructed dataset (2015+), not the general splits.
        from src.K.k_data import kicker_season_split, load_kicker_data
        from src.K.k_features import compute_k_features
        from src.K.k_targets import compute_k_targets

        k_df = load_kicker_data()
        k_df = compute_k_targets(k_df)
        compute_k_features(k_df)
        train_df, val_df, _ = kicker_season_split(k_df)
        full_df = pd.concat([train_df, val_df], ignore_index=True)
        folds = expanding_window_folds(full_df, min_train_season=2015)
    elif pos == "DST":
        # DST builds team-level data internally, not from the general splits.
        from src.config import TRAIN_SEASONS, VAL_SEASONS
        from src.DST.dst_data import build_dst_data
        from src.DST.dst_features import compute_dst_features
        from src.DST.dst_targets import compute_dst_targets

        dst_df = build_dst_data()
        dst_df = compute_dst_targets(dst_df)
        compute_dst_features(dst_df)
        train_df = dst_df[dst_df["season"].isin(TRAIN_SEASONS)].copy()
        val_df = dst_df[dst_df["season"].isin(VAL_SEASONS)].copy()
        full_df = pd.concat([train_df, val_df], ignore_index=True)
        folds = expanding_window_folds(full_df, min_train_season=min(TRAIN_SEASONS))
    else:
        train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
        val_df = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
        full_df = pd.concat([train_df, val_df], ignore_index=True)
        folds = expanding_window_folds(full_df)
    targets = cfg["targets"]

    folds_data = []
    for _fold_idx, fold_train_df, fold_val_df in folds:
        (X_train, X_val, _, y_train_dict, y_val_dict, _, _, _, _, feature_cols) = (
            _prepare_position_data(pos, cfg, fold_train_df, fold_val_df)
        )
        folds_data.append((X_train, X_val, y_train_dict, y_val_dict, feature_cols))

    print(
        f"  {len(folds_data)} folds prepared, {len(feature_cols)} features, "
        f"{len(targets)} targets: {targets}"
    )
    return folds_data, targets


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _make_objective(folds_data, targets, lgbm_objective):
    """Return an Optuna objective function that evaluates LightGBM on CV folds.

    ``lgbm_objective`` is the fixed loss family (``"huber"``, ``"fair"``, etc.)
    pulled from the position's cfg. Earlier revisions searched over
    ``{"huber", "fair", "regression"}`` and landed on Fair for QB/RB/WR/TE and
    Huber for K/DST — an undocumented split. PR 3 of the loss refactor
    unified RB/WR/TE/K/DST on ``"huber"``; QB stays on ``"fair"`` because its
    passing_yards heavy tail regresses ~0.2 pts/game under Huber's 90th-
    percentile-quantile quadratic zone. Respecting cfg keeps this explicit.
    """

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
            objective=lgbm_objective,
        )

        # --- Evaluate across CV folds ---
        fold_maes = []
        for fold_i, (X_train, X_val, y_train_dict, y_val_dict, feature_cols) in enumerate(
            folds_data
        ):
            model = LightGBMMultiTarget(target_names=targets, seed=42, **params)
            model.fit(X_train, y_train_dict, X_val, y_val_dict, feature_names=feature_cols)

            preds = model.predict(X_val)
            metrics = compute_target_metrics(y_val_dict, preds, targets)
            total_mae = metrics["total"]["mae"]
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

    if pos == "K":
        from src.K.k_data import kicker_season_split, load_kicker_data
        from src.K.k_features import compute_k_features
        from src.K.k_targets import compute_k_targets

        k_df = load_kicker_data()
        k_df = compute_k_targets(k_df)
        compute_k_features(k_df)
        train_df, val_df, test_df = kicker_season_split(k_df)
    elif pos == "DST":
        from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
        from src.DST.dst_data import build_dst_data
        from src.DST.dst_features import compute_dst_features
        from src.DST.dst_targets import compute_dst_targets

        dst_df = build_dst_data()
        dst_df = compute_dst_targets(dst_df)
        compute_dst_features(dst_df)
        train_df = dst_df[dst_df["season"].isin(TRAIN_SEASONS)].copy()
        val_df = dst_df[dst_df["season"].isin(VAL_SEASONS)].copy()
        test_df = dst_df[dst_df["season"].isin(TEST_SEASONS)].copy()
    else:
        train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
        val_df = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
        test_df = pd.read_parquet(f"{SPLITS_DIR}/test.parquet")

    targets = cfg["targets"]

    (
        X_train,
        X_val,
        X_test,
        y_train_dict,
        y_val_dict,
        y_test_dict,
        pos_train,
        pos_val,
        pos_test,
        feature_cols,
    ) = _prepare_position_data(pos, cfg, train_df, val_df, test_df)

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

    agg = cfg.get("aggregate_fn")

    def _total(preds):
        return agg(preds) if agg is not None else sum(preds[t] for t in targets)

    pos_test_old = pos_test.copy()
    pos_test_old["pred_lgbm_total"] = _total(old_preds)
    old_ranking = compute_ranking_metrics(pos_test_old, pred_col="pred_lgbm_total")

    # --- New (tuned) params ---
    new_model = LightGBMMultiTarget(target_names=targets, seed=42, **best_params)
    new_model.fit(X_train, y_train_dict, X_val, y_val_dict, feature_names=feature_cols)
    new_preds = new_model.predict(X_test)
    new_metrics = compute_target_metrics(y_test_dict, new_preds, targets)

    pos_test_new = pos_test.copy()
    pos_test_new["pred_lgbm_total"] = _total(new_preds)
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
    print(
        f"  {'Top-12 Hit Rate':<23} {old_hit:>10.3f} {new_hit:>10.3f} {sign_hit}{delta_hit:>9.3f}"
    )

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
                lines.append(f"{prefix}_LGBM_{const_suffix} = {val:.6g}")
            else:
                lines.append(f"{prefix}_LGBM_{const_suffix} = {val}")
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
    parser.add_argument("positions", nargs="+", help="Positions to tune (QB, RB, WR, TE, K, DST)")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds per position")
    parser.add_argument(
        "--print-best",
        action="store_true",
        help="Print best params from saved study without running new trials",
    )
    args = parser.parse_args()

    # Pull splits + raw data from S3 when running in the container. No-op
    # locally if the files already exist.
    if not args.print_best:
        _ensure_data_from_s3()

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
                print(
                    f"\n{pos} best trial #{study.best_trial.number} "
                    f"(CV MAE = {study.best_value:.4f}):"
                )
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

        lgbm_objective = cfg.get("lgbm_objective", "huber")
        objective = _make_objective(folds_data, targets, lgbm_objective)

        print(f"\n{'=' * 70}")
        print(f"  Tuning {pos} LightGBM — {args.n_trials} trials (objective={lgbm_objective})")
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

    # Save results (atomic: tmp file then rename so a crash mid-write can't
    # leave tune_lgbm_results.json partially written).
    if all_results:
        results_path = "tune_lgbm_results.json"
        tmp = f"{results_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        os.replace(tmp, results_path)
        print(f"\nResults saved to {results_path}")

        # Upload to S3 + echo as a delimited stdout block so the results
        # survive a --rm container. The retune workflow greps between markers
        # to extract structured params for PR 3b.
        bucket = os.environ.get("S3_BUCKET")
        if bucket:
            import boto3

            s3 = boto3.client("s3")
            s3_key = "tune_lgbm/tune_lgbm_results.json"
            s3.upload_file(results_path, bucket, s3_key)
            print(f"Uploaded results to s3://{bucket}/{s3_key}")

        print("\n==== BEST_PARAMS_JSON_START ====")
        print(json.dumps(all_results, indent=2, default=str))
        print("==== BEST_PARAMS_JSON_END ====")


if __name__ == "__main__":
    main()
