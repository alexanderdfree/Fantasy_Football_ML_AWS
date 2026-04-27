"""Benchmark WR Ridge variants: PCR and aggressive feature selection.

Tests 6 Ridge configurations on the same data split, reporting per-target MAE,
total MAE, R2, condition number, and feature count for each variant.

Usage:
    python WR/benchmark_ridge_variants.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import MIN_GAMES_PER_SEASON, SPLITS_DIR
from src.evaluation.metrics import compute_metrics
from src.shared.models import RidgeMultiTarget
from src.shared.pipeline import _tune_ridge_alphas_cv
from src.wr.config import WR_RIDGE_ALPHA_GRIDS, WR_SPECIFIC_FEATURES, WR_TARGETS
from src.wr.data import filter_to_wr
from src.wr.features import (
    add_wr_specific_features,
    fill_wr_nans,
    get_wr_feature_columns,
)
from src.wr.targets import compute_wr_targets

# ── Aggressive feature drops (on top of WR_INCLUDE_FEATURES whitelist) ────────
EXTRA_DROPS = {
    # Zero variance
    "is_home",
    # snap_pct L5 rolling — missed in L5 cleanup, r=0.993 with L8
    "rolling_mean_snap_pct_L5",
    "rolling_std_snap_pct_L5",
    "rolling_max_snap_pct_L5",
    # Redundant matchup features
    "opp_recv_pts_allowed_to_pos",  # r=0.993 with opp_fantasy_pts_allowed_to_pos
    "opp_rush_pts_allowed_to_pos",  # irrelevant for WR receiving
    # Prior-season receiving_yards — r>0.98 with fantasy_points priors
    "prior_season_mean_receiving_yards",
    "prior_season_std_receiving_yards",
    "prior_season_max_receiving_yards",
    # Prior-season receptions — r>0.97 with targets/fantasy_points priors
    "prior_season_mean_receptions",
    "prior_season_std_receptions",
    "prior_season_max_receptions",
    # Carry-related rolling — WRs rarely carry; carry_share is more informative
    "rolling_mean_carries_L3",
    "rolling_std_carries_L3",
    "rolling_max_carries_L3",
    "rolling_mean_carries_L8",
    "rolling_std_carries_L8",
    "rolling_max_carries_L8",
    # Prior-season carries
    "prior_season_mean_carries",
    "prior_season_std_carries",
    "prior_season_max_carries",
    # carry_share L5 — r=0.97 with carry_share_L3
    "carry_share_L5",
}


def _condition_number(X):
    """Condition number of scaled X."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    s = np.linalg.svd(X_s, compute_uv=False)
    return s[0] / s[-1] if s[-1] > 1e-15 else float("inf")


def _run_variant(
    name,
    feature_cols,
    X_train,
    X_test,
    y_train_dict,
    y_test_dict,
    pos_train,
    pca_n=None,
):
    """Train + evaluate a single Ridge variant. Returns metrics dict."""
    t0 = time.time()

    # Select features
    Xi_train = X_train[feature_cols].values.astype(np.float32)
    Xi_test = X_test[feature_cols].values.astype(np.float32)

    cond = _condition_number(Xi_train)

    # Tune alphas
    best_alphas = _tune_ridge_alphas_cv(
        Xi_train,
        y_train_dict,
        pos_train["season"].values,
        targets=WR_TARGETS,
        alpha_grids=WR_RIDGE_ALPHA_GRIDS,
        n_cv_folds=4,
        refine_points=5,
        pca_n_components=pca_n,
    )

    # Fit final model
    model = RidgeMultiTarget(
        target_names=WR_TARGETS,
        alpha=best_alphas,
        pca_n_components=pca_n,
    )
    model.fit(Xi_train, y_train_dict)

    # Predict
    preds = model.predict(Xi_test)
    preds["total"] = sum(preds[t] for t in WR_TARGETS)

    # Metrics
    target_metrics = {}
    for t in WR_TARGETS:
        target_metrics[t] = compute_metrics(y_test_dict[t], preds[t])
    target_metrics["total"] = compute_metrics(y_test_dict["total"], preds["total"])

    elapsed = time.time() - t0
    return {
        "name": name,
        "n_features": len(feature_cols),
        "pca_n": pca_n,
        "cond_number": cond,
        "best_alphas": best_alphas,
        "metrics": target_metrics,
        "elapsed": elapsed,
    }


def main():
    # ── Load data once ───────────────────────────────────────────────────────
    print("Loading data...")
    train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
    val_df = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
    test_df = pd.read_parquet(f"{SPLITS_DIR}/test.parquet")

    pos_train = filter_to_wr(train_df)
    pos_val = filter_to_wr(val_df)
    pos_test = filter_to_wr(test_df)

    games_per_season = pos_train.groupby(["player_id", "season"])["week"].transform("count")
    pos_train = pos_train[games_per_season >= MIN_GAMES_PER_SEASON].copy()

    pos_train = compute_wr_targets(pos_train)
    pos_val = compute_wr_targets(pos_val)
    pos_test = compute_wr_targets(pos_test)

    pos_train, pos_val, pos_test = add_wr_specific_features(pos_train, pos_val, pos_test)
    pos_train, pos_val, pos_test = fill_wr_nans(
        pos_train,
        pos_val,
        pos_test,
        WR_SPECIFIC_FEATURES,
    )

    # Base feature columns (current WR config)
    base_cols = get_wr_feature_columns()
    for df in [pos_train, pos_val, pos_test]:
        df[base_cols] = df[base_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Aggressive-drop columns
    aggressive_cols = [c for c in base_cols if c not in EXTRA_DROPS]

    # Target dicts
    y_train_dict = {t: pos_train[t].values for t in WR_TARGETS}
    y_val_dict = {t: pos_val[t].values for t in WR_TARGETS}
    y_test_dict = {t: pos_test[t].values for t in WR_TARGETS}
    y_train_dict["total"] = sum(pos_train[t].values for t in WR_TARGETS)
    y_val_dict["total"] = sum(pos_val[t].values for t in WR_TARGETS)
    y_test_dict["total"] = sum(pos_test[t].values for t in WR_TARGETS)

    print(f"Train: {len(pos_train)}, Val: {len(pos_val)}, Test: {len(pos_test)}")
    print(f"Base features: {len(base_cols)}, Aggressive features: {len(aggressive_cols)}")
    print(f"Dropped by aggressive: {len(base_cols) - len(aggressive_cols)}")
    dropped = sorted(set(base_cols) - set(aggressive_cols))
    for d in dropped:
        print(f"  - {d}")

    # ── Run variants ─────────────────────────────────────────────────────────
    variants = [
        ("1. baseline", base_cols, None),
        ("2. pcr_80", base_cols, 80),
        ("3. pcr_50", base_cols, 50),
        ("4. pcr_30", base_cols, 30),
        ("5. aggressive_drops", aggressive_cols, None),
        ("6. aggressive_drops+pcr", aggressive_cols, None),  # PCA TBD after 2-4
    ]

    results = []
    for name, cols, pca_n in variants:
        # For variant 6, use the best PCA from variants 2-4
        if name == "6. aggressive_drops+pcr" and results:
            pcr_results = [r for r in results if r["pca_n"] is not None]
            if pcr_results:
                best_pcr = min(pcr_results, key=lambda r: r["metrics"]["total"]["mae"])
                pca_n = best_pcr["pca_n"]
                print(f"\n  [auto] Using PCA({pca_n}) from best PCR variant")

        print(f"\n{'=' * 60}")
        print(f"  Variant: {name} (features={len(cols)}, PCA={pca_n})")
        print(f"{'=' * 60}")

        r = _run_variant(
            name,
            cols,
            pos_train,
            pos_test,
            y_train_dict,
            y_test_dict,
            pos_train,
            pca_n=pca_n,
        )
        results.append(r)
        m = r["metrics"]["total"]
        print(
            f"  -> MAE={m['mae']:.3f}  R2={m['r2']:.3f}  "
            f"cond={r['cond_number']:.2e}  ({r['elapsed']:.1f}s)"
        )

    # ── Comparison table ─────────────────────────────────────────────────────
    print(f"\n\n{'=' * 110}")
    print("  WR RIDGE VARIANT COMPARISON")
    print(f"{'=' * 110}")
    header = (
        f"{'Variant':<28} {'Feats':>5} {'PCA':>4} {'Cond#':>10} "
        f"{'Total MAE':>10} {'Total R2':>9} "
        f"{'recv_fl':>8} {'rush_fl':>8} {'td_pts':>8} {'Time':>6}"
    )
    print(header)
    print("-" * 110)

    baseline_mae = results[0]["metrics"]["total"]["mae"]
    for r in results:
        m = r["metrics"]
        pca_str = str(r["pca_n"]) if r["pca_n"] else "-"
        cond_str = f"{r['cond_number']:.1e}" if r["cond_number"] < 1e15 else "inf"
        delta = m["total"]["mae"] - baseline_mae
        delta_str = f"({delta:+.3f})" if r["name"] != "1. baseline" else ""
        print(
            f"{r['name']:<28} {r['n_features']:>5} {pca_str:>4} {cond_str:>10} "
            f"{m['total']['mae']:>10.3f}{delta_str:>8} {m['total']['r2']:>6.3f} "
            f"{m['receiving_tds']['mae']:>8.3f} {m['receiving_yards']['mae']:>8.3f} "
            f"{m['receptions']['mae']:>8.3f} {r['elapsed']:>5.1f}s"
        )

    print(f"{'=' * 110}")

    # Per-target R2 table
    print(f"\n{'=' * 80}")
    print("  PER-TARGET R\u00b2 BY VARIANT")
    print(f"{'=' * 80}")
    r2_hdr = f"{'Variant':<28} {'Total':>8} {'recv_td':>8} {'recv_yd':>8} {'recs':>8}"
    print(r2_hdr)
    print("-" * 80)
    for r in results:
        m = r["metrics"]
        print(
            f"{r['name']:<28} {m['total']['r2']:>8.3f} "
            f"{m['receiving_tds']['r2']:>8.3f} {m['receiving_yards']['r2']:>8.3f} "
            f"{m['receptions']['r2']:>8.3f}"
        )
    print(f"{'=' * 80}")

    # Best variant
    best = min(results, key=lambda r: r["metrics"]["total"]["mae"])
    print(
        f"\nBest variant: {best['name']} "
        f"(MAE={best['metrics']['total']['mae']:.3f}, "
        f"R2={best['metrics']['total']['r2']:.3f})"
    )
    print(f"  Best alphas: {best['best_alphas']}")


if __name__ == "__main__":
    main()
