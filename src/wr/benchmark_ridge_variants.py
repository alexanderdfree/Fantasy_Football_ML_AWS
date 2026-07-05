"""Benchmark WR Ridge variants: PCR and aggressive feature selection.

Tests 6 Ridge configurations on the same data split, reporting per-target MAE,
total MAE, R2, condition number, and feature count for each variant.

Usage:
    python -m src.wr.benchmark_ridge_variants
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import MIN_GAMES_PER_SEASON, SPLITS_DIR
from src.shared.evaluation import compute_metrics
from src.shared.models import RidgeMultiTarget
from src.shared.pipeline import _tune_ridge_alphas_cv
from src.wr.config import POSITION_CONFIG
from src.wr.data import filter_to_position
from src.wr.features import (
    add_specific_features,
    fill_nans,
    get_feature_columns,
)
from src.wr.targets import compute_targets

RIDGE_ALPHA_GRIDS = POSITION_CONFIG.ridge_alpha_grids
SPECIFIC_FEATURES = POSITION_CONFIG.specific_features
TARGETS = POSITION_CONFIG.targets

# ── Aggressive feature drops (on top of INCLUDE_FEATURES whitelist) ────────
EXTRA_DROPS = {
    # snap_pct L5 rolling — kept in production intentionally (see the
    # `if w != 5 or stat == "snap_pct"` guard in src/wr/config.py's
    # _INCLUDE_FEATURES). The aggressive variant drops it here to test the
    # r=0.993 redundancy-with-L8 claim; this is not an accidental retention.
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
    # Carry-related rolling — WRs rarely carry. (carry_share_L3/L5 were later
    # dropped from the WR whitelist entirely as sparse noise in PR #1320, so
    # they no longer need an EXTRA_DROPS entry here.)
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
    pca_n=None,
):
    """Train + evaluate a single Ridge variant. Returns metrics dict.

    ``X_train`` doubles as the CV-grouping source for ``season`` (the
    benchmark's train DataFrame already carries the season column the
    expanding-window splitter needs); we don't accept a separate
    ``pos_train`` to avoid the prior `pos_train` aliased-twice footgun.
    """
    t0 = time.time()

    # Select features
    Xi_train = X_train[feature_cols].values.astype(np.float32)
    Xi_test = X_test[feature_cols].values.astype(np.float32)

    cond = _condition_number(Xi_train)

    # Tune alphas — season grouping comes from X_train directly.
    best_alphas = _tune_ridge_alphas_cv(
        Xi_train,
        y_train_dict,
        X_train["season"].values,
        targets=TARGETS,
        alpha_grids=RIDGE_ALPHA_GRIDS,
        n_cv_folds=4,
        refine_points=5,
        pca_n_components=pca_n,
    )

    # Fit final model
    model = RidgeMultiTarget(
        target_names=TARGETS,
        alpha=best_alphas,
        pca_n_components=pca_n,
    )
    model.fit(Xi_train, y_train_dict)

    # Predict
    preds = model.predict(Xi_test)
    preds["total"] = sum(preds[t] for t in TARGETS)

    # Metrics
    target_metrics = {}
    for t in TARGETS:
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

    pos_train = filter_to_position(train_df)
    pos_val = filter_to_position(val_df)
    pos_test = filter_to_position(test_df)
    n_val = len(pos_val)  # row count only — the val split isn't scored below

    games_per_season = pos_train.groupby(["player_id", "season"])["week"].transform("count")
    # Match production's per-position floor (WR sets min_games_per_season=1); None → global.
    min_games = POSITION_CONFIG.min_games_per_season
    if min_games is None:
        min_games = MIN_GAMES_PER_SEASON
    pos_train = pos_train[games_per_season >= min_games].copy()

    pos_train = compute_targets(pos_train)
    pos_test = compute_targets(pos_test)

    # The val split is loaded for its row count only — this CLI trains on train
    # and evaluates on test (a Ridge-variant sweep, no early-stopping on val), so
    # feature-engineering the full val frame was dead work. Pass an empty
    # same-schema frame through the tri-frame helpers (their train-mean stats are
    # computed from ``pos_train``, so the train/test outputs are unchanged).
    empty_val = pos_train.iloc[0:0]
    pos_train, _empty_val, pos_test = add_specific_features(pos_train, empty_val, pos_test)
    pos_train, _empty_val, pos_test = fill_nans(
        pos_train,
        empty_val,
        pos_test,
        SPECIFIC_FEATURES,
    )

    # Base feature columns (current WR config). Intersect with the columns
    # actually present in the loaded splits: weather/Vegas cols (is_dome,
    # implied_team_total, wind_adjusted, temp_adjusted, implied_opp_total, ...)
    # are added downstream by ``merge_schedule_features`` at pipeline time and
    # are absent from the raw splits parquet this CLI reads directly — without
    # the intersect, ``df[base_cols]`` KeyErrors on a real run.
    base_cols = [c for c in get_feature_columns() if c in pos_train.columns]
    for df in [pos_train, pos_test]:
        df[base_cols] = df[base_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Aggressive-drop columns
    aggressive_cols = [c for c in base_cols if c not in EXTRA_DROPS]

    # Target dicts
    y_train_dict = {t: pos_train[t].values for t in TARGETS}
    y_test_dict = {t: pos_test[t].values for t in TARGETS}
    y_train_dict["total"] = sum(pos_train[t].values for t in TARGETS)
    y_test_dict["total"] = sum(pos_test[t].values for t in TARGETS)

    print(f"Train: {len(pos_train)}, Val: {n_val}, Test: {len(pos_test)}")
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
        # pca_n starts None; auto-selected from the best PCR variant (2-4)
        # by the block below (see lines ~204-209), not unfinished work.
        ("6. aggressive_drops+pcr", aggressive_cols, None),
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
            pca_n=pca_n,
        )
        results.append(r)
        m = r["metrics"]["total"]
        print(
            f"  -> MAE={m['mae']:.3f}  R2={m['r2']:.3f}  "
            f"cond={r['cond_number']:.2e}  ({r['elapsed']:.1f}s)"
        )

    # ── Comparison table ─────────────────────────────────────────────────────
    # Per-target columns iterate ``TARGETS`` so all four raw stats (incl.
    # ``fumbles_lost``) render; pre-2024 versions hardcoded three stale
    # fantasy-points-era column labels (recv_fl/rush_fl/td_pts) whose data
    # rows actually printed receiving_tds/receiving_yards/receptions MAEs.
    label_map = {
        "receiving_tds": "rec_tds",
        "receiving_yards": "rec_yds",
        "receptions": "recs",
        "fumbles_lost": "fum_lost",
    }
    # The "total" metric is the MAE/R2 of the SUM of the four raw-stat targets
    # (receiving_yards + receptions + receiving_tds + fumbles_lost) — NOT fantasy
    # points (those are computed post-prediction via predictions_to_fantasy_points
    # and weight the heads differently). Label it "RawSum" so the table isn't read
    # as a fantasy-point figure, per the raw-stat-targets convention.
    target_header = " ".join(f"{label_map.get(t, t[:8]):>8}" for t in TARGETS)
    header = (
        f"{'Variant':<28} {'Feats':>5} {'PCA':>4} {'Cond#':>10} "
        f"{'RawSum MAE':>10} {'RawSum R2':>9} "
        f"{target_header} {'Time':>6}"
    )
    table_width = len(header)
    print(f"\n\n{'=' * table_width}")
    print("  WR RIDGE VARIANT COMPARISON")
    print(f"{'=' * table_width}")
    print(header)
    print("-" * table_width)

    baseline_mae = results[0]["metrics"]["total"]["mae"]
    for r in results:
        m = r["metrics"]
        pca_str = str(r["pca_n"]) if r["pca_n"] else "-"
        cond_str = f"{r['cond_number']:.1e}" if r["cond_number"] < 1e15 else "inf"
        delta = m["total"]["mae"] - baseline_mae
        delta_str = f"({delta:+.3f})" if r["name"] != "1. baseline" else ""
        target_cells = " ".join(f"{m[t]['mae']:>8.3f}" for t in TARGETS)
        print(
            f"{r['name']:<28} {r['n_features']:>5} {pca_str:>4} {cond_str:>10} "
            f"{m['total']['mae']:>10.3f}{delta_str:>8} {m['total']['r2']:>6.3f} "
            f"{target_cells} {r['elapsed']:>5.1f}s"
        )

    print(f"{'=' * table_width}")

    # Per-target R2 table \u2014 iterate ``TARGETS`` so every raw stat (incl.
    # ``fumbles_lost``) renders. Pre-2024 versions hardcoded three columns
    # (recv_td/recv_yd/recs) and dropped fumbles_lost (M18 fix was incomplete
    # here \u2014 the MAE table above was migrated but this R2 table was missed).
    r2_target_header = " ".join(f"{label_map.get(t, t[:8]):>8}" for t in TARGETS)
    # "RawSum" = raw-stat-sum R2 (see the comparison-table note), not fantasy points.
    r2_hdr = f"{'Variant':<28} {'RawSum':>8} {r2_target_header}"
    r2_width = len(r2_hdr)
    print(f"\n{'=' * r2_width}")
    print("  PER-TARGET R\u00b2 BY VARIANT")
    print(f"{'=' * r2_width}")
    print(r2_hdr)
    print("-" * r2_width)
    for r in results:
        m = r["metrics"]
        r2_cells = " ".join(f"{m[t]['r2']:>8.3f}" for t in TARGETS)
        print(f"{r['name']:<28} {m['total']['r2']:>8.3f} {r2_cells}")
    print(f"{'=' * r2_width}")

    # Best variant (ranked by raw-stat-sum MAE, not fantasy points).
    best = min(results, key=lambda r: r["metrics"]["total"]["mae"])
    print(
        f"\nBest variant: {best['name']} "
        f"(RawSum MAE={best['metrics']['total']['mae']:.3f}, "
        f"RawSum R2={best['metrics']['total']['r2']:.3f})"
    )
    print(f"  Best alphas: {best['best_alphas']}")


if __name__ == "__main__":
    main()
