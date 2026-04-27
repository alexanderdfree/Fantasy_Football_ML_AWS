"""DEPRECATED: references pre-migration targets (rushing_floor/receiving_floor/td_points); do not run against current data.

RB Feature Signal Analysis: Weather, Vegas implied-odds, and depth chart columns.

Determines which of the 12 weather/Vegas features and depth_chart_rank are
genuine signal vs noise for the RB model, using:
  1. Per-target conditional correlations (Pearson + Spearman)
  2. Mutual information + depth chart deep dive
  3. Permutation importance on fitted Ridge
  4. Ablation study (expanding-window CV, 8 configurations)
  5. implied_team_total collinearity deep dive
  6. Final keep/drop recommendations
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from src.config import (
    CV_VAL_SEASONS,
    MIN_GAMES_PER_SEASON,
)
from src.data.loader import compute_all_floor_formats, compute_all_scoring_formats, load_raw_data
from src.data.split import expanding_window_folds, temporal_split
from src.features.engineer import build_features, flatten_include_features
from src.models.linear import RidgeModel
from src.rb.config import (
    INCLUDE_FEATURES,
    RIDGE_ALPHA_GRIDS,
    RIDGE_PCA_COMPONENTS,
    SPECIFIC_FEATURES,
    TARGETS,
)
from src.rb.data import filter_to_position
from src.rb.features import add_specific_features, fill_nans
from src.rb.targets import compute_targets
from src.shared.weather_features import WEATHER_FEATURES_ALL, merge_schedule_features

os.makedirs("analysis_output", exist_ok=True)

# The 13 features under investigation
DEPTH_FEATURE = "depth_chart_rank"
ALL_SIGNAL_FEATURES = WEATHER_FEATURES_ALL + [DEPTH_FEATURE]

TARGETS = TARGETS  # ["rushing_floor", "receiving_floor", "td_points"]

# Current production whitelist for weather_vegas
PRODUCTION_WEATHER = INCLUDE_FEATURES[
    "weather_vegas"
]  # ["implied_opp_total", "total_line", "rest_advantage"]


# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Data Loading
# ═══════════════════════════════════════════════════════════════════════════


def _build_feature_cols_with_all_weather():
    """Build feature list with ALL 12 weather features included."""
    include = dict(INCLUDE_FEATURES)
    include["weather_vegas"] = list(WEATHER_FEATURES_ALL)
    return flatten_include_features(include)


def _build_feature_cols_with_weather(weather_keep, keep_depth=True):
    """Build feature list with a specific set of weather features."""
    include = dict(INCLUDE_FEATURES)
    include["weather_vegas"] = list(weather_keep)
    if not keep_depth:
        include["contextual"] = [c for c in include["contextual"] if c != DEPTH_FEATURE]
    return flatten_include_features(include)


def load_and_prepare():
    """Load RB data with ALL 12 weather features (no drops) + depth_chart_rank."""
    print("=" * 70)
    print("  SECTION 1: Data Loading")
    print("=" * 70)

    print("Loading raw data...")
    df = load_raw_data()
    df = compute_all_scoring_formats(df)
    df = compute_all_floor_formats(df)
    print("Building features...")
    df = build_features(df)

    # Temporal split
    train_df, val_df, test_df = temporal_split(df)

    # Filter to RB
    rb_train = filter_to_position(train_df)
    rb_val = filter_to_position(val_df)
    rb_test = filter_to_position(test_df)

    # Min-games filter on training
    games = rb_train.groupby(["player_id", "season"])["week"].transform("count")
    rb_train = rb_train[games >= MIN_GAMES_PER_SEASON].copy()

    # Merge weather features — force re-merge by dropping implied_team_total
    # (build_features only sets implied_team_total; we need all 12)
    for split_name, _df in zip(["train", "val", "test"], [rb_train, rb_val, rb_test], strict=True):
        if "implied_team_total" in _df.columns:
            _df.drop(columns=["implied_team_total"], inplace=True)
        merge_schedule_features(_df, label=split_name)

    # Compute targets
    rb_train = compute_targets(rb_train)
    rb_val = compute_targets(rb_val)
    rb_test = compute_targets(rb_test)

    # Add RB-specific features
    rb_train, rb_val, rb_test = add_specific_features(rb_train, rb_val, rb_test)
    rb_train, rb_val, rb_test = fill_nans(rb_train, rb_val, rb_test, SPECIFIC_FEATURES)

    # Build feature columns with ALL weather features present
    feature_cols = _build_feature_cols_with_all_weather()

    # Fill NaN/inf in feature columns
    for _df in [rb_train, rb_val, rb_test]:
        for col in feature_cols:
            if col not in _df.columns:
                _df[col] = 0.0
        _df[feature_cols] = _df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  RB train: {len(rb_train)}, val: {len(rb_val)}, test: {len(rb_test)}")
    print(f"  Total features: {len(feature_cols)} (including all 12 weather + depth_chart_rank)")

    # Confirm signal features are present
    for f in ALL_SIGNAL_FEATURES:
        assert f in feature_cols, f"Missing feature: {f}"
        assert f in rb_train.columns, f"Missing column: {f}"

    return rb_train, rb_val, rb_test, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Per-Target Conditional Correlations
# ═══════════════════════════════════════════════════════════════════════════


def analyze_correlations(rb_train):
    print("\n" + "=" * 70)
    print("  SECTION 2: Per-Target Conditional Correlations")
    print("=" * 70)

    targets_plus_total = TARGETS + ["total"]
    rb_train = rb_train.copy()
    rb_train["total"] = sum(rb_train[t].values for t in TARGETS)

    # --- 2a: Full Pearson + Spearman ---
    print("\n--- Pearson & Spearman correlations (all RB rows) ---")
    rows = []
    for feat in ALL_SIGNAL_FEATURES:
        for tgt in targets_plus_total:
            x, y = rb_train[feat].values, rb_train[tgt].values
            mask = np.isfinite(x) & np.isfinite(y)
            if np.std(x[mask]) < 1e-10:
                pearson_r, pearson_p, spearman_r, spearman_p = np.nan, np.nan, np.nan, np.nan
            else:
                pearson_r, pearson_p = stats.pearsonr(x[mask], y[mask])
                spearman_r, spearman_p = stats.spearmanr(x[mask], y[mask])
            rows.append(
                {
                    "feature": feat,
                    "target": tgt,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                }
            )
    corr_df = pd.DataFrame(rows)

    # Pivot for display
    for metric in ["pearson_r", "spearman_r"]:
        pivot = corr_df.pivot(index="feature", columns="target", values=metric)
        pivot = pivot[targets_plus_total]
        pivot = pivot.reindex(ALL_SIGNAL_FEATURES)
        label = "Pearson r" if "pearson" in metric else "Spearman rho"
        print(f"\n  {label}:")
        print(pivot.round(4).to_string())

    # --- 2b: Conditional — is_grass on rushing_floor, outdoor only ---
    print("\n--- Conditional: is_grass -> rushing_floor (outdoor games only) ---")
    outdoor = rb_train[rb_train["is_dome"] == 0]
    if len(outdoor) > 100 and outdoor["is_grass"].std() > 1e-10:
        r, p = stats.pearsonr(outdoor["is_grass"], outdoor["rushing_floor"])
        print(f"  Outdoor rows: {len(outdoor)}")
        print(f"  Pearson r = {r:.4f}  (p = {p:.4f})")
        grass_mean = outdoor[outdoor["is_grass"] == 1]["rushing_floor"].mean()
        turf_mean = outdoor[outdoor["is_grass"] == 0]["rushing_floor"].mean()
        print(
            f"  Mean rushing_floor: grass={grass_mean:.3f}, turf={turf_mean:.3f}, diff={grass_mean - turf_mean:+.3f}"
        )
    else:
        print(f"  Outdoor rows: {len(outdoor)} (is_grass has no variance — check merge)")

    # --- 2c: Conditional — wind_adjusted on targets, outdoor non-zero-wind ---
    print("\n--- Conditional: wind_adjusted -> targets (outdoor, wind > 0) ---")
    windy = rb_train[(rb_train["is_dome"] == 0) & (rb_train["wind_adjusted"] > 0)]
    if len(windy) > 100:
        print(f"  Rows with wind > 0: {len(windy)}")
        for tgt in targets_plus_total:
            r, p = stats.pearsonr(windy["wind_adjusted"], windy[tgt])
            print(f"  wind_adjusted -> {tgt}: r={r:.4f} (p={p:.4f})")

    return corr_df


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Mutual Information + Depth Chart Deep Dive
# ═══════════════════════════════════════════════════════════════════════════


def analyze_mi_and_depth_chart(rb_train):
    print("\n" + "=" * 70)
    print("  SECTION 3: Mutual Information + Depth Chart Deep Dive")
    print("=" * 70)

    rb_train = rb_train.copy()
    rb_train["total"] = sum(rb_train[t].values for t in TARGETS)
    targets_plus_total = TARGETS + ["total"]

    # --- 3a: MI for all signal features ---
    X = rb_train[ALL_SIGNAL_FEATURES].values
    mi_results = {}
    for tgt in targets_plus_total:
        y = rb_train[tgt].values
        mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
        mi_results[tgt] = mi

    mi_df = pd.DataFrame(mi_results, index=ALL_SIGNAL_FEATURES)
    print("\n--- Mutual Information (nats) ---")
    print(mi_df.round(4).to_string())

    # --- 3b: Depth chart stratification ---
    print("\n--- Depth Chart: Mean fantasy points by rank ---")
    for rank in [1, 2, 3]:
        subset = rb_train[rb_train["depth_chart_rank"] == rank]
        if len(subset) > 0:
            means = {tgt: subset[tgt].mean() for tgt in targets_plus_total}
            n = len(subset)
            means_str = ", ".join(f"{t}={v:.2f}" for t, v in means.items())
            print(f"  Rank {rank} (n={n:,}): {means_str}")

    # ANOVA F-test
    print("\n--- Depth Chart: ANOVA F-statistic ---")
    groups_by_rank = [rb_train[rb_train["depth_chart_rank"] == r] for r in [1, 2, 3]]
    for tgt in targets_plus_total:
        group_vals = [g[tgt].dropna().values for g in groups_by_rank if len(g) > 0]
        if len(group_vals) >= 2:
            f_stat, p_val = stats.f_oneway(*group_vals)
            print(f"  {tgt}: F={f_stat:.1f}, p={p_val:.2e}")

    # --- 3c: MI comparison: depth_chart_rank vs snap_pct ---
    print("\n--- MI comparison: depth_chart_rank vs snap_pct ---")
    if "snap_pct" in rb_train.columns:
        for tgt in targets_plus_total:
            y = rb_train[tgt].values
            mi_depth = mutual_info_regression(
                rb_train[["depth_chart_rank"]].values, y, random_state=42, n_neighbors=5
            )[0]
            mi_snap = mutual_info_regression(
                rb_train[["snap_pct"]].values, y, random_state=42, n_neighbors=5
            )[0]
            print(f"  {tgt}: depth_chart_rank MI={mi_depth:.4f}, snap_pct MI={mi_snap:.4f}")

    return mi_df


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Permutation Importance on Fitted Ridge
# ═══════════════════════════════════════════════════════════════════════════


def analyze_permutation_importance(rb_train, rb_test, feature_cols):
    """Manual permutation importance: shuffle one feature at a time, measure MAE increase."""
    print("\n" + "=" * 70)
    print("  SECTION 4: Permutation Importance (Ridge, held-out test set)")
    print("=" * 70)

    signal_indices = {f: feature_cols.index(f) for f in ALL_SIGNAL_FEATURES}
    n_repeats = 30
    rng = np.random.RandomState(42)

    for tgt in TARGETS:
        print(f"\n--- {tgt} ---")
        X_train = rb_train[feature_cols].values.astype(np.float32)
        X_test = rb_test[feature_cols].values.astype(np.float32).copy()
        y_test = rb_test[tgt].values

        n_components = min(RIDGE_PCA_COMPONENTS, X_train.shape[1] - 1)
        model = RidgeModel(alpha=10.0, pca_n_components=n_components)
        model.fit(X_train, rb_train[tgt].values)

        baseline_mae = mean_absolute_error(y_test, model.predict(X_test))

        importances = {}
        for feat, col_idx in signal_indices.items():
            mae_increases = []
            for _ in range(n_repeats):
                X_perm = X_test.copy()
                X_perm[:, col_idx] = rng.permutation(X_perm[:, col_idx])
                perm_mae = mean_absolute_error(y_test, model.predict(X_perm))
                mae_increases.append(perm_mae - baseline_mae)
            importances[feat] = (np.mean(mae_increases), np.std(mae_increases))

        # Sort by mean importance (descending)
        sorted_feats = sorted(importances, key=lambda f: importances[f][0], reverse=True)
        for feat in sorted_feats:
            m, s = importances[feat]
            ci_low = m - 1.96 * s
            flag = " <-- CI includes 0" if ci_low <= 0 else ""
            print(f"  {feat:30s}  delta_MAE={m:+.4f}  std={s:.4f}{flag}")


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Ablation Study
# ═══════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    "baseline": {
        "weather_keep": PRODUCTION_WEATHER,
        "depth_chart": True,
        "desc": "Current production (3 weather + depth)",
    },
    "no_weather": {
        "weather_keep": [],
        "depth_chart": True,
        "desc": "No weather/Vegas features",
    },
    "all_weather": {
        "weather_keep": list(WEATHER_FEATURES_ALL),
        "depth_chart": True,
        "desc": "All 12 weather/Vegas features",
    },
    "vegas_only": {
        "weather_keep": ["implied_team_total", "implied_opp_total", "total_line"],
        "depth_chart": True,
        "desc": "3 Vegas features only",
    },
    "add_implied_team": {
        "weather_keep": PRODUCTION_WEATHER + ["implied_team_total"],
        "depth_chart": True,
        "desc": "Baseline + implied_team_total",
    },
    "add_grass": {
        "weather_keep": PRODUCTION_WEATHER + ["is_grass"],
        "depth_chart": True,
        "desc": "Baseline + is_grass",
    },
    "no_depth_chart": {
        "weather_keep": PRODUCTION_WEATHER,
        "depth_chart": False,
        "desc": "Baseline weather, no depth_chart",
    },
    "depth_chart_only": {
        "weather_keep": [],
        "depth_chart": True,
        "desc": "No weather, depth_chart only",
    },
}


def run_ablation(rb_full, all_feature_cols):
    """Expanding-window CV ablation across 8 feature configurations."""
    print("\n" + "=" * 70)
    print("  SECTION 5: Ablation Study (expanding-window CV)")
    print("=" * 70)

    targets_plus_total = TARGETS + ["total"]

    # Generate folds
    folds = expanding_window_folds(rb_full, val_seasons=CV_VAL_SEASONS)

    # Store per-fold MAEs for statistical tests
    results = {}  # config -> {target -> (mean, std)}
    fold_results = {}  # config -> {target -> [mae_per_fold]}

    for config_name, config in ABLATION_CONFIGS.items():
        config_cols = _build_feature_cols_with_weather(
            config["weather_keep"], config["depth_chart"]
        )
        # Only use columns that exist in the data
        config_cols = [c for c in config_cols if c in rb_full.columns]
        n_feats = len(config_cols)
        print(f"\n  Config: {config_name} ({n_feats} feats) -- {config['desc']}")

        fold_maes = {t: [] for t in targets_plus_total}

        for _fold_idx, fold_train, fold_val in folds:
            X_tr = fold_train[config_cols].values.astype(np.float32)
            X_va = fold_val[config_cols].values.astype(np.float32)
            pca_n = min(RIDGE_PCA_COMPONENTS, X_tr.shape[1] - 1)

            # Fit per-target, accumulate for total
            total_pred = np.zeros(len(fold_val))
            total_actual = np.zeros(len(fold_val))

            for tgt in TARGETS:
                y_tr = fold_train[tgt].values
                y_va = fold_val[tgt].values

                # Quick alpha search (coarse grid)
                best_alpha, best_mae = 10.0, float("inf")
                for alpha in RIDGE_ALPHA_GRIDS[tgt]:
                    m = RidgeModel(alpha=alpha, pca_n_components=pca_n)
                    m.fit(X_tr, y_tr)
                    mae = mean_absolute_error(y_va, m.predict(X_va))
                    if mae < best_mae:
                        best_mae = mae
                        best_alpha = alpha

                fold_maes[tgt].append(best_mae)

                # Re-fit with best alpha for total prediction
                m = RidgeModel(alpha=best_alpha, pca_n_components=pca_n)
                m.fit(X_tr, y_tr)
                total_pred += m.predict(X_va)
                total_actual += y_va

            fold_maes["total"].append(mean_absolute_error(total_actual, total_pred))

        results[config_name] = {
            t: (np.mean(fold_maes[t]), np.std(fold_maes[t])) for t in targets_plus_total
        }
        fold_results[config_name] = fold_maes

    # --- Print comparison table ---
    print("\n" + "-" * 100)
    print(f"  {'Config':<22s}", end="")
    for tgt in targets_plus_total:
        print(f"  {tgt:>18s}", end="")
    print()
    print("-" * 100)

    for config_name in ABLATION_CONFIGS:
        print(f"  {config_name:<22s}", end="")
        for tgt in targets_plus_total:
            mean, std = results[config_name][tgt]
            print(f"  {mean:>7.4f} +/- {std:>5.4f}", end="")
        print()

    # --- Best per target ---
    print("\n  Best config per target:")
    for tgt in targets_plus_total:
        best = min(results, key=lambda k: results[k][tgt][0])
        mean, std = results[best][tgt]
        print(f"    {tgt}: {best} ({mean:.4f} +/- {std:.4f})")

    # --- Paired t-tests vs baseline ---
    print("\n  Paired t-test vs baseline (4 folds, total MAE):")
    baseline_folds = fold_results["baseline"]["total"]
    for config_name in ABLATION_CONFIGS:
        if config_name == "baseline":
            continue
        config_folds = fold_results[config_name]["total"]
        if len(baseline_folds) == len(config_folds) and len(baseline_folds) >= 2:
            t_stat, p_val = stats.ttest_rel(config_folds, baseline_folds)
            delta_mean = np.mean(config_folds) - np.mean(baseline_folds)
            print(
                f"    {config_name:<22s}: delta={delta_mean:+.4f}, t={t_stat:+.2f}, p={p_val:.3f}"
            )

    _plot_ablation(results, targets_plus_total)

    return results, fold_results


def _plot_ablation(results, targets):
    """Grouped bar chart of MAE by config x target."""
    config_names = list(results.keys())
    n_configs = len(config_names)
    n_targets = len(targets)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_configs)
    width = 0.8 / n_targets

    for i, tgt in enumerate(targets):
        means = [results[c][tgt][0] for c in config_names]
        stds = [results[c][tgt][1] for c in config_names]
        offset = (i - n_targets / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=tgt, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [ABLATION_CONFIGS[c]["desc"] for c in config_names], rotation=35, ha="right", fontsize=8
    )
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("RB Feature Ablation: MAE by Configuration (4-fold expanding CV)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig("analysis_output/rb_feature_signal_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n  Saved: analysis_output/rb_feature_signal_ablation.png")


# ═══════════════════════════════════════════════════════════════════════════
# Section 6: implied_team_total Deep Dive
# ═══════════════════════════════════════════════════════════════════════════


def analyze_implied_team_total(rb_train):
    print("\n" + "=" * 70)
    print("  SECTION 6: implied_team_total Deep Dive")
    print("=" * 70)

    rb_train = rb_train.copy()
    rb_train["total"] = sum(rb_train[t].values for t in TARGETS)
    targets_plus_total = TARGETS + ["total"]

    vegas = ["implied_team_total", "implied_opp_total", "total_line"]

    # --- 6a: Inter-correlation ---
    print("\n--- Vegas feature inter-correlations ---")
    vcorr = rb_train[vegas].corr()
    print(vcorr.round(4).to_string())

    # --- 6b: VIF ---
    print("\n--- Variance Inflation Factors ---")
    from numpy.linalg import lstsq

    X_v = rb_train[vegas].values
    X_v_scaled = StandardScaler().fit_transform(X_v)
    for i, feat in enumerate(vegas):
        others = np.delete(X_v_scaled, i, axis=1)
        coef, _, _, _ = lstsq(others, X_v_scaled[:, i], rcond=None)
        y_pred = others @ coef
        ss_res = np.sum((X_v_scaled[:, i] - y_pred) ** 2)
        ss_tot = np.sum((X_v_scaled[:, i] - X_v_scaled[:, i].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        print(f"  {feat}: VIF = {vif:.1f} (R2 = {r2:.4f})")

    # --- 6c: Partial correlations ---
    print(
        "\n--- Partial correlation of implied_team_total with targets, controlling for total_line ---"
    )
    for tgt in targets_plus_total:
        x = rb_train["implied_team_total"].values
        y = rb_train[tgt].values
        z = rb_train["total_line"].values

        z_mat = np.column_stack([z, np.ones(len(z))])
        coef_x, _, _, _ = np.linalg.lstsq(z_mat, x, rcond=None)
        coef_y, _, _, _ = np.linalg.lstsq(z_mat, y, rcond=None)
        x_resid = x - z_mat @ coef_x
        y_resid = y - z_mat @ coef_y
        partial_r, partial_p = stats.pearsonr(x_resid, y_resid)
        print(f"  {tgt}: partial_r = {partial_r:.4f} (p = {partial_p:.4f})")

    # --- 6d: Direct comparison ---
    print("\n--- Direct correlation comparison ---")
    for feat in vegas:
        r_total, _ = stats.pearsonr(rb_train[feat], rb_train["total"])
        print(f"  {feat} -> total: r = {r_total:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Section 7: Recommendations
# ═══════════════════════════════════════════════════════════════════════════


def print_recommendations(ablation_results):
    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)

    baseline_total = ablation_results["baseline"]["total"][0]
    print(f"\n  Baseline total MAE: {baseline_total:.4f}")

    comparisons = [
        (
            "3 kept weather features",
            "baseline",
            "no_weather",
            "Removing implied_opp_total + total_line + rest_advantage",
        ),
        (
            "All 12 weather features",
            "all_weather",
            "baseline",
            "Adding back all 9 dropped weather features",
        ),
        (
            "implied_team_total",
            "add_implied_team",
            "baseline",
            "Adding implied_team_total to baseline",
        ),
        ("is_grass", "add_grass", "baseline", "Adding is_grass to baseline"),
        (
            "depth_chart_rank",
            "baseline",
            "no_depth_chart",
            "Removing depth_chart_rank from baseline",
        ),
        (
            "Vegas-only (3 lines)",
            "vegas_only",
            "no_weather",
            "Adding 3 Vegas features vs no weather",
        ),
    ]

    print()
    for _desc, config_a, config_b, explanation in comparisons:
        mae_a = ablation_results[config_a]["total"][0]
        mae_b = ablation_results[config_b]["total"][0]
        delta = mae_a - mae_b
        print(f"  {explanation}:")
        print(f"    {config_a}: {mae_a:.4f}  vs  {config_b}: {mae_b:.4f}  (delta = {delta:+.4f})")
        print()

    # Feature verdicts
    no_weather_mae = ablation_results["no_weather"]["total"][0]
    all_weather_mae = ablation_results["all_weather"]["total"][0]
    add_team_mae = ablation_results["add_implied_team"]["total"][0]
    add_grass_mae = ablation_results["add_grass"]["total"][0]
    no_depth_mae = ablation_results["no_depth_chart"]["total"][0]

    THRESHOLD = 0.005  # 0.005 pts MAE = meaningful signal

    print("  --- Feature-level verdicts (threshold = 0.005 MAE) ---\n")

    print("  Currently KEPT:")
    w_delta = baseline_total - no_weather_mae
    verdict = "KEEP" if w_delta < -THRESHOLD else ("DROP" if w_delta > THRESHOLD else "MARGINAL")
    print(
        f"    implied_opp_total + total_line + rest_advantage: {verdict} (delta = {w_delta:+.4f})"
    )

    d_delta = baseline_total - no_depth_mae
    verdict = "KEEP" if d_delta < -THRESHOLD else ("DROP" if d_delta > THRESHOLD else "MARGINAL")
    print(f"    depth_chart_rank: {verdict} (delta = {d_delta:+.4f})")

    print("\n  Currently DROPPED:")
    t_delta = add_team_mae - baseline_total
    verdict = "RESTORE" if t_delta < -THRESHOLD else "KEEP DROPPED"
    print(f"    implied_team_total: {verdict} (delta = {t_delta:+.4f})")

    g_delta = add_grass_mae - baseline_total
    verdict = "RESTORE" if g_delta < -THRESHOLD else "KEEP DROPPED"
    print(f"    is_grass: {verdict} (delta = {g_delta:+.4f})")

    a_delta = all_weather_mae - baseline_total
    verdict = "RECONSIDER" if a_delta < -THRESHOLD else "KEEP DROPPED"
    print(
        f"    Remaining 7 (dome, temp, wind, interactions, etc.): {verdict} (delta = {a_delta:+.4f})"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rb_train, rb_val, rb_test, feature_cols = load_and_prepare()

    corr_df = analyze_correlations(rb_train)
    mi_df = analyze_mi_and_depth_chart(rb_train)
    analyze_permutation_importance(rb_train, rb_test, feature_cols)

    # For ablation, combine train+val for expanding-window CV
    rb_full = pd.concat([rb_train, rb_val], ignore_index=True)
    ablation_results, fold_results = run_ablation(rb_full, feature_cols)

    analyze_implied_team_total(rb_train)
    print_recommendations(ablation_results)

    print("\n\nDone.")
