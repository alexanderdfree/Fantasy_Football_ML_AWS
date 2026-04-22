"""DEPRECATED: references pre-migration targets (rushing_floor/receiving_floor/td_points); do not run against current data.

Component analysis: Weather & Vegas implied-odds features vs multi-targets.

Produces per-position:
  1. Pearson correlation heatmap (weather/vegas features × targets)
  2. Mutual information scores (captures non-linear relationships)
  3. PCA variance explained by weather/vegas component group
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from QB.qb_targets import compute_qb_targets
from RB.rb_targets import compute_rb_targets
from shared.weather_features import (
    WEATHER_DROPS_BY_POSITION,
    merge_schedule_features,
)
from src.data.loader import compute_all_floor_formats, compute_all_scoring_formats, load_raw_data
from TE.te_targets import compute_te_targets
from WR.wr_targets import compute_wr_targets

# ── Config ──────────────────────────────────────────────────────────────────
POSITIONS = {
    "QB": {"fn": compute_qb_targets, "targets": ["passing_floor", "rushing_floor", "td_points"]},
    "RB": {"fn": compute_rb_targets, "targets": ["rushing_floor", "receiving_floor", "td_points"]},
    "WR": {"fn": compute_wr_targets, "targets": ["receiving_floor", "rushing_floor", "td_points"]},
    "TE": {"fn": compute_te_targets, "targets": ["receiving_floor", "rushing_floor", "td_points"]},
}

VEGAS_FEATURES = ["implied_team_total", "implied_opp_total", "total_line"]
VENUE_FEATURES = ["is_dome", "is_grass", "temp_adjusted", "wind_adjusted"]
CONTEXT_FEATURES = ["is_divisional", "days_rest_improved", "rest_advantage"]
INTERACTION_FEATURES = ["implied_total_x_wind"]

ALL_FEATURES = VEGAS_FEATURES + VENUE_FEATURES + CONTEXT_FEATURES + INTERACTION_FEATURES

FEATURE_GROUPS = {
    "Vegas Lines": VEGAS_FEATURES,
    "Venue/Weather": VENUE_FEATURES,
    "Context": CONTEXT_FEATURES,
    "Interactions": INTERACTION_FEATURES,
}


def main():
    # ── Load data once ──────────────────────────────────────────────────────
    print("Loading data...")
    df = load_raw_data()
    df = compute_all_scoring_formats(df)
    df = compute_all_floor_formats(df)

    print("Merging schedule/weather features...")
    df = merge_schedule_features(df)

    os.makedirs("analysis_output", exist_ok=True)

    # ── Per-position analysis ───────────────────────────────────────────────
    all_corrs = {}

    for pos, info in POSITIONS.items():
        print(f"\n{'=' * 60}")
        print(f"  {pos} Analysis")
        print(f"{'=' * 60}")

        pos_df = df[df["position"] == pos].copy()
        pos_df = info["fn"](pos_df)
        targets = info["targets"] + ["total"]
        pos_df["total"] = sum(pos_df[t].values for t in info["targets"])

        # Position-specific feature set
        drops = WEATHER_DROPS_BY_POSITION.get(pos, set())
        features = [f for f in ALL_FEATURES if f not in drops]

        # Drop rows with NaN in any feature or target
        cols_needed = features + targets
        subset = pos_df[cols_needed].dropna()
        print(f"  Rows: {len(subset):,} (after dropping NaN)")

        # ── 1. Pearson Correlation ──────────────────────────────────────
        corr_matrix = pd.DataFrame(index=features, columns=targets, dtype=float)
        for t in targets:
            for f in features:
                corr_matrix.loc[f, t] = subset[f].corr(subset[t])

        all_corrs[pos] = corr_matrix
        print("\n  Pearson correlations:")
        print(corr_matrix.round(4).to_string())

        # ── 2. Mutual Information ───────────────────────────────────────
        X = subset[features].values
        mi_results = {}
        for t in targets:
            y = subset[t].values
            mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
            mi_results[t] = mi
        mi_df = pd.DataFrame(mi_results, index=features)
        print("\n  Mutual Information (bits):")
        print(mi_df.round(4).to_string())

        # ── 3. PCA on weather/vegas features ────────────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X_scaled)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        print(f"\n  PCA cumulative variance: {[f'{v:.3f}' for v in cum_var]}")
        n_90 = np.searchsorted(cum_var, 0.90) + 1
        print(f"  Components for 90% variance: {n_90} / {len(features)}")

        # ── 4. R² from weather/vegas features alone ─────────────────────
        # Project targets onto PCA space of weather features
        X_pca = pca.transform(X_scaled)
        from sklearn.linear_model import LinearRegression

        r2_scores = {}
        for t in targets:
            y = subset[t].values
            lr = LinearRegression().fit(X_pca, y)
            r2_scores[t] = lr.score(X_pca, y)
        print("\n  R² (weather/vegas features alone → target):")
        for t, r2 in r2_scores.items():
            print(f"    {t}: {r2:.4f}")

        # ── Plot: Correlation heatmap ───────────────────────────────────
        fig, axes = plt.subplots(
            1, 2, figsize=(14, max(5, len(features) * 0.45)), gridspec_kw={"width_ratios": [1.2, 1]}
        )

        # Left: Pearson correlation
        ax = axes[0]
        sns.heatmap(
            corr_matrix.astype(float),
            annot=True,
            fmt=".3f",
            center=0,
            cmap="RdBu_r",
            vmin=-0.25,
            vmax=0.25,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"{pos} — Pearson Correlation\n(Weather/Vegas → Targets)", fontsize=12)
        ax.set_xlabel("Target")
        ax.set_ylabel("Feature")

        # Right: Mutual Information
        ax = axes[1]
        sns.heatmap(
            mi_df.astype(float),
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"{pos} — Mutual Information\n(Weather/Vegas → Targets)", fontsize=12)
        ax.set_xlabel("Target")
        ax.set_ylabel("")

        plt.tight_layout()
        fig.savefig(
            f"analysis_output/{pos}_weather_vegas_correlation.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"  Saved: analysis_output/{pos}_weather_vegas_correlation.png")

    # ── Summary: Cross-position comparison ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  CROSS-POSITION SUMMARY: Top correlations with 'total'")
    print(f"{'=' * 60}")

    summary_rows = []
    for pos, corr_matrix in all_corrs.items():
        if "total" in corr_matrix.columns:
            for feat in corr_matrix.index:
                summary_rows.append(
                    {
                        "position": pos,
                        "feature": feat,
                        "corr_total": corr_matrix.loc[feat, "total"],
                    }
                )

    summary = pd.DataFrame(summary_rows)
    # Pivot: features × positions
    pivot = summary.pivot(index="feature", columns="position", values="corr_total")
    pivot = pivot.reindex(columns=["QB", "RB", "WR", "TE"])

    # Sort by mean absolute correlation
    pivot["mean_abs"] = pivot.abs().mean(axis=1)
    pivot = pivot.sort_values("mean_abs", ascending=False)
    print(pivot.drop(columns="mean_abs").round(4).to_string())

    # Final heatmap: all positions × total
    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.45)))
    sns.heatmap(
        pivot.drop(columns="mean_abs").astype(float),
        annot=True,
        fmt=".3f",
        center=0,
        cmap="RdBu_r",
        vmin=-0.25,
        vmax=0.25,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(
        "Weather/Vegas Features → Total Fantasy Points\n(Pearson Correlation by Position)",
        fontsize=13,
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(
        "analysis_output/cross_position_weather_vegas_summary.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("\nSaved: analysis_output/cross_position_weather_vegas_summary.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
