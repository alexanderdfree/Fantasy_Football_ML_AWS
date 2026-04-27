"""RB error stratification analysis.

Runs the RB pipeline, then slices prediction errors by game context,
player usage, opponent quality, and scoring patterns.

Usage:
    python -m src.rb.analyze_errors
    python -m src.rb.analyze_errors --no-plots
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.rb.config import TARGETS
from src.rb.run_pipeline import run
from src.shared.error_analysis import (
    add_stratification_columns,
    find_top_error_sources,
    plot_bias_heatmap,
    plot_error_by_stratum,
    plot_td_zero_vs_scored,
    print_stratified_table,
    print_top_error_sources,
    run_stratified_analysis,
)

STRATA_COLS = [
    "snap_bucket",
    "opp_tier",
    "td_bucket",
    "week_phase",
    "volatility_q",
    "home_away",
]

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figures")


def build_model_pred_cols(df, targets):
    """Build the model_pred_cols dict from columns present in the DataFrame."""
    models = {}
    for prefix, name in [
        ("pred_ridge_", "Ridge"),
        ("pred_nn_", "NN"),
        ("pred_attn_nn_", "Attention NN"),
    ]:
        pred_map = {}
        for t in targets + ["total"]:
            col = f"{prefix}{t}"
            if col in df.columns:
                pred_map[t] = col
        if pred_map:
            models[name] = pred_map
    return models


def main():
    parser = argparse.ArgumentParser(description="RB Error Stratification Analysis")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    targets = TARGETS

    # Run pipeline to get test DataFrame with predictions
    print("Running RB pipeline...")
    result = run()
    df = result["test_df"].copy()

    # Compute actual total from target columns (not fantasy_points, which includes fumbles)
    df["actual_total"] = sum(df[t] for t in targets)

    # Add stratification columns
    add_stratification_columns(df, targets)

    # Build prediction column mappings
    target_cols = {t: t for t in targets}
    target_cols["total"] = "actual_total"
    model_pred_cols = build_model_pred_cols(df, targets)

    print(f"\nModels found: {list(model_pred_cols.keys())}")
    print(f"Test set size: {len(df)} rows")
    print(f"Strata: {STRATA_COLS}")

    # Run stratified analysis
    results = run_stratified_analysis(df, model_pred_cols, target_cols, STRATA_COLS)

    # Print tables for the best model (Attention NN, fall back to NN)
    primary_model = "Attention NN" if "Attention NN" in model_pred_cols else "NN"

    for target in targets + ["total"]:
        print_stratified_table(results, primary_model, target)

    # Top error sources
    sources = find_top_error_sources(results, primary_model, top_k=15)
    print_top_error_sources(sources, primary_model)

    # Comparison: print total MAE by stratum for Ridge vs Attention NN
    if "Ridge" in model_pred_cols and "total" in model_pred_cols["Ridge"]:
        print(f"\n{'=' * 80}")
        print("Model Comparison by Stratum (Total MAE)")
        print(f"{'=' * 80}")
        print(f"{'Stratum':<20} {'Bucket':<16} {'N':>5} {'Ridge':>8} {'Attn NN':>8} {'Delta':>8}")
        print("-" * 80)
        for stratum in STRATA_COLS:
            if stratum not in results:
                continue
            ridge_data = results[stratum].get("Ridge", {}).get("total")
            attn_data = results[stratum].get(primary_model, {}).get("total")
            if ridge_data is None or attn_data is None:
                continue
            merged = ridge_data.merge(
                attn_data, on=ridge_data.columns[0], suffixes=("_ridge", "_attn")
            )
            for _, row in merged.iterrows():
                delta = row["mae_attn"] - row["mae_ridge"]
                print(
                    f"{stratum:<20} {str(row.iloc[0]):<16} {int(row['n_ridge']):>5} "
                    f"{row['mae_ridge']:>8.3f} {row['mae_attn']:>8.3f} {delta:>+8.3f}"
                )
            print("-" * 80)

    # Generate figures
    if not args.no_plots:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        print(f"\nSaving figures to {FIGURE_DIR}/")

        all_targets = targets + ["total"]

        for stratum in STRATA_COLS:
            plot_error_by_stratum(
                results,
                primary_model,
                stratum,
                all_targets,
                os.path.join(FIGURE_DIR, f"rb_error_mae_by_{stratum}.png"),
            )

        plot_bias_heatmap(
            results,
            primary_model,
            STRATA_COLS,
            all_targets,
            os.path.join(FIGURE_DIR, "rb_error_bias_heatmap.png"),
        )

        # TD-specific analysis
        td_target = next((t for t in targets if "td" in t.lower()), None)
        if td_target:
            pred_col = model_pred_cols.get(primary_model, {}).get(td_target)
            if pred_col:
                plot_td_zero_vs_scored(
                    df,
                    pred_col,
                    td_target,
                    os.path.join(FIGURE_DIR, "rb_error_td_zero_vs_scored.png"),
                    title=f"RB TD Prediction Errors — {primary_model}",
                )

        print(f"  Saved {len(STRATA_COLS) + 2} figures.")

    print("\nDone.")


if __name__ == "__main__":
    main()
