"""Generic weekly backtest simulation for any position."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.evaluation.metrics import compute_metrics


def run_weekly_simulation(
    test_df: pd.DataFrame,
    pred_columns: dict,
    true_col: str = "fantasy_points",
    top_k: int = 12,
) -> dict:
    """Week-by-week simulation across the test season.

    Args:
        test_df: Position-filtered test DataFrame
        pred_columns: {"model_name": "pred_column_name", ...}
        true_col: Actual fantasy points column
        top_k: Top-K for ranking metrics

    Returns:
        {
            "weekly_metrics": {model_name: [{"week", "mae", "rmse", "r2"}, ...]},
            "weekly_ranking": {model_name: [{"week", "top_k_hit_rate", "spearman"}, ...]},
            "season_summary": {model_name: {"mae", "rmse", "r2"}},
        }
    """
    weekly_metrics = {name: [] for name in pred_columns}
    weekly_ranking = {name: [] for name in pred_columns}
    season_preds = {name: [] for name in pred_columns}
    season_true = []

    for week in sorted(test_df["week"].unique()):
        week_df = test_df[test_df["week"] == week]
        if len(week_df) == 0:
            continue

        y_true = week_df[true_col].values
        season_true.extend(y_true)

        for model_name, pred_col in pred_columns.items():
            y_pred = week_df[pred_col].values
            season_preds[model_name].extend(y_pred)

            metrics = compute_metrics(y_true, y_pred)
            metrics["week"] = week
            weekly_metrics[model_name].append(metrics)

            if len(week_df) >= top_k:
                actual_top_k = set(week_df.nlargest(top_k, true_col)["player_id"])
                pred_top_k = set(week_df.nlargest(top_k, pred_col)["player_id"])
                hit_rate = len(actual_top_k & pred_top_k) / top_k

                corr, _ = spearmanr(week_df[pred_col], week_df[true_col])
                if np.isnan(corr):
                    print(
                        f"  WARNING: Spearman NaN for {model_name} week {week} (n={len(week_df)})"
                    )

                weekly_ranking[model_name].append(
                    {
                        "week": week,
                        "top_k_hit_rate": hit_rate,
                        "spearman": corr,
                    }
                )

    season_summary = {}
    for model_name in pred_columns:
        if season_preds[model_name]:
            season_summary[model_name] = compute_metrics(
                np.array(season_true), np.array(season_preds[model_name])
            )
        else:
            season_summary[model_name] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
            }

    return {
        "weekly_metrics": weekly_metrics,
        "weekly_ranking": weekly_ranking,
        "season_summary": season_summary,
    }


def plot_weekly_accuracy(sim_results: dict, position: str, save_path: str) -> None:
    """Two-panel figure: weekly MAE and top-12 hit rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, weekly in sim_results["weekly_metrics"].items():
        weeks = [w["week"] for w in weekly]
        maes = [w["mae"] for w in weekly]
        ax1.plot(weeks, maes, label=model_name, marker="o", markersize=3)
    ax1.set_xlabel("Week")
    ax1.set_ylabel("MAE")
    ax1.set_title(f"{position} Weekly MAE by Model")
    ax1.legend()

    for model_name, weekly in sim_results["weekly_ranking"].items():
        if not weekly:
            continue
        weeks = [w["week"] for w in weekly]
        hit_rates = [w["top_k_hit_rate"] for w in weekly]
        ax2.plot(weeks, hit_rates, label=model_name, marker="o", markersize=3)
    ax2.set_xlabel("Week")
    ax2.set_ylabel("Top-12 Hit Rate")
    ax2.set_title(f"{position} Weekly Top-12 Hit Rate")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
