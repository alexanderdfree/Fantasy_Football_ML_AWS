"""Generic position evaluation utilities."""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.evaluation.metrics import compute_metrics


def compute_target_metrics(y_true_dict: dict, y_pred_dict: dict, target_names: list[str]) -> dict:
    """Compute per-target and total metrics.

    Returns:
        {"total": {mae, rmse, r2}, "target_1": {mae, rmse, r2}, ...}
    """
    results = {}
    for target in ["total"] + target_names:
        results[target] = compute_metrics(y_true_dict[target], y_pred_dict[target])
    return results


def compute_ranking_metrics(
    test_df: pd.DataFrame,
    pred_col: str = "pred_total",
    true_col: str = "fantasy_points",
    top_k: int = 12,
) -> dict:
    """Per-week ranking quality metrics.

    Returns:
        {
            "weekly": [{"week", "top_k_hit_rate", "spearman"}, ...],
            "season_avg_hit_rate": float,
            "season_avg_spearman": float,
        }
    """
    from scipy.stats import spearmanr

    weekly_results = []
    for week in sorted(test_df["week"].unique()):
        week_df = test_df[test_df["week"] == week]
        if len(week_df) < top_k:
            continue

        actual_top_k = set(week_df.nlargest(top_k, true_col)["player_id"])
        pred_top_k = set(week_df.nlargest(top_k, pred_col)["player_id"])
        hit_rate = len(actual_top_k & pred_top_k) / top_k

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = spearmanr(week_df[pred_col], week_df[true_col])

        weekly_results.append({
            "week": week,
            "top_k_hit_rate": hit_rate,
            "spearman": corr if not np.isnan(corr) else 0.0,
        })

    if weekly_results:
        avg_hit_rate = np.mean([r["top_k_hit_rate"] for r in weekly_results])
        avg_spearman = np.mean([r["spearman"] for r in weekly_results])
    else:
        avg_hit_rate = 0.0
        avg_spearman = 0.0

    return {
        "weekly": weekly_results,
        "season_avg_hit_rate": avg_hit_rate,
        "season_avg_spearman": avg_spearman,
    }


def print_comparison_table(results: dict, position: str, target_names: list[str]) -> None:
    """Pretty-print comparison of all models for a position."""
    print("\n" + "=" * 80)
    print(f"{position} Model Comparison -- Total Fantasy Points")
    print("=" * 80)
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 56)
    for model_name, metrics in results.items():
        m = metrics["total"]
        print(f"{model_name:<30} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['r2']:>8.3f}")

    # Per-target MAE
    labels = {t: t.replace("_", " ").title() for t in target_names}
    header = "".join(f"{labels[t]:>12}" for t in target_names)
    print(f"\n{'=' * 80}")
    print(f"{position} Model Comparison -- Per-Target MAE")
    print("=" * 80)
    print(f"{'Model':<30} {header}")
    print("-" * (30 + 12 * len(target_names)))
    for model_name, metrics in results.items():
        if target_names[0] in metrics:
            vals = "".join(f"{metrics[t]['mae']:>12.3f}" for t in target_names)
            print(f"{model_name:<30} {vals}")


def plot_pred_vs_actual(
    y_true_dict: dict,
    y_pred_dict: dict,
    target_names: list[str],
    model_name: str,
    save_path: str,
) -> None:
    """Scatter plots of predicted vs actual for each target + total."""
    targets = [("total", "Total Fantasy Points")]
    targets += [(t, t.replace("_", " ").title()) for t in target_names]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (target, title) in zip(axes.flat, targets):
        y_true = y_true_dict[target]
        y_pred = y_pred_dict[target]
        ax.scatter(y_true, y_pred, alpha=0.3, s=10)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{model_name}: {title}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
