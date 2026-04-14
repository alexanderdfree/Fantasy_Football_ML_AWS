import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.evaluation.metrics import compute_metrics


def compute_rb_metrics(y_true_dict: dict, y_pred_dict: dict) -> dict:
    """Compute per-target and total metrics for RB model.

    Args:
        y_true_dict: {"rushing_floor": ..., "receiving_floor": ..., "td_points": ..., "total": ...}
        y_pred_dict: same structure

    Returns:
        {
            "total": {"mae": float, "rmse": float, "r2": float},
            "rushing_floor": {"mae": float, "rmse": float, "r2": float},
            "receiving_floor": {"mae": float, "rmse": float, "r2": float},
            "td_points": {"mae": float, "rmse": float, "r2": float},
        }
    """
    results = {}
    for target in ["total", "rushing_floor", "receiving_floor", "td_points"]:
        results[target] = compute_metrics(y_true_dict[target], y_pred_dict[target])
    return results


def compute_rb_ranking_metrics(
    test_df: pd.DataFrame,
    pred_col: str = "pred_total",
    true_col: str = "fantasy_points",
    top_k: int = 12,
) -> dict:
    """Per-week ranking quality metrics for RB model.

    Returns:
        {
            "weekly": [{"week": int, "top_k_hit_rate": float, "spearman": float}, ...],
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

        # Actual top-K
        actual_top_k = set(week_df.nlargest(top_k, true_col)["player_id"])
        # Predicted top-K
        pred_top_k = set(week_df.nlargest(top_k, pred_col)["player_id"])

        hit_rate = len(actual_top_k & pred_top_k) / top_k

        # Spearman rank correlation (suppress ConstantInputWarning for constant predictions)
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


def print_rb_comparison_table(results: dict) -> None:
    """Pretty-print comparison of all RB models."""
    print("\n" + "=" * 80)
    print("RB Model Comparison -- Total Fantasy Points")
    print("=" * 80)
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 56)
    for model_name, metrics in results.items():
        m = metrics["total"]
        print(f"{model_name:<30} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['r2']:>8.3f}")

    print("\n" + "=" * 80)
    print("RB Model Comparison -- Per-Target MAE")
    print("=" * 80)
    print(f"{'Model':<30} {'Rush Floor':>12} {'Recv Floor':>12} {'TD Pts':>12}")
    print("-" * 68)
    for model_name, metrics in results.items():
        if "rushing_floor" in metrics:
            print(
                f"{model_name:<30} "
                f"{metrics['rushing_floor']['mae']:>12.3f} "
                f"{metrics['receiving_floor']['mae']:>12.3f} "
                f"{metrics['td_points']['mae']:>12.3f}"
            )


def plot_rb_pred_vs_actual(
    y_true_dict: dict,
    y_pred_dict: dict,
    model_name: str,
    save_path: str,
) -> None:
    """Scatter plots of predicted vs actual for each target + total."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    targets = [("total", "Total Fantasy Points"), ("rushing_floor", "Rushing Floor"),
               ("receiving_floor", "Receiving Floor"), ("td_points", "TD Points")]

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
