import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from src.evaluation.metrics import compute_metrics


def compute_multi_target_metrics(y_true_dict: dict, y_pred_dict: dict, target_names: list[str]) -> dict:
    results = {}
    for target in ["total"] + target_names:
        results[target] = compute_metrics(y_true_dict[target], y_pred_dict[target])
    return results


def compute_ranking_metrics(test_df, pred_col="pred_total", true_col="fantasy_points", top_k=12):
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

    avg_hit = np.mean([r["top_k_hit_rate"] for r in weekly_results]) if weekly_results else 0.0
    avg_spear = np.mean([r["spearman"] for r in weekly_results]) if weekly_results else 0.0
    return {"weekly": weekly_results, "season_avg_hit_rate": avg_hit, "season_avg_spearman": avg_spear}


def print_comparison_table(results: dict, target_names: list[str], pos_label: str = "") -> None:
    print(f"\n{'=' * 80}")
    print(f"{pos_label} Model Comparison -- Total Fantasy Points")
    print("=" * 80)
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 56)
    for model_name, metrics in results.items():
        m = metrics["total"]
        print(f"{model_name:<30} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['r2']:>8.3f}")

    print(f"\n{'=' * 80}")
    print(f"{pos_label} Model Comparison -- Per-Target MAE")
    print("=" * 80)
    header = f"{'Model':<30}" + "".join(f" {t:>14}" for t in target_names)
    print(header)
    print("-" * (30 + 15 * len(target_names)))
    for model_name, metrics in results.items():
        if target_names[0] in metrics:
            vals = "".join(f" {metrics[t]['mae']:>14.3f}" for t in target_names)
            print(f"{model_name:<30}{vals}")


def plot_pred_vs_actual(y_true_dict, y_pred_dict, target_names, model_name, save_path):
    targets = [("total", "Total Fantasy Points")] + [(t, t.replace("_", " ").title()) for t in target_names]
    n = len(targets)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flat if n > 1 else [axes]

    for i, (target, title) in enumerate(targets):
        ax = axes[i]
        y_true = y_true_dict[target]
        y_pred = y_pred_dict[target]
        ax.scatter(y_true, y_pred, alpha=0.3, s=10)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{model_name}: {title}")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
