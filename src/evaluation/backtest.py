import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from src.evaluation.metrics import compute_metrics
from src.config import TOP_K_RANKING


def run_weekly_simulation(
    test_df: pd.DataFrame,
    pred_columns: dict,
    true_col: str = "fantasy_points",
) -> dict:
    """Week-by-week prediction accuracy simulation across the test season."""
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

            # Ranking metrics per position
            for pos in week_df["position"].unique():
                pos_df = week_df[week_df["position"] == pos]
                if len(pos_df) < TOP_K_RANKING:
                    continue

                actual_top_k = set(pos_df.nlargest(TOP_K_RANKING, true_col)["player_id"])
                pred_top_k = set(pos_df.nlargest(TOP_K_RANKING, pred_col)["player_id"])
                hit_rate = len(actual_top_k & pred_top_k) / TOP_K_RANKING

                corr, _ = spearmanr(pos_df[pred_col], pos_df[true_col])

                weekly_ranking[model_name].append({
                    "week": week, "position": pos,
                    "top12_hit_rate": hit_rate, "spearman_corr": corr,
                })

    season_summary = {}
    for model_name in pred_columns:
        season_summary[model_name] = compute_metrics(
            np.array(season_true), np.array(season_preds[model_name])
        )

    return {
        "weekly_metrics": weekly_metrics,
        "weekly_ranking": weekly_ranking,
        "season_summary": season_summary,
    }


def plot_weekly_accuracy(sim_results: dict, save_path: str) -> None:
    """Two-panel figure: weekly MAE and top-12 hit rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, weekly in sim_results["weekly_metrics"].items():
        weeks = [w["week"] for w in weekly]
        maes = [w["mae"] for w in weekly]
        ax1.plot(weeks, maes, label=model_name, marker="o", markersize=3)
    ax1.set_xlabel("Week")
    ax1.set_ylabel("MAE")
    ax1.set_title("Weekly MAE by Model")
    ax1.legend()

    for model_name, weekly in sim_results["weekly_ranking"].items():
        df_rank = pd.DataFrame(weekly)
        if len(df_rank) == 0:
            continue
        avg_by_week = df_rank.groupby("week")["top12_hit_rate"].mean()
        ax2.plot(avg_by_week.index, avg_by_week.values, label=model_name, marker="o", markersize=3)
    ax2.set_xlabel("Week")
    ax2.set_ylabel("Top-12 Hit Rate")
    ax2.set_title("Weekly Top-12 Hit Rate (Avg Across Positions)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
