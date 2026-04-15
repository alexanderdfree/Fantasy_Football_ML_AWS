import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def compute_positional_metrics(df, pred_col, true_col) -> pd.DataFrame:
    """Returns DataFrame with columns: position, mae, rmse, r2, n_samples."""
    results = []
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        mask = df["position"] == pos
        if mask.sum() == 0:
            continue
        metrics = compute_metrics(df.loc[mask, true_col], df.loc[mask, pred_col])
        metrics["position"] = pos
        metrics["n_samples"] = mask.sum()
        results.append(metrics)
    return pd.DataFrame(results)


def print_comparison_table(results: dict) -> None:
    """Pretty-print model comparison table."""
    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 52)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['mae']:>8.3f} {metrics['rmse']:>8.3f} {metrics['r2']:>8.3f}")
    print("=" * 60)
