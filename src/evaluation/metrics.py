"""Numerical evaluation primitives; no data loading, training, plotting, or CLI imports.

Historical metric names, row selection, NaN behavior, and sklearn arithmetic are
preserved. The caller supplies the scored actual column and chosen cohort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

ACTUAL = "fantasy_points"
MODELS = {
    "Ridge": "pred_ridge_total",
    "NN": "pred_nn_total",
    "Attention NN": "pred_attn_nn_total",
    "LightGBM": "pred_lgbm_total",
}
CANONICAL_PRED_COLUMNS = {
    "Season Avg": "pred_baseline",
    "Ridge": "pred_ridge_total",
    "Neural Net": "pred_nn_total",
    "ElasticNet": "pred_enet_total",
    "Attention NN": "pred_attn_nn_total",
    "LightGBM": "pred_lgbm_total",
}


def pred_columns_from_test_df(test_df: pd.DataFrame) -> dict[str, str]:
    """Discover the canonical prediction columns actually present."""
    return {name: col for name, col in CANONICAL_PRED_COLUMNS.items() if col in test_df.columns}


def prediction_columns(df: pd.DataFrame) -> dict[str, str]:
    """Dynamic discovery preserving cohort reports' historical short NN label."""
    return {
        "NN" if name == "Neural Net" else name: col
        for name, col in pred_columns_from_test_df(df).items()
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # r2_score emits UndefinedMetricWarning when n<2; tiny e2e smoke tests can
    # hit that path via single-sample per-target slices, so skip it explicitly.
    y_true_arr = np.asarray(y_true)
    r2 = r2_score(y_true, y_pred) if y_true_arr.size >= 2 else float("nan")
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2,
    }


def available_models(df: pd.DataFrame, models: dict[str, str] | None = None) -> dict[str, str]:
    """Subset of ``models`` whose prediction column is present in ``df``."""
    models = models or MODELS
    return {name: col for name, col in models.items() if col in df.columns}


def per_model_metrics(
    df: pd.DataFrame, models: dict[str, str] | None = None, actual: str = ACTUAL
) -> dict[str, dict[str, float]]:
    """MAE / signed bias / RMSE / n for each model on ``df``.

    Bias = mean(pred - actual): positive means over-prediction.
    """
    models = models or available_models(df)
    if len(df) == 0:
        return {
            name: {"mae": float("nan"), "bias": float("nan"), "rmse": float("nan"), "n": 0}
            for name in models
        }
    y = df[actual].to_numpy(dtype=float)
    out: dict[str, dict[str, float]] = {}
    for name, col in models.items():
        p = df[col].to_numpy(dtype=float)
        m = compute_metrics(y, p)
        out[name] = {
            "mae": m["mae"],
            "bias": float(np.mean(p - y)),
            "rmse": m["rmse"],
            "n": int(len(df)),
        }
    return out


def bucket_model_table(
    df: pd.DataFrame,
    bucket_col: str,
    models: dict[str, str] | None = None,
    *,
    actual: str = ACTUAL,
) -> pd.DataFrame:
    """Uniform per-model MAE/RMSE/bias/n by bucket plus dMAE vs global."""
    models = models or prediction_columns(df)
    global_metrics = per_model_metrics(df, models, actual)
    out = []
    for name, col in models.items():
        for bucket, sub in df.groupby(bucket_col, observed=True, sort=True):
            m = per_model_metrics(sub, {name: col}, actual)[name]
            out.append(
                {
                    "model": name,
                    "bucket": str(bucket),
                    "n": int(m["n"]),
                    "mae": m["mae"],
                    "dmae": m["mae"] - global_metrics[name]["mae"],
                    "rmse": m["rmse"],
                    "bias": m["bias"],
                }
            )
    return pd.DataFrame(out)


def bias_corrected_mae(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str
) -> pd.Series:
    """Per-group mean(|error - mean(error)|), the bias-removed MAE."""
    tmp = df[[group_col, y_true_col, y_pred_col]].dropna().copy()
    tmp["_e"] = tmp[y_pred_col] - tmp[y_true_col]
    centered = tmp["_e"] - tmp.groupby(group_col, observed=True)["_e"].transform("mean")
    tmp["_abs_centered"] = centered.abs()
    return tmp.groupby(group_col, observed=True)["_abs_centered"].mean()


def best_model(df: pd.DataFrame, models: dict[str, str] | None = None) -> tuple[str | None, float]:
    """Lowest-overall-MAE model present on ``df``."""
    models = models or available_models(df)
    best_name, best_mae = None, float("inf")
    for name, col in models.items():
        mae = (df[col] - df[ACTUAL]).abs().mean()
        if mae < best_mae:
            best_name, best_mae = name, mae
    return best_name, (best_mae if best_name is not None else float("nan"))
