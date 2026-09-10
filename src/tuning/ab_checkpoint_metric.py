"""Compare checkpoint selectors with identical production recipes and seeds.

Use the eager harness: each arm changes only the selection criterion. This
preserves the same seeded optimization trajectory until its stopping point.
Default production cosine schedules do not consume the selection score.
Plateau configurations intentionally couple that score to LR scheduling too.
"""

import os

import numpy as np

from src.shared.aggregate_targets import predictions_to_fantasy_points
from src.shared.evaluation import compute_metrics
from src.shared.registry import get_config
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
SEEDS = [42, 123, 7]
SUPPORTS_STACKED = False  # Fixed-epoch stacking bypasses checkpoint selection.


def _select(cfg, metric):
    if int(os.environ.get("FF_NN_FIXED_EPOCHS", "0") or "0") > 0:
        raise ValueError("Checkpoint A/B requires early stopping; use --no-stacked-seeds")
    cfg["nn_selection_metric"] = metric
    return cfg


def _legacy_mae(cfg):
    return _select(cfg, "weighted_mae")


def _raw_rmse(cfg):
    return _select(cfg, "weighted_rmse")


def _fantasy_rmse(cfg):
    return _select(cfg, "fantasy_rmse_ppr")


VARIANTS = [
    Variant("baseline", cfg_mutator=_legacy_mae, label="Weighted raw-stat MAE"),
    Variant(
        "raw_rmse",
        cfg_mutator=_raw_rmse,
        expect_ridge_identical=True,
        label="Weighted raw-stat RMSE",
    ),
    Variant(
        "fantasy_rmse",
        cfg_mutator=_fantasy_rmse,
        expect_ridge_identical=True,
        label="PPR fantasy-point RMSE",
    ),
]


def metric_fn(result, position):
    """Score the same target components as reporting, with format/stat diagnostics."""
    frame = result["test_df"]
    targets = get_config(position)["targets"]
    truth = {t: frame[t].to_numpy() for t in targets}
    metrics = {}
    for model, label, history_key in (
        ("ridge", "Ridge", None),
        ("nn", "NN", "history"),
        ("attn_nn", "Attention NN", "attn_history"),
        ("lgbm", "LightGBM", None),
    ):
        if f"{model}_metrics" not in result:
            continue
        preds = {t: frame[f"pred_{model}_{t}"].to_numpy() for t in targets}
        values = {"n": len(frame)}
        for fmt in ("ppr", "half_ppr", "standard"):
            actual = predictions_to_fantasy_points(position, truth, fmt)
            predicted = predictions_to_fantasy_points(position, preds, fmt)
            prefix = "" if fmt == "ppr" else f"{fmt}_"
            values.update(
                {prefix + k: float(v) for k, v in compute_metrics(actual, predicted).items()}
            )
            values[prefix + "bias"] = float(np.mean(predicted - actual))
        for t in targets:
            stat_metrics = compute_metrics(truth[t], preds[t])
            values[f"{t}_mae"] = float(stat_metrics["mae"])
            values[f"{t}_rmse"] = float(stat_metrics["rmse"])
        selection = (result.get(history_key) or {}).get("checkpoint_selection")
        if selection:
            values["selected_epoch"] = selection["epoch"]
            values["val_fantasy_rmse_ppr"] = selection["validation_metrics"]["val_fantasy_rmse_ppr"]
        metrics[label] = values
    return metrics


if __name__ == "__main__":
    ab_main(__spec__.name)
