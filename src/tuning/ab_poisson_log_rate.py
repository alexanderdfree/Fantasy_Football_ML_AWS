"""Paired clamp-vs-log-rate validation for ungated Poisson NN heads.

Run a single eager cell first, then the affected five positions with three
seeds. K has no Poisson heads; its unchanged architecture is contract-tested.

    python -m src.tuning.ab_poisson_log_rate --positions RB --seeds 42 --no-stacked-seeds
    python -m src.tuning.launch_ab --spec src.tuning.ab_poisson_log_rate --positions RB --seeds 42

The spec retains all production model/loss settings except the output-link
switch. Full fantasy metrics accompany per-head bias, RMSE and zero fraction;
sparse-event MAE alone rewards a collapsed zero predictor.
"""

import numpy as np

from src.shared.registry import get_config
from src.tuning.ab_harness import Variant, ab_main, default_metric_fn

POSITIONS = ["QB", "RB", "WR", "TE", "DST"]
SEEDS = [42, 123, 7]


def _legacy(cfg):
    cfg["nn_poisson_log_rate"] = False
    return cfg


def _log_rate(cfg):
    cfg["nn_poisson_log_rate"] = True
    return cfg


VARIANTS = [
    Variant("baseline", cfg_mutator=_legacy),
    Variant("log_rate", cfg_mutator=_log_rate, expect_ridge_identical=True),
]


def metric_fn(result, position):
    metrics = default_metric_fn(result, position)
    frame = result["test_df"]
    cfg = get_config(position)
    poisson = {t for t, family in cfg["head_losses"].items() if family == "poisson_nll"}
    for model, prefix in (("NN", "nn"), ("Attention NN", "attn_nn")):
        for target in sorted(poisson):
            column = f"pred_{prefix}_{target}"
            if column not in frame:
                continue
            predicted = frame[column].to_numpy(dtype=float)
            actual = frame[target].to_numpy(dtype=float)
            error = predicted - actual
            metrics[f"{model}:{target}"] = {
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "bias": float(error.mean()),
                "zero_fraction": float((predicted == 0).mean()),
                "mean_prediction": float(predicted.mean()),
                "mean_actual": float(actual.mean()),
                "poisson_nll": float(
                    (predicted - actual * np.log(np.maximum(predicted, 1e-8))).mean()
                ),
                "n": len(actual),
            }
    return metrics


if __name__ == "__main__":
    ab_main(__spec__.name)
