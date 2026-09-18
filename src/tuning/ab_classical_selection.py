"""Compare legacy and PPR-aligned Ridge/LightGBM selection on matched inputs."""

from src.tuning.ab_checkpoint_metric import metric_fn as _base_metrics
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
SEEDS = [42, 123, 7]
SUPPORTS_STACKED = False  # Stacking repeats CPU-model seed-0 results across seeds.


def _legacy(cfg):
    cfg["ridge_selection_metric"] = "raw_mae"
    cfg["lgbm_selection_metric"] = "per_target"
    return cfg


def _aligned(cfg):
    cfg["ridge_selection_metric"] = "fantasy_rmse_ppr"
    cfg["lgbm_selection_metric"] = "fantasy_rmse_ppr"
    return cfg


VARIANTS = [
    Variant("baseline", cfg_mutator=_legacy, label="Legacy independent selection"),
    Variant("ppr_rmse", cfg_mutator=_aligned, label="Joint PPR RMSE selection"),
]


def metric_fn(result, position):
    metrics = _base_metrics(result, position)
    timings = result.get("phase_seconds", {})
    for model, phase in (("Ridge", "ridge_fit"), ("LightGBM", "lgbm_train")):
        if model in metrics and phase in timings:
            metrics[model]["fit_seconds"] = timings[phase]
    if "Ridge" in metrics and "ridge_tune" in timings:
        metrics["Ridge"]["tune_seconds"] = timings["ridge_tune"]
    return metrics


if __name__ == "__main__":
    ab_main(__spec__.name)
