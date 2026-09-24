"""Retrospective practice-reason screen, with production defaults unchanged.

All arms join the same cached weekly reports for identical cohort labels. Only
the treatments whitelist new inputs. These final weekly snapshots cannot prove
what was known 48/24 hours before kickoff; use the prospective practice archive
for that evidence. Keep candidate features off until both MAE and RMSE improve
without worsening protected cohorts. This is not a play-probability model.

    python -m src.tuning.ab_practice_context --list
    python -m src.tuning.ab_practice_context --positions RB --seeds 42 --only reasons_location --no-stacked-seeds
    python -m src.tuning.launch_ab --spec src.tuning.ab_practice_context --image-sha <sha> --positions RB --seeds 42 --only reasons_location
    python -m src.tuning.launch_ab --spec src.tuning.ab_practice_context --image-sha <sha>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, SEASONS
from src.features.practice_context import (
    LOCATION_FEATURES,
    PRACTICE_CONTEXT_FEATURES,
    REASON_FEATURES,
    attach_historical_context,
)
from src.training.context import raw_data_dir
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["QB", "RB", "WR", "TE"]
SEEDS = [42, 123, 7]
BASELINE = "baseline"
SUPPORTS_STACKED = False
SMALL_COHORT = 30


def inject_context(train, val, test):
    """Read the pinned raw release; never fetch a newer feed into one arm."""
    path = Path(raw_data_dir(CACHE_DIR)) / f"injuries_{min(SEASONS)}_{max(SEASONS)}.parquet"
    injuries = pd.read_parquet(path)
    required = {"gsis_id", "season", "week", "practice_primary_injury"}
    if not required <= set(injuries):
        raise ValueError(f"Practice experiment requires injury descriptions in {path}")
    frames = tuple(attach_historical_context(frame, injuries) for frame in (train, val, test))
    if not frames[0][list(LOCATION_FEATURES)].to_numpy().any():
        raise ValueError("Practice experiment has no injury-location signal in training")
    return frames


def _whitelist(columns):
    def mutate(cfg):
        original = cfg["get_feature_columns_fn"]
        if set(PRACTICE_CONTEXT_FEATURES) & set(original()):
            raise ValueError("Production baseline already includes candidate practice features")
        cfg["get_feature_columns_fn"] = lambda: [*original(), *columns]
        cfg["attn_static_features"] = [*cfg["attn_static_features"], *columns]
        return cfg

    return mutate


def metric_fn(result, position):
    from src.evaluation.metrics import available_models, per_model_metrics
    from src.shared.comparison_scoring import comparison_actuals, comparison_model_totals
    from src.shared.evaluation_cohorts import regular_season_rows

    frame = comparison_model_totals(regular_season_rows(result["test_df"]), position)
    frame["fantasy_points"] = comparison_actuals(frame, position)
    observed = np.isfinite(frame["fantasy_points"])
    scoring_coverage = {"n": int(observed.sum()), "unavailable_actuals": int((~observed).sum())}
    frame = frame[observed]
    required = {*PRACTICE_CONTEXT_FEATURES, "game_status", "is_returning_from_absence"}
    if missing := required - set(frame):
        raise ValueError(f"{position} practice cohorts missing columns: {sorted(missing)}")
    models = available_models(frame)
    output = dict(per_model_metrics(frame, models))
    output["scoring coverage"] = scoring_coverage
    injured = frame[list(LOCATION_FEATURES)].gt(0).any(axis=1)
    masks = {
        "injured": injured,
        "rest_only": frame["practice_rest_only"].eq(1),
        "illness": frame["practice_illness"].eq(1),
        "unknown": frame["practice_reason_unknown"].eq(1),
        "healthy": frame["game_status"].ge(1)
        & frame[list(PRACTICE_CONTEXT_FEATURES)].eq(0).all(axis=1),
        "returning": frame["is_returning_from_absence"].eq(1),
    }
    for name, mask in masks.items():
        output[f"{name} coverage"] = {
            "n": int(mask.sum()),
            "sparse": int(mask.sum() < SMALL_COHORT),
        }
        for model, metrics in per_model_metrics(frame[mask], models).items():
            output[f"{model} @{name}"] = metrics
    # Reuse the pipeline's component-matched ADR-0024 protected cohorts.
    # An unavailable pregame reference is reported, never silently dropped.
    for name in ("elite_top24", "weekly_reference_top24"):
        cohort = result.get("cohorts", {}).get(name, {})
        available = cohort.get("status") == "available"
        n = cohort.get("n") or 0
        output[f"{name} coverage"] = {
            "available": int(available),
            "n": n,
            "sparse": int(not available or n < SMALL_COHORT),
        }
        for model, metrics in cohort.get("models", {}).items():
            output[f"{model} @{name}"] = {
                key: float("nan") if value is None else value for key, value in metrics.items()
            }
    return output


VARIANTS = [
    Variant(
        "baseline", frame_injector=inject_context, label="production baseline; cohort labels only"
    ),
    Variant(
        "reasons",
        frame_injector=inject_context,
        cfg_mutator=_whitelist(REASON_FEATURES),
        expect_ridge_identical=False,
        label="+ rest / illness / unknown",
    ),
    Variant(
        "reasons_location",
        frame_injector=inject_context,
        cfg_mutator=_whitelist(PRACTICE_CONTEXT_FEATURES),
        expect_ridge_identical=False,
        label="+ reasons and injury location",
    ),
]


if __name__ == "__main__":
    ab_main(__spec__.name)
