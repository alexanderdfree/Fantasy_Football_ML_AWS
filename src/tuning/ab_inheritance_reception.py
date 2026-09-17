"""Isolate inheritance scaling, reception expectation, and their combination.

Uses production configurations through the shared isolated A/B harness. The
legacy arm disables both corrections; each other arm changes exactly the
named recipe settings. Held-out comparisons use shared projected components.
Saved models/scalers are reconstructed through the serving primitives in every
eager cell, so an apparently good training result cannot hide inference drift.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.analysis.artifact_eval import build_test_df_from_artifacts
from src.features.engineer import get_attn_static_columns
from src.shared.comparison_scoring import score_actual_components
from src.shared.evaluation_cohorts import load_reference, reference_selection, regular_season_rows
from src.shared.feature_build import scale_and_clip
from src.shared.registry import INFERENCE_REGISTRY
from src.training.context import RunContext, current_context
from src.tuning.ab_harness import Variant, ab_main, default_metric_fn, run_ab

POSITIONS = ["QB", "RB", "WR", "TE"]
SEEDS = [42, 123, 7]
BASELINE = "legacy"
_ARM = "unset"


def _recipe(config, arm, magnitude, expectation):
    global _ARM
    _ARM = arm
    config["nn_magnitude_features"] = ("inherited_opportunity",) if magnitude else ()
    config["nn_correct_ztnb_mean"] = expectation
    return config


def legacy(config):
    return _recipe(config, "legacy", False, False)


def magnitude_only(config):
    return _recipe(config, "magnitude_only", True, False)


def expectation_only(config):
    return _recipe(config, "expectation_only", False, True)


def both(config):
    return _recipe(config, "both", True, True)


VARIANTS = [
    Variant("legacy", cfg_mutator=legacy),
    Variant("magnitude_only", cfg_mutator=magnitude_only, expect_ridge_identical=True),
    Variant("expectation_only", cfg_mutator=expectation_only, expect_ridge_identical=True),
    Variant("both", cfg_mutator=both, expect_ridge_identical=True),
]


def _metrics(frame, col):
    error = frame[col].to_numpy() - frame.fantasy_points.to_numpy()
    return {
        "n": len(frame),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
    }


def metric_fn(result, position):
    context = current_context() or RunContext.defaults()
    frame = regular_season_rows(result["test_df"]).copy()
    frame["fantasy_points"] = score_actual_components(frame, position)
    if frame.fantasy_points.isna().any():
        raise ValueError("Shared actual components unavailable")
    out = default_metric_fn({**result, "test_df": frame}, position)
    top, reference = reference_selection(
        position, frame, load_reference(cache_dir=context.raw_root), 24
    )
    if reference["status"] != "available":
        raise ValueError(f"Pregame reference unavailable: {reference}")
    subsets = {"pregame24": top, "inheritors": frame.inherited_opportunity.gt(0)}
    for name, mask in subsets.items():
        sub = frame[mask]
        for model in ("ridge", "nn", "attn_nn", "lgbm"):
            col = f"pred_{model}_total"
            if col not in frame or not np.isfinite(frame[col]).all():
                raise ValueError(f"Missing/nonfinite {model} predictions")
            if sub.empty:
                raise ValueError(f"No rows in required {name} cohort")
            out[f"{model}_{name}"] = _metrics(sub, col)

    model_dir = context.output_dir(position) / "models"
    reg = INFERENCE_REGISTRY[position]
    feature_cols = reg["get_feature_columns_fn"]()
    static_cols = get_attn_static_columns(feature_cols, reg["attn_static_features"])
    for label, cols, filename in (
        ("nn", feature_cols, "nn_scaler.pkl"),
        ("attn", static_cols, "attention_nn_scaler.pkl"),
    ):
        scaler = joblib.load(model_dir / filename)
        raw = frame[cols].to_numpy(dtype=np.float32)
        encoded = scale_and_clip(scaler, raw)
        index = cols.index("inherited_opportunity")
        positive = raw[:, index] > 0
        raw_unique = len(np.unique(raw[positive, index]))
        encoded_unique = len(np.unique(encoded[positive, index]))
        if _ARM in {"magnitude_only", "both"} and raw_unique != encoded_unique:
            raise AssertionError(f"{position} {label}: distinct inheritance values collapsed")
        out[f"{label}_encoding"] = {
            "positive_n": int(positive.sum()),
            "raw_unique": raw_unique,
            "encoded_unique": encoded_unique,
            "encoded_min": float(encoded[positive, index].min()),
            "encoded_max": float(encoded[positive, index].max()),
        }

    # Reconstruct the just-saved artifacts with the same loaders/factories
    # used by serving; saved legacy modes must override current new defaults.
    splits = [
        pd.read_parquet(context.splits_dir / f"{name}.parquet") for name in ("train", "val", "test")
    ]
    restored = build_test_df_from_artifacts(position, *splits, model_dir=str(model_dir))
    keys = ["player_id", "season", "week"]
    restored = restored.set_index(keys).loc[pd.MultiIndex.from_frame(frame[keys])]
    out["inference_parity"] = {}
    for model in ("ridge", "nn", "attn_nn", "lgbm"):
        for col in (c for c in frame if c.startswith(f"pred_{model}_")):
            np.testing.assert_allclose(restored[col], frame[col], atol=2e-5, rtol=1e-6)
        col = f"pred_{model}_total"
        out["inference_parity"][model] = float(
            np.abs(restored[col].to_numpy() - frame[col].to_numpy()).max()
        )

    if folder := os.environ.get("FF_FIX_AB_OUTPUT"):
        destination = Path(folder)
        destination.mkdir(parents=True, exist_ok=True)
        stem = f"{position}_{_ARM}_{torch.initial_seed()}"
        frame.to_parquet(destination / f"{stem}.parquet", index=False)
        (destination / f"{stem}.json").write_text(
            json.dumps(
                {
                    "position": position,
                    "arm": _ARM,
                    "seed": torch.initial_seed(),
                    "metrics": out,
                    "reference": reference,
                },
                indent=2,
            )
        )
    return out


if __name__ == "__main__":
    import sys

    if "--list" in sys.argv:
        raise SystemExit(ab_main(__spec__.name))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", nargs="+", default=POSITIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("-j", type=int, default=2)
    args = parser.parse_args()
    aggregate = run_ab(
        __spec__.name,
        positions=args.positions,
        seeds=args.seeds,
        only=args.only,
        jobs=args.j,
        stacked_seeds=False,
    )
    if folder := os.environ.get("FF_FIX_AB_OUTPUT"):
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    if aggregate["failed"]:
        raise SystemExit(1)
