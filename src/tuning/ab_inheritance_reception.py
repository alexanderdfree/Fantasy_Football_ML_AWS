"""Isolate the corrected zero-truncated NB reception expectation (from #1575).

Two arms on production configurations through the shared isolated A/B harness.
``legacy`` reports ``sigmoid(gate) * mu`` (the pre-correction law);
``expectation_only`` reports ``sigmoid(gate) * E[Y | Y > 0]`` via
``nn_correct_ztnb_mean=True``. Only RB/WR/TE carry a ``hurdle_negbin``
reception head, so the arms are identical elsewhere. The full 2x2 grid
(inheritance magnitude scaling x expectation, plus ``both``) lives in #1575's
version of this module; the magnitude arms set component 1's
``nn_magnitude_features``, which this branch does not carry.

Held-out comparisons use shared projected components. Saved models/scalers
are reconstructed through the serving primitives in every eager cell, so an
apparently good training result cannot hide inference drift.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis.artifact_eval import build_test_df_from_artifacts
from src.shared.artifact_integrity import unwrap_state_dict
from src.shared.comparison_scoring import score_actual_components
from src.shared.evaluation_cohorts import load_reference, reference_selection, regular_season_rows
from src.shared.registry import INFERENCE_REGISTRY
from src.training.context import RunContext, current_context
from src.tuning.ab_harness import Variant, ab_main, default_metric_fn, run_ab

POSITIONS = ["RB", "WR", "TE"]
SEEDS = [42, 123, 7]
BASELINE = "legacy"
_ARM = "unset"


def _recipe(config, arm, expectation):
    global _ARM
    _ARM = arm
    config["nn_correct_ztnb_mean"] = expectation
    return config


def legacy(config):
    return _recipe(config, "legacy", False)


def expectation_only(config):
    return _recipe(config, "expectation_only", True)


VARIANTS = [
    Variant("legacy", cfg_mutator=legacy),
    Variant("expectation_only", cfg_mutator=expectation_only, expect_ridge_identical=True),
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

    # Knob-changed sentinel: the Ridge sentinel is blind to an NN-only no-op, so
    # the saved attention checkpoint must carry this arm's expectation version
    # on every gated NB head (0 = legacy law, 1 = corrected ZTNB mean).
    model_dir = context.output_dir(position) / "models"
    checkpoint = torch.load(
        model_dir / INFERENCE_REGISTRY[position]["attn_nn_file"],
        map_location="cpu",
        weights_only=True,
    )
    state_dict, _ = unwrap_state_dict(checkpoint)
    versions = {
        key: int(value.item())
        for key, value in state_dict.items()
        if key.endswith("_ztnb_mean_version")
    }
    expected_version = int(_ARM == "expectation_only")
    if not versions or any(version != expected_version for version in versions.values()):
        raise AssertionError(
            f"{position} {_ARM}: gated-head expectation versions {versions} != {expected_version}"
        )
    out["ztnb_mean_versions"] = versions

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
