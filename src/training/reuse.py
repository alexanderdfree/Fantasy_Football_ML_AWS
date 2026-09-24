"""Reuse complete fitted runs without skipping stages inside a new fit."""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import time

from src.training.result_store import ResultStore
from src.training.reuse_identity import Uncacheable, digest, fit_identity

LOG = logging.getLogger(__name__)


def _changed_fields(before, after):
    return ", ".join(
        key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
    )


def fresh_requested():
    return os.environ.get("FF_FRESH", "0").strip().lower() in {"1", "true", "yes"}


def _replay(payload, directory, recipe, context, manifest, lookup_seconds):
    from src.prediction.predictor import Predictor
    from src.shared.backtest import run_weekly_simulation
    from src.shared.evaluation import (
        build_gate_info,
        compute_target_metrics,
    )
    from src.shared.pipeline import (
        _reporting_baseline,
        _reporting_cohorts,
        _reporting_frame,
        _reporting_ranking,
        _reporting_scored,
    )
    from src.training.contracts import TrainingResult

    values, prepared = payload["values"], payload["prepared"]
    original_seconds = values.pop("phase_seconds", {})
    start = time.monotonic()
    frame = _reporting_frame(prepared.test, recipe, prepared.y_test, position=recipe.position)
    columns = {}
    labels = {
        "ridge": "Ridge",
        "nn": "Neural Net",
        "attn_nn": "Attention NN",
        "lgbm": "LightGBM",
        "elasticnet": "ElasticNet",
        "tabpfn": "TabPFN",
    }
    for family, predictions in values["per_target_preds"].items():
        gate_info = (
            build_gate_info(predictions, recipe.get("gated_targets") or [])
            if family == "attn_nn"
            else None
        )
        values[f"{family}_metrics"] = compute_target_metrics(
            prepared.y_test, predictions, recipe["targets"], gate_info=gate_info
        )
        column = f"pred_{family}_total"
        frame[column] = recipe["aggregate_fn"](predictions)
        for target in recipe["targets"]:
            frame[f"pred_{family}_{target}"] = predictions[target]
        values[f"{family}_ranking"] = _reporting_ranking(frame, recipe.position, column)
        columns[labels[family]] = column
    if "test_df" in values:
        values["test_df"] = frame
        if "sim_results" in values:
            frame["pred_baseline"], _ = _reporting_baseline(frame)
            columns["Season Avg"] = "pred_baseline"
            values["sim_results"] = run_weekly_simulation(
                _reporting_scored(frame, recipe.position),
                pred_columns=columns,
                true_col="actual_projected_total",
            )
    values["cohorts"] = _reporting_cohorts(
        recipe.position, frame, prior_frames=(prepared.train, prepared.val)
    )
    models = {
        family: Predictor.from_bundle(
            directory / "outputs/models",
            family,
            position=recipe.position,
            device=manifest["identity"]["execution"]["device"],
        ).model
        if present
        else None
        for family, present in payload["models"].items()
    }
    outputs = directory / "outputs"
    if outputs.is_dir():
        context.emit_artifacts(
            lambda: shutil.copytree(
                outputs, context.output_dir(recipe.position), dirs_exist_ok=True
            )
        )
    values["run_id"] = context.run_id
    values["execution"] = context.metadata(device=manifest["identity"]["execution"]["device"])
    values["phase_seconds"] = {
        "result_lookup": lookup_seconds,
        "result_evaluation": time.monotonic() - start,
    }
    values["reuse"] = {
        "cache_hit": True,
        "key": manifest["key"],
        "reused_from": manifest["source_run_id"],
        "source_phase_seconds": original_seconds,
        "fresh_training": False,
    }
    return TrainingResult(values, recipe, prepared, models, context.run_id)


def reuse_training(function, bound, context):
    """Called only when an analysis/experiment explicitly enables auto reuse."""
    position, recipe = bound.arguments["position"], bound.arguments["cfg"]
    inputs = dict(bound.arguments)
    # Preparation may attach columns/attrs. Keep the caller-owned identity
    # immutable while the numerical pipeline works on its private frames.
    for name, value in inputs.items():
        if name.endswith("_df") and value is not None:
            bound.arguments[name] = value.copy(deep=True)
    identity = None
    start = time.monotonic()
    try:
        identity = fit_identity(position, recipe, inputs, context, function.__name__)
        key = digest(identity)
        store = ResultStore.configured()
        if not fresh_requested():
            cached = store.lookup(key)
            if cached is not None:
                directory, manifest = cached
                with (directory / "result.pkl").open("rb") as stream:
                    payload = pickle.load(stream)
                result = _replay(
                    payload, directory, recipe, context, manifest, time.monotonic() - start
                )
                # Loading model artifacts must not hide a concurrently changed input.
                if fit_identity(position, recipe, inputs, context, function.__name__) != identity:
                    raise Uncacheable("Inputs changed during cache lookup")
                print(
                    f"[result_cache] hit {position}/{key[:12]} source={manifest['source_run_id']}"
                )
                return result
    except Exception as exc:
        # Cache failures cannot turn a requested real experiment into a failure.
        # The actual fit below is deliberately outside this exception boundary.
        LOG.warning("Result reuse unavailable for %s: %s", position, exc)
        identity = None

    result = function(*bound.args, **bound.kwargs)
    from src.training.contracts import TrainingResult

    if not isinstance(result, TrainingResult):
        return result
    values = dict(result)
    values["reuse"] = {
        "cache_hit": False,
        "fresh_training": True,
        "reason": "fresh" if fresh_requested() else ("miss" if identity else "unavailable"),
    }
    result = TrainingResult(values, result.recipe, result.prepared, result.models, result.run_id)
    if identity is None:
        return result
    try:
        after = fit_identity(position, recipe, inputs, context, function.__name__)
        if after != identity:
            raise Uncacheable("Inputs changed during fitting: " + _changed_fields(identity, after))
        output = context.output_dir(position)
        if (
            any(model is not None for model in result.models.values())
            and not (output / "models").is_dir()
        ):
            raise Uncacheable("Fitted model artifacts were not emitted")

        def write(directory):
            payload = {
                "values": dict(result),
                "prepared": result.prepared,
                "models": {name: model is not None for name, model in result.models.items()},
            }
            with (directory / "result.pkl").open("wb") as stream:
                pickle.dump(payload, stream, protocol=5)
            if output.is_dir():
                shutil.copytree(output, directory / "outputs")

        store.publish(key, write, source_run_id=result.run_id, identity=identity)
        print(f"[result_cache] stored {position}/{key[:12]}")
    except Exception as exc:
        LOG.warning("Result cache write skipped for %s: %s", position, exc)
    return result
