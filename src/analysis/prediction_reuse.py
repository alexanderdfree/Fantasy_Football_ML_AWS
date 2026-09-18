"""Cache raw predictions for a pinned model directory, never model selection."""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import replace
from pathlib import Path

from src.training.context import RunContext, current_context
from src.training.result_store import ResultStore
from src.training.reuse import fresh_requested
from src.training.reuse_identity import (
    data_identity,
    digest,
    execution_identity,
    fingerprint_file,
    source_identity,
    stable,
)

LOG = logging.getLogger(__name__)


def predict_reusing(
    position, train, val, test, spec, *, kicks=None, opponent_weekly=None, device=None, fresh=False
):
    from src.prediction.bundle import bundled_families
    from src.prediction.frames import SCORING_FORMATS, _details, predict_position
    from src.shared.aggregate_targets import predictions_to_fantasy_points

    context = current_context() or RunContext.defaults()
    directory = Path(spec["model_dir"]).resolve()

    def identity():
        return {
            "schema": 1,
            "kind": "prediction",
            "position": position,
            "models": {
                str(path.relative_to(directory)): fingerprint_file(path)
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            },
            "code": source_identity(position),
            "data": data_identity(context),
            "spec": stable({key: value for key, value in spec.items() if key != "model_dir"}),
            "frames": stable((train, val, test, kicks, opponent_weekly)),
            "execution": execution_identity(device),
        }

    before = None
    start = time.monotonic()
    try:
        before = identity()
        key = digest(before)
        store = ResultStore.configured()
        hit = None if fresh or fresh_requested() else store.lookup(key)
        if hit is not None:
            path, manifest = hit
            with (path / "prediction.pkl").open("rb") as stream:
                prediction = pickle.load(stream)
            if prediction.errors or identity() != before:
                raise ValueError("Incomplete or changed prediction generation")
            # Totals/metrics belong to this evaluation, never to a cached report.
            legacy = bundled_families(directory) is None
            signs = spec.get("target_signs") if legacy else None
            adjust = spec.get("compute_adjustment_fn") if legacy else None
            adjustment = adjust(prediction.frame).to_numpy() if adjust is not None else None
            totals = {}
            for family, raw in prediction.raw.items():
                if signs is not None or adjustment is not None:
                    value = sum(
                        raw[target] * (signs or {}).get(target, 1.0) for target in spec["targets"]
                    )
                    if adjustment is not None:
                        value = value + adjustment
                    totals[family] = {fmt: value for fmt in SCORING_FORMATS}
                else:
                    totals[family] = {
                        fmt: predictions_to_fantasy_points(position, raw, fmt)
                        for fmt in SCORING_FORMATS
                    }
            details = _details(
                prediction.frame,
                prediction.raw,
                totals,
                prediction.details["n_features"],
                spec["targets"],
            )
            prediction = replace(prediction, totals=totals, details=details)
            prediction.frame.attrs["reuse"] = {
                "cache_hit": True,
                "reused_from": manifest["source_run_id"],
                "key": key,
                "lookup_seconds": time.monotonic() - start,
                "fresh_training": False,
            }
            return prediction
    except Exception as exc:
        LOG.warning("Prediction reuse unavailable for %s: %s", position, exc)
        before = None

    prediction = predict_position(
        position,
        train,
        val,
        test,
        spec,
        kicks=kicks,
        opponent_weekly=opponent_weekly,
        device=device,
    )
    prediction.frame.attrs["reuse"] = {
        "cache_hit": False,
        "fresh_training": False,
        "reason": "fresh" if fresh or fresh_requested() else ("miss" if before else "unavailable"),
    }
    if before is not None and not prediction.errors:
        try:
            if identity() != before:
                raise ValueError("Inputs or artifacts changed during inference")

            def writer(path):
                with (path / "prediction.pkl").open("wb") as stream:
                    pickle.dump(prediction, stream, protocol=5)

            store.publish(key, writer, source_run_id=digest(before["models"]), identity=before)
        except Exception as exc:
            LOG.warning("Prediction cache write skipped for %s: %s", position, exc)
    return prediction
