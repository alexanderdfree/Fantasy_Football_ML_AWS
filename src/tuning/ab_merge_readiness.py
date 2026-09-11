"""Current NN policies with durable rows and a held-out QB pregame replay.

Uses the existing nonstacked ``ab_nn_correctness`` grid and Batch launcher.
For QB, supply FF_MERGE_READINESS_QB_REPLAY and its SHA256 in
FF_MERGE_READINESS_QB_REPLAY_SHA256. The already-local parquet contains unique
player_id/season/week plus is_top_available/inherited_opportunity, constructed
offline from the pinned pregame sources. Only those two TEST features change;
the just-fitted models and historical train/validation remain unchanged.

Evidence goes only to the launcher's ab_runs/<run>/readiness/<cell>/ prefix,
or FF_MERGE_READINESS_OUTPUT for local checks. Content-addressed objects retain
all four families' raw predictions, native/shared actuals, availability and row
identities. Cross-input comparisons must join these rows and fix their cohort
once; per-input inheritor flags are not a shared cohort by themselves.

An observational wrapper records the original trainer's full-step capture
return value and graph object without changing tensors, RNG, options or returns.
CUDA cells fail if either neural family did not actually capture the full step.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
from dataclasses import asdict
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.records import record_for_result
from src.prediction.bundle import canonical_json
from src.shared.comparison_scoring import comparison_actuals, comparison_model_totals
from src.shared.evaluation_cohorts import (
    load_reference,
    reference_path,
    reference_selection,
    regular_season_rows,
)
from src.training.context import current_context
from src.tuning import ab_nn_correctness
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ab_nn_correctness.POSITIONS
SEEDS = ab_nn_correctness.SEEDS
BASELINE = "baseline"
KEYS = ["player_id", "season", "week"]
QB_FIELDS = ["is_top_available", "inherited_opportunity"]
FAMILIES = ("ridge", "nn", "attn_nn", "lgbm")
REQUIRED_COHORTS = (
    "elite_top24",
    "weekly_reference_top24",
    "seasonal_actual_top24",
    "weekly_actual_top24",
)
_ARM = "unset"
_CAPTURE_EVENTS = []


def _observe_capture():
    """Observe the existing capture call, keeping numerical execution untouched."""
    from src.shared.training import MultiHeadTrainer

    current = MultiHeadTrainer._maybe_graph_full_step
    original = getattr(current, "_merge_readiness_original", current)

    @wraps(original)
    def observed(self, train_loader):
        captured = original(self, train_loader)
        _CAPTURE_EVENTS.append(
            {
                "family": "nn" if type(self) is MultiHeadTrainer else "attn_nn",
                "trainer": type(self).__name__,
                "model": type(self.model).__name__,
                "criterion": type(self.criterion).__name__,
                "device": str(self.device),
                "use_amp": bool(self._use_amp),
                "amp_dtype": str(self._amp_dtype),
                "returned_true": captured is True,
                "graph_present": self._graphed_step is not None,
                "capturable_loss": callable(
                    getattr(self.criterion, "compute_combined_capturable", None)
                ),
            }
        )
        return captured

    observed._merge_readiness_original = original
    MultiHeadTrainer._maybe_graph_full_step = observed


def _capture_evidence(result):
    device = str((result.get("execution") or {}).get("device", ""))
    expected = device.startswith("cuda") or os.environ.get("FF_DEVICE") == "cuda"
    events = list(_CAPTURE_EVENTS)
    if expected:
        engaged = {
            event["family"]
            for event in events
            if event["returned_true"] and event["graph_present"] and event["capturable_loss"]
        }
        if not {"nn", "attn_nn"}.issubset(engaged):
            raise ValueError(f"Required CUDA full-step capture did not engage: {events}")
    return {"required": expected, "events": events}


def _configure(config, arm):
    global _ARM
    if os.environ.get("FF_AB_STACKED", "").lower() in {"1", "true", "yes", "on"}:
        raise ValueError("Merge-readiness evidence requires nonstacked production fits")
    _ARM = arm
    _CAPTURE_EVENTS.clear()
    _observe_capture()
    return (ab_nn_correctness.legacy if arm == BASELINE else ab_nn_correctness.corrected)(config)


def baseline(config):
    return _configure(config, BASELINE)


def corrected(config):
    return _configure(config, "corrected")


VARIANTS = [
    Variant(BASELINE, cfg_mutator=baseline),
    Variant("corrected", cfg_mutator=corrected, expect_ridge_identical=True),
]


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _identity_frame(frame):
    if any(key not in frame for key in KEYS) or frame[KEYS].isna().any().any():
        raise ValueError("Readiness rows require non-null player_id/season/week")
    frame = frame.copy()
    frame["player_id"] = frame.player_id.astype(str)
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate readiness player-week identities")
    return frame


def _qb_replay(test):
    """Replace only the two declared QB features, with complete keyed coverage."""
    path = Path(os.environ["FF_MERGE_READINESS_QB_REPLAY"]).resolve()
    expected = os.environ["FF_MERGE_READINESS_QB_REPLAY_SHA256"]
    payload = path.read_bytes()
    if _sha256(payload) != expected:
        raise ValueError("QB replay SHA256 does not match the pinned input")
    source = pd.read_parquet(io.BytesIO(payload))
    if "position" in source:
        source = source.loc[source.position.eq("QB")]
    source = _identity_frame(source)
    if any(column not in source for column in QB_FIELDS):
        raise ValueError("QB replay lacks the two availability feature columns")
    if not np.isfinite(source[QB_FIELDS].to_numpy(dtype=float)).all():
        raise ValueError("QB replay features must be finite")
    result = test.copy()
    mask = result.position.eq("QB")
    selected = _identity_frame(result.loc[mask])
    keys = pd.MultiIndex.from_frame(selected[KEYS])
    lookup = source.set_index(KEYS)
    if not keys.isin(lookup.index).all():
        raise ValueError("QB replay is missing held-out player-week identities")
    for column in QB_FIELDS:
        result.loc[mask, column] = lookup.reindex(keys)[column].to_numpy()
    return result, {"path": str(path), "sha256": expected, "matched_rows": int(mask.sum())}


def _rows(frame, position, targets, reference):
    frame = _identity_frame(regular_season_rows(frame))
    required = [f"pred_{family}_{target}" for family in FAMILIES for target in (*targets, "total")]
    if any(column not in frame for column in required):
        raise ValueError("Readiness requires every family's raw heads and total")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Readiness predictions must be finite")
    top, status = reference_selection(position, frame, reference, 24)
    if status["status"] != "available":
        raise ValueError(f"Readiness archived reference unavailable: {status}")
    keep = [
        column
        for column in frame
        if column in {*KEYS, *targets, *QB_FIELDS, "position", "season_type", "recent_team"}
        or column.startswith(("pred_", "actual_", "fantasy_points"))
    ]
    rows = frame[keep].copy()
    rows["comparison_actual"] = comparison_actuals(frame, position)
    rows["comparison_available"] = np.isfinite(rows.comparison_actual)
    comparable = comparison_model_totals(frame, position)
    for family in FAMILIES:
        rows[f"comparison_{family}"] = comparable[f"pred_{family}_total"]
    rows["cohort_reference_top24"] = top
    rows["cohort_week1"] = rows.week.eq(1)
    if "inherited_opportunity" in rows:
        rows["cohort_native_inheritor"] = rows.inherited_opportunity.gt(0)
    return rows.sort_values(KEYS).reset_index(drop=True), status


def _row_metrics(rows, regime):
    metrics = {}
    for family in FAMILIES:
        prediction = rows[f"comparison_{family}"]
        valid = rows.comparison_available & np.isfinite(prediction)
        error = prediction[valid].to_numpy() - rows.loc[valid, "comparison_actual"].to_numpy()
        if not len(error):
            raise ValueError(f"No comparable {regime}/{family} observations")
        metrics[f"{regime}:{family}"] = {
            "n": len(error),
            "unavailable": int((~valid).sum()),
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "bias": float(error.mean()),
        }
    return metrics


def _evidence_sink(cell):
    """Write only content-addressed validation objects, never model prefixes."""
    run = os.environ.get("FF_AB_RUN_ID")
    if run:
        import boto3
        from botocore.exceptions import ClientError

        prefix = os.environ.get("FF_AB_S3_PREFIX", "").strip("/")
        allowed = prefix.split("/")[0] == "ab_runs" or prefix == (
            "experiments/merge-readiness/20260911T181252Z/ab_runs"
        )
        if not allowed or any(p in {"", ".", ".."} for p in prefix.split("/")):
            raise ValueError("Readiness S3 evidence requires an explicit ab_runs prefix")
        if "/" in run or run in {".", ".."}:
            raise ValueError("Readiness run ID must be one path component")
        bucket = os.environ["S3_BUCKET"]
        client = boto3.client("s3")
        root = f"{prefix}/{run}/readiness/{cell}"

        def write(name, payload):
            key = f"{root}/{name}"
            try:
                client.put_object(Bucket=bucket, Key=key, Body=payload, IfNoneMatch="*")
            except ClientError as exc:
                if exc.response["Error"]["Code"] != "PreconditionFailed":
                    raise
            return f"s3://{bucket}/{key}"

        return write
    root = Path(os.environ["FF_MERGE_READINESS_OUTPUT"]).resolve() / cell
    root.mkdir(parents=True, exist_ok=True)

    def write(name, payload):
        path = root / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Existing readiness evidence has different bytes")
        path.write_bytes(payload)
        return str(path)

    return write


def metric_fn(result, position, *, variant=None):
    context = current_context()
    arm = _ARM if variant is None else variant
    if context is None or _ARM not in {BASELINE, "corrected"}:
        raise ValueError("Readiness evidence requires the harness cell context and policy")
    if not arm or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in arm
    ):
        raise ValueError("Readiness variant must be a nonempty safe identifier")
    write = _evidence_sink(f"{position}-{arm}-{context.seed}")
    capture = _capture_evidence(result)
    cohorts = result.get("cohorts", {})
    unavailable = [
        name for name in REQUIRED_COHORTS if cohorts.get(name, {}).get("status") != "available"
    ]
    if unavailable:
        raise ValueError(f"Required readiness cohort blocks unavailable: {unavailable}")
    metrics = ab_nn_correctness.metric_fn(result, position)
    recipe = result.recipe
    targets = tuple(recipe["targets"])
    reference = load_reference(cache_dir=context.raw_root)
    native = result["test_df"].copy()
    for family, predictions in result["per_target_preds"].items():
        if predictions is not None:
            for target in targets:
                native[f"pred_{family}_{target}"] = predictions[target]
    frames = {"native": native}
    replay_input = None
    if position == "QB":
        from src.analysis.artifact_eval import build_test_df_from_artifacts

        train, val, test = (
            pd.read_parquet(context.splits_dir / f"{name}.parquet")
            for name in ("train", "val", "test")
        )
        replay_test, replay_input = _qb_replay(test)
        replay = build_test_df_from_artifacts(
            position,
            train,
            val,
            replay_test,
            model_dir=str(context.output_dir(position) / "models"),
        )
        if replay.attrs.get("prediction_errors"):
            raise ValueError(f"QB replay model errors: {replay.attrs['prediction_errors']}")
        frames["pregame_replay"] = replay
    evidence = {}
    for regime, frame in frames.items():
        rows, reference_status = _rows(frame, position, targets, reference)
        if regime != "native":
            pd.testing.assert_frame_equal(
                evidence["native"]["identity_frame"], rows[KEYS], check_exact=True
            )
            pd.testing.assert_frame_equal(
                evidence["native"]["truth_frame"],
                rows[[*targets, "comparison_actual", "comparison_available"]],
                check_exact=True,
                check_dtype=False,
            )
        payload = rows.to_parquet(index=False)
        digest = _sha256(payload)
        location = write(f"{regime}-{digest}.parquet", payload)
        metrics.update(_row_metrics(rows, regime))
        evidence[regime] = {
            "location": location,
            "sha256": digest,
            "n_rows": len(rows),
            "identity_frame": rows[KEYS],
            "truth_frame": rows[[*targets, "comparison_actual", "comparison_available"]],
            "reference": reference_status,
            "evaluation_record": record_for_result(
                position,
                {**result, "test_df": frame},
                actual_columns=targets,
                metric_definition=f"merge_readiness:{regime}",
                use_environment=True,
            ).to_dict(),
            "frame_metadata": dict(frame.attrs),
        }
    for entry in evidence.values():
        del entry["identity_frame"]
        del entry["truth_frame"]
    from src.shared.utils import cuda_graph_enabled, cuda_graph_full_enabled

    metadata = {
        "schema_version": 1,
        "position": position,
        "variant": arm,
        "seed": context.seed,
        "execution": result.get("execution"),
        "image_sha": os.environ.get("FF_TRAIN_GIT_SHA"),
        "data_release": os.environ.get("FF_DATA_RELEASE"),
        "prepared_data_id": result.get("data_id"),
        "recipe": {
            "features": asdict(recipe.features),
            "model": asdict(recipe.model),
            "training": {
                key: recipe[key] for key in recipe.training.values if not callable(recipe[key])
            },
        },
        "input_sha256": {
            **{
                f"splits/{name}.parquet": _sha256(
                    (context.splits_dir / f"{name}.parquet").read_bytes()
                )
                for name in ("train", "val", "test")
            },
            "archived_reference": _sha256(reference_path(cache_dir=context.raw_root).read_bytes()),
        },
        "replay_input": replay_input,
        "cuda_capture": {
            "enabled_gate": cuda_graph_enabled(),
            "full_step_enabled_gate": cuda_graph_full_enabled(),
            **capture,
        },
        "evaluations": evidence,
    }
    payload = canonical_json(metadata).encode()
    location = write(f"manifest-{_sha256(payload)}.json", payload)
    print(f"[merge-readiness] evidence {location}", flush=True)
    metrics["readiness"] = {
        "native_rows": evidence["native"]["n_rows"],
        "pregame_rows": evidence.get("pregame_replay", {}).get("n_rows", 0),
        "required_cohorts_available": len(REQUIRED_COHORTS),
        **{
            f"{family}_full_step_capture": sum(
                event["family"] == family
                and event["returned_true"]
                and event["graph_present"]
                and event["capturable_loss"]
                for event in capture["events"]
            )
            for family in ("nn", "attn_nn")
        },
    }
    return metrics


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    return ab_main("src.tuning.ab_merge_readiness", ["--no-stacked-seeds", *args])


if __name__ == "__main__":
    main()
