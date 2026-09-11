"""Matched nonstacked comparison of current-main and proposed PPR selectors.

Reuses the existing harness and the prior merge-readiness evidence layout.
Both arms keep current production data, features, targets, losses and schedules.
The baseline explicitly selects legacy MAE/independent CPU policies; ppr_rmse
selects all three PPR policies together. No NN count/inheritance corrections
or QB replay are included. Use immutable data/image pins and compare row hashes.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import asdict
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.records import record_for_result
from src.prediction.bundle import canonical_json
from src.shared.comparison_scoring import (
    comparison_actuals,
    comparison_model_totals,
    scoring_components,
)
from src.shared.evaluation_cohorts import (
    load_reference,
    reference_path,
    reference_selection,
    regular_season_rows,
)
from src.training.context import current_context
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
SEEDS = [42, 123, 7]
SUPPORTS_STACKED = False
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
    from src.shared.training import MultiHeadNestedHistoryTrainer, MultiHeadTrainer

    current = MultiHeadTrainer._maybe_graph_full_step
    original = getattr(current, "_rmse_readiness_original", current)

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

    observed._rmse_readiness_original = original
    MultiHeadTrainer._maybe_graph_full_step = observed
    nested_current = MultiHeadNestedHistoryTrainer._maybe_graph_full_step
    nested_original = getattr(nested_current, "_rmse_readiness_original", nested_current)

    @wraps(nested_original)
    def observed_nested(self, train_loader):
        captured = nested_original(self, train_loader)
        _CAPTURE_EVENTS.append(
            {
                "family": "attn_nn",
                "trainer": type(self).__name__,
                "device": str(self.device),
                "use_amp": bool(self._use_amp),
                "returned_true": captured is True,
                "graph_present": self._graphed_step is not None,
                "capturable_loss": callable(
                    getattr(self.criterion, "compute_combined_capturable", None)
                ),
            }
        )
        return captured

    observed_nested._rmse_readiness_original = nested_original
    MultiHeadNestedHistoryTrainer._maybe_graph_full_step = observed_nested


def _capture_evidence(result, position):
    device = str((result.get("execution") or {}).get("device", ""))
    expected = device.startswith("cuda") or os.environ.get("FF_DEVICE") == "cuda"
    events = list(_CAPTURE_EVENTS)
    if expected:
        engaged = {
            event["family"]
            for event in events
            if event["returned_true"] and event["graph_present"] and event["capturable_loss"]
        }
        required = {"nn"} if position == "K" else {"nn", "attn_nn"}
        if not required.issubset(engaged):
            raise ValueError(f"Required CUDA full-step capture did not engage: {events}")
    return {"required": expected, "events": events}


def _configure(config, arm):
    global _ARM
    if os.environ.get("FF_AB_STACKED", "").lower() in {"1", "true", "yes", "on"}:
        raise ValueError("RMSE evidence requires nonstacked production fits")
    if int(os.environ.get("FF_NN_FIXED_EPOCHS", "0") or "0") > 0:
        raise ValueError("RMSE selection comparison requires active early stopping")
    if (
        config.get("epoch_callback") is not None
        or config.get("scheduler_type") == "plateau"
        or config.get("attn_scheduler_type") == "plateau"
    ):
        raise ValueError(
            "Current-main comparison requires the production non-plateau schedule without tuning callbacks"
        )
    _ARM = arm
    _CAPTURE_EVENTS.clear()
    _observe_capture()
    config["nn_selection_metric"] = "weighted_mae" if arm == BASELINE else "fantasy_rmse_ppr"
    config["ridge_selection_metric"] = "raw_mae" if arm == BASELINE else "fantasy_rmse_ppr"
    config["lgbm_selection_metric"] = "per_target" if arm == BASELINE else "fantasy_rmse_ppr"
    return config


def baseline(config):
    return _configure(config, BASELINE)


def ppr_rmse(config):
    return _configure(config, "ppr_rmse")


VARIANTS = [
    Variant(BASELINE, cfg_mutator=baseline, label="Current-main selection policies"),
    Variant("ppr_rmse", cfg_mutator=ppr_rmse, label="Joint PPR RMSE selection policies"),
]


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _prepared_hashes(prepared):
    """Fingerprint actual transformed inputs independently of selector config."""
    hashes = {}
    for split in ("train", "val", "test"):
        frame = getattr(prepared, split)
        payload = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
        schema = repr([(str(c), str(t)) for c, t in zip(frame.columns, frame.dtypes, strict=True)])
        hashes[f"{split}_frame"] = _sha256(schema.encode() + payload)
        x = np.ascontiguousarray(getattr(prepared, f"X_{split}"))
        hashes[f"X_{split}"] = _sha256(str((x.shape, str(x.dtype))).encode() + x.tobytes())
        for target, values in (getattr(prepared, f"y_{split}") or {}).items():
            y = np.ascontiguousarray(values)
            hashes[f"y_{split}/{target}"] = _sha256(
                str((y.shape, str(y.dtype))).encode() + y.tobytes()
            )
    return hashes


def _identity_frame(frame):
    if any(key not in frame for key in KEYS) or frame[KEYS].isna().any().any():
        raise ValueError("Readiness rows require non-null player_id/season/week")
    frame = frame.copy()
    frame["player_id"] = frame.player_id.astype(str)
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate readiness player-week identities")
    return frame


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
        if column
        in {
            *KEYS,
            *targets,
            *scoring_components(position),
            *QB_FIELDS,
            "position",
            "season_type",
            "recent_team",
        }
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
        allowed = prefix.split("/")[0] == "ab_runs"
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
    root = Path(os.environ["FF_RMSE_READINESS_OUTPUT"]).resolve() / cell
    root.mkdir(parents=True, exist_ok=True)

    def write(name, payload):
        path = root / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Existing readiness evidence has different bytes")
        path.write_bytes(payload)
        return str(path)

    return write


def _finite(values):
    try:
        return bool(np.isfinite(np.asarray(values, dtype=float)).all())
    except (ValueError, TypeError):
        return False


def _cohort_evidence(cohorts):
    for name in REQUIRED_COHORTS:
        block = cohorts.get(name, {})
        if block.get("status") != "available" or not (block.get("n") or 0) > 0:
            raise ValueError(f"Required cohort has no available observations: {name}")
        for family in ("Ridge", "NN", "Attention NN", "LightGBM"):
            metrics = block.get("models", {}).get(family, {})
            count_key = "n_weeks" if name == "weekly_actual_top24" else "n"
            value_keys = (
                ("hit_rate", "points_captured", "lineup_regret")
                if name == "weekly_actual_top24"
                else ("mae", "rmse", "bias")
            )
            if not (metrics.get(count_key) or 0) > 0 or not _finite(
                [metrics.get(key) for key in value_keys]
            ):
                raise ValueError(f"Required cohort lacks populated metrics: {name}/{family}")
    return cohorts


def _selection_evidence(result):
    aligned = _ARM == "ppr_rmse"
    recipe = result.recipe
    expected = {
        "nn_selection_metric": "fantasy_rmse_ppr" if aligned else "weighted_mae",
        "ridge_selection_metric": "fantasy_rmse_ppr" if aligned else "raw_mae",
        "lgbm_selection_metric": "fantasy_rmse_ppr" if aligned else "per_target",
    }
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError("Resolved recipe does not contain the intended selection policies")
    evidence = {}
    for family, history_key in (("nn", "history"), ("attn_nn", "attn_history")):
        report = (result.get(history_key) or {}).get("checkpoint_selection") or {}
        curve = report.get("validation_curve") or []
        epoch = report.get("epoch")
        if (
            report.get("metric") != expected["nn_selection_metric"]
            or report.get("fixed_epochs")
            or not isinstance(epoch, int)
            or not 1 <= epoch <= len(curve)
            or not _finite(curve)
            or not _finite([report.get("score")])
            or report["score"] != curve[epoch - 1]
            or report["score"] != min(curve)
            or not _finite([report.get("validation_metrics", {}).get("val_fantasy_rmse_ppr")])
        ):
            raise ValueError(f"Missing or inconsistent selected-checkpoint evidence: {family}")
        if aligned and (
            report.get("scoring_format") != "ppr"
            or report["score"] != report["validation_metrics"]["val_fantasy_rmse_ppr"]
        ):
            raise ValueError(f"Checkpoint did not select PPR RMSE: {family}")
        evidence[family] = report
    ridge = result.models["ridge"]
    lgbm = result.models["lgbm"]
    alphas = dict(ridge._alphas)
    if set(alphas) != set(recipe["targets"]) or not _finite(list(alphas.values())):
        raise ValueError("Fitted Ridge does not expose every selected alpha")
    if lgbm.selection_metric != expected["lgbm_selection_metric"]:
        raise ValueError("Fitted LightGBM did not use the intended selector")
    evidence["fitted_ridge_alphas"] = alphas
    evidence["fitted_lgbm_iterations"] = {
        target: int(lgbm.selected_iterations.get(target, model.best_iteration_))
        for target, model in lgbm._models.items()
    }
    if set(evidence["fitted_lgbm_iterations"]) != set(recipe["targets"]) or any(
        value <= 0 for value in evidence["fitted_lgbm_iterations"].values()
    ):
        raise ValueError("Fitted LightGBM lacks selected validation prefixes")
    for family, metric, count_key in (
        ("ridge", "mean_cv_fantasy_rmse_ppr", "n_folds"),
        ("lgbm", "fantasy_rmse_ppr", "n_validation_rows"),
    ):
        report = result.get(f"{family}_selection")
        if not aligned:
            if report:
                raise ValueError(f"Legacy {family} unexpectedly reports PPR selection")
            evidence[family] = None
            continue
        report = report or {}
        curve = report.get("score_history") or []
        if (
            report.get("metric") != metric
            or report.get("scoring_format") != "ppr"
            or not (report.get(count_key) or 0) > 0
            or not curve
            or not _finite(curve)
            or not _finite([report.get("score")])
        ):
            raise ValueError(f"Missing or inconsistent PPR selection evidence: {family}")
        if family == "ridge":
            special = set(recipe.get("two_stage_targets", {})) | set(
                recipe.get("classification_targets", {})
            )
            tuned = set(recipe["targets"]) - special
            if set(report.get("alphas", {})) != tuned or any(
                report["alphas"][t] != alphas[t] for t in tuned
            ):
                raise ValueError("Ridge selected alphas do not match fitted models")
        elif report.get("iterations") != evidence["fitted_lgbm_iterations"]:
            raise ValueError("LightGBM selected prefixes do not match fitted models")
        evidence[family] = report
    return evidence


def metric_fn(result, position):
    from src.tuning.ab_classical_selection import metric_fn as selection_metrics

    context = current_context()
    if context is None or _ARM not in {BASELINE, "ppr_rmse"}:
        raise ValueError("Readiness evidence requires the harness cell context and policy")
    write = _evidence_sink(f"{position}-{_ARM}-{context.seed}")
    capture = _capture_evidence(result, position)
    cohorts = _cohort_evidence(result.get("cohorts", {}))
    selection = _selection_evidence(result)
    recipe = result.recipe
    targets = tuple(recipe["targets"])
    reference = load_reference(cache_dir=context.raw_root)
    native = result["test_df"].copy()
    for family, predictions in result["per_target_preds"].items():
        if predictions is not None:
            for target in targets:
                native[f"pred_{family}_{target}"] = predictions[target]
    metrics = selection_metrics({**result, "test_df": native}, position)
    frames = {"native": native}
    evidence = {}
    for regime, frame in frames.items():
        rows, reference_status = _rows(frame, position, targets, reference)
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
                metric_definition=f"rmse_readiness:{regime}",
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
        "variant": _ARM,
        "seed": context.seed,
        "execution": result.get("execution"),
        "image_sha": os.environ.get("FF_TRAIN_GIT_SHA"),
        "data_release": os.environ.get("FF_DATA_RELEASE"),
        "prepared_data_id": result.get("data_id"),
        "prepared_inputs_sha256": _prepared_hashes(result.prepared),
        "cohorts": cohorts,
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
        "selection": selection,
        "cuda_capture": {
            "enabled_gate": cuda_graph_enabled(),
            "full_step_enabled_gate": cuda_graph_full_enabled(),
            **capture,
        },
        "evaluations": evidence,
    }
    payload = canonical_json(metadata).encode()
    location = write(f"manifest-{_sha256(payload)}.json", payload)
    print(f"[rmse-readiness] evidence {location}", flush=True)
    metrics["readiness"] = {
        "native_rows": evidence["native"]["n_rows"],
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
    return ab_main("src.tuning.ab_rmse_readiness", ["--no-stacked-seeds", *args])


if __name__ == "__main__":
    main()
