"""Versioned evaluation identity, without reinterpreting historical metrics."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CohortIdentity:
    name: str
    definition: str | None
    status: str | None
    cohort_hash: str | None
    reference_hash: str | None
    reference_version: str | None
    actual_basis: str | None
    scoring_components: tuple[str, ...]
    n: int | None


@dataclass(frozen=True)
class EvaluationRecord:
    """Identity alongside existing metric payloads; unknown provenance is null.

    ``evaluation_data_id`` hashes the supplied held-out identities/actuals.
    ``training_data_id`` identifies fitted preparation and is never substituted
    for evaluation rows. Cohorts retain their individual scoring definitions;
    custom A/B metrics are not relabeled as the production cohort contract.
    """

    position: str
    metric_definition: str
    scoring: str | None
    actual_basis: str | None
    sample_basis: str | None
    source_ids: tuple[str, ...]
    cohorts: tuple[CohortIdentity, ...]
    evaluation_data_id: str | None
    training_data_id: str | None
    dataset_id: str | None
    model_bundle_ids: tuple[tuple[str, str], ...]
    execution_regime: str | None
    run_id: str | None
    build_plan_id: str | None
    code_id: str | None
    schema_version: int = 1

    def to_dict(self) -> dict:
        result = asdict(self)
        result["source_ids"] = list(self.source_ids)
        result["model_bundle_ids"] = dict(self.model_bundle_ids)
        result["cohorts"] = {
            item.name: {**asdict(item), "scoring_components": list(item.scoring_components)}
            for item in self.cohorts
        }
        return result


def evaluation_data_identity(frame) -> str | None:
    """Hash row identity and actuals, excluding predictions and fitted features."""
    if frame is None or not hasattr(frame, "columns"):
        return None
    identity = [key for key in ("player_id", "season", "week", "position") if key in frame]
    actuals = sorted(
        key for key in frame.columns if key == "fantasy_points" or key.startswith("actual_")
    )
    if not identity or not actuals:
        return None
    selected = frame[[*identity, *actuals]].sort_values(identity)
    payload = selected.to_json(orient="split", index=False, double_precision=15)
    return hashlib.sha256(payload.encode()).hexdigest()


def record_for_result(
    position: str,
    result: Mapping,
    *,
    cohorts: Mapping | None = None,
    source_ids=(),
    execution_regime: str | None = None,
    metric_definition: str = "pipeline_metrics",
    use_environment: bool = False,
) -> EvaluationRecord:
    """Read explicit producer identities; environment applies only to live runs."""
    frame = result.get("test_df")
    attrs = getattr(frame, "attrs", {})
    prepared = getattr(result, "prepared", None)
    cohort_map = cohorts if cohorts is not None else result.get("cohorts", {})
    identities = tuple(
        CohortIdentity(
            name=name,
            definition=block.get("definition"),
            status=block.get("status"),
            cohort_hash=block.get("cohort_hash"),
            reference_hash=block.get("reference_hash"),
            reference_version=block.get("reference_version"),
            actual_basis=block.get("actual_basis"),
            scoring_components=tuple(block.get("scoring_components", ())),
            n=block.get("n"),
        )
        for name, block in sorted((cohort_map or {}).items())
    )

    def identity(key, variable):
        return result.get(key) or (os.environ.get(variable) if use_environment else None)

    bundles = result.get("model_bundle_ids") or attrs.get("model_bundle_ids") or {}
    return EvaluationRecord(
        position=position,
        metric_definition=metric_definition,
        scoring=result.get("scoring"),
        actual_basis=result.get("actual_basis"),
        sample_basis=result.get("sample_basis"),
        source_ids=tuple(source_ids)
        or tuple(
            key.removesuffix("_metrics")
            for key in result
            if key.endswith("_metrics") and key != "cv_metrics"
        ),
        cohorts=identities,
        evaluation_data_id=evaluation_data_identity(frame),
        training_data_id=result.get("prepared_data_id")
        or result.get("data_id")
        or getattr(prepared, "data_id", None),
        dataset_id=identity("dataset_id", "FF_DATASET_ID"),
        model_bundle_ids=tuple(sorted(bundles.items())),
        execution_regime=execution_regime
        or result.get("execution_regime")
        or (result.get("execution") or {}).get("regime"),
        run_id=result.get("run_id") or getattr(result, "run_id", None),
        build_plan_id=identity("build_plan_id", "FF_BUILD_PLAN_ID"),
        code_id=result.get("code_id")
        or result.get("git_sha")
        or identity("code_id", "FF_TRAIN_GIT_SHA"),
    )
