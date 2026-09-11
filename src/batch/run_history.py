"""Immutable per-run results, published by the last completing training job.

The launcher registers the expected positions before submitting work. Full/merge
jobs save their own metrics after uploading their artifacts, then attempt the
roll-up. S3's strong read consistency means the last writer can see every result;
conditional writes make concurrent completion and retries idempotent. Publication
does not depend on the launcher's wait or on the mutable serving manifests.
"""

import hashlib
import json
import os
import re
import uuid

from botocore.exceptions import ClientError

from src.shared.benchmark_utils import summarize_pipeline_result, utc_now_iso
from src.shared.registry import ALL_POSITIONS


def _prefix():
    return os.environ.get("FF_MODEL_S3_PREFIX", "models").strip("/") or "models"


def _run_key(run_id, filename):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", run_id):
        raise ValueError("run_id must be a safe, nonempty S3 path component")
    return f"{_prefix()}/training_runs/{run_id}/{filename}"


def _read(s3, bucket, key):
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def _write_once(s3, bucket, key, value):
    for attempt in range(3):
        try:
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(value, indent=2).encode(),
                ContentType="application/json",
                IfNoneMatch="*",
            )
            return value
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in {"412", "PreconditionFailed"}:
                existing = _read(s3, bucket, key)
                if existing is not None:
                    return existing
            elif code not in {"409", "ConditionalRequestConflict"}:
                raise
            if attempt == 2:
                raise


def create_run(
    s3,
    bucket,
    positions,
    *,
    run_id=None,
    git_sha=None,
    pr_number=None,
    seed=42,
    note="AWS Batch training run",
    build_plan_id=None,
    dataset_id=None,
    data_release=None,
    legacy_run_id=None,
):
    """Register a complete run before any position can finish or a wait can expire."""
    if not positions or set(positions) - set(ALL_POSITIONS):
        raise ValueError("A history run requires known positions")
    if data_release not in {None, "legacy"} and dataset_id != data_release:
        raise ValueError("History data release and dataset identity must match")
    run_id = run_id or f"batch-{uuid.uuid4().hex}"
    descriptor = {
        "schema_version": 1,
        "run_id": run_id,
        "positions": [p for p in ALL_POSITIONS if p in positions],
        "git_sha": git_sha or None,
        "pr_number": pr_number,
        "seed": seed,
        "note": note,
        "build_plan_id": build_plan_id,
        "dataset_id": dataset_id,
        "data_release": data_release,
        "legacy_run_id": legacy_run_id,
        "created_at": utc_now_iso(),
    }
    existing = _write_once(s3, bucket, _run_key(run_id, "run.json"), descriptor)
    if any(existing.get(k) != v for k, v in descriptor.items() if k != "created_at"):
        raise ValueError(f"History run {run_id} already describes different work")
    return run_id


def _descriptor(s3, bucket, run_id):
    descriptor = _read(s3, bucket, _run_key(run_id, "run.json"))
    if descriptor is None:
        raise ValueError(f"History run {run_id} was not registered")
    return descriptor


def _check_sha(expected, actual):
    if expected and (
        not actual
        or min(len(expected), len(actual)) < 7
        or not (expected.startswith(actual) or actual.startswith(expected))
    ):
        raise ValueError(f"Training history SHA mismatch: expected {expected}, got {actual}")


def publish_position(
    s3,
    bucket,
    run_id,
    position,
    metrics,
    artifact_key,
    *,
    smoke_passed=True,
    publication_status=None,
):
    """Save this job's result and publish the complete run when all positions arrive."""
    descriptor = _descriptor(s3, bucket, run_id)
    if position not in descriptor["positions"]:
        raise ValueError(f"Unexpected position {position} for history run {run_id}")
    _check_sha(descriptor["git_sha"], metrics.get("git_sha"))
    for name in ("build_plan_id", "dataset_id", "data_release"):
        if descriptor.get(name) is not None and descriptor[name] != metrics.get(name):
            raise ValueError(f"Training history {name} differs from its registered request")
    if descriptor.get("legacy_run_id") is not None and (
        metrics.get("publication_intent", {}).get("run_id") != descriptor["legacy_run_id"]
    ):
        raise ValueError("Training history publication run differs from its registered request")
    result = {
        "run_id": run_id,
        "position": position,
        "completed_at": utc_now_iso(),
        "artifact_key": artifact_key,
        "metrics": metrics,
        "smoke_passed": smoke_passed,
        "validation_status": "accepted" if smoke_passed else "validation_failed",
        "publication_status": publication_status,
    }
    if smoke_passed:
        # Callers supply metrics from the canonical accepted output receipt.
        _write_once(s3, bucket, _run_key(run_id, f"{position}.json"), result)
    else:
        attempt_id = hashlib.sha256(artifact_key.encode()).hexdigest()
        attempt = _write_once(
            s3, bucket, _run_key(run_id, f"attempts/{position}/{attempt_id}.json"), result
        )
        # Failed evidence never occupies the accepted-result slot. A first
        # failed presentation remains available until a successful retry lands.
        _write_once(s3, bucket, _run_key(run_id, f"{position}.failed.json"), attempt)
    return complete_run(s3, bucket, run_id)


def complete_run(s3, bucket, run_id, *, positions=None, git_sha=None, data_release=None):
    """Return/publish an immutable summary, or None while any position is missing.

    Also used by the CLI to retrieve exactly its own run for the git history
    commit, and to retry publication after a transient failure without retraining.
    """
    descriptor = _descriptor(s3, bucket, run_id)
    if positions is not None and set(positions) != set(descriptor["positions"]):
        raise ValueError("Requested positions do not match the registered history run")
    _check_sha(git_sha, descriptor["git_sha"])
    if data_release is not None and descriptor.get("data_release") != data_release:
        raise ValueError("Requested data release differs from registered history run")
    results = []
    for position in descriptor["positions"]:
        result = _read(s3, bucket, _run_key(run_id, f"{position}.json"))
        if result is None:
            result = _read(s3, bucket, _run_key(run_id, f"{position}.failed.json"))
        if result is None:
            return None
        if result["run_id"] != run_id or result["position"] != position:
            raise ValueError("Training result belongs to another run or position")
        _check_sha(descriptor["git_sha"], result["metrics"].get("git_sha"))
        for name in ("build_plan_id", "dataset_id", "data_release"):
            if descriptor.get(name) is not None and result["metrics"].get(name) != descriptor[name]:
                raise ValueError(f"Training history {name} differs from its registered request")
        results.append(result)

    # Reuse the existing hardware label; local import avoids CLI import cycles.
    from src.batch.benchmark import _derive_instance_label

    metrics = {r["position"]: r["metrics"] for r in results}
    timestamp = max(r["completed_at"] for r in results)
    git_short = (descriptor["git_sha"] or "unknown")[:7]
    entry = {
        "run_id": f"{timestamp}_{git_short}_{run_id}",
        "training_run_id": run_id,
        "timestamp": timestamp,
        "git_hash": git_short,
        "pr_number": descriptor["pr_number"],
        "backend": "batch",
        "instance_type": _derive_instance_label(metrics, backend="batch", fallback="AWS Batch"),
        "note": descriptor["note"],
        "positions": descriptor["positions"],
        "data_release": descriptor.get("data_release"),
        "dataset_id": descriptor.get("dataset_id"),
        "build_plan_id": descriptor.get("build_plan_id"),
        "results": [summarize_pipeline_result(p, metrics[p]) for p in descriptor["positions"]],
        "artifacts": {r["position"]: r["artifact_key"] for r in results},
        "accepted_positions": [r["position"] for r in results if r.get("smoke_passed", True)],
        "validation_status": (
            "accepted" if all(r.get("smoke_passed", True) for r in results) else "validation_failed"
        ),
        "position_outcomes": {
            r["position"]: {
                "smoke_passed": r.get("smoke_passed", True),
                "validation_status": r.get("validation_status", "accepted"),
                "publication_status": r.get("publication_status"),
            }
            for r in results
        },
    }
    if entry["validation_status"] == "validation_failed":
        # Immutable partial presentations must not collide with a later
        # successful retry that completes within the same timestamp second.
        identity = json.dumps(entry["artifacts"], sort_keys=True) + json.dumps(
            entry["accepted_positions"]
        )
        entry["run_id"] += (
            "_validation_failed_" + hashlib.sha256(identity.encode()).hexdigest()[:12]
        )
    filename = entry["run_id"].replace(":", "-") + ".json"
    return _write_once(s3, bucket, f"{_prefix()}/benchmark_history/{filename}", entry)
