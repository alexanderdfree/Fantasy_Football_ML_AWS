"""Immutable per-run results, published by the last completing training job.

The launcher registers the expected positions before submitting work. Full/merge
jobs save their own metrics after uploading their artifacts, then attempt the
roll-up. S3's strong read consistency means the last writer can see every result;
conditional writes make concurrent completion and retries idempotent. Publication
does not depend on the launcher's wait or on the mutable serving manifests.
"""

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
    data_release=None,
    pr_number=None,
    seed=42,
    note="AWS Batch training run",
):
    """Register a complete run before any position can finish or a wait can expire."""
    if not positions or set(positions) - set(ALL_POSITIONS):
        raise ValueError("A history run requires known positions")
    run_id = run_id or f"batch-{uuid.uuid4().hex}"
    descriptor = {
        "schema_version": 1,
        "run_id": run_id,
        "positions": [p for p in ALL_POSITIONS if p in positions],
        "git_sha": git_sha or None,
        "data_release": data_release or None,
        "pr_number": pr_number,
        "seed": seed,
        "note": note,
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


def _check_data_release(expected, actual):
    # Legacy descriptors/results without this field remain readable. A pinned
    # result must never enter an unknown run, nor may an unknown result enter a
    # pinned run; both would falsely describe one coherent input generation.
    if (expected or None) != (actual or None):
        raise ValueError(
            f"Training history data release mismatch: expected {expected!r}, got {actual!r}"
        )


def publish_position(s3, bucket, run_id, position, metrics, artifact_key):
    """Save this job's result and publish the complete run when all positions arrive."""
    descriptor = _descriptor(s3, bucket, run_id)
    if position not in descriptor["positions"]:
        raise ValueError(f"Unexpected position {position} for history run {run_id}")
    _check_sha(descriptor["git_sha"], metrics.get("git_sha"))
    _check_data_release(descriptor.get("data_release"), metrics.get("data_release"))
    _write_once(
        s3,
        bucket,
        _run_key(run_id, f"{position}.json"),
        {
            "run_id": run_id,
            "position": position,
            "completed_at": utc_now_iso(),
            "artifact_key": artifact_key,
            "metrics": metrics,
        },
    )
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
    if data_release is not None:
        _check_data_release(descriptor.get("data_release"), data_release)
    results = []
    for position in descriptor["positions"]:
        result = _read(s3, bucket, _run_key(run_id, f"{position}.json"))
        if result is None:
            return None
        if result["run_id"] != run_id or result["position"] != position:
            raise ValueError("Training result belongs to another run or position")
        _check_sha(descriptor["git_sha"], result["metrics"].get("git_sha"))
        _check_data_release(descriptor.get("data_release"), result["metrics"].get("data_release"))
        results.append(result)

    # Reuse the existing hardware label; local import avoids CLI import cycles.
    from src.batch.benchmark import _derive_instance_label

    metrics = {r["position"]: r["metrics"] for r in results}
    timestamp = max(r["completed_at"] for r in results)
    git_short = (descriptor["git_sha"] or "unknown")[:7]
    # The generic numeric summary does not own run provenance. Attach the
    # already-validated snapshot to both the aggregate and each position.
    data_metadata = (
        {"data_release": descriptor["data_release"]} if descriptor.get("data_release") else {}
    )
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
        "results": [
            {**summarize_pipeline_result(p, metrics[p]), **data_metadata}
            for p in descriptor["positions"]
        ],
        "artifacts": {r["position"]: r["artifact_key"] for r in results},
    }
    entry.update(data_metadata)
    filename = entry["run_id"].replace(":", "-") + ".json"
    return _write_once(s3, bucket, f"{_prefix()}/benchmark_history/{filename}", entry)
