"""Resolve and verify the exact artifacts published by one identified build plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.artifacts.intent import validate_intent
from src.artifacts.model_sync import history_prefix
from src.artifacts.source import source_key
from src.orchestration.build_plan import load_plan
from src.orchestration.datasets import (
    DatasetError,
    canonical_bytes,
    file_digest,
    read_json,
    require_id,
)

POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST"}


def receipt_key(plan_id: str, position: str) -> str:
    require_id(plan_id, "build plan ID")
    if position not in POSITIONS:
        raise DatasetError(f"Invalid receipt position: {position}")
    return f"build-plans/{plan_id}/artifacts/{position}.json"


def run_receipt_key(prefix, source_sha, position, run_id):
    source_key(prefix, source_sha)
    if position not in POSITIONS or not isinstance(run_id, str) or not run_id:
        raise DatasetError("Invalid legacy run receipt identity")
    digest = hashlib.sha256(run_id.encode()).hexdigest()
    return f"{prefix.strip('/')}/releases/v3/run-outputs/{source_sha}/{position}/{digest}.json"


def _validate_common(s3, bucket, prefix, receipt, position, source_sha, dataset_id, intent):
    if receipt.get("smoke_passed") is not True:
        raise DatasetError("Artifact receipt smoke_passed must be true")
    if not isinstance(intent, dict) or any(
        intent.get(key) != value
        for key, value in {
            "position": position,
            "source_sha": source_sha,
            "dataset_id": dataset_id,
        }.items()
    ):
        raise DatasetError("Artifact receipt intent differs from its source/position/dataset")
    for name, expected in {
        "schema_version": 1,
        "position": position,
        "git_sha": source_sha,
        "dataset_id": dataset_id,
        "smoke_passed": True,
        "publication_intent": intent,
        "publication_revision": intent.get("publication_revision"),
    }.items():
        if receipt.get(name) != expected:
            raise DatasetError(f"Artifact receipt {name} mismatch for {position}")
    release = receipt.get("data_release")
    if release not in {None, "legacy"} or (
        not receipt.get("build_plan_id") and dataset_id is not None
    ):
        require_id(dataset_id, "data release ID")
        if release != dataset_id or receipt.get("data_format") != "data-release-v1":
            raise DatasetError("Artifact receipt data release differs from its dataset binding")
    validate_intent(s3, bucket, prefix, intent)  # Historical receipts need not remain latest.
    require_id(receipt.get("sha256"), "artifact checksum")
    if (
        not isinstance(receipt.get("key"), str)
        or not receipt["key"].startswith(history_prefix(prefix, position))
        or type(receipt.get("bytes")) is not int
        or receipt["bytes"] < 0
    ):
        raise DatasetError("Artifact receipt has invalid protected object metadata")
    return receipt


def _validate_plan_receipt(s3, bucket, plan_id, position, receipt):
    plan = load_plan(s3, bucket, plan_id)
    if receipt.get("build_plan_id") != plan_id or position not in plan["positions"]:
        raise DatasetError("Artifact receipt belongs to another plan or position")
    from src.orchestration.build_plan import plan_data_format

    if plan_data_format(plan) == "data-release-v1":
        if (
            receipt.get("data_release") != plan["dataset_id"]
            or receipt.get("data_format") != "data-release-v1"
        ):
            raise DatasetError("Artifact receipt data release differs from its immutable plan")
    elif receipt.get("data_release"):
        raise DatasetError("Legacy dataset receipt cannot select a separate data release")
    intent = plan.get("intents", {}).get(position)
    if not isinstance(intent, dict):
        raise DatasetError("Build plan has no publication intent")
    if plan.get("publication_revisions", {}).get(position) != intent.get("publication_revision"):
        raise DatasetError("Build plan rollback epoch differs from its reserved intent")
    return _validate_common(
        s3,
        bucket,
        plan.get("model_prefix", "models"),
        receipt,
        position,
        plan["git_sha"],
        plan["dataset_id"],
        intent,
    )


def load_receipt(s3, bucket: str, plan_id: str, position: str) -> dict:
    receipt = read_json(s3, bucket, receipt_key(plan_id, position))
    return _validate_plan_receipt(s3, bucket, plan_id, position, receipt)


def load_run_receipt(s3, bucket, prefix, source_sha, position, run_id, *, expected_dataset_id=None):
    receipt = read_json(s3, bucket, run_receipt_key(prefix, source_sha, position, run_id))
    intent = receipt.get("publication_intent")
    if not isinstance(intent, dict) or intent.get("run_id") != run_id:
        raise DatasetError("Artifact receipt belongs to another legacy run")
    if expected_dataset_id is not None and intent.get("dataset_id") != expected_dataset_id:
        raise DatasetError("Artifact run receipt differs from the selected data release")
    return _validate_common(
        s3, bucket, prefix, receipt, position, source_sha, intent.get("dataset_id"), intent
    )


def claim_successful_output(s3, bucket, prefix, position, entry):
    """First accepted output wins before pointer mutation; retries use that output.

    Failed smoke candidates never occupy the accepted-output slot. Publication
    status belongs to the mutable model pointer, not this immutable receipt.
    """
    from botocore.exceptions import ClientError

    if entry.get("smoke_passed") is not True:
        raise DatasetError("Failed-smoke artifacts cannot claim accepted output receipts")
    plan_id = entry.get("build_plan_id")
    intent = entry["publication_intent"]
    key = (
        receipt_key(plan_id, position)
        if plan_id
        else run_receipt_key(prefix, entry["git_sha"], position, intent["run_id"])
    )
    document = {"schema_version": 1, "position": position, **entry}
    document.pop("promoted", None)
    if plan_id:
        plan = load_plan(s3, bucket, plan_id)
        if plan.get("model_prefix", "models") != prefix:
            raise DatasetError("Output claim namespace differs from its immutable plan")
        _validate_plan_receipt(s3, bucket, plan_id, position, document)
    _validate_common(
        s3, bucket, prefix, document, position, entry["git_sha"], entry.get("dataset_id"), intent
    )
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=canonical_bytes(document),
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {
            "PreconditionFailed",
            "ConditionalRequestConflict",
            "412",
            "409",
        }:
            raise
    canonical = (
        load_receipt(s3, bucket, plan_id, position)
        if plan_id
        else load_run_receipt(s3, bucket, prefix, entry["git_sha"], position, intent["run_id"])
    )
    return canonical


def publish_receipt(s3, bucket: str, plan_id: str, position: str, entry: dict) -> dict:
    """Compatibility wrapper around canonical first-success output claiming."""
    plan = load_plan(s3, bucket, plan_id)
    return claim_successful_output(
        s3,
        bucket,
        plan.get("model_prefix", "models"),
        position,
        {**entry, "build_plan_id": plan_id},
    )


def download_receipt_artifact(s3, bucket: str, plan_id: str, position: str, target: Path) -> dict:
    receipt = load_receipt(s3, bucket, plan_id, position)
    return download_output_artifact(s3, bucket, receipt, target)


def download_run_artifact(
    s3, bucket, prefix, source_sha, position, run_id, target: Path, *, expected_dataset_id=None
):
    receipt = load_run_receipt(
        s3, bucket, prefix, source_sha, position, run_id, expected_dataset_id=expected_dataset_id
    )
    return download_output_artifact(s3, bucket, receipt, target)


def download_output_artifact(s3, bucket, receipt, target: Path):
    position = receipt["position"]
    s3.download_file(bucket, receipt["key"], str(target))
    if target.stat().st_size != receipt["bytes"] or file_digest(target) != receipt["sha256"]:
        raise DatasetError(f"Published artifact checksum mismatch for {position}")
    with tarfile.open(target, "r:gz") as archive:
        stream = archive.extractfile("benchmark_metrics.json")
        if stream is None:
            raise DatasetError(f"No benchmark metrics in published artifact for {position}")
        metrics = json.load(stream)
    for name in (
        "build_plan_id",
        "dataset_id",
        "data_release",
        "data_format",
        "git_sha",
        "publication_intent",
        "publication_revision",
        "position",
    ):
        if metrics.get(name) != receipt.get(name):
            raise DatasetError(f"Artifact metrics {name} mismatch for {position}")
    return metrics


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--plan-id", required=True)
    args = parser.parse_args(argv)
    import boto3

    s3 = boto3.client("s3")
    plan = load_plan(s3, args.bucket, args.plan_id)

    def verify(position):
        with tempfile.TemporaryDirectory(prefix="verify-artifact-") as temp:
            download_receipt_artifact(
                s3, args.bucket, args.plan_id, position, Path(temp) / "model.tar.gz"
            )
        return position

    with ThreadPoolExecutor(max_workers=len(plan["positions"])) as pool:
        positions = list(pool.map(verify, plan["positions"]))
    print(json.dumps({"build_plan_id": args.plan_id, "verified_positions": positions}))


if __name__ == "__main__":
    main()
