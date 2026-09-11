"""Immutable Git ancestry for ordering releases from the actual training image."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

_SHA = re.compile(r"[0-9a-f]{40}\Z")
_CONFLICT = {"PreconditionFailed", "ConditionalRequestConflict", "412", "409"}


def source_key(prefix: str, source_sha: str) -> str:
    if not isinstance(source_sha, str) or not _SHA.fullmatch(source_sha):
        raise RuntimeError("Publication requires the actual full 40-character training source SHA")
    return f"{prefix.strip('/')}/source-revisions/{source_sha}.json"


def register_source(s3, bucket: str, prefix: str, source_sha: str, repo: str = ".") -> dict:
    """Capture ancestry from a full checkout before dispatching the identified image.

    Production sources must belong to main's first-parent history. A sandbox
    prefix permits a branch lineage, but publication still checks descent from
    its previous source. Completion timestamps never determine source order.
    """
    from botocore.exceptions import ClientError

    prefix = prefix.strip("/")
    key = source_key(prefix, source_sha)
    shallow = subprocess.check_output(
        ["git", "rev-parse", "--is-shallow-repository"], cwd=repo, text=True
    ).strip()
    if shallow != "false":
        raise RuntimeError("Source registration requires full Git history (fetch-depth: 0)")
    history = subprocess.check_output(
        ["git", "rev-list", "--first-parent", "origin/main" if prefix == "models" else source_sha],
        cwd=repo,
        text=True,
    ).splitlines()
    if source_sha not in history:
        raise RuntimeError(
            f"Source {source_sha} is not on origin/main's first-parent history; "
            "fetch current main and register the actual built image SHA. "
            "Use explicit promotion for a production rollback."
        )
    lineage = history[history.index(source_sha) :]
    record = {"source_sha": source_sha, "source_order": len(lineage), "lineage": lineage}
    body = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    try:
        s3.put_object(
            Bucket=bucket, Key=key, Body=body, ContentType="application/json", IfNoneMatch="*"
        )
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in _CONFLICT:
            raise
        if json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read()) != record:
            raise RuntimeError(
                f"Immutable source record disagrees with Git ancestry: {key}"
            ) from error
    return record


def image_source_sha(*, root: Path | None = None) -> str:
    """Read build-time identity; the requested runtime environment cannot override it."""
    root = root if root is not None else Path(__file__).resolve().parents[2]
    baked = root / ".training-source-sha"
    value = (
        baked.read_text().strip()
        if baked.exists()
        else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    )
    source_key("models", value)
    return value


def load_source(s3, bucket: str, prefix: str, source_sha: str) -> dict:
    """Require a registered lineage matching the executable image/checkout."""
    source_key(prefix, source_sha)
    if image_source_sha() != source_sha:
        raise RuntimeError("FF_TRAIN_GIT_SHA disagrees with the actual image/checkout source SHA")
    return read_source(s3, bucket, prefix, source_sha)


def read_source(s3, bucket: str, prefix: str, source_sha: str) -> dict:
    """Validate stored ancestry without claiming it describes this operator's executable."""
    from botocore.exceptions import ClientError

    key = source_key(prefix, source_sha)
    try:
        record = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"NoSuchKey", "404", "NotFound"}:
            raise
        raise RuntimeError(
            f"Missing publication source {key}; register the built image SHA with "
            "src.scripts.register_training_source before launching training"
        ) from error
    lineage = record.get("lineage") if isinstance(record, dict) else None
    if (
        not isinstance(lineage, list)
        or not lineage
        or any(not isinstance(sha, str) or not _SHA.fullmatch(sha) for sha in lineage)
        or lineage[0] != source_sha
        or record.get("source_sha") != source_sha
        or type(record.get("source_order")) is not int
        or record["source_order"] != len(lineage)
        or len(set(lineage)) != len(lineage)
    ):
        raise RuntimeError(f"Malformed publication source record: {key}")
    return record
