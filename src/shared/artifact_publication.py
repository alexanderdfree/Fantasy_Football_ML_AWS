"""Conditional model publication, isolated from pre-v3 writers and garbage collection.

Only a successful pointer CAS retires objects. Publishers always create fresh
physical keys; rollback copies a retained object instead of resurrecting a key.
Consequently delayed cleanup may delete its retirement set without racing a
later publisher. Unpublished uploads are deliberately excluded from online GC.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

from botocore.exceptions import ClientError

from src.shared.model_sync import (
    build_manifest,
    history_prefix,
    load_manifest,
    manifest_key,
    new_history_key,
    write_manifest,
)

_SHA = re.compile(r"[0-9a-f]{40}\Z")
_MISSING = {"NoSuchKey", "404"}
_CONFLICT = {"PreconditionFailed", "ConditionalRequestConflict", "412", "409"}


class PublicationSuperseded(RuntimeError):
    """A newer source or manual rollback won; do not retry or aggregate its metrics."""


def source_key(prefix: str, source_sha: str) -> str:
    if not _SHA.fullmatch(source_sha):
        raise RuntimeError("Publication requires FF_TRAIN_GIT_SHA as a full 40-character SHA")
    return f"{prefix}/source-revisions/{source_sha}.json"


def register_source(s3, bucket: str, prefix: str, source_sha: str, repo: str = ".") -> dict:
    """Record the actual image revision's main ancestry before submitting jobs.

    Full checkout required. Wall-clock timestamps, workflow ids and the current
    checkout HEAD cannot establish source order across delayed image builds.
    """
    key = source_key(prefix, source_sha)
    main = subprocess.check_output(
        ["git", "rev-list", "--first-parent", "origin/main" if prefix == "models" else source_sha],
        cwd=repo,
        text=True,
    ).splitlines()
    if source_sha not in main:
        raise RuntimeError(
            f"Source {source_sha} is not on origin/main's first-parent history; "
            "fetch full origin/main history and use the actual built image SHA. "
            "Use the explicit promote command for a production rollback."
        )
    lineage = main[main.index(source_sha) :]
    if (
        subprocess.check_output(
            ["git", "rev-parse", "--is-shallow-repository"], cwd=repo, text=True
        ).strip()
        != "false"
    ):
        raise RuntimeError("Source registration requires a full git checkout (fetch-depth: 0)")
    record = {"source_sha": source_sha, "source_order": len(lineage), "lineage": lineage}
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(record).encode(),
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in _CONFLICT:
            raise
        if json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read()) != record:
            raise RuntimeError(
                f"Immutable source record disagrees with git ancestry: {key}"
            ) from exc
    return record


def load_source(s3, bucket: str, prefix: str, source_sha: str) -> dict:
    key = source_key(prefix, source_sha)
    if image_source_sha() != source_sha:
        raise RuntimeError("FF_TRAIN_GIT_SHA disagrees with the actual image/checkout source SHA")
    try:
        source = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in _MISSING:
            raise
        raise RuntimeError(
            f"Missing publication source record {key}; register the built image SHA "
            "with src.scripts.register_training_source before launching training."
        ) from exc
    lineage = source.get("lineage", [])
    if (
        not lineage
        or lineage[0] != source_sha
        or source.get("source_sha") != source_sha
        or source.get("source_order") != len(lineage)
        or len(set(lineage)) != len(lineage)
        or any(not _SHA.fullmatch(s) for s in lineage)
    ):
        raise RuntimeError(f"Malformed publication source record: {key}")
    return source


def image_source_sha() -> str:
    root = Path(__file__).resolve().parents[2]
    baked = root / ".training-source-sha"
    if baked.exists():
        return baked.read_text().strip()
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def snapshot(s3, bucket: str, prefix: str, position: str) -> tuple[dict | None, str | None]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=manifest_key(prefix, position))
    except ClientError as exc:
        if exc.response["Error"]["Code"] in _MISSING:
            return None, None
        raise
    return json.loads(obj["Body"].read()), obj["ETag"]


def references(manifest: dict | None) -> set[str]:
    manifest = manifest or {}
    return set(manifest.get("history") or []) | {
        entry["key"]
        for label in ("current", "stable", "previous")
        if (entry := manifest.get(label)) and entry.get("key")
    }


def _protect_legacy(s3, bucket: str, prefix: str, position: str, old: dict, source: dict) -> dict:
    """Copy retained legacy bytes before cutover; legacy GC cannot reach copies."""
    copies = {}
    for key in references(old):
        data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            metrics_file = tar.extractfile("benchmark_metrics.json")
            metrics = json.load(metrics_file) if metrics_file else {}
        if metrics.get("git_sha") not in source["lineage"]:
            raise RuntimeError(
                f"Cannot migrate {key}: artifact git_sha is missing or is not an ancestor "
                "of the publishing image. Publish from a main descendant of that artifact "
                "or restore its verified provenance before retrying."
            )
        # Preserve the final content-hash segment used by the operator CLI.
        version, sha7 = key.rsplit("/", 2)[-2].rsplit("-", 1)
        protected = new_history_key(prefix, position, version, sha7)
        s3.put_object(Bucket=bucket, Key=protected, Body=data, IfNoneMatch="*")
        copies[key] = protected
    result = dict(old)
    for label in ("current", "stable", "previous"):
        if entry := old.get(label):
            result[label] = {**entry, "key": copies[entry["key"]]}
    result["history"] = [copies[key] for key in old.get("history", [])]
    if "stable" not in old:
        result["stable"] = result.get("current")
    return result


def publish_artifact(
    s3,
    bucket: str,
    prefix: str,
    position: str,
    *,
    new_key: str,
    sha7: str,
    bytes_: int,
    uploaded_at: str,
    smoke_passed: bool,
    source: dict,
    initialize_only: bool = False,
) -> dict | None:
    """Publish a validated unique upload, returning None when safely superseded."""
    if initialize_only and not smoke_passed:
        raise RuntimeError(
            "Initial artifact smoke test failed; refusing to create a serving manifest"
        )
    if not new_key.startswith(history_prefix(prefix, position)):
        raise RuntimeError("Publication requires an isolated v3 artifact key")
    migrated = None
    for _ in range(12):
        old, etag = snapshot(s3, bucket, prefix, position)
        if initialize_only and (old is not None or load_manifest(s3, bucket, prefix, position)):
            return None
        if old is None:
            if migrated is None:
                legacy = load_manifest(s3, bucket, prefix, position)
                migrated = (
                    _protect_legacy(s3, bucket, prefix, position, legacy, source) if legacy else {}
                )
            old = migrated
        prior_source = old.get("publication_source")
        if prior_source:
            rank = prior_source["source_order"]
            if source["source_order"] < rank:
                return None
            if source["source_order"] == rank and old.get("promotion_mode") == "rollback":
                return None
            if prior_source["source_sha"] not in source["lineage"]:
                raise RuntimeError("Publication source is not a descendant of the active source")
        new = build_manifest(new_key, sha7, bytes_, uploaded_at, old, smoke_passed)
        new["publication_source"] = {k: source[k] for k in ("source_sha", "source_order")}
        new["current"].update(new["publication_source"])
        # This exact removal set is safe even if a newer pointer wins before GC.
        # No producer can introduce an existing physical key again.
        new["retired"] = sorted(references(old) - references(new))
        try:
            write_manifest(s3, bucket, prefix, position, new, etag=etag)
        except ClientError as exc:
            if exc.response["Error"]["Code"] in _CONFLICT:
                continue
            raise
        return new
    raise RuntimeError("Publication conflicted repeatedly; pointer unchanged by this attempt")
