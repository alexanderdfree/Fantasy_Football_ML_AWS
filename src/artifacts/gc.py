"""Operator artifact collection, coordinated with manifest publication.

Dry-run is the default. Execution holds a manifest CAS lock with no expiration.
The collector never deletes plan-owned artifacts or dataset objects. A crashed
collector leaves its lock visible for explicit recovery after it has stopped.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta

from src.artifacts.model_sync import (
    HISTORY_KEEP_N,
    history_prefix,
    load_manifest_snapshot,
    write_manifest,
)
from src.artifacts.receipts import POSITIONS
from src.orchestration.datasets import DatasetError, read_json


def _snapshot(s3, bucket, prefix, position, *, allow_locked=False):
    manifest, etag = load_manifest_snapshot(s3, bucket, prefix, position, allow_locked=allow_locked)
    if manifest is None:
        raise DatasetError(f"No manifest for {position}; collection cannot establish references")
    return manifest, etag


def _candidates(s3, bucket, prefix, position, manifest, keep_n, cutoff):
    protected = {
        entry["key"]
        for label in ("stable", "previous_stable", "current", "previous")
        if (entry := manifest.get(label)) and entry.get("key")
    }
    protected.update((manifest.get("history") or [])[:keep_n])
    protected.update(_snapshot_model_roots(s3, bucket, prefix, position))
    # Existing receipts can predate the object-metadata protection. Preserve
    # their references too; plan/receipt expiration is a separate operator policy.
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="build-plans/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(f"/artifacts/{position}.json"):
                receipt = read_json(s3, bucket, obj["Key"])
                if receipt.get("key"):
                    protected.add(receipt["key"])
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/releases/v3/run-outputs/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                receipt = read_json(s3, bucket, obj["Key"])
                if receipt.get("position") == position and receipt.get("key"):
                    protected.add(receipt["key"])
    candidates = []
    for page in paginator.paginate(Bucket=bucket, Prefix=history_prefix(prefix, position)):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key in protected:
                continue
            head = s3.head_object(Bucket=bucket, Key=key)
            # A delayed receipt writer may still be running after several later
            # promotions. Ownership stamped atomically with upload protects it.
            if head.get("Metadata", {}).get("build-plan-id") or head.get("Metadata", {}).get(
                "publication-intent"
            ):
                continue
            modified = head.get("LastModified")
            if modified is None or modified > cutoff:
                continue
            candidates.append(key)
    return sorted(candidates)


def _snapshot_model_roots(s3, bucket, prefix, position):
    """Retained complete/pending serving generations own their captured models.

    A builder may have completed model checks before GC acquired the lock and
    still be about to publish its snapshot pointer. Its immutable generation
    manifest was written before those checks, so protect it even before it is
    current. Generation expiration remains a separate retention policy.
    """
    from botocore.exceptions import ClientError

    base = f"{prefix}/predictions_cache/generations/"
    keys = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=base):
        keys.update(
            obj["Key"] for obj in page.get("Contents", []) if obj["Key"].endswith("/manifest.json")
        )
    try:
        pointer = read_json(s3, bucket, f"{prefix}/predictions_cache/current.json")
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"NoSuchKey", "404", "NotFound"}:
            raise
    else:
        expected = f"{base}{pointer.get('generation')}/manifest.json"
        if pointer.get("manifest") != expected:
            raise DatasetError("Cannot establish current serving snapshot references")
        keys.add(expected)
    protected = set()
    for key in keys:
        snapshot = read_json(s3, bucket, key)
        digest = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if key != f"{base}{digest}/manifest.json" or not isinstance(snapshot.get("models"), dict):
            raise DatasetError("Cannot establish retained serving snapshot references")
        model_key = snapshot["models"].get(position)
        if model_key:
            protected.add(model_key)
    return protected


def _require_owner(s3, bucket, prefix, position, owner, revision):
    manifest, etag = _snapshot(s3, bucket, prefix, position, allow_locked=True)
    if (manifest.get("gc_lock") or {}).get("owner") != owner or manifest.get(
        "revision"
    ) != revision:
        raise DatasetError("Collector lock changed; refusing further deletion or automatic unlock")
    return manifest, etag


def collect(
    s3,
    bucket: str,
    prefix: str,
    position: str,
    *,
    execute: bool = False,
    keep_n: int = HISTORY_KEEP_N,
    grace_seconds: int = 86400,
    now: datetime | None = None,
) -> dict:
    if position not in POSITIONS or keep_n < 1 or grace_seconds < 0:
        raise DatasetError("Invalid collection position, keep count or grace period")
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(seconds=grace_seconds)
    manifest, etag = _snapshot(s3, bucket, prefix, position)
    if not execute:
        return {
            "position": position,
            "dry_run": True,
            "candidates": _candidates(s3, bucket, prefix, position, manifest, keep_n, cutoff),
            "deleted": [],
        }

    owner = uuid.uuid4().hex
    locked = copy.deepcopy(manifest)
    locked["gc_lock"] = {
        "owner": owner,
        "acquired_at": now.isoformat(),
        "grace_seconds": grace_seconds,
    }
    write_manifest(s3, bucket, prefix, position, locked, expected_etag=etag)
    revision = locked["revision"]
    deleted = []
    try:
        _require_owner(s3, bucket, prefix, position, owner, revision)
        candidates = _candidates(s3, bucket, prefix, position, locked, keep_n, cutoff)
        retired = set(candidates)
        history = locked.get("history") or []
        retained_history = [key for key in history if key not in retired]
        if retained_history != history:
            current, etag = _require_owner(s3, bucket, prefix, position, owner, revision)
            current["history"] = retained_history
            # Remove the advertised references before destructive work. If a
            # delete fails, retryable unreferenced bytes remain; an operator's
            # history listing never points at something this collector deleted.
            # Pointer/receipt/ownership roots excluded from candidates retain
            # their history entries even when they are beyond keep_n.
            write_manifest(s3, bucket, prefix, position, current, expected_etag=etag)
            revision = current["revision"]
        for start in range(0, len(candidates), 1000):
            _require_owner(s3, bucket, prefix, position, owner, revision)
            keys = candidates[start : start + 1000]
            result = s3.delete_objects(
                Bucket=bucket, Delete={"Objects": [{"Key": key} for key in keys], "Quiet": True}
            )
            if result.get("Errors"):
                raise DatasetError(f"S3 rejected artifact deletions: {result['Errors']}")
            deleted.extend(keys)
        return {
            "position": position,
            "dry_run": False,
            "owner": owner,
            "candidates": candidates,
            "deleted": deleted,
        }
    finally:
        current, etag = _require_owner(s3, bucket, prefix, position, owner, revision)
        del current["gc_lock"]
        write_manifest(s3, bucket, prefix, position, current, expected_etag=etag)


def recover_lock(s3, bucket, prefix, position, owner, *, collector_stopped: bool = False) -> None:
    """Operator recovery only after independently confirming the collector stopped."""
    if not collector_stopped:
        raise DatasetError("Confirm the original collector has stopped before clearing its lock")
    manifest, etag = _snapshot(s3, bucket, prefix, position, allow_locked=True)
    if (manifest.get("gc_lock") or {}).get("owner") != owner:
        raise DatasetError("Collector recovery token does not match the current lock")
    del manifest["gc_lock"]
    write_manifest(s3, bucket, prefix, position, manifest, expected_etag=etag)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="models")
    parser.add_argument("--position", choices=sorted(POSITIONS), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--keep", type=int, default=HISTORY_KEEP_N)
    parser.add_argument("--grace-hours", type=int, default=24)
    parser.add_argument("--recover-lock")
    parser.add_argument("--confirm-collector-stopped", action="store_true")
    parser.add_argument("--output", help="Optional JSON inventory/deletion report path")
    args = parser.parse_args(argv)
    import boto3

    s3 = boto3.client("s3")
    if args.recover_lock:
        if args.execute:
            parser.error("--recover-lock cannot be combined with --execute")
        recover_lock(
            s3,
            args.bucket,
            args.prefix,
            args.position,
            args.recover_lock,
            collector_stopped=args.confirm_collector_stopped,
        )
        print("Collector lock released")
        return
    report = collect(
        s3,
        args.bucket,
        args.prefix,
        args.position,
        execute=args.execute,
        keep_n=args.keep,
        grace_seconds=args.grace_hours * 3600,
    )
    if args.output:
        with open(args.output, "w") as stream:
            json.dump(report, stream, indent=2)
    print(
        json.dumps(
            {**report, "candidates": len(report["candidates"]), "deleted": len(report["deleted"])}
        )
    )


if __name__ == "__main__":
    main()
