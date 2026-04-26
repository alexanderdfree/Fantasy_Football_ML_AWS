"""Retention for versioned artifacts under ``{prefix}/{POS}/history/``.

Invoked best-effort by ``batch/train.py::upload_artifacts`` after the new
``manifest.json`` is written. A prune failure does not fail the training run
— retention is cleanup, not correctness. Training is flock-serialized on the
EC2 host so concurrent producers can't race on the same position.
"""

from __future__ import annotations

from shared.model_sync import HISTORY_KEEP_N, history_prefix


def prune(
    s3_client,
    bucket: str,
    prefix: str,
    pos: str,
    manifest: dict,
    keep_n: int = HISTORY_KEEP_N,
) -> list[str]:
    """Delete keys under ``history_prefix(prefix, pos)`` that are NOT in the
    manifest's keep set: {current.key, stable.key, previous.key, *history[:keep_n]}.

    Source of truth for "what exists" is the S3 listing (not the manifest),
    so orphan keys from abandoned uploads also get swept up. Returns the list
    of keys that were deleted. Idempotent on retry — a re-run with the same
    manifest deletes nothing new.

    ``stable`` is exempted regardless of its history-list position so the
    last-known-good artifact survives even after ``keep_n`` newer (possibly
    smoke-failing) uploads have rolled into the top of ``history``. This is
    the conservation guarantee the consumer's stable-first fallback depends on.
    """
    keep: set[str] = set()
    for label in ("current", "stable", "previous"):
        entry = manifest.get(label)
        if entry and entry.get("key"):
            keep.add(entry["key"])
    for k in (manifest.get("history") or [])[:keep_n]:
        keep.add(k)

    paginator = s3_client.get_paginator("list_objects_v2")
    to_delete: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=history_prefix(prefix, pos)):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key not in keep:
                to_delete.append(key)

    # S3 delete_objects handles batches of up to 1000 keys per call.
    deleted: list[str] = []
    for i in range(0, len(to_delete), 1000):
        chunk = to_delete[i : i + 1000]
        s3_client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
        )
        deleted.extend(chunk)
    return deleted
