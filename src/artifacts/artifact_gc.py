"""Compatibility entrypoint for the suspended artifact retention policy.

Independent Batch producers can publish the same position concurrently. A
publisher's manifest snapshot cannot authorize deletion: an unreferenced object
may belong to another in-flight upload, or become referenced after a fresh read.
Automatic publisher-side collection remains disabled. Explicit operator
collection is available in src.artifacts.gc and acquires the manifest lock;
this legacy snapshot-based entrypoint has no authority to delete objects.
"""

from __future__ import annotations

import warnings

from src.artifacts.model_sync import HISTORY_KEEP_N


def prune(
    s3_client,
    bucket: str,
    prefix: str,
    pos: str,
    manifest: dict,
    keep_n: int = HISTORY_KEEP_N,
) -> list[str]:
    """Retain all objects; preserve the old callable without unsafe deletion.

    ``manifest`` and ``keep_n`` are retained for callers during migration, but
    neither is sufficient authority to delete objects with concurrent writers.
    Returns an empty deletion list and makes no S3 requests.
    """
    warnings.warn(
        "Artifact pruning is disabled for publisher snapshots; "
        "use the coordinated src.artifacts.gc operator command. All objects retained.",
        RuntimeWarning,
        stacklevel=2,
    )
    return []
