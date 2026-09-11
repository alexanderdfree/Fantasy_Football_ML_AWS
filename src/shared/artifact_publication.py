"""Read/source compatibility for the predecessor publication module.

The old writer accepted upload-time source claims without a pre-training intent.
It cannot be adapted to the current publication protocol without losing dataset
ordering, rollback fences and canonical receipts. Callers must use the training
entrypoint, or the explicit initialize-only operator seeder for a fresh bucket.
"""

from src.artifacts.model_sync import load_manifest_snapshot as snapshot
from src.artifacts.publication import references
from src.artifacts.source import (
    image_source_sha,
    load_source,
    read_source,
    register_source,
    source_key,
)

__all__ = [
    "PublicationSuperseded",
    "image_source_sha",
    "load_source",
    "publish_artifact",
    "read_source",
    "references",
    "register_source",
    "snapshot",
    "source_key",
]


class PublicationSuperseded(RuntimeError):
    """Legacy exception retained for import compatibility."""


def publish_artifact(*args, **kwargs):
    """Reject an unbound predecessor writer before any reads or writes."""
    raise RuntimeError(
        "The predecessor publish_artifact API has no pre-training intent or canonical receipt. "
        "Use src.batch.train with an immutable build plan/identified run, or "
        "src.scripts.seed_s3_models for initialize-only operator import."
    )
