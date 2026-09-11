"""Download and read upcoming projections without importing prediction code."""

import json
import os
import threading

from src.artifacts import snapshot_runtime as core
from src.artifacts import upcoming_transfer

_ARTIFACT_NAME = "upcoming_week.json"
_ENV_BUCKET = "FF_MODEL_S3_BUCKET"
_ENV_PREFIX = "FF_MODEL_S3_PREFIX"
_DOWNLOAD_INTERVAL_S = int(os.environ.get("FF_UPCOMING_SYNC_INTERVAL_S", "60"))


def _artifact_path() -> str:
    return os.path.join(core._PREDICTIONS_CACHE_DIR, _ARTIFACT_NAME)


def read_cached_artifact() -> dict | None:
    """Return the on-disk artifact, or ``None`` if absent/unreadable."""
    path = _artifact_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _s3_artifact_key() -> str:
    prefix = os.environ.get(_ENV_PREFIX, "models").strip("/")
    return f"{prefix}/predictions_cache/{_ARTIFACT_NAME}"


def sync_artifact_from_s3() -> bool:
    """Read a validated generation, preserving last-good data and version recovery."""
    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    if not bucket:
        return False
    return upcoming_transfer.sync(_artifact_path(), bucket, _s3_artifact_key())


def start_artifact_download_poller(
    interval_s: int | None = None, stop_event: threading.Event | None = None
) -> threading.Thread | None:
    """Daemon thread that re-pulls the CI-built artifact from S3 so a fresh build
    appears without a redeploy.

    Interval from ``FF_UPCOMING_SYNC_INTERVAL_S`` (default 60s); ``0`` disables.
    Cheap (a single S3 GET) — none of the build/PBP/OOM cost that the in-serving
    build had. Broad try/except so a transient S3 error never kills the loop.
    """
    interval = _DOWNLOAD_INTERVAL_S if interval_s is None else interval_s
    if interval <= 0:
        return None
    ev = stop_event or threading.Event()

    def _loop():
        while not ev.is_set():
            try:
                sync_artifact_from_s3()
            except Exception as e:  # noqa: BLE001 - poller must never die
                print(f"[upcoming_week] artifact sync failed: {e!r}")
            ev.wait(interval)

    t = threading.Thread(target=_loop, name="upcoming-week-sync", daemon=True)
    t.start()
    return t
