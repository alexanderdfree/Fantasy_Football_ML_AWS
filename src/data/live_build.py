"""Process-isolated mutable raw cache for the offline upcoming-week builder."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from src.data.release import DataReleaseError

HISTORICAL_CACHE_ENV = "FF_LIVE_HISTORICAL_CACHE_DIR"


def historical_cache_dir(default: str) -> str:
    return os.environ.get(HISTORICAL_CACHE_ENV, default)


def require_mutable_live_cache(cache_dir: str) -> None:
    """A direct in-process caller must not alter a pinned historical directory."""
    cache = Path(cache_dir).resolve()
    original = os.environ.get(HISTORICAL_CACHE_ENV)
    if (cache / ".release.json").exists() or (original and cache == Path(original).resolve()):
        raise DataReleaseError(
            "Live history requires a separate mutable cache; run python -m src.prediction.upcoming"
        )


def run_in_live_overlay(source_cache: str, command: Sequence[str]) -> int:
    """Run once in a child initialized against copies, preserving caller output paths.

    CWD, model/split paths, stdout/stderr and exit status are retained. No module
    globals or process-wide cwd change under serving's loader worker threads.
    The historical pointer stays pinned; only explicitly scoped live source
    caches may fetch missing data. Copies deliberately exclude the release
    marker because the overlay's schedule/team rollups are mutable.
    """
    source = Path(source_cache).resolve()
    with tempfile.TemporaryDirectory(prefix="ff-live-build-") as temporary:
        overlay = Path(temporary) / "raw"
        shutil.copytree(source, overlay, ignore=shutil.ignore_patterns(".release.json"))
        env = os.environ.copy()
        env["FF_CACHE_DIR"] = str(overlay)
        env[HISTORICAL_CACHE_ENV] = str(source)
        marker = source / ".release.json"
        if marker.is_file():
            env["FF_DATA_RELEASE"] = json.loads(marker.read_text())["release_id"]
        print("[upcoming_week] building in an isolated copy of historical raw inputs", flush=True)
        return subprocess.run(list(command), env=env, check=False).returncode
