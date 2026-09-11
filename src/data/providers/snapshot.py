"""Capture provider responses once; replay exact responses for identified training.

Capture belongs to the release producer. Captured releases replay exact source
responses; older releases use only their complete derived caches. Neither mode
permits missing historical inputs to trigger a network request.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import inspect
import json
import numbers
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ProviderSnapshotRecord:
    """On-disk provider transport record, distinct from a derived SourceResult.

    Field names/status strings preserve the existing capture/replay format.
    The consumer-facing outcome contract lives in src.data.source_result.
    """

    provider: str
    provider_version: str
    loader: str
    request: dict
    retrieved_at: str
    status: Literal["observed", "empty", "unavailable"]
    content_digest: str | None
    rows: int | None
    error: str | None = None


class SourceUnavailable(RuntimeError):
    """A provider response is unavailable in the selected snapshot."""


_missing: set[str] = set()
_lock = threading.Lock()
_request_locks: dict[str, threading.RLock] = {}
_live_sources = ContextVar("live_provider_sources", default=False)
_capture_directory = ContextVar("capture_provider_sources", default=None)


@contextmanager
def capture_provider_sources(directory):
    """Capture the producer's source calls into its eventual sealed raw root."""
    token = _capture_directory.set(str(Path(directory).resolve()))
    try:
        yield
    finally:
        _capture_directory.reset(token)


@contextmanager
def live_provider_sources():
    """Auxiliary/live forecasts have independent provenance from training inputs."""
    token = _live_sources.set(True)
    try:
        yield
    finally:
        _live_sources.reset(token)


def _plain(value):
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, dict):
        return {name: _plain(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _raw_root() -> Path:
    from src.config import CACHE_DIR
    from src.training.context import raw_data_dir

    return Path(raw_data_dir(CACHE_DIR)).resolve()


def provider_replay_mode() -> str | None:
    """Resolve captured versus derived-only replay from the selected data format."""
    if _live_sources.get():
        return None
    raw = _raw_root()
    marker = raw / ".release.json"
    selected_release = os.environ.get("FF_DATA_RELEASE") not in (None, "", "legacy")
    if (
        selected_release
        or marker.is_file()
        or os.environ.get("FF_DATA_FORMAT") == "data-release-v1"
    ):
        if marker.is_file():
            mode = json.loads(marker.read_text()).get("provider_sources")
            if mode in {"captured", "derived_only"}:
                return mode
        return "captured" if (raw / "provider_sources").is_dir() else "derived_only"
    return "captured" if os.environ.get("FF_DATASET_ID") else None


def assert_snapshot_sources_complete() -> None:
    prefix = str(_raw_root() / "provider_sources") + ":"
    missing = sorted(name for name in _missing if name.startswith(prefix))
    if provider_replay_mode() == "captured" and missing:
        raise SourceUnavailable(f"Selected dataset lacks required provider responses: {missing}")


def verify_provider_snapshot_files(directory: Path) -> None:
    """Validate captured transport bytes before the data release can be sealed."""
    if not directory.is_dir():
        return
    expected_parquet = set()
    for metadata in directory.glob("*.json"):
        doc = json.loads(metadata.read_text())
        identity = {"loader": doc["loader"], "request": doc["request"]}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if metadata.stem != key or doc.get("status") not in {"observed", "empty", "unavailable"}:
            raise SourceUnavailable(f"Invalid captured provider metadata: {metadata}")
        if doc["status"] == "unavailable":
            continue
        parquet = metadata.with_suffix(".parquet")
        if not parquet.is_file() or _digest(parquet) != doc["content_digest"]:
            raise SourceUnavailable(f"Invalid captured provider content: {parquet}")
        expected_parquet.add(parquet)
    if set(directory.glob("*.parquet")) != expected_parquet:
        raise SourceUnavailable("Captured provider directory has unpaired response files")


def snapshot_source(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        capture = (
            (_capture_directory.get() or os.environ.get("FF_CAPTURE_PROVIDER_SOURCES"))
            if not _live_sources.get()
            else None
        )
        mode = provider_replay_mode()
        if mode == "derived_only":
            from src.data.release import DataReleaseError

            raise DataReleaseError(
                f"Historical release has derived caches only; provider call {function.__name__} "
                "cannot fetch missing inputs. Rebuild the release with current producers."
            )
        replay = mode == "captured"
        if not capture and not replay:
            return function(*args, **kwargs)
        request = signature.bind(*args, **kwargs)
        request.apply_defaults()
        request = _plain(dict(request.arguments))
        identity = {"loader": function.__name__, "request": request}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        directory = Path(capture) if capture and not replay else _raw_root() / "provider_sources"
        metadata, parquet = directory / f"{key}.json", directory / f"{key}.parquet"
        missing_key = str(directory) + ":" + function.__name__ + ":" + key
        with _lock:
            request_lock = _request_locks.setdefault(key, threading.RLock())
        with request_lock:
            if metadata.exists():
                doc = json.loads(metadata.read_text())
                if doc["loader"] != function.__name__ or doc["request"] != request:
                    with _lock:
                        _missing.add(missing_key)
                    raise SourceUnavailable(
                        f"Provider snapshot identity mismatch: {function.__name__}"
                    )
                if doc["status"] == "unavailable":
                    raise SourceUnavailable(
                        f"Captured provider was unavailable: {doc.get('error')}"
                    )
                if not parquet.exists() or _digest(parquet) != doc["content_digest"]:
                    with _lock:
                        _missing.add(missing_key)
                    raise SourceUnavailable(
                        f"Provider snapshot content mismatch: {function.__name__}"
                    )
                import pandas as pd

                return pd.read_parquet(parquet)
            if replay:
                with _lock:
                    _missing.add(missing_key)
                raise SourceUnavailable(
                    f"Provider response absent from selected dataset: {identity}"
                )
            directory.mkdir(parents=True, exist_ok=True)
            retrieved = datetime.now(UTC).isoformat()
            version = importlib.metadata.version("nflreadpy")
            try:
                frame = function(*args, **kwargs)
            except Exception as error:
                result = ProviderSnapshotRecord(
                    "nflverse",
                    version,
                    function.__name__,
                    request,
                    retrieved,
                    "unavailable",
                    None,
                    None,
                    str(error),
                )
                metadata.write_text(json.dumps(asdict(result), sort_keys=True))
                raise
            frame.to_parquet(parquet, index=False)
            result = ProviderSnapshotRecord(
                "nflverse",
                version,
                function.__name__,
                request,
                retrieved,
                "empty" if frame.empty else "observed",
                _digest(parquet),
                len(frame),
            )
            metadata.write_text(json.dumps(asdict(result), sort_keys=True))
            return frame

    return wrapped
