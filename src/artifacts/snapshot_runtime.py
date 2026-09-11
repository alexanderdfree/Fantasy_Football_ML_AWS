"""Read-only serving of verified prediction generations; no ML imports."""

import io
import json
import os
from datetime import UTC, datetime

import pandas as pd
from werkzeug.exceptions import ServiceUnavailable

from src.artifacts import serving_snapshot
from src.artifacts import snapshot_state as app_pkg
from src.artifacts.position_metadata import _ALL_POSITIONS
from src.artifacts.serving_snapshot import CACHE_SCHEMA_VERSION as _PREDICTIONS_CACHE_SCHEMA_VERSION
from src.contracts.serialization import _VALID_SCORING, _records_to_player_rows

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PREDICTIONS_CACHE_DIR = os.path.join(_REPO_ROOT, "data", "serving_cache")
_PREDICTIONS_PARQUET = "predictions.parquet"
_METRICS_JSON = "metrics.json"
_FINGERPRINT_JSON = "fingerprint.json"
_SNAPSHOT_JSON = "snapshot.json"


def _artifact_only():
    return True


def _positions_pending():
    return False


def _any_position_sentinel_advanced():
    return False


def _compute_models_fingerprint():
    return app_pkg._cache.get("prediction_inputs_fingerprint"), {}


def _ensure_metrics():
    _discard_invalidated_generation()
    if app_pkg.current_snapshot() is not None:
        return
    with app_pkg._cache_lock:
        try:
            directory = _cache_read_directory()
            if (
                app_pkg._cache.get("snapshot_generation") == directory.name
                and "metrics_by_format" in app_pkg._cache
            ):
                return
            if _try_hydrate_from_disk():
                return
        except (OSError, ValueError, KeyError, TypeError):
            pass
    raise ServiceUnavailable("Serving snapshot is warming")


def _ensure_base_data():
    _ensure_metrics()


def _ensure_position_loaded(_position):
    _ensure_metrics()


def _ensure_all_positions_loaded():
    _ensure_metrics()


def _degraded_positions() -> list[str]:
    """Positions with any recorded model-load error (fully failed OR partial
    per-model failure). Returned sorted so the frontend banner has a stable
    ordering across requests.
    """
    errs = app_pkg._cache.get("position_load_errors", {})
    if not errs:
        return []
    degraded: set[str] = set()
    # Snapshot the keys: pre-warm workers insert ``{pos}_{model}`` error keys
    # into the live dict without a lock, so iterating it directly can raise
    # "dictionary changed size during iteration" during the cold-start window.
    for key in list(errs):
        for p in _ALL_POSITIONS:
            if key == p or key.startswith(f"{p}_"):
                degraded.add(p)
                break
    return sorted(degraded)


def _cache_read_directory():
    from src.artifacts.serving_snapshot import active_directory

    return active_directory(_PREDICTIONS_CACHE_DIR)


def _snapshot_bytes():
    """Serialize browser data from the same completed results as API responses."""
    results = app_pkg._cache.get("results")
    if results is None:
        return None
    try:
        return json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "weeks": sorted(int(w) for w in results["week"].unique()),
                "degraded_positions": _degraded_positions(),
                "scoring": {
                    fmt: _records_to_player_rows(results, scoring=fmt) for fmt in _VALID_SCORING
                },
            }
        ).encode()
    except Exception as exc:
        print(f"[snapshot] serialization failed: {exc!r}")
        return None


def _verified_cache_bytes(generation=None):
    """Read one generation; only explicit runtime inference checks local inputs."""
    directory, files = serving_snapshot.read_generation(
        _PREDICTIONS_CACHE_DIR,
        generation,
        expected_dataset_id=os.environ.get("FF_DATA_RELEASE") if _artifact_only() else None,
    )
    fingerprint = json.loads(files[_FINGERPRINT_JSON])
    if (
        not isinstance(fingerprint, dict)
        or fingerprint.get("schema_version") != _PREDICTIONS_CACHE_SCHEMA_VERSION
    ):
        raise ValueError("Serving prediction cache schema mismatch")
    if not _artifact_only() and fingerprint.get("sha256") != _compute_models_fingerprint()[0]:
        raise ValueError("Serving prediction cache fingerprint mismatch")
    return directory, files


def _snapshot_response_bytes():
    """Return verified captured bytes, never a path reopened after verification."""
    captured = app_pkg.current_snapshot()
    generation = captured.cache.get("snapshot_generation") if captured is not None else None
    if captured is not None and generation is None:
        # A local in-memory publication need not have a disk representation.
        return _snapshot_bytes(), None
    try:
        directory, files = _verified_cache_bytes(generation)
        if serving_snapshot.is_invalidated(_PREDICTIONS_CACHE_DIR, directory.name):
            return None, None
        return files[_SNAPSHOT_JSON], directory.name
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


def _try_hydrate_from_disk():
    """Parse verified generation bytes and publish one complete owned snapshot."""
    try:
        mtimes = {}
        directory, files = _verified_cache_bytes()
        stored = json.loads(files[_FINGERPRINT_JSON])
        results = pd.read_parquet(io.BytesIO(files[_PREDICTIONS_PARQUET]))
        metrics_payload = json.loads(files[_METRICS_JSON])
        metrics_by_format = metrics_payload["metrics_by_format"]
        position_details = metrics_payload.get("position_details") or {}
        position_load_errors = metrics_payload.get("position_load_errors") or {}
        if not _artifact_only() and _compute_models_fingerprint()[0] != stored.get("sha256"):
            return False
        if serving_snapshot.is_invalidated(_PREDICTIONS_CACHE_DIR, directory.name):
            return False
    except Exception as exc:
        print(f"[predcache] generation unavailable: {exc!r}")
        return False
    cache = app_pkg.current_state().cache
    cache.update(
        {
            "results": results,
            "metrics_by_format": metrics_by_format,
            "metrics": metrics_by_format.get("ppr", {}),
            "snapshot_generation": directory.name,
            "model_bundle_ids": metrics_payload.get("model_bundle_ids", {}),
            "model_metadata": metrics_payload.get("model_metadata", {}),
            "comparison_snapshot": metrics_payload.get("comparison_snapshot"),
            "positions_loaded": set(_ALL_POSITIONS)
            - {key.split("_", 1)[0] for key in position_load_errors},
            "positions_failed": set(),
            "positions_failed_mtime": {},
            "positions_mtime": mtimes,
            "base_loaded": True,
            "position_details": position_details,
            "position_load_errors": position_load_errors,
            "prediction_inputs_fingerprint": stored.get("sha256"),
        }
    )
    cache.pop("base_load_error", None)
    print(f"[predcache] hydrated generation {directory.name[:12]} (rows={len(results)})")
    if _artifact_only() or not _positions_pending():
        app_pkg.current_state().publish()
    return True


def _discard_invalidated_generation():
    """Stop using a generation another worker revoked, regardless of local mtimes."""
    owner = app_pkg.current_state()
    generation = owner.cache.get("snapshot_generation")
    if generation is None or not serving_snapshot.is_invalidated(
        _PREDICTIONS_CACHE_DIR, generation
    ):
        return
    with owner.cache_lock:
        generation = owner.cache.get("snapshot_generation")
        if generation is None or not serving_snapshot.is_invalidated(
            _PREDICTIONS_CACHE_DIR, generation
        ):
            return
        _invalidate_metrics_cache(reason="shared-generation-invalidation")
        owner.cache.pop("snapshot_generation", None)
        owner.cache.pop("prediction_inputs_fingerprint", None)
        for key in ("positions_loaded", "positions_failed"):
            owner.cache[key] = set()
        for key in (
            "positions_mtime",
            "positions_failed_mtime",
            "position_load_errors",
            "position_details",
        ):
            owner.cache[key] = {}


def _invalidate_metrics_cache(*, reason: str) -> None:
    """Revoke this exact consumed generation without deleting another reader's files."""
    owner = app_pkg.current_state()
    owner.cache.pop("metrics_by_format", None)
    owner.cache.pop("metrics", None)
    generation = owner.cache.get("snapshot_generation")
    if generation is not None:
        serving_snapshot.invalidate_generation(_PREDICTIONS_CACHE_DIR, generation)
    owner.snapshots.discard(generation)
    print(f"[predcache] invalidated generation ({reason})")


def _get_data(scoring="ppr"):
    """Full load: all positions + metrics for the requested scoring format."""
    _ensure_metrics()
    metrics_by_format = app_pkg._cache["metrics_by_format"]
    metrics = metrics_by_format.get(scoring) or metrics_by_format.get("ppr", {})
    return app_pkg._cache["results"], metrics
