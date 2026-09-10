"""Unit tests for the serving prediction-cache layer in ``src/serving/app.py``.

Covers:
- ``_compute_models_fingerprint`` reflects content + size changes (NOT mtime —
  that was the original design but caused systematic cache misses across ECS
  task replacements; see ``_compute_models_fingerprint`` docstring).
- ``_persist_cache_to_disk`` + ``_try_hydrate_from_disk`` round-trip the
  results DataFrame and metrics dict.
- ``_try_hydrate_from_disk`` returns False on fingerprint mismatch or
  when any of the three cache files are missing.
- The ``post_fork`` hook in ``gunicorn.conf.py`` spawns a daemon thread
  and returns immediately (so the worker isn't blocked from accepting
  requests by a slow warm).
- Atomic publication: concurrent writers expose one complete immutable
  prediction/metrics/fingerprint/snapshot generation.

The fingerprint test files live under ``tmp_path``; ``_iter_fingerprint_paths``
is monkeypatched to yield them, which avoids needing a real
``src/{pos}/outputs/models/`` tree on the test host. ``_PREDICTIONS_CACHE_DIR``
is also redirected to ``tmp_path`` so the production cache dir is never
touched.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.serving.core as core
from src.shared.prediction_cache import current_generation, publish_generation, read_generation

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_results(n: int = 4) -> pd.DataFrame:
    """Tiny DataFrame shaped like the real ``_cache['results']``: integer
    index, a position column, and pred + actual columns for two scoring
    formats. Enough to confirm parquet round-trip preserves shape + values.
    """
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "player_id": [f"P{i:03d}" for i in range(n)],
            "player_display_name": [f"Player {i}" for i in range(n)],
            "position": ["QB", "RB", "WR", "TE"][:n],
            "week": rng.integers(1, 18, size=n),
            "fantasy_points": rng.uniform(0, 30, size=n).round(2),
            "fantasy_points_half_ppr": rng.uniform(0, 30, size=n).round(2),
            "fantasy_points_standard": rng.uniform(0, 30, size=n).round(2),
            "ridge_pred_ppr": rng.uniform(0, 30, size=n).round(2),
            "nn_pred_ppr": rng.uniform(0, 30, size=n).round(2),
            "nflcom_pred_ppr": rng.uniform(0, 30, size=n).round(2),
            "rotowire_pred_ppr": rng.uniform(0, 30, size=n).round(2),
            "nflcom_pred_half_ppr": rng.uniform(0, 30, size=n).round(2),
            "rotowire_pred_half_ppr": rng.uniform(0, 30, size=n).round(2),
            "nflcom_pred_standard": rng.uniform(0, 30, size=n).round(2),
            "rotowire_pred_standard": rng.uniform(0, 30, size=n).round(2),
        }
    )


def _fake_metrics() -> dict:
    return {
        "ppr": {
            "Ridge Regression": {
                "overall": {"mae": 5.1, "rmse": 6.7, "r2": 0.22},
                "by_position": [{"position": "QB", "mae": 4.8}],
            }
        },
        "half_ppr": {"Ridge Regression": {"overall": None, "by_position": []}},
        "standard": {"Ridge Regression": {"overall": None, "by_position": []}},
    }


def _persist_fixture_cache():
    """The synthetic cache was computed against these unchanged model inputs."""
    core.app_pkg._cache.setdefault(
        "prediction_inputs_fingerprint", core._compute_models_fingerprint()[0]
    )
    core._persist_cache_to_disk()


def _generation(cache_dir: Path) -> Path:
    directory = current_generation(cache_dir)
    assert directory is not None, "a complete generation must have been committed"
    return directory


def test_sentinel_only_invalidation_rejects_old_generation_without_deleting_readers(
    cache_dir, fingerprint_files, monkeypatch
):
    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    core.app_pkg._cache["results"] = _fake_results()
    core.app_pkg._cache["metrics_by_format"] = _fake_metrics()
    _persist_fixture_cache()
    old, files = read_generation(cache_dir)
    core._invalidate_metrics_cache(reason="sentinel-only-refresh")
    assert old.is_dir()
    assert core._snapshot_path() is None
    assert core._try_hydrate_from_disk() is False

    # Another process can finish a new generation without our invalidation
    # deleting or quarantining its pointer, even when input bytes are unchanged.
    snapshot = json.loads(files["snapshot.json"])
    snapshot["generated_at"] = "2026-09-10T13:00:00+00:00"
    publish_generation(cache_dir, {**files, "snapshot.json": json.dumps(snapshot).encode()})
    assert core._snapshot_path() is not None
    assert core._try_hydrate_from_disk() is True


def test_replacement_worker_cannot_hydrate_or_upload_invalidated_generation(
    cache_dir, fingerprint_files, monkeypatch
):
    from src.shared import prediction_cache

    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    core.app_pkg._cache.update(results=_fake_results(), metrics_by_format=_fake_metrics())
    _persist_fixture_cache()
    old, files = read_generation(cache_dir)
    core._invalidate_metrics_cache(reason="sentinel-only-refresh")

    # Replacement gunicorn worker has no predecessor's in-memory error marker.
    core.app_pkg._cache.clear()
    assert core._try_hydrate_from_disk() is False
    assert core._snapshot_path() is None
    with pytest.raises(ValueError):
        prediction_cache.bundle_generation(cache_dir)
    assert (old / "predictions.parquet").read_bytes() == files["predictions.parquet"]


def test_delayed_invalidation_preserves_another_workers_new_generation(
    cache_dir, fingerprint_files, monkeypatch
):
    from src.shared import prediction_cache

    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    core.app_pkg._cache.update(results=_fake_results(), metrics_by_format=_fake_metrics())
    _persist_fixture_cache()
    worker_a = dict(core.app_pkg._cache)
    old, _ = read_generation(cache_dir)

    # B consumes the same initial generation, recomputes, and commits first.
    core.app_pkg._cache.clear()
    assert core._try_hydrate_from_disk()
    _persist_fixture_cache()
    new, new_files = read_generation(cache_dir)
    assert new != old
    core.app_pkg._cache.clear()
    core.app_pkg._cache.update(worker_a)
    core._invalidate_metrics_cache(reason="worker-a-delayed-refresh")

    assert prediction_cache.is_invalidated(cache_dir, old.name)
    assert not prediction_cache.is_invalidated(cache_dir, new.name)
    assert read_generation(cache_dir) == (new, new_files)
    core.app_pkg._cache.clear()
    assert core._try_hydrate_from_disk()
    assert core.app_pkg._cache["prediction_cache_generation"] == new.name


def test_recompute_identical_predictions_publishes_new_generation_after_invalidation(
    cache_dir, fingerprint_files, monkeypatch
):
    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    # Even an absent/unchanged optional snapshot cannot be the uniqueness key.
    monkeypatch.setattr(core, "_snapshot_bytes", lambda: None)
    results, metrics = _fake_results(), _fake_metrics()
    core.app_pkg._cache.update(results=results, metrics_by_format=metrics)
    _persist_fixture_cache()
    old, old_files = read_generation(cache_dir)
    core._invalidate_metrics_cache(reason="sentinel-only-refresh")
    core.app_pkg._cache["metrics_by_format"] = metrics
    _persist_fixture_cache()
    new, new_files = read_generation(cache_dir)

    assert new != old
    assert new_files["predictions.parquet"] == old_files["predictions.parquet"]
    assert new_files["metrics.json"] == old_files["metrics.json"]
    old_fp, new_fp = (json.loads(files["fingerprint.json"]) for files in (old_files, new_files))
    assert old_fp["sha256"] == new_fp["sha256"]
    assert old_fp["computation_id"] != new_fp["computation_id"]
    core.app_pkg._cache.clear()
    assert core._try_hydrate_from_disk()


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Redirect the module-level ``_PREDICTIONS_CACHE_DIR`` to a tmp dir so
    persist/hydrate don't write to ``<repo>/data/serving_cache/``.
    """
    import src.serving.app as app_mod

    target = tmp_path / "serving_cache"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(target))
    monkeypatch.setattr(app_mod, "_cache", {})
    return target


@pytest.fixture
def fingerprint_files(tmp_path, monkeypatch):
    """Three synthetic 'model files' whose mtime + size define the live
    fingerprint. ``_iter_fingerprint_paths`` is monkeypatched to yield them
    in place of walking the real model tree.
    """
    import src.serving.app as app_mod

    files = []
    for name, content in (
        ("model_a.pkl", b"alpha"),
        ("model_b.pt", b"beta-data"),
        ("test_split.parquet", b"gamma!"),
    ):
        p = tmp_path / name
        p.write_bytes(content)
        files.append(str(p))

    def _iter():
        yield from files

    monkeypatch.setattr(core, "_iter_fingerprint_paths", _iter)
    return files


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_stable_across_mtime_bump_when_content_unchanged(fingerprint_files):
    """Fingerprint must be content-driven, not mtime-driven. boto3's
    ``download_file`` stamps the destination with the *download* time, so
    every fresh ECS task saw a different fingerprint and missed the cache
    on boot — see ``_compute_models_fingerprint`` docstring for the
    full reasoning. This test pins the new contract: mtime change with
    identical content must NOT change the fingerprint.
    """
    import src.serving.app as app_mod

    sha1, files1 = core._compute_models_fingerprint()
    assert isinstance(sha1, str) and len(sha1) == 64
    assert len(files1) == 3

    # Bump mtime on one file (size + content unchanged).
    target = fingerprint_files[1]
    new_mtime = os.stat(target).st_mtime + 10.0
    os.utime(target, (new_mtime, new_mtime))

    sha2, _ = core._compute_models_fingerprint()
    assert sha2 == sha1, "fingerprint must NOT change when only mtime changes (content-hash bug)"


def test_fingerprint_changes_on_content_change(fingerprint_files):
    """Modifying the head bytes of a fingerprint input MUST change the
    fingerprint — the content-hash sampling reads the first 64 KB, and every
    file in the fingerprint set is much shorter than that, so any byte
    change in the file is fully visible to the hash.
    """
    import src.serving.app as app_mod

    sha1, _ = core._compute_models_fingerprint()
    # Mutate content (same length so size doesn't carry the signal).
    target = Path(fingerprint_files[1])
    original = target.read_bytes()
    target.write_bytes(b"X" * len(original))

    sha2, _ = core._compute_models_fingerprint()
    assert sha2 != sha1, "fingerprint must change when file content changes"


def test_fingerprint_changes_on_size_change(fingerprint_files):
    import src.serving.app as app_mod

    sha1, _ = core._compute_models_fingerprint()
    # Rewrite one file with different content (different size).
    Path(fingerprint_files[0]).write_bytes(b"alpha-extended")
    sha2, _ = core._compute_models_fingerprint()
    assert sha2 != sha1


@pytest.mark.parametrize("position", core._ALL_POSITIONS)
def test_manifest_identity_changes_fingerprint_even_with_identical_models(
    tmp_path, monkeypatch, position
):
    monkeypatch.setattr(core, "_REPO_ROOT", str(tmp_path))
    outputs = tmp_path / "src" / position.lower() / "outputs"
    models = outputs / "models"
    models.mkdir(parents=True)
    (models / "weights.pt").write_bytes(b"unchanged model bytes")
    sidecar = outputs / ".manifest-etag"
    sidecar.write_text('"manifest-a"')
    before, paths = core._compute_models_fingerprint()
    assert any(entry["path"].endswith("outputs/.manifest-etag") for entry in paths)

    # Equal-length identity changes must affect content hash, not just size.
    sidecar.write_text('"manifest-b"')
    after, _ = core._compute_models_fingerprint()
    assert after != before


def test_equal_manifest_identity_hydrates_across_containers_with_different_timestamps(
    cache_dir, tmp_path, monkeypatch
):
    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    roots = [tmp_path / "producer", tmp_path / "consumer"]
    for root, timestamp in zip(roots, (100.0, 900.0), strict=True):
        for position in core._ALL_POSITIONS:
            outputs = root / "src" / position.lower() / "outputs"
            (outputs / "models").mkdir(parents=True)
            (outputs / "models/weights.pt").write_bytes(position.encode())
            sidecar = outputs / ".manifest-etag"
            sidecar.write_text(f'"{position}-same-manifest"')
            os.utime(sidecar, (timestamp, timestamp))
            sentinel = outputs / ".refreshed_at"
            sentinel.write_text(str(timestamp))
            os.utime(sentinel, (timestamp, timestamp))

    monkeypatch.setattr(core, "_REPO_ROOT", str(roots[0]))
    core.app_pkg._cache.update(results=_fake_results(), metrics_by_format=_fake_metrics())
    _persist_fixture_cache()
    producer_fingerprint = core._compute_models_fingerprint()[0]
    monkeypatch.setattr(core, "_REPO_ROOT", str(roots[1]))
    assert core._compute_models_fingerprint()[0] == producer_fingerprint
    core.app_pkg._cache.clear()
    assert core._try_hydrate_from_disk()

    # A changed manifest in another container rejects the otherwise same bundle.
    (roots[1] / "src/qb/outputs/.manifest-etag").write_text('"QB-new-manifest"')
    core.app_pkg._cache.clear()
    assert core._try_hydrate_from_disk() is False
    assert core._snapshot_path() is None


def test_fingerprint_skips_missing_paths(tmp_path, monkeypatch):
    """A path that disappears between iteration and stat must not raise —
    cache invalidation will trigger naturally because the fingerprint just
    differs.
    """
    import src.serving.app as app_mod

    real = tmp_path / "real.pkl"
    real.write_bytes(b"x")
    ghost = tmp_path / "missing.pkl"

    def _iter():
        yield str(real)
        yield str(ghost)

    monkeypatch.setattr(core, "_iter_fingerprint_paths", _iter)
    sha, files = core._compute_models_fingerprint()
    assert isinstance(sha, str)
    rels = {f["path"] for f in files}
    assert any("real.pkl" in r for r in rels)
    assert not any("missing.pkl" in r for r in rels)


def test_fingerprint_ignores_non_serving_raw_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_REPO_ROOT", str(tmp_path))
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    serving_raw = {
        "depth_charts_v2_2012_2025.parquet",
        "injuries_2012_2025.parquet",
        "kicker_pbp_2015_2024.parquet",
        "kicker_kicks_pbp_2015_2025.parquet",
        "rosters_2012_2025.parquet",
        "schedules_2012_2025.parquet",
        "snap_counts_2012_2025.parquet",
        "team_stats_2012_2025.parquet",
        "weekly_2012_2025.parquet",
    }
    local_only_raw = {
        "contracts_2012_2025.parquet",
        "depth_charts_2012_2025.parquet",
        "ff_opportunity_2012_2025.parquet",
        "player_ids_2012_2025.parquet",
        "qbr_weekly_v2_2012_2025.parquet",
        "redzone_pbp_2012_2025.parquet",
        "weekly_2023_2023.parquet",
        "nflcom_projections_v1_2025_2025_w1-18.parquet",
        "nflcom_projections_joined_v1_2025_2025_mr90.parquet",
        "sleeper_projections_v2_2025_2025_w1-18_DEF-QB-RB-TE-WR.parquet",
        "sleeper_projections_joined_v2_2025_2025.parquet",
    }
    for name in serving_raw | local_only_raw:
        (raw_dir / name).write_bytes(b"x")

    rels = {os.path.relpath(path, tmp_path) for path in core._iter_fingerprint_paths()}

    assert {f"data/raw/{name}" for name in serving_raw}.issubset(rels)
    for name in local_only_raw:
        assert f"data/raw/{name}" not in rels
    assert not any("nflcom_projections" in rel for rel in rels)
    assert not any("sleeper_projections" in rel for rel in rels)


def test_fingerprint_sha_ignores_extra_local_raw_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_REPO_ROOT", str(tmp_path))
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "weekly_2012_2025.parquet").write_bytes(b"weekly")

    sha_before, _ = core._compute_models_fingerprint()

    for name in (
        "contracts_2012_2025.parquet",
        "redzone_pbp_2012_2025.parquet",
        "nflcom_projections_v1_2025_2025_w1-18.parquet",
        "sleeper_projections_v2_2025_2025_w1-18_DEF-QB-RB-TE-WR.parquet",
    ):
        (raw_dir / name).write_bytes(b"local-only")

    sha_after, files = core._compute_models_fingerprint()

    assert sha_after == sha_before
    assert [f["path"] for f in files] == ["data/raw/weekly_2012_2025.parquet"]


# ---------------------------------------------------------------------------
# Persist + hydrate round-trip
# ---------------------------------------------------------------------------


def test_persist_then_hydrate_round_trips_results_and_metrics(
    cache_dir, fingerprint_files, monkeypatch
):
    import src.serving.app as app_mod

    # Avoid touching real S3 — upload helper is best-effort but still issues
    # a print; replacing with a no-op keeps the test output clean.
    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)

    results = _fake_results()
    metrics = _fake_metrics()
    app_mod._cache["results"] = results
    app_mod._cache["metrics_by_format"] = metrics
    app_mod._cache["metrics"] = metrics["ppr"]

    _persist_fixture_cache()

    # All artifacts share the same immutable generation.
    generation = _generation(cache_dir)
    for name in ("predictions.parquet", "metrics.json", "fingerprint.json"):
        assert (generation / name).is_file()
        assert not (cache_dir / name).exists()
    with open(generation / "fingerprint.json") as f:
        fp = json.load(f)
    assert fp["schema_version"] == core._PREDICTIONS_CACHE_SCHEMA_VERSION

    # Clear the in-memory cache and hydrate from disk.
    app_mod._cache.clear()
    app_mod._cache["base_load_error"] = "Shared data initialization failed"
    assert core._try_hydrate_from_disk() is True

    assert "results" in app_mod._cache
    assert "metrics_by_format" in app_mod._cache
    assert app_mod._cache["metrics"] == metrics["ppr"]
    assert app_mod._cache["positions_loaded"] == set(app_mod._ALL_POSITIONS)
    assert app_mod._cache.get("base_loaded") is True
    assert "base_load_error" not in app_mod._cache

    pd.testing.assert_frame_equal(
        app_mod._cache["results"].reset_index(drop=True),
        results.reset_index(drop=True),
    )


def test_espn_outage_cannot_publish_a_reusable_null_cache(
    cache_dir, fingerprint_files, monkeypatch
):
    import src.serving.app as app_mod

    uploads = []
    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: uploads.append(True))
    results = _fake_results()
    results.attrs["espn_complete"] = False
    app_mod._cache["results"] = results
    app_mod._cache["metrics_by_format"] = _fake_metrics()
    _persist_fixture_cache()
    assert current_generation(cache_dir) is None
    assert uploads == []
    assert core._try_hydrate_from_disk() is False

    # Once a retry succeeds, persistence/hydration resume. A later failed
    # refresh must also leave this complete on-disk snapshot untouched.
    results.attrs["espn_complete"] = True
    _persist_fixture_cache()
    before = _generation(cache_dir)
    before_files = read_generation(cache_dir)[1]
    assert uploads == [True]
    results.attrs["espn_complete"] = False
    _persist_fixture_cache()
    assert _generation(cache_dir) == before
    assert read_generation(cache_dir)[1] == before_files
    assert uploads == [True]
    app_mod._cache.clear()
    assert core._try_hydrate_from_disk() is True
    assert app_mod._cache["results"].attrs["espn_complete"] is True


def test_hydrate_returns_false_on_fingerprint_mismatch(cache_dir, fingerprint_files, monkeypatch):
    import src.serving.app as app_mod

    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    app_mod._cache["results"] = _fake_results()
    app_mod._cache["metrics_by_format"] = _fake_metrics()
    _persist_fixture_cache()

    # Mutate a fingerprint input — live fingerprint will diverge from the
    # one written into fingerprint.json.
    Path(fingerprint_files[0]).write_bytes(b"changed-content")

    app_mod._cache.clear()
    assert core._try_hydrate_from_disk() is False
    # Cache stayed empty — no partial state.
    assert "results" not in app_mod._cache
    assert "metrics_by_format" not in app_mod._cache
    assert app_mod.app.test_client().get("/api/snapshot").status_code == 404


def test_persist_requires_pre_inference_fingerprint(cache_dir, fingerprint_files, monkeypatch):
    uploads = []
    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: uploads.append(True))
    core.app_pkg._cache.update(results=_fake_results(), metrics_by_format=_fake_metrics())

    core._persist_cache_to_disk()

    assert current_generation(cache_dir) is None
    assert uploads == []


@pytest.mark.parametrize("change_during_serialization", [False, True])
def test_changed_model_inputs_cannot_relabel_existing_predictions(
    cache_dir, fingerprint_files, monkeypatch, change_during_serialization
):
    uploads = []
    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: uploads.append(True))
    core.app_pkg._cache.update(results=_fake_results(), metrics_by_format=_fake_metrics())
    _persist_fixture_cache()
    previous, previous_files = read_generation(cache_dir)
    assert uploads == [True]

    if change_during_serialization:
        serialize = pd.DataFrame.to_parquet

        def replace_model_after_serialize(frame, *args, **kwargs):
            result = serialize(frame, *args, **kwargs)
            Path(fingerprint_files[0]).write_bytes(b"new-model-during-serialization")
            return result

        monkeypatch.setattr(pd.DataFrame, "to_parquet", replace_model_after_serialize)
    else:
        Path(fingerprint_files[0]).write_bytes(b"new-model-since-inference")

    # The old predictions retain the old pre-inference fingerprint.
    core._persist_cache_to_disk()

    assert read_generation(cache_dir) == (previous, previous_files)
    assert uploads == [True]


def test_hydrate_rejects_model_change_during_parse(cache_dir, fingerprint_files, monkeypatch):
    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    core.app_pkg._cache.update(results=_fake_results(), metrics_by_format=_fake_metrics())
    _persist_fixture_cache()
    core.app_pkg._cache.clear()
    read_parquet = pd.read_parquet

    def model_changes_while_reading(*args, **kwargs):
        results = read_parquet(*args, **kwargs)
        Path(fingerprint_files[0]).write_bytes(b"new-model-during-hydration")
        return results

    monkeypatch.setattr(pd, "read_parquet", model_changes_while_reading)
    assert core._try_hydrate_from_disk() is False
    assert not core.app_pkg._cache


def test_hydrate_returns_false_on_old_cache_schema(cache_dir, fingerprint_files, monkeypatch):
    import src.serving.app as app_mod

    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    app_mod._cache["results"] = _fake_results()
    app_mod._cache["metrics_by_format"] = _fake_metrics()
    _persist_fixture_cache()

    # A well-formed generation with an old application schema must be refused,
    # independently of storage integrity/checksum validation.
    _, files = read_generation(cache_dir)
    fp = json.loads(files["fingerprint.json"])
    fp.pop("schema_version")
    files["fingerprint.json"] = json.dumps(fp).encode()
    old_schema = publish_generation(cache_dir, files)
    assert (old_schema / "snapshot.json").is_file()

    app_mod._cache.clear()
    assert core._try_hydrate_from_disk() is False
    assert "results" not in app_mod._cache
    assert app_mod.app.test_client().get("/api/snapshot").status_code == 404
    # Readers can still finish consuming immutable old generations.
    assert (old_schema / "snapshot.json").is_file()


@pytest.mark.parametrize(
    "drop",
    ["predictions.parquet", "metrics.json", "fingerprint.json"],
)
def test_hydrate_returns_false_when_any_cache_file_missing(
    cache_dir, fingerprint_files, monkeypatch, drop
):
    import src.serving.app as app_mod

    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    app_mod._cache["results"] = _fake_results()
    app_mod._cache["metrics_by_format"] = _fake_metrics()
    _persist_fixture_cache()

    (_generation(cache_dir) / drop).unlink()
    app_mod._cache.clear()
    assert core._try_hydrate_from_disk() is False
    assert app_mod.app.test_client().get("/api/snapshot").status_code == 404


def test_hydrate_returns_false_when_fingerprint_unreadable(
    cache_dir, fingerprint_files, monkeypatch
):
    """A corrupt fingerprint.json must not crash the boot path."""
    import src.serving.app as app_mod

    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    app_mod._cache["results"] = _fake_results()
    app_mod._cache["metrics_by_format"] = _fake_metrics()
    _persist_fixture_cache()

    _, files = read_generation(cache_dir)
    files["fingerprint.json"] = b"not-valid-json"
    publish_generation(cache_dir, files)
    app_mod._cache.clear()
    assert core._try_hydrate_from_disk() is False


@pytest.mark.parametrize(
    "member", ["predictions.parquet", "metrics.json", "fingerprint.json", "snapshot.json"]
)
def test_hydrate_and_snapshot_refuse_changed_immutable_members(
    cache_dir, fingerprint_files, monkeypatch, member
):
    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    core.app_pkg._cache.update(results=_fake_results(), metrics_by_format=_fake_metrics())
    _persist_fixture_cache()
    (_generation(cache_dir) / member).write_bytes(b"tampered after commit")
    core.app_pkg._cache.clear()

    assert core._try_hydrate_from_disk() is False
    assert not core.app_pkg._cache
    assert core.app_pkg.app.test_client().get("/api/snapshot").status_code == 404


def test_hydrate_ignores_valid_legacy_loose_files(cache_dir, fingerprint_files, monkeypatch):
    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    core.app_pkg._cache.update(results=_fake_results(), metrics_by_format=_fake_metrics())
    _persist_fixture_cache()
    _, files = read_generation(cache_dir)
    for name, content in files.items():
        (cache_dir / name).write_bytes(content)
    (cache_dir / "current.json").unlink()
    core.app_pkg._cache.clear()

    assert core._try_hydrate_from_disk() is False
    assert not core.app_pkg._cache
    assert core.app_pkg.app.test_client().get("/api/snapshot").status_code == 404


# ---------------------------------------------------------------------------
# Atomic write under concurrency
# ---------------------------------------------------------------------------


def test_atomic_write_survives_concurrent_persist(cache_dir, fingerprint_files, monkeypatch):
    """Two workers with different predictions commit whole browser/API views."""
    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    worker = threading.local()

    class IsolatedWorker:
        # Gunicorn has a cache per process; thread-local state models that
        # isolation while both writers share the real publication directory.
        @property
        def _cache(self):
            return worker.cache

    monkeypatch.setattr(core, "app_pkg", IsolatedWorker())

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def _writer(value):
        try:
            results = _fake_results()
            results["ridge_pred_ppr"] = float(value)
            metrics = _fake_metrics()
            metrics["ppr"]["Ridge Regression"]["overall"]["mae"] = value
            worker.cache = {"results": results, "metrics_by_format": metrics}
            barrier.wait(timeout=5)
            _persist_fixture_cache()
        except BaseException as e:  # noqa: BLE001 — record + re-raise outside thread
            errors.append(e)

    threads = [threading.Thread(target=_writer, args=(value,)) for value in (11, 99)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive()
    assert not errors, f"writer raised: {errors!r}"

    # Both complete generations survive. Every constituent matches the same
    # writer; checking parseability alone would miss the original mixing bug.
    directories = list((cache_dir / "generations").iterdir())
    assert len(directories) == 2
    values = set()
    for directory in directories:
        predictions = pd.read_parquet(directory / "predictions.parquet")
        value = predictions["ridge_pred_ppr"].iloc[0]
        values.add(value)
        assert predictions["ridge_pred_ppr"].eq(value).all()
        metrics = json.loads((directory / "metrics.json").read_bytes())
        assert metrics["metrics_by_format"]["ppr"]["Ridge Regression"]["overall"]["mae"] == value
        snapshot = json.loads((directory / "snapshot.json").read_bytes())
        assert {row["ridge_pred"] for row in snapshot["scoring"]["ppr"]} == {value}
    assert values == {11, 99}

    selected, files = read_generation(cache_dir)
    assert selected in directories
    assert len(pd.read_parquet(io.BytesIO(files["predictions.parquet"]))) == 4
    assert not list(cache_dir.rglob(".staging-*"))
    assert not list(cache_dir.glob(".current-*"))


# ---------------------------------------------------------------------------
# Gunicorn post_fork hook
# ---------------------------------------------------------------------------


def _load_gunicorn_conf():
    """Load ``gunicorn.conf.py`` by file path (it's not on sys.path as a
    module). Returns the loaded module object.
    """
    conf_path = PROJECT_ROOT / "gunicorn.conf.py"
    spec = importlib.util.spec_from_file_location("gunicorn_conf_test", conf_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_post_fork_starts_daemon_thread_and_returns_quickly(monkeypatch):
    import src.serving.app as app_mod

    gunicorn_conf = _load_gunicorn_conf()

    # Slow stand-in for _ensure_metrics — verifies post_fork doesn't block
    # on it. The thread should still be alive when post_fork returns.
    started = threading.Event()
    release = threading.Event()

    def _slow_warm():
        started.set()
        # Block until the test releases — proves the thread is running
        # async to post_fork.
        release.wait(timeout=5)

    monkeypatch.setattr(core, "_ensure_metrics", _slow_warm)

    fake_worker = mock.MagicMock()
    fake_server = mock.MagicMock()

    t0 = time.monotonic()
    gunicorn_conf.post_fork(fake_server, fake_worker)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0, f"post_fork took {elapsed:.2f}s — should return immediately"

    # The background thread did start.
    assert started.wait(timeout=2), "pre-warm thread never entered _ensure_metrics"

    # Let the warm thread finish so it doesn't linger past the test.
    release.set()


def test_post_fork_swallows_warm_exception(monkeypatch):
    """A failure inside the pre-warm thread must NOT propagate to gunicorn —
    the first user request will retry the load through the normal lazy path.
    """
    import src.serving.app as app_mod

    gunicorn_conf = _load_gunicorn_conf()

    raised = threading.Event()

    def _boom():
        raised.set()
        raise RuntimeError("simulated warm failure")

    monkeypatch.setattr(core, "_ensure_metrics", _boom)

    fake_worker = mock.MagicMock()
    fake_server = mock.MagicMock()

    # Should not raise.
    gunicorn_conf.post_fork(fake_server, fake_worker)

    assert raised.wait(timeout=2)
    # worker.log.warning called with the exception repr.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if fake_worker.log.warning.called:
            break
        time.sleep(0.01)
    assert fake_worker.log.warning.called, "exception should be logged via worker.log.warning"


# ---------------------------------------------------------------------------
# Browser snapshot (static first-paint payload) + /api/snapshot route
# ---------------------------------------------------------------------------


def test_persist_writes_browser_snapshot(cache_dir, fingerprint_files, monkeypatch):
    """``_persist_cache_to_disk`` emits ``snapshot.json`` with all three scoring
    formats, the week list, and rows identical to the ``/api/predictions``
    serializer — so the static snapshot can never drift from the live API.
    """
    import src.serving.app as app_mod

    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    results = _fake_results()
    app_mod._cache["results"] = results
    app_mod._cache["metrics_by_format"] = _fake_metrics()

    _persist_fixture_cache()

    snap_path = _generation(cache_dir) / "snapshot.json"
    assert snap_path.is_file()
    snap = json.loads(snap_path.read_text())
    assert set(snap["scoring"]) == {"ppr", "half_ppr", "standard"}
    assert snap["weeks"] == sorted(int(w) for w in results["week"].unique())
    assert snap["degraded_positions"] == []
    # Rows are exactly what /api/predictions would serialize for each format.
    for fmt in ("ppr", "half_ppr", "standard"):
        assert snap["scoring"][fmt] == app_mod._records_to_player_rows(results, scoring=fmt)
        assert {"nflcom_pred", "rotowire_pred"}.issubset(snap["scoring"][fmt][0])


def test_hydrate_regenerates_snapshot_when_absent(cache_dir, fingerprint_files, monkeypatch):
    """A valid generation whose manifest omits the optional snapshot regenerates
    it locally from the selected generation so
    ``/api/snapshot`` serves without waiting for the next retrain.
    """
    import src.serving.app as app_mod

    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    app_mod._cache["results"] = _fake_results()
    app_mod._cache["metrics_by_format"] = _fake_metrics()
    _persist_fixture_cache()

    # Commit the supported snapshot-absent shape. Deleting a member listed in
    # an immutable manifest instead constitutes corruption and must be refused.
    _, files = read_generation(cache_dir)
    files.pop("snapshot.json")
    without_snapshot = publish_generation(cache_dir, files)
    assert not (without_snapshot / "snapshot.json").exists()

    app_mod._cache.clear()
    assert core._try_hydrate_from_disk() is True
    regenerated = _generation(cache_dir)
    assert regenerated != without_snapshot
    assert (regenerated / "snapshot.json").is_file()
    assert not (without_snapshot / "snapshot.json").exists()
    assert app_mod.app.test_client().get("/api/snapshot").status_code == 200


def test_snapshot_route_serves_file_without_triggering_compute(
    cache_dir, fingerprint_files, monkeypatch
):
    """``/api/snapshot`` serves straight off disk and MUST NOT call the heavy
    ``_ensure_metrics`` model-load path — that decoupling is the whole point.
    """
    import src.serving.app as app_mod

    def _boom():
        raise AssertionError("/api/snapshot must not call _ensure_metrics")

    monkeypatch.setattr(core, "_ensure_metrics", _boom)
    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    app_mod._cache["results"] = _fake_results()
    app_mod._cache["metrics_by_format"] = _fake_metrics()
    _persist_fixture_cache()

    payload = json.loads((_generation(cache_dir) / "snapshot.json").read_bytes())

    resp = app_mod.app.test_client().get("/api/snapshot")
    assert resp.status_code == 200
    assert resp.get_json() == payload
    assert resp.headers["Cache-Control"] == "no-cache"


def test_snapshot_route_404_when_absent(cache_dir, monkeypatch):
    """No snapshot on disk -> 404 (frontend falls back to /api/predictions),
    still without invoking the compute path.
    """
    import src.serving.app as app_mod

    def _boom():
        raise AssertionError("/api/snapshot must not call _ensure_metrics")

    monkeypatch.setattr(core, "_ensure_metrics", _boom)

    resp = app_mod.app.test_client().get("/api/snapshot")
    assert resp.status_code == 404
