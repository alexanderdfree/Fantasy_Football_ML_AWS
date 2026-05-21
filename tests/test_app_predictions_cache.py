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
- Atomic write: two concurrent ``_persist_cache_to_disk`` calls leave
  the destination triple in a consistent, parseable state.

The fingerprint test files live under ``tmp_path``; ``_iter_fingerprint_paths``
is monkeypatched to yield them, which avoids needing a real
``src/{pos}/outputs/models/`` tree on the test host. ``_PREDICTIONS_CACHE_DIR``
is also redirected to ``tmp_path`` so the production cache dir is never
touched.
"""

from __future__ import annotations

import importlib.util
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


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Redirect the module-level ``_PREDICTIONS_CACHE_DIR`` to a tmp dir so
    persist/hydrate don't write to ``<repo>/data/serving_cache/``.
    """
    import src.serving.app as app_mod

    target = tmp_path / "serving_cache"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_mod, "_PREDICTIONS_CACHE_DIR", str(target))
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

    monkeypatch.setattr(app_mod, "_iter_fingerprint_paths", _iter)
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

    sha1, files1 = app_mod._compute_models_fingerprint()
    assert isinstance(sha1, str) and len(sha1) == 64
    assert len(files1) == 3

    # Bump mtime on one file (size + content unchanged).
    target = fingerprint_files[1]
    new_mtime = os.stat(target).st_mtime + 10.0
    os.utime(target, (new_mtime, new_mtime))

    sha2, _ = app_mod._compute_models_fingerprint()
    assert sha2 == sha1, "fingerprint must NOT change when only mtime changes (content-hash bug)"


def test_fingerprint_changes_on_content_change(fingerprint_files):
    """Modifying the head bytes of a fingerprint input MUST change the
    fingerprint — the content-hash sampling reads the first 64 KB, and every
    file in the fingerprint set is much shorter than that, so any byte
    change in the file is fully visible to the hash.
    """
    import src.serving.app as app_mod

    sha1, _ = app_mod._compute_models_fingerprint()
    # Mutate content (same length so size doesn't carry the signal).
    target = Path(fingerprint_files[1])
    original = target.read_bytes()
    target.write_bytes(b"X" * len(original))

    sha2, _ = app_mod._compute_models_fingerprint()
    assert sha2 != sha1, "fingerprint must change when file content changes"


def test_fingerprint_changes_on_size_change(fingerprint_files):
    import src.serving.app as app_mod

    sha1, _ = app_mod._compute_models_fingerprint()
    # Rewrite one file with different content (different size).
    Path(fingerprint_files[0]).write_bytes(b"alpha-extended")
    sha2, _ = app_mod._compute_models_fingerprint()
    assert sha2 != sha1


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

    monkeypatch.setattr(app_mod, "_iter_fingerprint_paths", _iter)
    sha, files = app_mod._compute_models_fingerprint()
    assert isinstance(sha, str)
    rels = {f["path"] for f in files}
    assert any("real.pkl" in r for r in rels)
    assert not any("missing.pkl" in r for r in rels)


# ---------------------------------------------------------------------------
# Persist + hydrate round-trip
# ---------------------------------------------------------------------------


def test_persist_then_hydrate_round_trips_results_and_metrics(
    cache_dir, fingerprint_files, monkeypatch
):
    import src.serving.app as app_mod

    # Avoid touching real S3 — upload helper is best-effort but still issues
    # a print; replacing with a no-op keeps the test output clean.
    monkeypatch.setattr(app_mod, "upload_predictions_cache_to_s3", lambda: None)

    results = _fake_results()
    metrics = _fake_metrics()
    app_mod._cache["results"] = results
    app_mod._cache["metrics_by_format"] = metrics
    app_mod._cache["metrics"] = metrics["ppr"]

    app_mod._persist_cache_to_disk()

    # All three artifacts present after persist.
    for name in ("predictions.parquet", "metrics.json", "fingerprint.json"):
        assert (cache_dir / name).is_file()

    # Clear the in-memory cache and hydrate from disk.
    app_mod._cache.clear()
    assert app_mod._try_hydrate_from_disk() is True

    assert "results" in app_mod._cache
    assert "metrics_by_format" in app_mod._cache
    assert app_mod._cache["metrics"] == metrics["ppr"]
    assert app_mod._cache["positions_loaded"] == set(app_mod._ALL_POSITIONS)
    assert app_mod._cache.get("base_loaded") is True

    pd.testing.assert_frame_equal(
        app_mod._cache["results"].reset_index(drop=True),
        results.reset_index(drop=True),
    )


def test_hydrate_returns_false_on_fingerprint_mismatch(cache_dir, fingerprint_files, monkeypatch):
    import src.serving.app as app_mod

    monkeypatch.setattr(app_mod, "upload_predictions_cache_to_s3", lambda: None)
    app_mod._cache["results"] = _fake_results()
    app_mod._cache["metrics_by_format"] = _fake_metrics()
    app_mod._persist_cache_to_disk()

    # Mutate a fingerprint input — live fingerprint will diverge from the
    # one written into fingerprint.json.
    Path(fingerprint_files[0]).write_bytes(b"changed-content")

    app_mod._cache.clear()
    assert app_mod._try_hydrate_from_disk() is False
    # Cache stayed empty — no partial state.
    assert "results" not in app_mod._cache
    assert "metrics_by_format" not in app_mod._cache


@pytest.mark.parametrize(
    "drop",
    ["predictions.parquet", "metrics.json", "fingerprint.json"],
)
def test_hydrate_returns_false_when_any_cache_file_missing(
    cache_dir, fingerprint_files, monkeypatch, drop
):
    import src.serving.app as app_mod

    monkeypatch.setattr(app_mod, "upload_predictions_cache_to_s3", lambda: None)
    app_mod._cache["results"] = _fake_results()
    app_mod._cache["metrics_by_format"] = _fake_metrics()
    app_mod._persist_cache_to_disk()

    (cache_dir / drop).unlink()
    app_mod._cache.clear()
    assert app_mod._try_hydrate_from_disk() is False


def test_hydrate_returns_false_when_fingerprint_unreadable(
    cache_dir, fingerprint_files, monkeypatch
):
    """A corrupt fingerprint.json must not crash the boot path."""
    import src.serving.app as app_mod

    monkeypatch.setattr(app_mod, "upload_predictions_cache_to_s3", lambda: None)
    app_mod._cache["results"] = _fake_results()
    app_mod._cache["metrics_by_format"] = _fake_metrics()
    app_mod._persist_cache_to_disk()

    (cache_dir / "fingerprint.json").write_text("not-valid-json")
    app_mod._cache.clear()
    assert app_mod._try_hydrate_from_disk() is False


# ---------------------------------------------------------------------------
# Atomic write under concurrency
# ---------------------------------------------------------------------------


def test_atomic_write_survives_concurrent_persist(cache_dir, fingerprint_files, monkeypatch):
    """Two threads calling ``_persist_cache_to_disk`` must leave the
    destination triple in a consistent, parseable state (no zero-length or
    half-written files).
    """
    import src.serving.app as app_mod

    monkeypatch.setattr(app_mod, "upload_predictions_cache_to_s3", lambda: None)

    app_mod._cache["results"] = _fake_results()
    app_mod._cache["metrics_by_format"] = _fake_metrics()

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def _writer():
        try:
            barrier.wait(timeout=5)
            app_mod._persist_cache_to_disk()
        except BaseException as e:  # noqa: BLE001 — record + re-raise outside thread
            errors.append(e)

    threads = [threading.Thread(target=_writer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert not errors, f"writer raised: {errors!r}"

    # All three files parseable.
    df = pd.read_parquet(cache_dir / "predictions.parquet")
    assert len(df) == 4
    with open(cache_dir / "metrics.json") as f:
        json.load(f)
    with open(cache_dir / "fingerprint.json") as f:
        fp = json.load(f)
    assert isinstance(fp.get("sha256"), str)

    # No leftover temp files (each writer cleans up its own on success or
    # failure, and atomic replace consumes the temp).
    leftovers = [p.name for p in cache_dir.iterdir() if p.name.endswith(".tmp")]
    assert not leftovers, f"unexpected tmp leftovers: {leftovers}"


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

    monkeypatch.setattr(app_mod, "_ensure_metrics", _slow_warm)

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

    monkeypatch.setattr(app_mod, "_ensure_metrics", _boom)

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
