"""Tests for src/batch/train.py's smoke-test wiring into upload_artifacts.

The smoke test itself is unit-tested in tests/shared/test_smoke_test.py.
These tests focus on the *integration* between upload_artifacts and the
manifest's ``stable`` slot:

- A passing smoke test promotes the new entry into ``stable``.
- A failing smoke test pins ``stable`` to the previous good pointer (or
  leaves it null on first upload). Eligible current/history candidates remain
  available for forensic inspection in the protected namespace.
- Smoke failure does not claim an accepted output receipt and preserves stable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.shared.smoke_test import SmokeTestFailed
from tests.batch.test_train import _FakeS3Producer

pytestmark = pytest.mark.unit


@pytest.fixture
def publication_run(monkeypatch, tmp_path):
    """Prepare real source/plan/intent bindings; only smoke prediction is mocked."""
    from tests.batch.test_train import _publication_metrics
    from tests.batch.test_train import _write_fake_model_dir as write_model_directory

    sequence = 0

    def prepare(store, directory, position="RB", **extra_metrics):
        nonlocal sequence
        sequence += 1
        metrics = _publication_metrics(
            store,
            monkeypatch,
            tmp_path / "images",
            position=position,
            bucket="my-bucket",
            run_id=f"smoke-{sequence}",
        )
        write_model_directory(directory, position, metrics={**metrics, **extra_metrics})
        return metrics

    return prepare


@mock.patch("src.batch.train.boto3.client")
@mock.patch("src.batch.train.run_smoke_test")
def test_smoke_pass_advances_stable_on_first_upload(
    mock_smoke, mock_boto, tmp_path, capsys, publication_run
):
    """Happy path: first-ever upload, smoke test passes → manifest.stable
    points at the new key (== current), making the output eligible for serving."""
    from src.batch.train import upload_artifacts

    fake_s3 = _FakeS3Producer()
    mock_boto.return_value = fake_s3
    mock_smoke.return_value = None  # success — no exception

    d = tmp_path / "model"
    d.mkdir()
    publication_run(fake_s3, d)

    upload_artifacts("my-bucket", "RB", str(d))

    manifest = json.loads(fake_s3.objects["models/releases/v3/RB/manifest.json"])
    assert manifest["schema_version"] == 3
    assert manifest["stable"] is not None
    assert manifest["stable"]["key"] == manifest["current"]["key"]
    assert "[smoke_test] RB: PASS" in capsys.readouterr().out


@mock.patch("src.batch.train.boto3.client")
@mock.patch("src.batch.train.run_smoke_test")
def test_smoke_fail_pins_stable_to_old_value(
    mock_smoke, mock_boto, tmp_path, capsys, publication_run
):
    """A new (broken) artifact uploads after a previously-stable one. The
    new artifact is structurally valid (passes _validate_remote_tarball)
    but smoke-fails → ``stable`` must NOT advance. ``current`` and
    ``previous`` advance as usual; the broken artifact is in S3 for triage."""
    from src.batch.train import upload_artifacts

    fake_s3 = _FakeS3Producer()
    mock_boto.return_value = fake_s3

    d = tmp_path / "model"
    d.mkdir()
    publication_run(fake_s3, d)

    # Upload #1 — smoke passes, stable advances.
    mock_smoke.side_effect = [None]
    upload_artifacts("my-bucket", "RB", str(d))
    first_manifest = json.loads(fake_s3.objects["models/releases/v3/RB/manifest.json"])
    first_stable_key = first_manifest["stable"]["key"]
    first_current_key = first_manifest["current"]["key"]
    assert first_stable_key == first_current_key

    # Upload #2 — smoke fails. Different bytes so the new history key differs.
    publication_run(fake_s3, d, r=2)
    mock_smoke.side_effect = SmokeTestFailed("simulated NaN prediction")
    upload_artifacts("my-bucket", "RB", str(d))

    second_manifest = json.loads(fake_s3.objects["models/releases/v3/RB/manifest.json"])
    # Stable is pinned to upload #1.
    assert second_manifest["stable"]["key"] == first_stable_key
    # Current advanced to the broken upload.
    assert second_manifest["current"]["key"] != first_current_key
    # Previous demoted from upload #1's current.
    assert second_manifest["previous"]["key"] == first_current_key
    # Both bytes still in S3 — the consumer can fall back if stable somehow
    # disappears.
    assert first_stable_key in fake_s3.objects
    assert second_manifest["current"]["key"] in fake_s3.objects

    out = capsys.readouterr().out
    assert "[smoke_test] RB: FAIL" in out
    assert "stable pointer NOT advanced" in out


@mock.patch("src.batch.train.boto3.client")
@mock.patch("src.batch.train.run_smoke_test")
def test_smoke_fail_first_run_leaves_stable_null(mock_smoke, mock_boto, tmp_path, publication_run):
    """First-ever upload AND smoke fails → stable stays null. The consumer
    fails closed until a successful smoke test establishes an approved stable."""
    from src.batch.train import upload_artifacts

    fake_s3 = _FakeS3Producer()
    mock_boto.return_value = fake_s3
    mock_smoke.side_effect = SmokeTestFailed("simulated load error")

    d = tmp_path / "model"
    d.mkdir()
    publication_run(fake_s3, d)

    upload_artifacts("my-bucket", "RB", str(d))

    manifest = json.loads(fake_s3.objects["models/releases/v3/RB/manifest.json"])
    assert manifest["stable"] is None
    assert manifest["current"] is not None  # still recorded for forensics


@mock.patch("src.batch.train.boto3.client")
@mock.patch("src.batch.train.run_smoke_test")
def test_smoke_fail_does_not_block_upload_completion(
    mock_smoke, mock_boto, tmp_path, publication_run
):
    """A smoke-test failure must NOT abort upload_artifacts — the artifact
    still lands in history/ and the manifest still gets written. We only
    gate the ``stable`` pointer advance."""
    from src.batch.train import upload_artifacts

    fake_s3 = _FakeS3Producer()
    mock_boto.return_value = fake_s3
    mock_smoke.side_effect = SmokeTestFailed("simulated")

    d = tmp_path / "model"
    d.mkdir()
    publication_run(fake_s3, d)

    # No exception expected.
    upload_artifacts("my-bucket", "RB", str(d))

    # Both producer writes still happened: history key + manifest. The legacy
    # mirror is no longer written (Layer C of the race fix).
    history_keys = [k for k in fake_s3.objects if k.startswith("models/releases/v3/RB/history/")]
    assert len(history_keys) == 1
    assert "models/releases/v3/RB/manifest.json" in fake_s3.objects
    assert "models/RB/model.tar.gz" not in fake_s3.objects


@mock.patch("src.batch.train.boto3.client")
@mock.patch("src.batch.train.run_smoke_test")
def test_unexpected_smoke_exception_is_treated_as_failure(
    mock_smoke, mock_boto, tmp_path, capsys, publication_run
):
    """If run_smoke_test raises a non-SmokeTestFailed exception (e.g.
    ImportError, OOM), the producer treats it as a smoke-test failure and
    pins stable. Better to be conservative than to promote an artifact we
    couldn't validate."""
    from src.batch.train import upload_artifacts

    fake_s3 = _FakeS3Producer()
    mock_boto.return_value = fake_s3
    mock_smoke.side_effect = ImportError("torch not installed in some weird env")

    d = tmp_path / "model"
    d.mkdir()
    publication_run(fake_s3, d)

    upload_artifacts("my-bucket", "RB", str(d))

    manifest = json.loads(fake_s3.objects["models/releases/v3/RB/manifest.json"])
    assert manifest["stable"] is None
    out = capsys.readouterr().out
    assert "UNEXPECTED" in out
    assert "stable pointer NOT advanced" in out


@mock.patch("src.batch.train.boto3.client")
@mock.patch("src.batch.train.run_smoke_test")
def test_smoke_pass_after_prior_failure_recovers_stable(
    mock_smoke, mock_boto, tmp_path, publication_run
):
    """A retrain that finally passes smoke after a stretch of failures
    advances stable to the new key, healing the pin. This is the recovery
    path: operator pushes a fix → next train passes smoke → frontend serves
    fresh predictions again on the next ECS task restart."""
    from src.batch.train import upload_artifacts

    fake_s3 = _FakeS3Producer()
    mock_boto.return_value = fake_s3

    d = tmp_path / "model"
    d.mkdir()
    publication_run(fake_s3, d)

    # #1 passes — stable=v1.
    mock_smoke.side_effect = [None]
    upload_artifacts("my-bucket", "RB", str(d))
    v1_stable = json.loads(fake_s3.objects["models/releases/v3/RB/manifest.json"])["stable"]["key"]

    # #2 fails — stable still v1.
    publication_run(fake_s3, d, r=2)
    mock_smoke.side_effect = SmokeTestFailed("broken")
    upload_artifacts("my-bucket", "RB", str(d))
    v2 = json.loads(fake_s3.objects["models/releases/v3/RB/manifest.json"])
    assert v2["stable"]["key"] == v1_stable

    # #3 passes — stable advances to #3's key.
    publication_run(fake_s3, d, r=3)
    mock_smoke.side_effect = [None]
    upload_artifacts("my-bucket", "RB", str(d))
    v3 = json.loads(fake_s3.objects["models/releases/v3/RB/manifest.json"])
    assert v3["stable"]["key"] == v3["current"]["key"]
    assert v3["stable"]["key"] != v1_stable
