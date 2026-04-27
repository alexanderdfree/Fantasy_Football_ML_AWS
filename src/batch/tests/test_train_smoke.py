"""Tests for src/batch/train.py's smoke-test wiring into upload_artifacts.

The smoke test itself is unit-tested in src/shared/tests/test_smoke_test.py.
These tests focus on the *integration* between upload_artifacts and the
manifest's ``stable`` slot:

- A passing smoke test promotes the new entry into ``stable``.
- A failing smoke test pins ``stable`` to the previous good pointer (or
  leaves it null on first upload), but ``current`` and ``history`` advance
  unconditionally so the artifact is still auditable in S3.
- Smoke test failure is non-fatal — upload still completes (manifest, legacy
  mirror, GC).
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path
from unittest import mock

import pytest
from botocore.exceptions import ClientError

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.shared.smoke_test import SmokeTestFailed


def _nosuchkey_error(key: str) -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": "NoSuchKey", "Message": f"{key} not found"}},
        operation_name="GetObject",
    )


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakePaginator:
    def __init__(self, objects: dict):
        self._objects = objects

    def paginate(self, Bucket: str, Prefix: str):  # noqa: N803
        contents = [{"Key": k} for k in self._objects if k.startswith(Prefix)]
        yield {"Contents": contents}


class _FakeS3Producer:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.ops: list[tuple[str, str]] = []

    def upload_file(self, local_path, Bucket, Key):  # noqa: N803
        with open(local_path, "rb") as f:
            self.objects[Key] = f.read()
        self.ops.append(("upload_file", Key))

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        if hasattr(Body, "read"):
            Body = Body.read()
        self.objects[Key] = Body
        self.ops.append(("put_object", Key))

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise _nosuchkey_error(Key)
        return {"Body": _FakeBody(self.objects[Key])}

    def get_paginator(self, op):
        assert op == "list_objects_v2"
        return _FakePaginator(self.objects)

    def delete_objects(self, Bucket, Delete):  # noqa: N803
        for obj in Delete["Objects"]:
            self.objects.pop(obj["Key"], None)
            self.ops.append(("delete", obj["Key"]))


def _write_fake_model_dir(d: Path, pos: str) -> None:
    """Same shape as the helper in test_train.py — fake bytes for every
    file the inference registry expects, plus benchmark_metrics.json. The
    bytes deliberately can't be deserialized; tests mock run_smoke_test
    directly to control the smoke-test outcome independently of the bytes.
    """
    from src.shared.registry import INFERENCE_REGISTRY

    reg = INFERENCE_REGISTRY[pos]
    files = {
        reg["nn_file"]: b"fake-nn-weights",
        "nn_scaler.pkl": b"fake-scaler",
        "nn_scaler_meta.json": b"{}",
        "benchmark_metrics.json": b'{"position":"' + pos.encode() + b'"}',
    }
    if reg.get("train_attention_nn") and reg.get("attn_nn_file"):
        files[reg["attn_nn_file"]] = b"fake-attn-weights"
        files["attention_nn_scaler.pkl"] = b"fake-attn-scaler"
        files["attention_nn_scaler_meta.json"] = b"{}"
    files["ridge_model.pkl"] = b"fake-ridge"
    for name, data in files.items():
        (d / name).write_bytes(data)


@mock.patch("src.batch.train.boto3.client")
@mock.patch("src.batch.train.run_smoke_test")
def test_smoke_pass_advances_stable_on_first_upload(mock_smoke, mock_boto, tmp_path, capsys):
    """Happy path: first-ever upload, smoke test passes → manifest.stable
    points at the new key (== current). This is the migration-window-end
    state: once any smoke test passes, the consumer can switch from
    falling-through to current to actually serving stable."""
    from src.batch.train import upload_artifacts

    fake_s3 = _FakeS3Producer()
    mock_boto.return_value = fake_s3
    mock_smoke.return_value = None  # success — no exception

    d = tmp_path / "model"
    d.mkdir()
    _write_fake_model_dir(d, "RB")

    upload_artifacts("my-bucket", "RB", str(d))

    manifest = json.loads(fake_s3.objects["models/RB/manifest.json"])
    assert manifest["schema_version"] == 2
    assert manifest["stable"] is not None
    assert manifest["stable"]["key"] == manifest["current"]["key"]
    assert "[smoke_test] RB: PASS" in capsys.readouterr().out


@mock.patch("src.batch.train.boto3.client")
@mock.patch("src.batch.train.run_smoke_test")
def test_smoke_fail_pins_stable_to_old_value(mock_smoke, mock_boto, tmp_path, capsys):
    """A new (broken) artifact uploads after a previously-stable one. The
    new artifact is structurally valid (passes _validate_remote_tarball)
    but smoke-fails → ``stable`` must NOT advance. ``current`` and
    ``previous`` advance as usual; the broken artifact is in S3 for triage."""
    from src.batch.train import upload_artifacts

    fake_s3 = _FakeS3Producer()
    mock_boto.return_value = fake_s3

    d = tmp_path / "model"
    d.mkdir()
    _write_fake_model_dir(d, "RB")

    # Upload #1 — smoke passes, stable advances.
    mock_smoke.side_effect = [None]
    upload_artifacts("my-bucket", "RB", str(d))
    first_manifest = json.loads(fake_s3.objects["models/RB/manifest.json"])
    first_stable_key = first_manifest["stable"]["key"]
    first_current_key = first_manifest["current"]["key"]
    assert first_stable_key == first_current_key

    # Upload #2 — smoke fails. Different bytes so the new history key differs.
    (d / "benchmark_metrics.json").write_bytes(b'{"position":"RB","r":2}')
    mock_smoke.side_effect = SmokeTestFailed("simulated NaN prediction")
    upload_artifacts("my-bucket", "RB", str(d))

    second_manifest = json.loads(fake_s3.objects["models/RB/manifest.json"])
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
def test_smoke_fail_first_run_leaves_stable_null(mock_smoke, mock_boto, tmp_path):
    """First-ever upload AND smoke fails → stable stays null. The consumer
    falls through to current via the v1-compat path until the next
    successful smoke test populates stable."""
    from src.batch.train import upload_artifacts

    fake_s3 = _FakeS3Producer()
    mock_boto.return_value = fake_s3
    mock_smoke.side_effect = SmokeTestFailed("simulated load error")

    d = tmp_path / "model"
    d.mkdir()
    _write_fake_model_dir(d, "RB")

    upload_artifacts("my-bucket", "RB", str(d))

    manifest = json.loads(fake_s3.objects["models/RB/manifest.json"])
    assert manifest["stable"] is None
    assert manifest["current"] is not None  # still recorded for forensics


@mock.patch("src.batch.train.boto3.client")
@mock.patch("src.batch.train.run_smoke_test")
def test_smoke_fail_does_not_block_upload_completion(mock_smoke, mock_boto, tmp_path):
    """A smoke-test failure must NOT abort upload_artifacts — the artifact
    still lands in history/ and the legacy mirror still updates. We only
    gate the ``stable`` pointer advance."""
    from src.batch.train import upload_artifacts

    fake_s3 = _FakeS3Producer()
    mock_boto.return_value = fake_s3
    mock_smoke.side_effect = SmokeTestFailed("simulated")

    d = tmp_path / "model"
    d.mkdir()
    _write_fake_model_dir(d, "RB")

    # No exception expected.
    upload_artifacts("my-bucket", "RB", str(d))

    # All three writes still happened: history key, manifest, legacy mirror.
    history_keys = [k for k in fake_s3.objects if k.startswith("models/RB/history/")]
    assert len(history_keys) == 1
    assert "models/RB/manifest.json" in fake_s3.objects
    assert "models/RB/model.tar.gz" in fake_s3.objects


@mock.patch("src.batch.train.boto3.client")
@mock.patch("src.batch.train.run_smoke_test")
def test_unexpected_smoke_exception_is_treated_as_failure(mock_smoke, mock_boto, tmp_path, capsys):
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
    _write_fake_model_dir(d, "RB")

    upload_artifacts("my-bucket", "RB", str(d))

    manifest = json.loads(fake_s3.objects["models/RB/manifest.json"])
    assert manifest["stable"] is None
    out = capsys.readouterr().out
    assert "UNEXPECTED" in out
    assert "stable pointer NOT advanced" in out


@mock.patch("src.batch.train.boto3.client")
@mock.patch("src.batch.train.run_smoke_test")
def test_smoke_pass_after_prior_failure_recovers_stable(mock_smoke, mock_boto, tmp_path):
    """A retrain that finally passes smoke after a stretch of failures
    advances stable to the new key, healing the pin. This is the recovery
    path: operator pushes a fix → next train passes smoke → frontend serves
    fresh predictions again on the next ECS task restart."""
    from src.batch.train import upload_artifacts

    fake_s3 = _FakeS3Producer()
    mock_boto.return_value = fake_s3

    d = tmp_path / "model"
    d.mkdir()
    _write_fake_model_dir(d, "RB")

    # #1 passes — stable=v1.
    mock_smoke.side_effect = [None]
    upload_artifacts("my-bucket", "RB", str(d))
    v1_stable = json.loads(fake_s3.objects["models/RB/manifest.json"])["stable"]["key"]

    # #2 fails — stable still v1.
    (d / "benchmark_metrics.json").write_bytes(b'{"r":2}')
    mock_smoke.side_effect = SmokeTestFailed("broken")
    upload_artifacts("my-bucket", "RB", str(d))
    v2 = json.loads(fake_s3.objects["models/RB/manifest.json"])
    assert v2["stable"]["key"] == v1_stable

    # #3 passes — stable advances to #3's key.
    (d / "benchmark_metrics.json").write_bytes(b'{"r":3}')
    mock_smoke.side_effect = [None]
    upload_artifacts("my-bucket", "RB", str(d))
    v3 = json.loads(fake_s3.objects["models/RB/manifest.json"])
    assert v3["stable"]["key"] == v3["current"]["key"]
    assert v3["stable"]["key"] != v1_stable
