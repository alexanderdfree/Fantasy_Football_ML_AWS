"""Tests for shared.model_sync — S3 tarball sync at container boot."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared import model_sync


def _make_tarball(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeS3:
    """Returns per-position tarballs keyed by the exact S3 key."""

    def __init__(self, objects: dict[str, bytes]):
        self._objects = objects
        self.calls: list[tuple[str, str]] = []

    def get_object(self, Bucket: str, Key: str):  # noqa: N803 (boto3 convention)
        self.calls.append((Bucket, Key))
        if Key not in self._objects:
            raise KeyError(f"NoSuchKey: {Key}")
        return {"Body": _FakeBody(self._objects[Key])}


def test_sync_noop_when_bucket_unset(monkeypatch, capsys):
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    result = model_sync.sync_models_from_s3()
    assert result is None
    assert "unset" in capsys.readouterr().out


def test_sync_noop_when_bucket_blank(monkeypatch):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "   ")
    assert model_sync.sync_models_from_s3() is None


def test_sync_extracts_all_positions(monkeypatch, tmp_path):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    objects = {
        f"models/{pos}/model.tar.gz": _make_tarball(
            {f"{pos.lower()}_marker.pkl": b"payload-" + pos.encode()}
        )
        for pos in model_sync.POSITIONS
    }
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_models_from_s3()

    assert summary is not None
    assert {r["pos"] for r in summary["positions"]} == set(model_sync.POSITIONS)
    for pos in model_sync.POSITIONS:
        extracted = tmp_path / pos / "outputs" / "models" / f"{pos.lower()}_marker.pkl"
        assert extracted.is_file()
        assert extracted.read_bytes() == b"payload-" + pos.encode()


def test_sync_honors_custom_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "nightly/v2")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    objects = {
        f"nightly/v2/{pos}/model.tar.gz": _make_tarball({"file.pkl": b"x"})
        for pos in model_sync.POSITIONS
    }
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        model_sync.sync_models_from_s3()

    keys_called = {key for _, key in fake_s3.calls}
    assert keys_called == set(objects.keys())


def test_sync_raises_on_missing_key(monkeypatch, tmp_path):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    fake_s3 = _FakeS3(objects={})
    with mock.patch("boto3.client", return_value=fake_s3):
        with pytest.raises(KeyError, match="NoSuchKey"):
            model_sync.sync_models_from_s3()


def test_extract_rejects_path_traversal(tmp_path):
    malicious = _make_tarball({"../../../etc/evil.pkl": b"pwn"})
    with pytest.raises(RuntimeError, match="escape"):
        model_sync._extract_tarball(malicious, tmp_path / "dest")


def test_extract_allows_nested_subdirs(tmp_path):
    data = _make_tarball(
        {
            "nn_scaler.pkl": b"a",
            "lightgbm/receiving_yards.pkl": b"b",
        }
    )
    dest = tmp_path / "dest"
    model_sync._extract_tarball(data, dest)
    assert (dest / "nn_scaler.pkl").read_bytes() == b"a"
    assert (dest / "lightgbm" / "receiving_yards.pkl").read_bytes() == b"b"
