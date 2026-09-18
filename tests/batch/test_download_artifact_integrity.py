"""Post-training downloads replace model generations without retaining sidecars."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from src.batch.launch import download_artifacts
from src.shared.models import RidgeModel

pytestmark = pytest.mark.unit


class LocalS3:
    def __init__(self, objects, *, current="new", stable=None):
        self.objects = objects
        self.current = current
        self.stable = stable

    def get_object(self, **kwargs):
        manifest = {
            "schema_version": 2,
            "current": {"key": self.current},
            "stable": {"key": self.stable} if self.stable else None,
        }
        return {"Body": io.BytesIO(json.dumps(manifest).encode())}

    def head_object(self, **kwargs):
        return {}

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[key])


def _archive(files, *, escape=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        if escape:
            member = tarfile.TarInfo("../outside-models")
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
    return stream.getvalue()


def test_download_non_pca_generation_loads_without_previous_pca(tmp_path, monkeypatch):
    inputs = np.array([[1.0, 2.0], [2.0, 5.0], [4.0, 1.0], [8.0, 7.0]])
    targets = np.array([1.0, 3.0, 4.0, 8.0])
    objects = {}
    expected = None
    for name, components in (("old", 1), ("new", None)):
        source = tmp_path / name / "passing_yards"
        model = RidgeModel(pca_n_components=components)
        model.fit(inputs, targets)
        model.save(str(source))
        objects[name] = _archive(
            {f"passing_yards/{file.name}": file.read_bytes() for file in source.iterdir()}
        )
        if name == "new":
            fresh = RidgeModel()
            fresh.load(str(source))
            expected = fresh.predict(inputs)

    s3 = LocalS3(objects, current="old")
    monkeypatch.chdir(tmp_path)
    download_artifacts(["QB"], s3_client=s3)
    destination = tmp_path / "qb/outputs/models/passing_yards"
    assert (destination / "pca.pkl").exists()

    s3.current = "new"
    download_artifacts(["QB"], s3_client=s3)
    loaded = RidgeModel()
    loaded.load(str(destination))
    np.testing.assert_allclose(loaded.predict(inputs), expected)
    assert not (destination / "pca.pkl").exists()


@pytest.mark.parametrize("invalid", ["not_a_tar", "escape_after_file"])
def test_invalid_download_preserves_existing_artifacts(tmp_path, monkeypatch, invalid):
    destination = tmp_path / "qb/outputs/models"
    destination.mkdir(parents=True)
    (destination / "kept.pkl").write_bytes(b"original")
    payload = (
        b"not a tarball"
        if invalid == "not_a_tar"
        else _archive({"kept.pkl": b"partial replacement"}, escape=True)
    )
    s3 = LocalS3({"new": payload})
    monkeypatch.chdir(tmp_path)

    download_artifacts(["QB"], s3_client=s3)

    assert (destination / "kept.pkl").read_bytes() == b"original"
    assert not (destination.parent / "outside-models").exists()


def test_invalid_candidate_cannot_contaminate_fallback(tmp_path, monkeypatch):
    s3 = LocalS3(
        {
            "bad": _archive({"poison.pkl": b"rejected candidate"}, escape=True),
            "new": _archive({"model.pkl": b"complete fallback"}),
        },
        stable="bad",
    )
    monkeypatch.chdir(tmp_path)

    download_artifacts(["QB"], s3_client=s3)

    destination = tmp_path / "qb/outputs/models"
    assert sorted(file.name for file in destination.iterdir()) == ["model.pkl"]
    assert (destination / "model.pkl").read_bytes() == b"complete fallback"
