"""Publication and consumption of complete immutable prediction generations."""

from __future__ import annotations

import io
import json
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.shared import prediction_cache

pytestmark = pytest.mark.unit


def _files(value: int, *, snapshot: bool = True) -> dict[str, bytes]:
    files = {
        "predictions.parquet": str(value).encode(),
        "metrics.json": json.dumps({"value": value}).encode(),
        "fingerprint.json": json.dumps({"sha256": f"model-{value}"}).encode(),
    }
    if snapshot:
        files["snapshot.json"] = json.dumps({"value": value}).encode()
    return files


def _assert_generation_matches(files: dict[str, bytes], expected: int) -> None:
    assert int(files["predictions.parquet"]) == expected
    assert json.loads(files["metrics.json"])["value"] == expected
    assert json.loads(files["fingerprint.json"])["sha256"] == f"model-{expected}"
    if "snapshot.json" in files:
        assert json.loads(files["snapshot.json"])["value"] == expected


def _archive(members: list[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return stream.getvalue()


@pytest.mark.parametrize("snapshot", [True, False])
def test_bundle_round_trip_publishes_complete_generation(tmp_path, snapshot):
    source, destination = tmp_path / "source", tmp_path / "destination"
    files = _files(11, snapshot=snapshot)
    original = prediction_cache.publish_generation(source, files)
    generation_id, bundle = prediction_cache.bundle_generation(source)

    installed = prediction_cache.install_bundle(destination, bundle)

    assert generation_id == original.name == installed.name
    assert prediction_cache.current_generation(destination) == installed
    directory, actual = prediction_cache.read_generation(destination)
    assert directory == installed
    assert actual == files
    assert not (destination / "predictions.parquet").exists()


def test_concurrent_distinct_writers_publish_whole_generations(tmp_path, monkeypatch):
    """Force competing final commits with different predictions/metrics/snapshots."""
    old = prediction_cache.publish_generation(tmp_path, _files(1))
    replace = prediction_cache.os.replace
    commits = threading.Barrier(2)

    def concurrent_commit(source, destination):
        if Path(destination).name == "current.json":
            commits.wait(timeout=5)
        return replace(source, destination)

    monkeypatch.setattr(prediction_cache.os, "replace", concurrent_commit)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(prediction_cache.publish_generation, tmp_path, _files(v)) for v in (11, 99)
        ]
        directories = [future.result(timeout=10) for future in futures]

    selected, files = prediction_cache.read_generation(tmp_path)
    assert selected in directories
    expected = int(files["predictions.parquet"])
    assert expected in (11, 99)
    _assert_generation_matches(files, expected)
    for directory, value in zip(directories, (11, 99), strict=True):
        assert {name: (directory / name).read_bytes() for name in _files(value)} == _files(value)
    assert (old / "predictions.parquet").read_bytes() == b"1"
    assert not list(tmp_path.rglob(".staging-*"))
    assert not list(tmp_path.glob(".current-*"))


def test_identical_concurrent_writers_converge(tmp_path, monkeypatch):
    rename = prediction_cache.os.rename
    installs = threading.Barrier(2)

    def concurrent_install(source, destination):
        installs.wait(timeout=5)
        return rename(source, destination)

    monkeypatch.setattr(prediction_cache.os, "rename", concurrent_install)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(prediction_cache.publish_generation, tmp_path, _files(11)) for _ in range(2)
        ]
        directories = [future.result(timeout=10) for future in futures]
    assert directories[0] == directories[1]
    assert prediction_cache.read_generation(tmp_path)[1] == _files(11)
    assert not list(tmp_path.rglob(".staging-*"))


def test_reader_retains_selected_generation_when_pointer_changes(tmp_path, monkeypatch):
    old = prediction_cache.publish_generation(tmp_path, _files(11))
    selected, resume = threading.Event(), threading.Event()
    read_bytes = Path.read_bytes

    def pause_after_selection(path):
        if path == old / "predictions.parquet":
            selected.set()
            assert resume.wait(timeout=5)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", pause_after_selection)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reader = pool.submit(prediction_cache.read_generation, tmp_path)
        try:
            assert selected.wait(timeout=5)
            new = prediction_cache.publish_generation(tmp_path, _files(99))
        finally:
            resume.set()
        directory, captured = reader.result(timeout=5)
    assert directory == old
    _assert_generation_matches(captured, 11)
    assert prediction_cache.current_generation(tmp_path) == new
    _assert_generation_matches(prediction_cache.read_generation(tmp_path)[1], 99)


@pytest.mark.parametrize("damage", ["missing", "modified"])
@pytest.mark.parametrize(
    "member", sorted(prediction_cache.REQUIRED_FILES | prediction_cache.OPTIONAL_FILES)
)
def test_damaged_immutable_member_is_rejected(tmp_path, damage, member):
    directory = prediction_cache.publish_generation(tmp_path, _files(11))
    if damage == "missing":
        (directory / member).unlink()
    else:
        (directory / member).write_bytes(b"corrupted")
    with pytest.raises((ValueError, OSError)):
        prediction_cache.read_generation(tmp_path)
    with pytest.raises((ValueError, OSError)):
        prediction_cache.bundle_generation(tmp_path)


@pytest.mark.parametrize(
    "damage", ["truncated", "missing", "checksum", "duplicate", "unexpected", "traversal"]
)
def test_bad_bundle_preserves_previously_committed_generation(tmp_path, damage):
    source, destination = tmp_path / "source", tmp_path / "destination"
    previous = prediction_cache.publish_generation(destination, _files(11))
    fresh = prediction_cache.publish_generation(source, _files(99))
    members = [(name, (fresh / name).read_bytes()) for name in (*_files(99), "generation.json")]
    if damage == "missing":
        members = [(name, data) for name, data in members if name != "metrics.json"]
    elif damage == "checksum":
        members = [(name, b"{}" if name == "metrics.json" else data) for name, data in members]
    elif damage == "duplicate":
        members.append(members[0])
    elif damage == "unexpected":
        members.append(("surprise.json", b"{}"))
    elif damage == "traversal":
        members.append(("../outside", b"not allowed"))
    bundle = _archive(members)
    if damage == "truncated":
        bundle = bundle[: len(bundle) // 2]

    with pytest.raises((ValueError, tarfile.TarError, EOFError)):
        prediction_cache.install_bundle(destination, bundle)

    assert prediction_cache.current_generation(destination) == previous
    assert prediction_cache.read_generation(destination)[1] == _files(11)
    assert not (tmp_path / "outside").exists()


def test_failed_pointer_commit_preserves_previous_generation(tmp_path, monkeypatch):
    previous = prediction_cache.publish_generation(tmp_path, _files(11))

    def reject_commit(*args):
        raise OSError("filesystem temporarily unavailable")

    monkeypatch.setattr(prediction_cache.os, "replace", reject_commit)
    with pytest.raises(OSError, match="temporarily unavailable"):
        prediction_cache.publish_generation(tmp_path, _files(99))
    assert prediction_cache.current_generation(tmp_path) == previous
    assert prediction_cache.read_generation(tmp_path)[1] == _files(11)
    assert not list(tmp_path.rglob(".staging-*"))
    assert not list(tmp_path.glob(".current-*"))


def test_legacy_loose_writer_cannot_replace_new_generation_or_bundle(tmp_path):
    published = prediction_cache.publish_generation(tmp_path, _files(99))
    for name, data in _files(11).items():
        (tmp_path / name).write_bytes(data)

    assert prediction_cache.current_generation(tmp_path) == published
    assert prediction_cache.read_generation(tmp_path)[1] == _files(99)
    _, bundle = prediction_cache.bundle_generation(tmp_path)
    destination = tmp_path / "consumer"
    prediction_cache.install_bundle(destination, bundle)
    assert prediction_cache.read_generation(destination)[1] == _files(99)


def test_legacy_loose_files_alone_are_not_a_committed_generation(tmp_path):
    for name, data in _files(11).items():
        (tmp_path / name).write_bytes(data)
    assert prediction_cache.current_generation(tmp_path) is None
    with pytest.raises(ValueError, match="No committed"):
        prediction_cache.read_generation(tmp_path)


def test_shared_invalidation_blocks_reads_uploads_and_reinstall_of_old_bundle(tmp_path):
    old = prediction_cache.publish_generation(tmp_path, _files(11))
    _, old_bundle = prediction_cache.bundle_generation(tmp_path)
    prediction_cache.invalidate_generation(tmp_path, old.name)
    assert prediction_cache.current_generation(tmp_path) is None
    for reader in (prediction_cache.read_generation, prediction_cache.bundle_generation):
        with pytest.raises(ValueError):
            reader(tmp_path)
    assert (old / "predictions.parquet").read_bytes() == b"11"

    new = prediction_cache.publish_generation(tmp_path, _files(99))
    with pytest.raises(ValueError, match="invalidated"):
        prediction_cache.install_bundle(tmp_path, old_bundle)
    assert prediction_cache.current_generation(tmp_path) == new
    assert prediction_cache.read_generation(tmp_path)[1] == _files(99)


def test_invalidation_while_reading_rejects_selected_generation(tmp_path, monkeypatch):
    old = prediction_cache.publish_generation(tmp_path, _files(11))
    selected, resume = threading.Event(), threading.Event()
    read_bytes = Path.read_bytes

    def pause_after_selection(path):
        if path == old / "predictions.parquet":
            selected.set()
            assert resume.wait(timeout=5)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", pause_after_selection)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reader = pool.submit(prediction_cache.read_generation, tmp_path)
        try:
            assert selected.wait(timeout=5)
            prediction_cache.invalidate_generation(tmp_path, old.name)
        finally:
            resume.set()
        with pytest.raises(ValueError, match="invalidated during read"):
            reader.result(timeout=5)
    assert (old / "predictions.parquet").read_bytes() == b"11"


def test_invalidation_during_bundle_creation_prevents_upload(tmp_path, monkeypatch):
    old = prediction_cache.publish_generation(tmp_path, _files(11))
    addfile = tarfile.TarFile.addfile

    def invalidate_during_compression(archive, member, stream):
        result = addfile(archive, member, stream)
        prediction_cache.invalidate_generation(tmp_path, old.name)
        return result

    monkeypatch.setattr(tarfile.TarFile, "addfile", invalidate_during_compression)
    with pytest.raises(ValueError, match="invalidated before upload"):
        prediction_cache.bundle_generation(tmp_path)


def test_invalidation_before_commit_preserves_existing_pointer(tmp_path, monkeypatch):
    previous = prediction_cache.publish_generation(tmp_path, _files(11))
    rename = prediction_cache.os.rename

    def invalidate_staged_generation(source, destination):
        result = rename(source, destination)
        prediction_cache.invalidate_generation(tmp_path, Path(destination).name)
        return result

    monkeypatch.setattr(prediction_cache.os, "rename", invalidate_staged_generation)
    with pytest.raises(ValueError, match="invalidated before publication"):
        prediction_cache.publish_generation(tmp_path, _files(99))
    assert prediction_cache.current_generation(tmp_path) == previous
    assert prediction_cache.read_generation(tmp_path)[1] == _files(11)
