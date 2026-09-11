"""Publication, selection and materialization at the dataset boundary."""

import hashlib
import io
import json
import subprocess
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from src.orchestration import datasets

pytestmark = pytest.mark.unit


class MemoryS3:
    def __init__(self):
        self.objects = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body = self.objects[Key]
        return {"Body": io.BytesIO(body), "ETag": hashlib.sha256(body).hexdigest()}

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.get_object(Bucket, Key)["Body"].read())}

    def put_object(self, Bucket, Key, Body, **kwargs):
        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if kwargs.get("IfMatch") and (
            Key not in self.objects
            or hashlib.sha256(self.objects[Key]).hexdigest() != kwargs["IfMatch"]
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body

    def upload_file(self, path, bucket, key):
        self.objects[key] = Path(path).read_bytes()

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.get_object(bucket, key)["Body"].read())


PRODUCER_HASHES = {"src/config.py": "a" * 64}


def write_release(s3, producer_hashes=PRODUCER_HASHES, *, files=None):
    """Create a canonical remote release fixture, without inventing a second index."""
    from src.data.release import producer_fingerprint

    files = files or {
        **{f"splits/{name}": name.encode() for name in datasets.SPLITS},
        "raw/weekly.parquet": b"weekly raw",
    }
    manifest = {
        "schema_version": 1,
        "data_producer_sha256": producer_fingerprint(producer_hashes),
        "producer": producer_hashes,
        "runtime": {},
        "coverage": {},
        "files": {
            name: {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
            for name, body in files.items()
        },
    }
    selected = datasets.content_id(manifest)
    base = f"data/releases/{selected}"
    s3.objects[f"{base}/manifest.json"] = datasets.canonical_bytes(manifest)
    for name, body in files.items():
        s3.objects[f"{base}/{name}"] = body
    pointer = datasets.canonical_bytes({"schema_version": 1, "release_id": selected})
    s3.objects[f"data/by-producer/{manifest['data_producer_sha256']}/manifest.json"] = pointer
    s3.objects["data/manifest.json"] = pointer
    return selected


@pytest.fixture
def published(tmp_path, monkeypatch):
    from src.data.release import producer_fingerprint
    from src.scripts import wait_data_release

    s3 = MemoryS3()
    selected = write_release(s3)
    monkeypatch.setattr(
        wait_data_release, "producer_hashes_at_revision", lambda *_, **__: PRODUCER_HASHES
    )
    return s3, producer_fingerprint(PRODUCER_HASHES), selected


def test_snapshot_survives_later_mutable_source_update(published, tmp_path):
    s3, source, selected = published
    manifest = datasets.load_dataset(s3, "bucket", selected)
    assert datasets.select_dataset(s3, "bucket", source, timeout=0) == selected
    # Publishing another dataset changes the source selection, never the
    # previously selected content-addressed manifest or objects.
    newer = write_release(
        s3,
        files={
            **{f"splits/{name}": name.encode() for name in datasets.SPLITS},
            "raw/weekly.parquet": b"new weekly raw",
        },
    )
    assert selected != newer
    assert datasets.load_dataset(s3, "bucket", selected) == manifest
    datasets.materialize_dataset(
        s3, "bucket", selected, splits_dir=tmp_path / "job/splits", raw_dir=tmp_path / "job/raw"
    )
    assert (tmp_path / "job/raw/weekly.parquet").read_bytes() == b"weekly raw"


def test_missing_source_never_falls_back_to_mutable_data():
    s3 = MemoryS3()
    s3.objects["data/train.parquet"] = b"old mutable data"
    with pytest.raises(datasets.DatasetError, match="fallback is disabled"):
        datasets.select_dataset(s3, "bucket", "f" * 64, timeout=0)


def test_waits_for_delayed_completed_snapshot(published):
    s3, source, selected = published
    key = f"data/by-producer/{source}/manifest.json"
    ready = s3.objects.pop(key)
    now = [0]

    def sleep(duration):
        now[0] += duration
        s3.objects[key] = ready

    assert (
        datasets.select_dataset(
            s3, "bucket", source, timeout=2, poll=1, clock=lambda: now[0], sleep=sleep
        )
        == selected
    )


def test_corrupt_object_cannot_replace_existing_inputs(published, tmp_path):
    s3, _, selected = published
    manifest = datasets.load_dataset(s3, "bucket", selected)
    s3.objects[manifest["files"][0]["key"]] = b"corrupt"
    raw, splits = tmp_path / "job/raw", tmp_path / "job/splits"
    raw.mkdir(parents=True)
    (raw / "old").write_bytes(b"untouched")
    with pytest.raises(datasets.DatasetError, match="checksum mismatch"):
        datasets.materialize_dataset(s3, "bucket", selected, splits_dir=splits, raw_dir=raw)
    assert (raw / "old").read_bytes() == b"untouched"
    assert not splits.exists()


@pytest.mark.parametrize(
    "path", ["/tmp/escape", "data/raw/../../escape", "data/raw/../x", "data/splits/other", None]
)
def test_manifest_rejects_unsafe_destinations(published, path):
    s3, _, selected = published
    manifest = datasets.load_dataset(s3, "bucket", selected)
    manifest["files"][0]["path"] = path
    with pytest.raises(datasets.DatasetError, match="Unsafe"):
        datasets.validate_manifest(manifest, datasets.content_id(manifest))


def test_wrong_manifest_digest_is_rejected(published):
    s3, _, selected = published
    key = f"data/releases/{selected}/manifest.json"
    doc = json.loads(s3.objects[key])
    doc["source_id"] = "b" * 64
    s3.objects[key] = json.dumps(doc).encode()
    with pytest.raises(datasets.DatasetError, match="checksum"):
        datasets.load_dataset(s3, "bucket", selected)


def test_unsealed_upload_never_exposes_ready_source(tmp_path):
    from src.data.release import data_producer_hashes, producer_fingerprint

    s3 = MemoryS3()
    for name in datasets.SPLITS:
        file = tmp_path / "data/splits" / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"split")
    file = tmp_path / "data/raw/raw.parquet"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"raw")
    recipe = producer_fingerprint(data_producer_hashes(tmp_path))
    with pytest.raises(RuntimeError, match="Unsealed"):
        datasets.publish_dataset(s3, "bucket", tmp_path, recipe)
    assert s3.objects == {}


def test_legacy_dataset_requires_explicit_immutable_format(tmp_path):
    s3 = MemoryS3()
    files = []
    for name, body in {
        **{f"data/splits/{name}": name.encode() for name in datasets.SPLITS},
        "data/raw/weekly.parquet": b"old raw",
    }.items():
        digest = hashlib.sha256(body).hexdigest()
        key = f"datasets/objects/{digest}"
        s3.objects[key] = body
        files.append({"path": name, "key": key, "sha256": digest, "bytes": len(body)})
    manifest = {"schema_version": 1, "source_id": "a" * 64, "files": files}
    selected = datasets.content_id(manifest)
    s3.objects[f"datasets/manifests/{selected}.json"] = datasets.canonical_bytes(manifest)
    s3.objects[f"datasets/sources/{'a' * 64}.json"] = datasets.canonical_bytes(
        {"dataset_id": selected}
    )
    with pytest.raises(ClientError):
        datasets.load_dataset(s3, "bucket", selected)
    assert datasets.load_dataset(s3, "bucket", selected, data_format="dataset-v1") == manifest
    datasets.materialize_dataset(
        s3,
        "bucket",
        selected,
        data_format="dataset-v1",
        raw_dir=tmp_path / "raw",
        splits_dir=tmp_path / "splits",
    )
    assert (tmp_path / "raw/weekly.parquet").read_bytes() == b"old raw"
    with pytest.raises(datasets.DatasetError, match="fallback is disabled"):
        datasets.select_dataset(s3, "bucket", "a" * 64, timeout=0)


def test_source_dependency_paths_match_refresh_workflow():
    import yaml

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/refresh-splits.yml").read_text())
    paths = (workflow.get("on") or workflow[True])["push"]["paths"]
    for source in datasets.SOURCE_PATHS:
        assert source in paths or source + "/**" in paths, source


def test_artifact_compatibility_alias_is_same_module():
    from src.artifacts import model_sync as implementation
    from src.shared import model_sync as compatibility

    assert implementation is compatibility


def test_source_identity_is_bound_to_image_revision_not_latest_checkout(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Dataset test")
    git("config", "user.email", "dataset@example.test")
    git("config", "commit.gpgsign", "false")
    source = tmp_path / "src/config.py"
    source.parent.mkdir()
    source.write_text("SEASONS = [2025]\n")
    git("add", ".")
    git("commit", "-qm", "first")
    revision = git("rev-parse", "HEAD")
    first = datasets.source_identity(tmp_path, revision)
    source.write_text("SEASONS = [2026]\n")
    with pytest.raises(datasets.DatasetError, match="checkout differs"):
        datasets.assert_source_checkout(tmp_path, revision)
    git("commit", "-qam", "second")
    assert datasets.source_identity(tmp_path, revision) == first
    assert datasets.source_identity(tmp_path, "HEAD") != first
