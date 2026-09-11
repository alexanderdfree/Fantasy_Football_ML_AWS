"""Adversarial serving-cache publication, readiness and local-reader contracts."""

import hashlib
import io
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from src.artifacts import model_sync
from src.artifacts import serving_snapshot as snapshots

pytestmark = pytest.mark.unit

POINTER = "models/predictions_cache/current.json"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class MemoryS3:
    def __init__(self):
        self.objects = {
            model_sync.manifest_key("models", position): encoded(
                {
                    "schema_version": 3,
                    "stable": {
                        "key": model_sync.history_prefix("models", position)
                        + "original/model.tar.gz"
                    },
                }
            )
            for position in snapshots.POSITIONS
        }
        self.before_put = lambda key, body: None
        self.calls = []

    def get_object(self, Bucket, Key):
        self.calls.append(("get", Key))
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        content = self.objects[Key]
        return {"Body": io.BytesIO(content), "ETag": hashlib.sha256(content).hexdigest()}

    def put_object(self, Bucket, Key, Body, IfMatch=None, IfNoneMatch=None, **_):
        self.before_put(Key, Body)
        existing = self.objects.get(Key)
        etag = hashlib.sha256(existing).hexdigest() if existing is not None else None
        if (IfMatch is not None and IfMatch != etag) or (
            IfNoneMatch == "*" and existing is not None
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.calls.append(("put", Key))
        self.objects[Key] = Body

    def advance_model(self, position="QB"):
        self.objects[model_sync.manifest_key("models", position)] = encoded(
            {
                "schema_version": 3,
                "stable": {
                    "key": model_sync.history_prefix("models", position) + "newer/model.tar.gz"
                },
            }
        )


def cache_directory(tmp_path, label):
    directory = tmp_path / label
    directory.mkdir()
    (directory / "predictions.parquet").write_bytes(f"predictions:{label}".encode())
    (directory / "metrics.json").write_bytes(encoded({"generation_label": label}))
    (directory / "fingerprint.json").write_bytes(
        encoded({"schema_version": snapshots.CACHE_SCHEMA_VERSION, "label": label})
    )
    (directory / "snapshot.json").write_bytes(encoded({"label": label}))
    return directory


def publish(s3, tmp_path, label):
    return snapshots.publish(
        s3, "bucket", cache_directory(tmp_path, label), snapshots.begin_build(s3, "bucket")
    )


def test_concurrent_builders_cannot_replace_winning_snapshot(tmp_path):
    s3 = MemoryS3()
    first, second = snapshots.begin_build(s3, "bucket"), snapshots.begin_build(s3, "bucket")
    winner = snapshots.publish(s3, "bucket", cache_directory(tmp_path, "winner"), first)
    with pytest.raises(ClientError, match="PreconditionFailed"):
        snapshots.publish(s3, "bucket", cache_directory(tmp_path, "loser"), second)
    assert json.loads(s3.objects[POINTER]) == winner


def test_model_change_during_payload_upload_blocks_publication(tmp_path):
    s3 = MemoryS3()
    original = publish(s3, tmp_path, "original")
    build = snapshots.begin_build(s3, "bucket")

    def advance_during_upload(key, _):
        if key.endswith("/metrics.json"):
            s3.advance_model()

    s3.before_put = advance_during_upload
    with pytest.raises(RuntimeError, match="advanced|superseded|generation"):
        snapshots.publish(s3, "bucket", cache_directory(tmp_path, "candidate"), build)
    assert json.loads(s3.objects[POINTER]) == original


def test_late_model_candidate_does_not_mix_captured_serving_release(tmp_path):
    s3 = MemoryS3()
    original = publish(s3, tmp_path, "original")
    build = snapshots.begin_build(s3, "bucket")

    def advance_at_commit(key, _):
        if key == POINTER:
            s3.advance_model()

    s3.before_put = advance_at_commit
    pointer = snapshots.publish(s3, "bucket", cache_directory(tmp_path, "candidate"), build)
    manifest = json.loads(s3.objects[pointer["manifest"]])
    # Model heads are candidates, not the served release. The snapshot owns
    # its complete captured model set even if a future candidate just arrived.
    assert pointer != original
    assert manifest["models"]["QB"].endswith("/original/model.tar.gz")
    assert json.loads(s3.objects[model_sync.manifest_key("models", "QB")])["stable"][
        "key"
    ].endswith("/newer/model.tar.gz")


def test_new_snapshot_builder_does_not_capture_a_legacy_only_head():
    s3 = MemoryS3()
    for position in snapshots.POSITIONS:
        protected = model_sync.manifest_key("models", position)
        s3.objects[model_sync.legacy_manifest_key("models", position)] = s3.objects.pop(protected)
    with pytest.raises(RuntimeError, match="No approved"):
        snapshots.begin_build(s3, "bucket")


def test_corrupted_remote_payload_retains_previous_local_generation(tmp_path):
    s3 = MemoryS3()
    first = publish(s3, tmp_path, "first")
    local = tmp_path / "local"
    snapshots.sync(s3, "bucket", local)
    captured = snapshots.active_directory(local)
    second = publish(s3, tmp_path, "second")
    base = second["manifest"].rsplit("/", 1)[0]
    s3.objects[f"{base}/predictions.parquet"] = b"corrupt"
    with pytest.raises(ValueError, match="checksum"):
        snapshots.sync(s3, "bucket", local)
    assert snapshots.active_directory(local) == captured
    assert json.loads((local / "current.json").read_text())["generation"] == first["generation"]


def test_corrupt_existing_generation_is_repaired_or_rejected(tmp_path):
    s3 = MemoryS3()
    pointer = publish(s3, tmp_path, "candidate")
    local = tmp_path / "local"
    existing = local / "generations" / pointer["generation"]
    existing.mkdir(parents=True)
    (existing / "predictions.parquet").write_bytes(b"corrupt pre-existing directory")
    try:
        snapshots.sync(s3, "bucket", local)
    except (ValueError, OSError):
        assert not (local / "current.json").exists()
    else:
        active = snapshots.active_directory(local)
        assert (active / "predictions.parquet").read_bytes() == b"predictions:candidate"
        assert all((active / name).is_file() for name in snapshots.FILES)


def test_unchanged_sync_revalidates_local_bytes_without_payload_downloads(tmp_path, monkeypatch):
    s3 = MemoryS3()
    pointer = publish(s3, tmp_path, "current")
    local = tmp_path / "local"
    snapshots.sync(s3, "bucket", local)
    reads = []
    get_object = s3.get_object

    def tracked(**kwargs):
        reads.append(kwargs["Key"])
        return get_object(**kwargs)

    monkeypatch.setattr(s3, "get_object", tracked)
    monkeypatch.setattr(
        snapshots.tempfile, "mkdtemp", lambda **_: pytest.fail("Unchanged generation staged again")
    )
    result = snapshots.sync(s3, "bucket", local)
    assert result == {"generation": pointer["generation"], "files": len(snapshots.FILES)}
    assert reads == [POINTER, pointer["manifest"]]
    directory, files = snapshots.read_generation(local)
    assert files["predictions.parquet"] == b"predictions:current"

    # An installed generation is not trusted merely because its ID matches.
    (directory / "predictions.parquet").write_bytes(b"damaged local bytes")
    with pytest.raises(ValueError, match="checksum"):
        snapshots.sync(s3, "bucket", local)
    assert reads == [POINTER, pointer["manifest"]] * 2


def test_new_generation_fetches_payloads_after_unchanged_sync(tmp_path, monkeypatch):
    s3 = MemoryS3()
    publish(s3, tmp_path, "old")
    local = tmp_path / "local"
    snapshots.sync(s3, "bucket", local)
    snapshots.sync(s3, "bucket", local)
    newer = publish(s3, tmp_path, "new")
    reads = []
    get_object = s3.get_object

    def tracked(**kwargs):
        reads.append(kwargs["Key"])
        return get_object(**kwargs)

    monkeypatch.setattr(s3, "get_object", tracked)
    assert snapshots.sync(s3, "bucket", local)["generation"] == newer["generation"]
    base = newer["manifest"].rsplit("/", 1)[0]
    assert set(reads) == {
        POINTER,
        newer["manifest"],
        *(f"{base}/{name}" for name in snapshots.FILES),
    }
    assert snapshots.read_generation(local)[1]["predictions.parquet"] == b"predictions:new"


def test_local_reader_captures_complete_old_or_new_generation(tmp_path):
    s3 = MemoryS3()
    publish(s3, tmp_path, "old")
    root = tmp_path / "local"
    snapshots.sync(s3, "bucket", root)
    captured = snapshots.active_directory(root)
    publish(s3, tmp_path, "new")
    snapshots.sync(s3, "bucket", root)
    new = snapshots.active_directory(root)
    assert captured != new
    assert (captured / "predictions.parquet").read_bytes() == b"predictions:old"
    assert json.loads((captured / "snapshot.json").read_text())["label"] == "old"
    assert (new / "predictions.parquet").read_bytes() == b"predictions:new"
    assert json.loads((new / "snapshot.json").read_text())["label"] == "new"


def test_publisher_uses_confirmed_generation_even_after_local_pointer_advances(tmp_path):
    s3 = MemoryS3()
    root = tmp_path / "local"
    old = cache_directory(tmp_path, "confirmed")
    confirmed = snapshots.publish_local(
        root, {name: (old / name).read_bytes() for name in snapshots.FILES}
    )
    newer = cache_directory(tmp_path, "unrelated")
    snapshots.publish_local(root, {name: (newer / name).read_bytes() for name in snapshots.FILES})
    pointer = snapshots.publish(
        s3, "bucket", root, snapshots.begin_build(s3, "bucket"), generation=confirmed.name
    )
    base = pointer["manifest"].rsplit("/", 1)[0]
    assert json.loads(s3.objects[f"{base}/snapshot.json"])["label"] == "confirmed"


def test_missing_pointer_is_the_only_legacy_fallback(tmp_path):
    s3 = MemoryS3()
    assert snapshots.sync(s3, "bucket", tmp_path) is None
    assert snapshots.active_directory(tmp_path) == tmp_path
    s3.objects[POINTER] = encoded({"schema_version": 1, "generation": "../invalid"})
    with pytest.raises(ValueError, match="Invalid"):
        snapshots.sync(s3, "bucket", tmp_path)


def test_local_concurrent_writers_keep_complete_generations(tmp_path, monkeypatch):
    files = []
    for label in ("one", "two"):
        source = cache_directory(tmp_path, label)
        files.append({name: (source / name).read_bytes() for name in snapshots.FILES})
    root = tmp_path / "local"
    barrier = threading.Barrier(2)
    replace = snapshots.os.replace

    def commit(source, destination):
        if Path(destination) == root / "current.json":
            barrier.wait(timeout=5)
        return replace(source, destination)

    monkeypatch.setattr(snapshots.os, "replace", commit)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(snapshots.publish_local, root, payload) for payload in files]
        directories = [future.result(timeout=10) for future in futures]
    selected, captured = snapshots.read_generation(root)
    assert selected in directories
    assert captured == files[directories.index(selected)]
    for directory, expected in zip(directories, files, strict=True):
        assert snapshots.read_generation(root, directory.name)[1] == expected


def test_schema9_revocation_blocks_sync_upload_and_delayed_reads(tmp_path, monkeypatch):
    s3 = MemoryS3()
    pointer = publish(s3, tmp_path, "old")
    root = tmp_path / "local"
    snapshots.sync(s3, "bucket", root)
    directory, files = snapshots.read_generation(root)
    snapshots.invalidate_generation(root, directory.name)
    with pytest.raises(ValueError, match="invalidated"):
        snapshots.read_generation(root)
    with pytest.raises(ValueError, match="invalidated"):
        snapshots.sync(s3, "bucket", root)
    with pytest.raises(ValueError, match="invalidated"):
        snapshots.publish(s3, "bucket", root, snapshots.begin_build(s3, "bucket"))
    assert json.loads(s3.objects[POINTER]) == pointer
    assert (directory / "snapshot.json").read_bytes() == files["snapshot.json"]
    publish(s3, tmp_path, "new")
    snapshots.sync(s3, "bucket", root)
    assert snapshots.read_generation(root)[0] != directory
    assert snapshots.is_invalidated(root, directory.name)


def test_revocation_during_verified_read_is_rejected(tmp_path, monkeypatch):
    source = cache_directory(tmp_path, "source")
    root = tmp_path / "local"
    directory = snapshots.publish_local(
        root, {name: (source / name).read_bytes() for name in snapshots.FILES}
    )
    read = Path.read_bytes
    revoked = []

    def invalidate_during_read(path):
        result = read(path)
        if path == directory / "predictions.parquet":
            snapshots.invalidate_generation(root, directory.name)
            revoked.append(True)
        return result

    monkeypatch.setattr(Path, "read_bytes", invalidate_during_read)
    with pytest.raises(ValueError, match="invalidated during read"):
        snapshots.read_generation(root)
    assert revoked == [True]


def test_legacy_cache_transport_cannot_replace_schema9_pointer(tmp_path):
    from src.shared import prediction_cache

    source = cache_directory(tmp_path, "source")
    root = tmp_path / "local"
    payload = {name: (source / name).read_bytes() for name in snapshots.FILES}
    current = snapshots.publish_local(root, payload)
    legacy = prediction_cache.publish_generation(root, payload)
    assert legacy.parent.parent == root / "legacy-v1"
    assert snapshots.read_generation(root) == (current, payload)
    assert prediction_cache.read_generation(root) == (legacy, payload)


@pytest.mark.parametrize("pointer", [None, [], {"schema_version": True, "generation": "a" * 64}])
def test_local_pointer_shape_is_validated_before_use(tmp_path, pointer):
    (tmp_path / "current.json").write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="pointer"):
        snapshots.read_generation(tmp_path)


def test_data_pinned_worker_recovers_rollback_and_accepts_same_data_retraining(
    tmp_path, monkeypatch
):
    s3 = MemoryS3()
    monkeypatch.setenv("FF_DATASET_ID", "a" * 64)
    original = publish(s3, tmp_path, "original-data-a")
    monkeypatch.setenv("FF_DATASET_ID", "b" * 64)
    publish(s3, tmp_path, "new-data-b")
    root = tmp_path / "replacement-worker"
    result = snapshots.sync(
        s3,
        "bucket",
        root,
        expected_dataset_id="a" * 64,
        pinned_generation=original["generation"],
    )
    assert result["generation"] == original["generation"]
    assert snapshots.read_generation(root, expected_dataset_id="a" * 64)[1][
        "snapshot.json"
    ] == encoded({"label": "original-data-a"})
    monkeypatch.setenv("FF_DATASET_ID", "a" * 64)
    retrained = publish(s3, tmp_path, "retrained-data-a")
    result = snapshots.sync(
        s3,
        "bucket",
        root,
        expected_dataset_id="a" * 64,
        pinned_generation=original["generation"],
    )
    assert result["generation"] == retrained["generation"]


def test_generation_from_different_data_pin_is_rejected(tmp_path, monkeypatch):
    s3 = MemoryS3()
    monkeypatch.setenv("FF_DATASET_ID", "a" * 64)
    pointer = publish(s3, tmp_path, "data-a")
    with pytest.raises(ValueError, match="different data release"):
        snapshots.sync(s3, "bucket", tmp_path / "worker", expected_dataset_id="b" * 64)
    snapshots.sync(s3, "bucket", tmp_path / "worker", expected_dataset_id="a" * 64)
    with pytest.raises(ValueError, match="different data release"):
        snapshots.read_generation(
            tmp_path / "worker", pointer["generation"], expected_dataset_id="b" * 64
        )


@pytest.mark.parametrize(
    "damage", ["missing_manifest", "corrupt_manifest", "missing_payload", "corrupt_payload"]
)
def test_pinned_fallback_never_installs_missing_or_tampered_bytes(tmp_path, monkeypatch, damage):
    s3 = MemoryS3()
    monkeypatch.setenv("FF_DATASET_ID", "a" * 64)
    old = publish(s3, tmp_path, "old-data-a")
    warm = tmp_path / "warm-worker"
    snapshots.sync(s3, "bucket", warm, expected_dataset_id="a" * 64)
    original = snapshots.read_generation(warm)
    monkeypatch.setenv("FF_DATASET_ID", "b" * 64)
    publish(s3, tmp_path, "new-data-b")
    key = (
        old["manifest"]
        if damage.endswith("manifest")
        else old["manifest"].replace("manifest.json", "snapshot.json")
    )
    if damage.startswith("missing"):
        del s3.objects[key]
    else:
        s3.objects[key] = b"{}"

    def sync(target):
        return snapshots.sync(
            s3,
            "bucket",
            target,
            expected_dataset_id="a" * 64,
            pinned_generation=old["generation"],
        )

    if damage.endswith("manifest"):
        with pytest.raises((ValueError, ClientError)):
            sync(warm)
    else:
        # Warm workers revalidate their intact local bytes against the remote
        # manifest. Remote payload damage cannot replace that verified copy.
        assert sync(warm)["generation"] == old["generation"]
    with pytest.raises((ValueError, ClientError)):
        sync(tmp_path / "cold-worker")
    assert snapshots.read_generation(warm) == original
    assert not (tmp_path / "cold-worker/current.json").exists()


def test_consumer_rejects_incompatible_cache_schema(tmp_path):
    s3 = MemoryS3()
    pointer = publish(s3, tmp_path, "old")
    manifest = json.loads(s3.objects[pointer["manifest"]])
    manifest["cache_schema_version"] = snapshots.CACHE_SCHEMA_VERSION + 1
    generation = hashlib.sha256(encoded(manifest)).hexdigest()
    base = f"models/predictions_cache/generations/{generation}"
    original_base = pointer["manifest"].rsplit("/", 1)[0]
    for name in snapshots.FILES:
        s3.objects[f"{base}/{name}"] = s3.objects[f"{original_base}/{name}"]
    s3.objects[f"{base}/manifest.json"] = encoded(manifest)
    s3.objects[POINTER] = encoded(
        {"schema_version": 1, "generation": generation, "manifest": f"{base}/manifest.json"}
    )
    with pytest.raises(ValueError, match="schema|incompatible|compatible"):
        snapshots.sync(s3, "bucket", tmp_path / "local")


@pytest.mark.parametrize("payload_state", ["valid", "missing", "corrupt"])
def test_readiness_verifies_payloads_before_accepting_generation(
    tmp_path, monkeypatch, payload_state
):
    s3 = MemoryS3()
    pointer = publish(s3, tmp_path, "candidate")
    base = pointer["manifest"].rsplit("/", 1)[0]
    if payload_state == "missing":
        del s3.objects[f"{base}/snapshot.json"]
    elif payload_state == "corrupt":
        s3.objects[f"{base}/snapshot.json"] = b"corrupt"

    def fake_aws(command, **_):
        key = command[command.index("--key") + 1]
        if key not in s3.objects:
            return subprocess.CompletedProcess(command, 1, "", "NoSuchKey")
        Path(command[-1]).write_bytes(s3.objects[key])
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(subprocess, "run", fake_aws)
    monkeypatch.setattr(
        sys, "argv", ["serving_snapshot", "wait", "--bucket", "bucket", "--timeout", "0"]
    )
    if payload_state == "valid":
        snapshots.main()
    else:
        with pytest.raises(RuntimeError, match="snapshot|payload|compatible|checksum"):
            snapshots.main()


@pytest.mark.parametrize("invalid", ["manifest_schema", "manifest_location"])
def test_readiness_rejects_generation_that_sync_would_refuse(tmp_path, monkeypatch, invalid):
    s3 = MemoryS3()
    pointer = publish(s3, tmp_path, "candidate")
    original_base = pointer["manifest"].rsplit("/", 1)[0]
    manifest = json.loads(s3.objects[pointer["manifest"]])
    if invalid == "manifest_schema":
        manifest["schema_version"] = 2
        generation = hashlib.sha256(encoded(manifest)).hexdigest()
        base = f"models/predictions_cache/generations/{generation}"
    else:
        generation = pointer["generation"]
        base = f"other-prefix/predictions_cache/generations/{generation}"
    for name in snapshots.FILES:
        s3.objects[f"{base}/{name}"] = s3.objects[f"{original_base}/{name}"]
    s3.objects[f"{base}/manifest.json"] = encoded(manifest)
    s3.objects[POINTER] = encoded(
        {"schema_version": 1, "generation": generation, "manifest": f"{base}/manifest.json"}
    )

    def fake_aws(command, **_):
        key = command[command.index("--key") + 1]
        Path(command[-1]).write_bytes(s3.objects[key])
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(subprocess, "run", fake_aws)
    monkeypatch.setattr(
        sys, "argv", ["serving_snapshot", "wait", "--bucket", "bucket", "--timeout", "0"]
    )
    with pytest.raises(ValueError):
        snapshots.sync(s3, "bucket", tmp_path / "local")
    with pytest.raises(RuntimeError, match="snapshot|compatible"):
        snapshots.main()
