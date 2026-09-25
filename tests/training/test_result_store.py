import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
from botocore.exceptions import ClientError

from src.training.result_store import ResultStore, configure_s3_expiration

pytestmark = pytest.mark.unit
KEY = "a" * 64


class MemoryS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, *, Key, Body, **kwargs):
        if Key in self.objects and kwargs.get("IfNoneMatch") == "*":
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body.read()

    def get_object(self, *, Key, **kwargs):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": BytesIO(self.objects[Key])}


def publish(store, key=KEY, content="payload"):
    store.publish(
        key,
        lambda root: (root / "value").write_text(content),
        source_run_id="original",
        identity={"seed": 42},
    )


def test_local_remote_round_trip_preserves_provenance(tmp_path):
    s3 = MemoryS3()
    original = ResultStore(tmp_path / "a", s3=s3, bucket="training")
    publish(original)
    restored = ResultStore(tmp_path / "b", s3=s3, bucket="training")
    path, manifest = restored.lookup(KEY)
    assert (path / "value").read_text() == "payload"
    assert manifest["source_run_id"] == "original"
    assert manifest["identity"] == {"seed": 42}


def test_first_complete_writer_wins_and_partial_writes_are_invisible(tmp_path):
    store = ResultStore(tmp_path)

    def writer(root):
        (root / "value").write_text("unfinished")
        assert store.lookup(KEY) is None
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        store.publish(KEY, writer, source_run_id="bad", identity={})
    assert store.lookup(KEY) is None
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: publish(store, content=str(i)), range(4)))
    path, manifest = store.lookup(KEY)
    assert (path / "value").read_text() in {"0", "1", "2", "3"}
    assert manifest["files"]


def test_corrupt_entry_is_recomputed_not_read(tmp_path):
    store = ResultStore(tmp_path)
    publish(store)
    (store.path(KEY) / "value").write_text("changed")
    assert store.lookup(KEY) is None
    publish(store, content="repaired")
    assert (store.lookup(KEY)[0] / "value").read_text() == "repaired"


def test_expiration_and_lru_affect_only_cache_entries(tmp_path):
    store = ResultStore(tmp_path, max_bytes=20)
    (tmp_path / "retained-report.json").write_text("keep")
    publish(store, content="0123456789")
    publish(store, "b" * 64, "0123456789")
    publish(store, "c" * 64, "0123456789")
    assert not store.path(KEY).exists()
    path = store.path("b" * 64) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["created_at"] = 0
    path.write_text(json.dumps(manifest))
    assert store.lookup("b" * 64) is None
    store.prune()
    assert not store.path("b" * 64).exists()
    assert (tmp_path / "retained-report.json").read_text() == "keep"


def test_remote_archive_cannot_escape_cache(tmp_path):
    s3 = MemoryS3()
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("../../outside", "bad")
    s3.objects[f"experiment-cache/v1/{KEY}.zip"] = stream.getvalue()
    assert ResultStore(tmp_path / "cache", s3=s3, bucket="training").lookup(KEY) is None
    assert not (tmp_path / "outside").exists()


def test_remote_hit_only_workload_obeys_local_budget(tmp_path):
    s3 = MemoryS3()
    source = ResultStore(tmp_path / "source", s3=s3, bucket="training")
    for key in ("a" * 64, "b" * 64, "c" * 64):
        publish(source, key, "x" * 1000)
    destination = ResultStore(tmp_path / "destination", s3=s3, bucket="training", max_bytes=2500)
    for key in ("a" * 64, "b" * 64, "c" * 64):
        assert destination.lookup(key) is not None
    entries = list(destination.root.glob("*/manifest.json"))
    assert sum(json.loads(path.read_text())["bytes"] for path in entries) <= 2500
    assert destination.path("c" * 64).is_dir()


def test_key_cannot_select_an_arbitrary_path(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        ResultStore(tmp_path).lookup("../other")


def test_s3_expiration_preserves_unrelated_rules():
    from unittest.mock import Mock

    s3 = Mock()
    existing = {
        "ID": "retained",
        "Status": "Enabled",
        "Filter": {"Prefix": "logs/"},
        "Expiration": {"Days": 90},
    }
    s3.get_bucket_lifecycle_configuration.return_value = {"Rules": [existing]}
    assert configure_s3_expiration(s3, "bucket")
    rules = s3.put_bucket_lifecycle_configuration.call_args.kwargs["LifecycleConfiguration"][
        "Rules"
    ]
    assert existing in rules
    assert rules[-1]["Filter"]["Prefix"] == "experiment-cache/v1/"
    s3.get_bucket_lifecycle_configuration.return_value = {"Rules": rules}
    assert not configure_s3_expiration(s3, "bucket")
    assert s3.put_bucket_lifecycle_configuration.call_count == 1
