import hashlib
import io
import json
from datetime import UTC, datetime, timedelta

import pytest
from botocore.exceptions import ClientError

from src.artifacts import gc, model_sync
from src.orchestration.datasets import DatasetError
from src.scripts import promote

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 10, tzinfo=UTC)
MANIFEST = model_sync.manifest_key("models", "QB")


def key(label):
    return model_sync.history_prefix("models", "QB") + f"{label}/model.tar.gz"


class Store:
    def __init__(self):
        self.objects = {
            key(name): b"artifact" for name in ("stable", "orphan", "planned", "recent", "receipt")
        }
        self.objects[MANIFEST] = json.dumps(
            {"stable": {"key": key("stable")}, "history": [key("stable")], "revision": "initial"}
        ).encode()
        self.objects["build-plans/retained/artifacts/QB.json"] = json.dumps(
            {"key": key("receipt")}
        ).encode()
        self.metadata = {key("planned"): {"build-plan-id": "retained-plan"}}
        self.before_delete = lambda: None
        self.deletes = []

    def etag(self, name):
        return hashlib.sha256(self.objects[name]).hexdigest()

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key]), "ETag": self.etag(Key)}

    def put_object(self, Bucket, Key, Body, IfMatch=None, IfNoneMatch=None, **_):
        if IfMatch != self.etag(Key):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body

    def get_paginator(self, _):
        store = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                yield {
                    "Contents": [
                        {"Key": name} for name in list(store.objects) if name.startswith(Prefix)
                    ]
                }

        return Paginator()

    def head_object(self, Bucket, Key):
        return {
            "Metadata": self.metadata.get(Key, {}),
            "LastModified": NOW if Key == key("recent") else NOW - timedelta(days=2),
        }

    def delete_objects(self, Bucket, Delete):
        self.before_delete()
        for item in Delete["Objects"]:
            self.deletes.append(item["Key"])
            self.objects.pop(item["Key"])
        return {}


def test_dry_run_has_no_mutations_and_protects_plans_receipts_and_grace():
    store = Store()
    before = store.objects.copy()
    report = gc.collect(store, "bucket", "models", "QB", now=NOW)
    assert report["candidates"] == [key("orphan")]
    assert report["deleted"] == []
    assert store.objects == before


def test_keep_trims_history_before_delete_and_promotion_list_stays_consistent():
    store = Store()
    store.objects[key("approved-old")] = b"prior approved artifact"
    manifest = json.loads(store.objects[MANIFEST])
    manifest.update(
        history=[
            key(name)
            for name in ("stable", "orphan", "approved-old", "planned", "receipt", "recent")
        ],
        previous_stable={"key": key("approved-old")},
        source_frontier={"source_sha": "a" * 40, "source_order": 12},
        intent_frontier={"sequence": 4},
        rollback_epoch="operator-rollback",
    )
    store.objects[MANIFEST] = json.dumps(manifest).encode()

    def before_delete():
        current = json.loads(store.objects[MANIFEST])
        assert current["gc_lock"]
        assert key("orphan") not in current["history"]
        assert key("orphan") in store.objects

    store.before_delete = before_delete
    report = gc.collect(store, "bucket", "models", "QB", keep_n=1, execute=True, now=NOW)
    current = model_sync.load_manifest(store, "bucket", "models", "QB")
    assert report["deleted"] == [key("orphan")]
    assert current["history"] == [
        key(name) for name in ("stable", "approved-old", "planned", "receipt", "recent")
    ]
    assert all(name in store.objects for name in current["history"])
    assert key("orphan") not in promote.list_history(current)
    for name in (
        "stable",
        "previous_stable",
        "source_frontier",
        "intent_frontier",
        "rollback_epoch",
    ):
        assert current[name] == manifest[name]
    assert not current.get("gc_lock")


def test_history_trim_conflict_prevents_deletion_and_keeps_changed_lock():
    class RacingStore(Store):
        def put_object(self, **kwargs):
            proposed = json.loads(kwargs["Body"])
            existing = json.loads(self.objects[MANIFEST])
            if existing.get("gc_lock") and proposed.get("history") != existing["history"]:
                existing["revision"] = "changed-during-history-trim"
                self.objects[MANIFEST] = json.dumps(existing).encode()
            return super().put_object(**kwargs)

    store = RacingStore()
    manifest = json.loads(store.objects[MANIFEST])
    manifest["history"].append(key("orphan"))
    store.objects[MANIFEST] = json.dumps(manifest).encode()
    with pytest.raises(DatasetError, match="lock changed"):
        gc.collect(store, "bucket", "models", "QB", keep_n=1, execute=True, now=NOW)
    assert not store.deletes
    current = json.loads(store.objects[MANIFEST])
    assert current["gc_lock"]
    assert current["history"] == manifest["history"]
    assert key("orphan") in store.objects


def test_failed_delete_keeps_trimmed_history_and_retry_cleans_unreferenced_bytes():
    store = Store()
    manifest = json.loads(store.objects[MANIFEST])
    manifest["history"].append(key("orphan"))
    store.objects[MANIFEST] = json.dumps(manifest).encode()

    def interrupted():
        raise OSError("delete interrupted after history trim")

    store.before_delete = interrupted
    with pytest.raises(OSError, match="interrupted"):
        gc.collect(store, "bucket", "models", "QB", keep_n=1, execute=True, now=NOW)
    assert json.loads(store.objects[MANIFEST])["history"] == [key("stable")]
    assert key("orphan") in store.objects
    store.before_delete = lambda: None
    report = gc.collect(store, "bucket", "models", "QB", keep_n=1, execute=True, now=NOW)
    assert report["deleted"] == [key("orphan")]


def test_protected_collector_never_deletes_legacy_history():
    store = Store()
    legacy = "models/QB/history/legacy-orphan/model.tar.gz"
    store.objects[legacy] = b"legacy writer still owns this namespace"
    report = gc.collect(store, "bucket", "models", "QB", execute=True, now=NOW)
    assert report["deleted"] == [key("orphan")]
    assert legacy in store.objects


@pytest.mark.parametrize("make_current", [False, True])
def test_retained_serving_generation_protects_model_even_before_pointer_publish(make_current):
    store = Store()
    manifest = {"schema_version": 1, "models": {"QB": key("orphan")}, "files": {}}
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    generation = hashlib.sha256(payload).hexdigest()
    manifest_key = f"models/predictions_cache/generations/{generation}/manifest.json"
    store.objects[manifest_key] = payload
    if make_current:
        store.objects["models/predictions_cache/current.json"] = json.dumps(
            {
                "schema_version": 1,
                "generation": generation,
                "manifest": manifest_key,
            }
        ).encode()
    report = gc.collect(store, "bucket", "models", "QB", execute=True, now=NOW)
    assert report["deleted"] == []
    assert key("orphan") in store.objects


def test_pre_lock_publisher_fails_cas_even_after_unlock():
    store = Store()
    before, etag = model_sync.load_manifest_snapshot(store, "bucket", "models", "QB")
    gc.collect(store, "bucket", "models", "QB", execute=True, now=NOW)
    assert "gc_lock" not in json.loads(store.objects[MANIFEST])
    assert json.loads(store.objects[MANIFEST])["revision"] != before["revision"]
    candidate = model_sync.build_manifest(
        key("stale-publisher"), "a", 1, "now", before, smoke_passed=True
    )
    with pytest.raises(ClientError, match="PreconditionFailed"):
        model_sync.write_manifest(store, "bucket", "models", "QB", candidate, expected_etag=etag)
    assert store.deletes == [key("orphan")]


def test_new_publishers_and_manual_promotion_refuse_held_lock():
    store = Store()

    def during_lock():
        with pytest.raises(model_sync.ManifestLockedError):
            model_sync.load_manifest_snapshot(store, "bucket", "models", "QB")
        locked = model_sync.load_manifest(store, "bucket", "models", "QB")
        with pytest.raises(model_sync.ManifestLockedError):
            model_sync.build_manifest("candidate", "a", 1, "now", locked)
        with pytest.raises(promote.PromotionError, match="lock is held"):
            promote.promote(store, "bucket", "models", "QB", key("stable"))

    store.before_delete = during_lock
    gc.collect(store, "bucket", "models", "QB", execute=True, now=NOW)


def test_abort_releases_owned_lock_without_restoring_old_etag():
    store = Store()
    before = store.etag(MANIFEST)

    def failure():
        raise OSError("delete interrupted")

    store.before_delete = failure
    with pytest.raises(OSError, match="interrupted"):
        gc.collect(store, "bucket", "models", "QB", execute=True, now=NOW)
    assert not json.loads(store.objects[MANIFEST]).get("gc_lock")
    assert store.etag(MANIFEST) != before
    assert key("orphan") in store.objects


def test_changed_lock_is_never_unconditionally_released():
    store = Store()

    def concurrent_change():
        manifest = json.loads(store.objects[MANIFEST])
        manifest["revision"] = "external-change"
        store.objects[MANIFEST] = json.dumps(manifest).encode()

    store.before_delete = concurrent_change
    with pytest.raises(DatasetError, match="lock changed"):
        gc.collect(store, "bucket", "models", "QB", execute=True, now=NOW)
    assert json.loads(store.objects[MANIFEST])["revision"] == "external-change"
    assert json.loads(store.objects[MANIFEST])["gc_lock"]


def test_concurrent_publisher_prevents_lock_acquisition_and_deletion():
    class RacingStore(Store):
        def put_object(self, **kwargs):
            candidate = json.loads(kwargs["Body"])
            if candidate.get("gc_lock"):
                newer = json.loads(self.objects[MANIFEST])
                newer["revision"] = "concurrent-publication"
                self.objects[MANIFEST] = json.dumps(newer).encode()
            return super().put_object(**kwargs)

    store = RacingStore()
    with pytest.raises(ClientError, match="PreconditionFailed"):
        gc.collect(store, "bucket", "models", "QB", execute=True, now=NOW)
    assert store.deletes == []
    assert json.loads(store.objects[MANIFEST])["revision"] == "concurrent-publication"


def test_crashed_lock_requires_exact_token_and_confirmed_stopped_collector():
    store = Store()
    manifest = json.loads(store.objects[MANIFEST])
    manifest["gc_lock"] = {"owner": "crashed-collector", "acquired_at": "yesterday"}
    store.objects[MANIFEST] = json.dumps(manifest).encode()
    with pytest.raises(model_sync.ManifestLockedError):
        gc.collect(store, "bucket", "models", "QB", execute=True, now=NOW)
    with pytest.raises(DatasetError, match="Confirm"):
        gc.recover_lock(store, "bucket", "models", "QB", "crashed-collector")
    with pytest.raises(DatasetError, match="token"):
        gc.recover_lock(store, "bucket", "models", "QB", "wrong", collector_stopped=True)
    before = store.etag(MANIFEST)
    gc.recover_lock(store, "bucket", "models", "QB", "crashed-collector", collector_stopped=True)
    assert not json.loads(store.objects[MANIFEST]).get("gc_lock")
    assert store.etag(MANIFEST) != before
