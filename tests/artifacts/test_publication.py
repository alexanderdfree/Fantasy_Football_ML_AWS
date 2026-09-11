"""Adversarial source/intent ordering, accepted-output retries and legacy isolation."""

import hashlib
import io
import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from src.artifacts import model_sync, publication, receipts, source
from src.artifacts.intent import reserve_intent
from src.orchestration.datasets import canonical_bytes, content_id

pytestmark = pytest.mark.unit
OLDER, NEWER = "a" * 40, "b" * 40


class Store:
    def __init__(self):
        self.objects, self.metadata, self.deleted = {}, {}, []

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body = self.objects[Key]
        return {"Body": io.BytesIO(body), "ETag": hashlib.sha256(body).hexdigest()}

    def put_object(self, Bucket, Key, Body, IfMatch=None, IfNoneMatch=None, **kwargs):
        prior = self.objects.get(Key)
        if (
            IfMatch is not None and (prior is None or hashlib.sha256(prior).hexdigest() != IfMatch)
        ) or (IfNoneMatch == "*" and prior is not None):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body
        self.metadata[Key] = kwargs.get("Metadata", {})

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.objects[key] = Path(path).read_bytes()
        self.metadata[key] = (ExtraArgs or {}).get("Metadata", {})

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.get_object(bucket, key)["Body"].read())

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        store = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                yield {
                    "Contents": [
                        {"Key": key} for key in list(store.objects) if key.startswith(Prefix)
                    ]
                }

        return Paginator()

    def delete_objects(self, Bucket, Delete):
        for entry in Delete["Objects"]:
            self.objects.pop(entry["Key"], None)
            self.deleted.append(entry["Key"])
        return {}


@pytest.fixture
def world(monkeypatch):
    from src.batch import train

    s3 = Store()
    actual = [OLDER]
    monkeypatch.setattr(source, "image_source_sha", lambda: actual[0])
    monkeypatch.setattr(publication, "image_source_sha", lambda: actual[0])
    monkeypatch.setattr(train.boto3, "client", lambda *_: s3)
    monkeypatch.setattr(train, "_try_smoke_test", lambda *_: True)
    monkeypatch.delenv("FF_MODEL_S3_PREFIX", raising=False)
    for sha, lineage in ((OLDER, [OLDER]), (NEWER, [NEWER, OLDER])):
        s3.objects[source.source_key("models", sha)] = canonical_bytes(
            {
                "source_sha": sha,
                "source_order": len(lineage),
                "lineage": lineage,
            }
        )
    return s3, actual


def prepare_run(s3, sha, run_id, *, dataset_id=None, epoch=None):
    intent = reserve_intent(
        s3, "bucket", "models", "RB", sha, dataset_id, run_id, publication_revision=epoch
    )
    metrics = {
        "position": "RB",
        "git_sha": sha,
        "publication_intent": intent,
        "publication_revision": epoch,
    }
    if dataset_id is not None:
        plan = {
            "schema_version": 1,
            "git_sha": sha,
            "dataset_id": dataset_id,
            "source_id": "c" * 64,
            "model_prefix": "models",
            "run_id": run_id,
            "positions": ["RB"],
            "seed": 42,
            "intents": {"RB": intent},
            "publication_revisions": {"RB": epoch},
            "job_definitions": {},
        }
        plan_id = content_id(plan)
        s3.objects[f"build-plans/{plan_id}.json"] = canonical_bytes(plan)
        metrics.update(dataset_id=dataset_id, build_plan_id=plan_id)
    return metrics


def test_standalone_publication_intent_binds_selected_release(world, monkeypatch):
    s3, actual = world
    release = "e" * 64
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", actual[0])
    monkeypatch.setenv("FF_LEGACY_RUN_ID", "standalone-release")
    monkeypatch.setenv("FF_DATA_RELEASE", release)
    monkeypatch.setenv("FF_DATASET_ID", release)
    for name in ("FF_PUBLICATION_INTENT", "FF_PUBLICATION_REVISION"):
        monkeypatch.delenv(name, raising=False)
    context = publication.prepare_training(s3, "bucket", "models", "RB")
    assert context["intent"]["dataset_id"] == release
    assert context["intent"]["run_id"] == "standalone-release"


def test_standalone_conflicting_data_aliases_do_not_reserve_an_intent(world, monkeypatch):
    s3, actual = world
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", actual[0])
    monkeypatch.setenv("FF_DATA_RELEASE", "d" * 64)
    monkeypatch.setenv("FF_DATASET_ID", "e" * 64)
    before = dict(s3.objects)
    with pytest.raises(RuntimeError, match="equal canonical data aliases"):
        publication.prepare_training(s3, "bucket", "models", "RB")
    assert s3.objects == before


def test_remote_unidentified_data_requires_explicit_legacy_mode(world, monkeypatch):
    s3, actual = world
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", actual[0])
    monkeypatch.setenv("AWS_BATCH_JOB_ID", "unplanned")
    monkeypatch.delenv("FF_DATA_RELEASE", raising=False)
    monkeypatch.delenv("FF_DATASET_ID", raising=False)
    with pytest.raises(RuntimeError, match="explicit legacy mode"):
        publication.prepare_training(s3, "bucket", "models", "RB")
    monkeypatch.setenv("FF_DATA_RELEASE", "legacy")
    for name in ("FF_PUBLICATION_INTENT", "FF_PUBLICATION_REVISION"):
        monkeypatch.delenv(name, raising=False)
    assert (
        publication.prepare_training(s3, "bucket", "models", "RB")["intent"]["dataset_id"] is None
    )


def model_directory(root, metrics, label):
    from src.shared.registry import INFERENCE_REGISTRY

    directory = root / label
    directory.mkdir()
    registry = INFERENCE_REGISTRY["RB"]
    files = [registry["nn_file"], "nn_scaler.pkl", "nn_scaler_meta.json"]
    if registry.get("train_attention_nn"):
        files += [
            registry["attn_nn_file"],
            "attention_nn_scaler.pkl",
            "attention_nn_scaler_meta.json",
        ]
    for name in files:
        (directory / name).write_bytes(label.encode())
    (directory / "benchmark_metrics.json").write_text(json.dumps(metrics))
    return directory


def upload(world, tmp_path, metrics, label):
    from src.batch.train import upload_artifacts

    s3, actual = world
    actual[0] = metrics["git_sha"]
    directory = model_directory(tmp_path, metrics, label)
    upload_artifacts("bucket", "RB", str(directory))
    return directory


def current(s3):
    return json.loads(s3.objects[model_sync.manifest_key("models", "RB")])


def test_older_source_finishing_after_descendant_cannot_regress_stable(world, tmp_path):
    s3, _ = world
    old = prepare_run(s3, OLDER, "old-image")
    new = prepare_run(s3, NEWER, "new-image")
    upload(world, tmp_path, new, "new")
    winner = current(s3)["stable"]["key"]
    upload(world, tmp_path, old, "old")
    assert current(s3)["stable"]["key"] == winner
    assert current(s3)["source_frontier"]["source_sha"] == NEWER
    old_output = receipts.load_run_receipt(s3, "bucket", "models", OLDER, "RB", "old-image")
    assert old_output["key"] != winner


def test_same_source_older_dataset_intent_cannot_regress_stable(world, tmp_path):
    s3, _ = world
    old = prepare_run(s3, OLDER, "old-data", dataset_id="d" * 64)
    new = prepare_run(s3, OLDER, "new-data", dataset_id="e" * 64)
    upload(world, tmp_path, new, "new-data")
    winner = current(s3)["stable"]["key"]
    upload(world, tmp_path, old, "old-data")
    assert current(s3)["stable"]["key"] == winner
    assert current(s3)["intent_frontier"] == new["publication_intent"]
    assert receipts.load_receipt(s3, "bucket", old["build_plan_id"], "RB")["dataset_id"] == "d" * 64


def test_same_intent_retry_reuses_canonical_output_before_promoting(world, tmp_path):
    s3, _ = world
    metrics = prepare_run(s3, OLDER, "same-run", dataset_id="d" * 64)
    upload(world, tmp_path, metrics, "first-attempt")
    first = receipts.load_receipt(s3, "bucket", metrics["build_plan_id"], "RB")
    upload(world, tmp_path, metrics, "retry-different-tar-bytes")
    assert current(s3)["stable"]["key"] == first["key"]
    assert receipts.load_receipt(s3, "bucket", metrics["build_plan_id"], "RB") == first


def test_concurrent_same_intent_uploads_publish_only_the_first_claimed_output(
    world, tmp_path, monkeypatch
):
    s3, _ = world
    metrics = prepare_run(s3, OLDER, "concurrent-run", dataset_id="d" * 64)
    slot = receipts.receipt_key(metrics["build_plan_id"], "RB")
    original = s3.put_object
    entered = [False]

    def interleave(**kwargs):
        if kwargs["Key"] == slot and not entered[0]:
            entered[0] = True
            upload(world, tmp_path, metrics, "concurrent-winner")
        return original(**kwargs)

    monkeypatch.setattr(s3, "put_object", interleave)
    upload(world, tmp_path, metrics, "late-claim")
    canonical = receipts.load_receipt(s3, "bucket", metrics["build_plan_id"], "RB")
    assert current(s3)["stable"]["key"] == canonical["key"]
    assert len([key for key in s3.objects if key == slot]) == 1


def test_upload_rejects_forged_image_and_wrong_position_intent_before_writes(world, tmp_path):
    from src.batch.train import upload_artifacts

    s3, actual = world
    metrics = prepare_run(s3, NEWER, "declared-newer")
    actual[0] = OLDER
    directory = model_directory(tmp_path, metrics, "forged-image")
    with pytest.raises(RuntimeError, match="actual image|source SHA"):
        upload_artifacts("bucket", "RB", str(directory))
    assert not any(key.startswith(model_sync.history_prefix("models", "RB")) for key in s3.objects)
    wrong = reserve_intent(s3, "bucket", "models", "QB", OLDER, None, "wrong-position")
    metrics = {
        "position": "RB",
        "git_sha": OLDER,
        "publication_intent": wrong,
        "publication_revision": None,
    }
    directory = model_directory(tmp_path, metrics, "wrong-position")
    with pytest.raises(RuntimeError, match="matching pre-training"):
        upload_artifacts("bucket", "RB", str(directory))
    assert not any(key.startswith(model_sync.history_prefix("models", "RB")) for key in s3.objects)


def test_crash_after_output_claim_recovers_without_a_second_canonical_output(
    world, tmp_path, monkeypatch
):
    s3, _ = world
    metrics = prepare_run(s3, OLDER, "crash-run", dataset_id="d" * 64)
    original = publication.publish_candidate
    monkeypatch.setattr(
        publication,
        "publish_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("crashed")),
    )
    with pytest.raises(OSError, match="crashed"):
        upload(world, tmp_path, metrics, "claimed-before-crash")
    first = receipts.load_receipt(s3, "bucket", metrics["build_plan_id"], "RB")
    assert model_sync.manifest_key("models", "RB") not in s3.objects
    monkeypatch.setattr(publication, "publish_candidate", original)
    upload(world, tmp_path, metrics, "retry")
    assert current(s3)["stable"]["key"] == first["key"]


def test_failed_smoke_cannot_poison_a_later_successful_retry(world, tmp_path, monkeypatch):
    from src.batch import train

    s3, _ = world
    metrics = prepare_run(s3, OLDER, "smoke-retry", dataset_id="d" * 64)
    monkeypatch.setattr(train, "_try_smoke_test", lambda *_: False)
    upload(world, tmp_path, metrics, "failed-smoke")
    assert receipts.receipt_key(metrics["build_plan_id"], "RB") not in s3.objects
    assert current(s3)["stable"] is None
    monkeypatch.setattr(train, "_try_smoke_test", lambda *_: True)
    upload(world, tmp_path, metrics, "passed-smoke")
    assert (
        receipts.load_receipt(s3, "bucket", metrics["build_plan_id"], "RB")["key"]
        == current(s3)["stable"]["key"]
    )


def test_corrupt_canonical_output_is_not_promoted_by_retry(world, tmp_path, monkeypatch):
    s3, _ = world
    metrics = prepare_run(s3, OLDER, "corrupt-output", dataset_id="d" * 64)
    original = publication.publish_candidate
    monkeypatch.setattr(
        publication,
        "publish_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("crashed")),
    )
    with pytest.raises(OSError):
        upload(world, tmp_path, metrics, "first")
    receipt = receipts.load_receipt(s3, "bucket", metrics["build_plan_id"], "RB")
    s3.objects[receipt["key"]] = b"corrupt"
    monkeypatch.setattr(publication, "publish_candidate", original)
    with pytest.raises(RuntimeError, match="checksum"):
        upload(world, tmp_path, metrics, "retry")
    assert model_sync.manifest_key("models", "RB") not in s3.objects


def test_authenticated_other_plan_intent_cannot_poison_output_slot(world, tmp_path):
    s3, _ = world
    first = prepare_run(s3, OLDER, "first-plan", dataset_id="d" * 64)
    second = prepare_run(s3, OLDER, "second-plan", dataset_id="e" * 64)
    upload(world, tmp_path, second, "second-output")
    entry = receipts.load_receipt(s3, "bucket", second["build_plan_id"], "RB")
    entry["build_plan_id"] = first["build_plan_id"]
    with pytest.raises(RuntimeError, match="receipt"):
        receipts.claim_successful_output(s3, "bucket", "models", "RB", entry)
    assert receipts.receipt_key(first["build_plan_id"], "RB") not in s3.objects


def test_actual_v2_writer_and_pruner_cannot_mutate_v3_or_its_migrated_backup(world, tmp_path):
    from tests.artifacts import _legacy_v2_gc as legacy
    from tests.shared._helpers import make_tarball

    s3, _ = world
    legacy_key = "models/RB/history/old/model.tar.gz"
    s3.objects[legacy_key] = make_tarball(
        {
            "benchmark_metrics.json": json.dumps({"position": "RB", "git_sha": OLDER}).encode(),
            "legacy-marker": b"approved older model",
        }
    )
    legacy_manifest = {
        "schema_version": 2,
        "stable": {"key": legacy_key},
        "current": {"key": legacy_key},
        "history": [legacy_key],
        "previous": None,
    }
    legacy.write_manifest(s3, "bucket", "models", "RB", legacy_manifest)
    new = prepare_run(s3, NEWER, "cutover")
    upload(world, tmp_path, new, "protected")
    protected = current(s3)
    backup_key = protected["previous_stable"]["key"]
    assert backup_key.startswith(model_sync.history_prefix("models", "RB"))
    assert backup_key != legacy_key
    # A queued old image knows only its original physical namespace.
    late = {"schema_version": 2, "current": None, "stable": None, "previous": None, "history": []}
    legacy.write_manifest(s3, "bucket", "models", "RB", late)
    legacy.prune(s3, "bucket", "models", "RB", late)
    assert legacy_key in s3.deleted
    assert protected["stable"]["key"] in s3.objects
    assert backup_key in s3.objects
    assert model_sync.load_manifest(s3, "bucket", "models", "RB") == protected
    # The approved fallback is physically safe even after v2 deletes the original.
    s3.objects[protected["stable"]["key"]] = b"corrupt latest bytes"
    result = model_sync._sync_one(s3, "bucket", "models", "RB", tmp_path / "consumer")
    assert result["key"] == backup_key
    assert (
        tmp_path / "consumer/src/rb/outputs/models/legacy-marker"
    ).read_bytes() == b"approved older model"


def test_rollback_epoch_fences_old_work_but_gc_revision_does_not_fence_new_intent(world, tmp_path):
    s3, _ = world
    first = prepare_run(s3, OLDER, "before-rollback")
    upload(world, tmp_path, first, "first")
    before = current(s3)
    rolled = {**before, "rollback_epoch": "operator-rollback", "promotion_mode": "rollback"}
    etag = s3.get_object(Bucket="bucket", Key=model_sync.manifest_key("models", "RB"))["ETag"]
    model_sync.write_manifest(s3, "bucket", "models", "RB", rolled, expected_etag=etag)
    # Ordinary CAS/GC nonce churn is independent of the manual rollback epoch.
    after = prepare_run(s3, OLDER, "after-rollback", epoch="operator-rollback")
    rolled = current(s3)
    etag = s3.get_object(Bucket="bucket", Key=model_sync.manifest_key("models", "RB"))["ETag"]
    model_sync.write_manifest(s3, "bucket", "models", "RB", rolled, expected_etag=etag)
    upload(world, tmp_path, after, "after-gc")
    winner = current(s3)["stable"]["key"]
    assert current(s3)["rollback_epoch"] == "operator-rollback"
    upload(world, tmp_path, first, "old-retry")
    assert current(s3)["stable"]["key"] == winner
