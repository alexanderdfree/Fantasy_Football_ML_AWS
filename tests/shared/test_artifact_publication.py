"""Compatibility and migration from the protected predecessor publication protocol.

Current source/intent/CAS and coordinated retention interleavings are covered by
artifacts/test_publication.py and test_gc.py. These cases exercise the actual
old namespace, its source high-water mark, and the intentionally read-only facade.
"""

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from src.artifacts import model_sync, source
from src.artifacts import publication as current
from src.scripts import promote
from src.shared import artifact_publication as publication
from src.shared.registry import ALL_POSITIONS
from tests.artifacts.test_publication import Store

pytestmark = pytest.mark.unit
OLD, NEW, LATEST = "a" * 40, "b" * 40, "c" * 40


class S3(Store):
    def head_object(self, Bucket, Key):
        obj = self.get_object(Bucket, Key)
        return {"ContentLength": len(obj["Body"].read()), "ETag": obj["ETag"]}


def record(sha):
    lineage = [LATEST, NEW, OLD]
    lineage = lineage[lineage.index(sha) :]
    return {"source_sha": sha, "source_order": len(lineage), "lineage": lineage}


@pytest.fixture
def boundary(monkeypatch):
    s3 = S3()
    for sha in (OLD, NEW, LATEST):
        s3.objects[source.source_key("models", sha)] = json.dumps(record(sha)).encode()
    actual = [LATEST]
    monkeypatch.setattr(source, "image_source_sha", lambda **_: actual[0])
    monkeypatch.setattr(current, "image_source_sha", lambda: actual[0])
    monkeypatch.delenv("FF_TRAIN_GIT_SHA", raising=False)
    return s3, actual


def predecessor(s3, position="QB", *, sha=OLD, frontier=NEW, rollback=False, stable=True):
    key = f"models/{position}/releases/history/2026-09-10-unique-{sha[:7]}/model.tar.gz"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        body = json.dumps({"git_sha": sha, "position": position}).encode()
        member = tarfile.TarInfo("benchmark_metrics.json")
        member.size = len(body)
        tar.addfile(member, io.BytesIO(body))
    s3.objects[key] = buffer.getvalue()
    entry = {"key": key, "bytes": len(buffer.getvalue()), "sha7": sha[:7]}
    manifest = {
        "schema_version": 3,
        "current": entry,
        "stable": entry if stable else None,
        "history": [key],
        "publication_source": {
            name: record(frontier)[name] for name in ("source_sha", "source_order")
        },
    }
    if rollback:
        manifest["promotion_mode"] = "rollback"
    s3.objects[model_sync.previous_protocol_manifest_key("models", position)] = json.dumps(
        manifest
    ).encode()
    return manifest


def candidate(s3, position, sha, run_id):
    intent = current.reserve_intent(
        s3,
        "b",
        "models",
        position,
        sha,
        None,
        run_id,
        publication_revision=(
            model_sync.load_publication_manifest(s3, "b", "models", position) or {}
        ).get("rollback_epoch"),
    )
    entry = {
        "key": model_sync.new_history_key("models", position, run_id, "d" * 64),
        "git_sha": sha,
        "publication_intent": intent,
        "dataset_id": None,
        "sha7": "d" * 7,
        "bytes": 1,
        "uploaded_at": "today",
        "smoke_passed": True,
    }
    s3.objects[entry["key"]] = b"new candidate"
    context = {"source": record(sha), "intent": intent}
    return entry, context, intent["publication_revision"]


def publish(s3, position, prepared):
    entry, context, epoch = prepared
    return current.publish_candidate(
        s3, "b", "models", position, entry=entry, context=context, initial_revision=epoch
    )


@pytest.mark.parametrize("position", ALL_POSITIONS)
def test_main_protocol_precedes_v2_and_preserves_frontier_and_copied_bytes(boundary, position):
    s3, _ = boundary
    old = predecessor(s3, position)
    s3.objects[model_sync.legacy_manifest_key("models", position)] = b'{"stable": null}'
    assert model_sync.load_manifest(s3, "b", "models", position)["stable"] == old["stable"]
    migrated = publish(s3, position, candidate(s3, position, LATEST, "new-run"))
    protected = migrated["previous_stable"]["key"]
    assert protected.startswith(model_sync.history_prefix("models", position))
    assert s3.objects[protected] == s3.objects[old["stable"]["key"]]
    assert migrated["source_frontier"]["source_sha"] == LATEST
    # Queued main writers/GC can mutate only their independent namespace.
    s3.objects[model_sync.previous_protocol_manifest_key("models", position)] = b'{"stable": null}'
    s3.objects.pop(old["stable"]["key"])
    assert model_sync.load_manifest(s3, "b", "models", position) == migrated
    assert protected in s3.objects


def test_main_rollback_high_water_is_not_replaced_by_older_artifact_source(boundary):
    s3, _ = boundary
    predecessor(s3, rollback=True)
    migrated = current.protect_legacy(s3, "b", "models", "QB", record(LATEST))
    assert migrated["source_frontier"] == {"source_sha": NEW, "source_order": 2}
    assert migrated["rollback_source_barrier"] == migrated["source_frontier"]
    assert (
        migrated["rollback_epoch"]
        == model_sync.load_publication_manifest(s3, "b", "models", "QB")["rollback_epoch"]
    )
    assert publish(s3, "QB", candidate(s3, "QB", NEW, "same-source")) is None
    assert publish(s3, "QB", candidate(s3, "QB", LATEST, "descendant")) is not None


def test_precompute_epoch_rejects_work_queued_before_main_rollback(boundary, monkeypatch):
    s3, _ = boundary
    predecessor(s3)
    monkeypatch.setenv("FF_LEGACY_RUN_ID", "queued")
    context = current.prepare_training(s3, "b", "models", "QB")
    entry, _, _ = candidate(s3, "QB", LATEST, "queued")
    predecessor(s3, rollback=True)
    assert (
        current.publish_candidate(
            s3,
            "b",
            "models",
            "QB",
            entry=entry,
            context=context,
            initial_revision=context["initial_revision"],
        )
        is None
    )
    monkeypatch.setenv("FF_LEGACY_RUN_ID", "after-rollback")
    after = current.prepare_training(s3, "b", "models", "QB")
    assert after["initial_revision"].startswith("previous-protocol:")
    assert (
        after["initial_revision"]
        == current.protect_legacy(s3, "b", "models", "QB", record(LATEST))["rollback_epoch"]
    )


@pytest.mark.parametrize("failure", ["AccessDenied", "corrupt", "null"])
def test_broken_main_pointer_never_falls_back_to_stale_v2(boundary, monkeypatch, failure):
    s3, _ = boundary
    key = model_sync.previous_protocol_manifest_key("models", "QB")
    s3.objects[model_sync.legacy_manifest_key("models", "QB")] = b'{"stable": {"key": "stale"}}'
    if failure == "AccessDenied":
        original = s3.get_object

        def get(Bucket, Key):
            if Key == key:
                raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
            return original(Bucket, Key)

        monkeypatch.setattr(s3, "get_object", get)
    else:
        s3.objects[key] = b"invalid JSON" if failure == "corrupt" else b"null"
    with pytest.raises((ClientError, RuntimeError, ValueError)):
        model_sync.load_manifest(s3, "b", "models", "QB")


def test_main_candidate_is_not_silently_approved(boundary, tmp_path):
    s3, _ = boundary
    predecessor(s3, stable=False)
    with pytest.raises(RuntimeError, match="all manifest entries failed"):
        model_sync._resolve_manifest_extract(s3, "b", "models", "QB", tmp_path)


def test_manual_migration_dry_run_is_read_only_and_adopts_epoch_semantics(boundary, monkeypatch):
    from src.shared import smoke_test

    s3, _ = boundary
    old = predecessor(s3, rollback=True)
    monkeypatch.setattr(smoke_test, "run_smoke_test", lambda *_: None)
    before = dict(s3.objects)
    preview = promote.promote(s3, "b", "models", "QB", old["stable"]["key"], dry_run=True)
    assert s3.objects == before
    assert "rollback_source_barrier" not in preview
    committed = promote.promote(s3, "b", "models", "QB", old["stable"]["key"])
    assert (
        committed["rollback_epoch"]
        != model_sync.load_legacy_manifest(s3, "b", "models", "QB")["rollback_epoch"]
    )
    assert committed["source_frontier"]["source_sha"] == NEW


def test_legacy_writer_facade_cannot_bypass_intents_or_make_writes(boundary):
    s3, _ = boundary
    before = dict(s3.objects)
    with pytest.raises(RuntimeError, match="no pre-training intent or canonical receipt"):
        publication.publish_artifact(
            s3, "b", "models", "QB", source=record(LATEST), initialize_only=True
        )
    assert s3.objects == before
    assert publication.register_source is source.register_source
    assert publication.load_source is source.load_source
    assert publication.snapshot(s3, "b", "models", "QB") == (None, None)


def test_source_registration_uses_requested_revision_not_checkout_head(tmp_path):
    repo = tmp_path / "git"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    commits = []
    for n in range(3):
        (repo / "file").write_text(str(n))
        git("add", "file")
        git("commit", "-qm", str(n))
        commits.append(git("rev-parse", "HEAD"))
    git("update-ref", "refs/remotes/origin/main", commits[-1])
    s3 = S3()
    old = publication.register_source(s3, "b", "models", commits[0], str(repo))
    new = publication.register_source(s3, "b", "models", commits[-1], str(repo))
    assert old["source_order"] == 1
    assert new["source_order"] == 3
    assert publication.register_source(s3, "b", "models", commits[0], str(repo)) == old


def test_main_rollback_during_copy_is_rechecked_before_cutover(boundary, monkeypatch):
    s3, _ = boundary
    predecessor(s3)
    prepared = candidate(s3, "QB", LATEST, "queued-before-rollback")
    put = s3.put_object
    rolled_back = []

    def interleaved_put(**kwargs):
        result = put(**kwargs)
        if "/history/legacy-" in kwargs["Key"] and not rolled_back:
            rolled_back.append(predecessor(s3, rollback=True))
        return result

    monkeypatch.setattr(s3, "put_object", interleaved_put)
    assert publish(s3, "QB", prepared) is None
    assert rolled_back
    assert model_sync.manifest_key("models", "QB") not in s3.objects
