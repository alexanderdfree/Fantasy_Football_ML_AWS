"""Exercise publication callers with real tarballs and conditional S3 semantics."""

import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from src.batch import train
from src.scripts import promote
from src.shared import artifact_publication as publication
from src.shared.artifact_gc import prune
from src.shared.model_sync import load_manifest, manifest_key
from src.shared.registry import ALL_POSITIONS, INFERENCE_REGISTRY

pytestmark = pytest.mark.unit
OLD, NEW, LATEST = "a" * 40, "b" * 40, "c" * 40


class S3:
    def __init__(self):
        self.objects = {}
        self.before_put = None
        self.deletes = []

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        data = self.objects[Key]
        return {"Body": io.BytesIO(data), "ETag": hashlib.sha256(data).hexdigest()}

    def put_object(self, Bucket, Key, Body, **kwargs):
        if self.before_put and Key.endswith("manifest.json"):
            callback, self.before_put = self.before_put, None
            callback()
        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if "IfMatch" in kwargs and self.get_object(Bucket, Key)["ETag"] != kwargs["IfMatch"]:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.read()
        return {}

    def upload_file(self, Filename, Bucket, Key):
        self.objects[Key] = Path(Filename).read_bytes()

    def delete_objects(self, Bucket, Delete):
        for entry in Delete["Objects"]:
            self.objects.pop(entry["Key"], None)
            self.deletes.append(entry["Key"])
        return {}

    def head_object(self, Bucket, Key):
        obj = self.get_object(Bucket, Key)
        return {"ContentLength": len(obj["Body"].read()), "ETag": obj["ETag"]}


def source(sha):
    lineage = [LATEST, NEW, OLD]
    lineage = lineage[lineage.index(sha) :]
    return {"source_sha": sha, "source_order": len(lineage), "lineage": lineage}


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    s3 = S3()
    for sha in (OLD, NEW, LATEST):
        s3.objects[publication.source_key("models", sha)] = json.dumps(source(sha)).encode()
    monkeypatch.setattr(train.boto3, "client", lambda *_a, **_kw: s3)
    monkeypatch.setattr(publication, "image_source_sha", lambda: os.environ["FF_TRAIN_GIT_SHA"])
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", NEW)
    monkeypatch.delenv("FF_MODEL_S3_PREFIX", raising=False)
    monkeypatch.setattr(train, "_try_smoke_test", lambda *_: True)
    return s3, tmp_path


def model_dir(tmp_path, position, sha=NEW, tag=""):
    directory = tmp_path / f"{position}-{sha}-{tag}"
    directory.mkdir(exist_ok=True)
    reg = INFERENCE_REGISTRY[position]
    for name in (reg["nn_file"], "nn_scaler.pkl", "nn_scaler_meta.json"):
        (directory / name).write_bytes(b"fixture")
    if reg.get("train_attention_nn") and reg.get("attn_nn_file"):
        for name in (
            reg["attn_nn_file"],
            "attention_nn_scaler.pkl",
            "attention_nn_scaler_meta.json",
        ):
            (directory / name).write_bytes(b"fixture")
    (directory / "benchmark_metrics.json").write_text(json.dumps({"git_sha": sha, "tag": tag}))
    return str(directory)


def upload(boundary, monkeypatch, position="QB", sha=NEW, tag="", **kwargs):
    _s3, tmp = boundary
    with monkeypatch.context() as mp:
        mp.setenv("FF_TRAIN_GIT_SHA", sha)
        return train.upload_artifacts("b", position, model_dir(tmp, position, sha, tag), **kwargs)


@pytest.mark.parametrize("position", ALL_POSITIONS)
def test_actual_upload_rejects_older_source_for_every_position(boundary, monkeypatch, position):
    s3, _ = boundary
    newest = upload(boundary, monkeypatch, position, NEW)
    with pytest.raises(publication.PublicationSuperseded):
        upload(boundary, monkeypatch, position, OLD)
    assert load_manifest(s3, "b", "models", position) == newest
    assert newest["stable"] == newest["current"]
    assert "releases/history/" in newest["stable"]["key"]


def test_cas_conflict_rechecks_order_instead_of_overwriting(boundary, monkeypatch):
    s3, _ = boundary
    upload(boundary, monkeypatch, sha=OLD)
    s3.before_put = lambda: upload(boundary, monkeypatch, sha=LATEST)
    with pytest.raises(publication.PublicationSuperseded):
        upload(boundary, monkeypatch, sha=NEW)
    assert load_manifest(s3, "b", "models", "QB")["publication_source"]["source_sha"] == LATEST


def test_initial_creation_conflict_cannot_overwrite_newer_winner(boundary, monkeypatch):
    s3, _ = boundary
    s3.before_put = lambda: upload(boundary, monkeypatch, sha=LATEST)
    with pytest.raises(publication.PublicationSuperseded):
        upload(boundary, monkeypatch, sha=NEW)
    assert load_manifest(s3, "b", "models", "QB")["publication_source"]["source_sha"] == LATEST


def test_delayed_cleanup_preserves_later_promotion_and_unpublished_upload(boundary, monkeypatch):
    s3, _ = boundary
    with monkeypatch.context() as mp:
        mp.setattr(train, "_gc_prune", lambda *_: [])
        for n in range(7):
            stale = upload(boundary, monkeypatch, tag=str(n))
        latest = upload(boundary, monkeypatch, tag="latest")
    pending = "models/QB/releases/history/pending/model.tar.gz"
    s3.objects[pending] = b"in-flight"
    assert stale["retired"]
    prune(s3, "b", "models", "QB", stale)
    assert latest["stable"]["key"] in s3.objects
    assert pending in s3.objects
    assert all(key in s3.objects for key in publication.references(latest))


def test_failed_smoke_keeps_stable_across_retention_window(boundary, monkeypatch):
    s3, _ = boundary
    good = upload(boundary, monkeypatch, sha=OLD)
    monkeypatch.setattr(train, "_try_smoke_test", lambda *_: False)
    for n in range(8):
        new = upload(boundary, monkeypatch, tag=str(n))
    assert new["stable"] == good["stable"]
    assert new["stable"]["key"] in s3.objects


def test_initial_seed_requires_smoke_and_cannot_race_training(boundary, monkeypatch):
    s3, _ = boundary
    monkeypatch.setattr(train, "_try_smoke_test", lambda *_: False)
    with pytest.raises(RuntimeError, match="smoke test failed"):
        upload(boundary, monkeypatch, initialize_only=True)
    assert load_manifest(s3, "b", "models", "QB") is None
    monkeypatch.setattr(train, "_try_smoke_test", lambda *_: True)
    s3.before_put = lambda: upload(boundary, monkeypatch, sha=LATEST)
    assert upload(boundary, monkeypatch, initialize_only=True) is None


def test_rollback_uses_fresh_bytes_preserves_high_water_and_survives_delayed_gc(
    boundary, monkeypatch
):
    s3, _ = boundary
    older = upload(boundary, monkeypatch, sha=OLD)
    newer = upload(boundary, monkeypatch, sha=NEW)
    rolled = promote.promote(s3, "b", "models", "QB", older["stable"]["key"])
    assert rolled["stable"]["key"] != older["stable"]["key"]
    assert rolled["stable"] == rolled["current"]
    assert rolled["publication_source"] == newer["publication_source"]
    with pytest.raises(publication.PublicationSuperseded):
        upload(boundary, monkeypatch, sha=OLD)
    with pytest.raises(publication.PublicationSuperseded):
        upload(boundary, monkeypatch, sha=NEW)
    assert load_manifest(s3, "b", "models", "QB") == rolled
    assert upload(boundary, monkeypatch, sha=LATEST) is not None


def legacy_manifest(s3, sha=OLD, version=2):
    key = "models/QB/history/2026-09-10T00-00-00Z-abc1234/model.tar.gz"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        data = json.dumps({"git_sha": sha}).encode()
        entry = tarfile.TarInfo("benchmark_metrics.json")
        entry.size = len(data)
        tar.addfile(entry, io.BytesIO(data))
    s3.objects[key] = stream.getvalue()
    entry = {"key": key, "bytes": len(stream.getvalue()), "sha7": "abc1234"}
    manifest = {"schema_version": version, "current": entry, "previous": None, "history": [key]}
    if version == 2:
        manifest["stable"] = entry
    s3.objects["models/QB/manifest.json"] = json.dumps(manifest).encode()
    return manifest


@pytest.mark.parametrize("version", [1, 2])
def test_migration_protects_legacy_fallback_from_old_image_gc(boundary, monkeypatch, version):
    s3, _ = boundary
    legacy = legacy_manifest(s3, version=version)
    monkeypatch.setattr(train, "_try_smoke_test", lambda *_: False)
    migrated = upload(boundary, monkeypatch)
    # Old binaries may still overwrite their pointer and sweep all legacy history.
    s3.objects["models/QB/manifest.json"] = b'{"current": null}'
    s3.objects.pop(legacy["current"]["key"])
    assert load_manifest(s3, "b", "models", "QB") == migrated
    assert migrated["stable"]["key"] in s3.objects
    assert migrated["stable"]["key"] != legacy["current"]["key"]


@pytest.mark.parametrize("sha", [None, LATEST])
def test_missing_or_newer_legacy_provenance_refuses_migration(boundary, monkeypatch, sha):
    s3, _ = boundary
    legacy_manifest(s3, sha=sha)
    with pytest.raises(RuntimeError, match="Cannot migrate"):
        upload(boundary, monkeypatch)
    assert manifest_key("models", "QB") not in s3.objects


def test_actual_image_sha_must_match_supplied_source(boundary, monkeypatch):
    monkeypatch.setattr(publication, "image_source_sha", lambda: OLD)
    with pytest.raises(RuntimeError, match="actual image"):
        upload(boundary, monkeypatch)


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


def test_first_poller_observation_compares_actual_boot_manifest(boundary, monkeypatch):
    from src.shared import model_sync

    s3, tmp = boundary
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "b")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp)
    legacy_manifest(s3)
    boot = model_sync._sync_one(s3, "b", "models", "QB", tmp)
    assert boot["manifest_etag"]
    upload(boundary, monkeypatch)
    etag, refreshed = model_sync.refresh_position("QB", None, s3_client=s3)
    assert refreshed
    metrics = tmp / "src/qb/outputs/models/benchmark_metrics.json"
    assert json.loads(metrics.read_text())["git_sha"] == NEW
    assert model_sync.refresh_position("QB", etag, s3_client=s3) == (etag, False)


@pytest.mark.parametrize("split", [False, True])
def test_launcher_registers_actual_source_before_submitting_all_positions(monkeypatch, split):
    from src.batch import launch

    calls = []
    registered = False

    def register(_s3, bucket, prefix, sha):
        nonlocal registered
        assert sha == NEW
        registered = True

    class Batch:
        def submit_job(self, **kwargs):
            assert registered
            env = {
                entry["name"]: entry["value"]
                for entry in kwargs["containerOverrides"]["environment"]
            }
            assert env["FF_TRAIN_GIT_SHA"] == NEW
            calls.append(kwargs)
            return {"jobId": str(len(calls))}

    monkeypatch.setattr(publication, "register_source", register)
    monkeypatch.setattr(launch, "TRAIN_GIT_SHA", NEW)
    monkeypatch.setattr(launch, "JOB_IDS_FILE", None)
    monkeypatch.setattr(launch, "JOB_DEFINITION_CPU", "cpu-def")
    monkeypatch.setattr(launch, "JOB_DEFINITION_REVISION", "100")
    monkeypatch.setattr(launch, "JOB_DEFINITION_CPU_REVISION", "101")
    monkeypatch.setattr(launch, "JOB_QUEUE_CPU", "cpu-queue")
    monkeypatch.setattr(
        launch.boto3, "client", lambda service, **_: Batch() if service == "batch" else S3()
    )
    args = ["launch", "--positions", *ALL_POSITIONS, "--skip-upload", "--wait", "false"]
    if split:
        args += ["--split", "--split-run-id", "test-run"]
    monkeypatch.setattr("sys.argv", args)
    launch.main()
    assert len(calls) == len(ALL_POSITIONS) * (3 if split else 1)


def test_training_workflows_supply_immutable_source_identity():
    import yaml

    root = Path(__file__).resolve().parents[2]
    build = yaml.safe_load((root / ".github/workflows/batch-image.yml").read_text())
    steps = [step for job in build["jobs"].values() for step in job.get("steps", [])]
    image_step = next(step for step in steps if step.get("name") == "Build & push training image")
    assert '--build-arg TRAIN_GIT_SHA="$IMAGE_TAG"' in image_step["run"]
    assert image_step["env"]["IMAGE_TAG"] == "${{ github.sha }}"
    ec2 = yaml.safe_load((root / ".github/workflows/train-ec2.yml").read_text())
    steps = ec2["jobs"]["train"]["steps"]
    step = next(s for s in steps if s.get("name") == "Run training for all positions (sequential)")
    assert step["run"].index("register_training_source") < step["run"].index("REMOTE_CMD=")
    assert "workflow_run.head_sha" in step["env"]["FF_TRAIN_GIT_SHA"]
    runner = (root / "infra/ec2/user-data.sh").read_text()
    assert 'IMAGE="\\${IMAGE%:*}:\\$FF_TRAIN_GIT_SHA"' in runner
    assert 'imageTag="\\${FF_TRAIN_GIT_SHA:-latest}"' in runner


def test_same_source_cas_retry_preserves_other_publisher_in_history(boundary, monkeypatch):
    s3, _ = boundary
    upload(boundary, monkeypatch, sha=OLD)
    winner = {}

    def race():
        winner.update(upload(boundary, monkeypatch, sha=NEW, tag="winner"))

    s3.before_put = race
    final = upload(boundary, monkeypatch, sha=NEW, tag="retry")
    assert final["previous"] == winner["current"]
    assert winner["current"]["key"] in final["history"]
    assert final["current"]["key"] != winner["current"]["key"]


def test_rollback_conflict_cannot_resurrect_a_retired_target(boundary, monkeypatch):
    s3, _ = boundary
    first = upload(boundary, monkeypatch, tag="first")
    target = first["current"]["key"]

    def retire_target():
        for index in range(6):
            upload(boundary, monkeypatch, tag=f"new-{index}")

    s3.before_put = retire_target
    with pytest.raises(promote.PromotionError, match="not in manifest.history"):
        promote.promote(s3, "b", "models", "QB", target)
    assert target not in s3.objects
    assert load_manifest(s3, "b", "models", "QB")["stable"]["key"] != target


def test_v3_rollback_timestamp_excludes_uniqueness_token():
    key = f"models/QB/releases/history/2026-09-10T12-00-00Z-{'a' * 32}-b123456/model.tar.gz"
    assert promote._parse_version_from_key(key) == ("2026-09-10T12-00-00Z", "b123456")
