"""Bootstrap IAM reconciliation and manifest-based seeding contracts."""

from __future__ import annotations

import fnmatch
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from src.scripts import seed_s3_models as seed
from src.shared import model_sync
from tests.artifacts.test_publication import Store

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


class FakeS3(Store):
    def __init__(self):
        super().__init__()
        self.reads = []

    def get_object(self, *, Bucket, Key):
        self.reads.append(Key)
        return super().get_object(Bucket, Key)


def put_models(s3, position="QB", *, prefix="models", legacy_manifest=False):
    key = model_sync.new_history_key(prefix, position, "2026-09-10T00-00-00Z", "1234567")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = b'{"ready": true}'
        member = tarfile.TarInfo("benchmark_metrics.json")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    s3.objects[key] = buffer.getvalue()
    manifest = model_sync.build_manifest(
        key, "1234567", len(buffer.getvalue()), "today", smoke_passed=True
    )
    pointer = (
        f"{prefix}/{position}/manifest.json"
        if legacy_manifest
        else model_sync.manifest_key(prefix, position)
    )
    s3.objects[pointer] = json.dumps(manifest).encode()
    return key


def check_extracted_models(position, directory):
    assert json.loads((directory / "benchmark_metrics.json").read_text()) == {"ready": True}


def test_verify_uses_real_consumer_and_ignores_legacy_mirror(monkeypatch):
    s3 = FakeS3()
    expected = put_models(s3)
    s3.objects["models/QB/model.tar.gz"] = b"obsolete and corrupt"
    monkeypatch.setattr(seed, "_check_models", check_extracted_models)

    result = seed.verify_models(s3, "bucket", "models", ("QB",))

    assert result[0]["source"] == "stable"
    assert result[0]["key"] == expected
    assert "models/QB/model.tar.gz" not in s3.reads


def test_legacy_mirror_alone_is_not_boot_ready():
    s3 = FakeS3()
    s3.objects["models/QB/model.tar.gz"] = b"legacy tarball"
    with pytest.raises(RuntimeError, match="no manifest"):
        seed.verify_models(s3, "bucket", "models", ("QB",))


def test_manifest_pointing_to_missing_tarball_is_not_boot_ready():
    s3 = FakeS3()
    key = put_models(s3)
    del s3.objects[key]
    with pytest.raises(RuntimeError, match="all manifest entries failed"):
        seed.verify_models(s3, "bucket", "models", ("QB",))


def test_model_smoke_failure_prevents_readiness():
    s3 = FakeS3()
    # The archive extracts successfully, but has metrics and no model files.
    # The real CPU load/predict smoke test must reject it as boot-ready.
    put_models(s3)
    with pytest.raises(RuntimeError, match="QB ridge"):
        seed.verify_models(s3, "bucket", "models", ("QB",))


@pytest.mark.parametrize("metrics", [None, "[]", "not-json"])
def test_invalid_metrics_prevent_model_load(monkeypatch, tmp_path, metrics):
    from src.shared import smoke_test

    if metrics is not None:
        (tmp_path / "benchmark_metrics.json").write_text(metrics)
    monkeypatch.setattr(smoke_test, "run_smoke_test", lambda *_: pytest.fail("loaded invalid seed"))
    with pytest.raises((RuntimeError, json.JSONDecodeError)):
        seed._check_models("QB", tmp_path)


@pytest.mark.parametrize("legacy_manifest", [False, True])
def test_seed_preserves_existing_manifest_without_local_models(
    monkeypatch, tmp_path, legacy_manifest
):
    s3 = FakeS3()
    put_models(s3, legacy_manifest=legacy_manifest)
    before = dict(s3.objects)
    monkeypatch.setattr(seed, "_check_models", check_extracted_models)
    monkeypatch.setattr(
        seed, "_register_source", lambda *_: pytest.fail("registered existing seed")
    )
    monkeypatch.setattr(
        seed, "_upload_initial_artifact", lambda *_: pytest.fail("overwrote existing seed")
    )

    seed.seed_models(s3, "bucket", "models", tmp_path, "sha", ("QB",))

    assert s3.objects == before


def test_all_candidates_validate_before_first_s3_write(monkeypatch, tmp_path):
    validated = []

    def validate(position, _):
        validated.append(position)
        if position == "DST":
            raise RuntimeError("invalid DST model")

    monkeypatch.setattr(seed, "_check_models", validate)
    monkeypatch.setattr(
        seed, "_register_source", lambda *_: pytest.fail("registered before all validated")
    )
    monkeypatch.setattr(
        seed, "_upload_initial_artifact", lambda *_: pytest.fail("partial publication")
    )
    with pytest.raises(RuntimeError, match="invalid DST model"):
        seed.seed_models(FakeS3(), "bucket", "models", tmp_path, "sha")
    assert validated == list(model_sync.POSITIONS)


def pending_models(root, position="QB", sha="a" * 40):
    directory = root / "src" / position.lower() / "outputs/models"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "benchmark_metrics.json").write_text(
        json.dumps({"position": position, "git_sha": sha})
    )
    return directory


@pytest.fixture
def seed_world(monkeypatch):
    from src.artifacts import source
    from src.shared import smoke_test

    s3 = FakeS3()
    sha = "a" * 40
    monkeypatch.setattr(source, "image_source_sha", lambda **_: sha)
    monkeypatch.setattr(smoke_test, "run_smoke_test", lambda *_: None)

    def register(client, bucket, prefix, source_sha, root):
        assert client is s3 and source_sha == sha
        s3.objects[source.source_key(prefix, sha)] = json.dumps(
            {"source_sha": sha, "source_order": 1, "lineage": [sha]}
        ).encode()

    monkeypatch.setattr(seed, "_register_source", register)
    return s3, sha


def test_seed_initializes_then_verifies_without_mutating_environment(
    seed_world, monkeypatch, tmp_path
):
    s3, sha = seed_world
    pending_models(tmp_path)
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "original")
    monkeypatch.delenv("FF_TRAIN_GIT_SHA", raising=False)
    result = seed.seed_models(s3, "bucket", "custom", tmp_path, sha, ("QB",))
    assert result[0]["source"] == "stable"
    manifest = model_sync.load_manifest(s3, "bucket", "custom", "QB")
    assert manifest["stable"]["origin"] == "operator-seed"
    assert manifest["source_frontier"] == {"source_sha": sha, "source_order": 1}
    assert manifest["stable"]["key"].startswith("custom/releases/v3/QB/history/")
    assert not any("intents/" in key or "run-outputs/" in key for key in s3.objects)
    assert os.environ["FF_MODEL_S3_PREFIX"] == "original"
    assert "FF_TRAIN_GIT_SHA" not in os.environ


def test_all_captured_candidates_validate_before_source_registration(
    seed_world, monkeypatch, tmp_path
):
    s3, sha = seed_world
    pending_models(tmp_path, "QB")
    pending_models(tmp_path, "DST", sha="b" * 40)
    monkeypatch.setattr(
        seed, "_register_source", lambda *_: pytest.fail("registered invalid request")
    )
    with pytest.raises(RuntimeError, match="actual source and position"):
        seed.seed_models(s3, "bucket", "models", tmp_path, sha, ("QB", "DST"))
    assert not s3.objects


def test_initialize_seed_loses_create_race_without_overwriting_winner(
    seed_world, monkeypatch, tmp_path
):
    s3, sha = seed_world
    pending_models(tmp_path)
    put = s3.put_object
    winner = []

    def racing_put(**kwargs):
        if kwargs["Key"] == model_sync.manifest_key("models", "QB"):
            assert kwargs["IfNoneMatch"] == "*"
            winner.append(put_models(s3))
        return put(**kwargs)

    monkeypatch.setattr(s3, "put_object", racing_put)
    seed.seed_models(s3, "bucket", "models", tmp_path, sha, ("QB",))
    assert model_sync.load_manifest(s3, "bucket", "models", "QB")["stable"]["key"] == winner[0]


@pytest.mark.parametrize("protocol", ["ours", "previous", "v2"])
def test_initialize_seed_rechecks_every_protocol_after_validation(
    seed_world, monkeypatch, tmp_path, protocol
):
    from src.artifacts import publication, source

    s3, sha = seed_world
    pending_models(tmp_path)
    original = publication.validate_seed
    calls = []
    winner = []

    def validate(*args):
        original(*args)
        calls.append(True)
        if len(calls) == 2:  # after whole-request preflight, before initial upload
            key = put_models(s3)
            pointer = s3.objects.pop(model_sync.manifest_key("models", "QB"))
            destination = {
                "ours": model_sync.manifest_key,
                "previous": model_sync.previous_protocol_manifest_key,
                "v2": model_sync.legacy_manifest_key,
            }[protocol]("models", "QB")
            s3.objects[destination] = pointer
            winner.append(key)

    monkeypatch.setattr(publication, "validate_seed", validate)
    seed.seed_models(s3, "bucket", "models", tmp_path, sha, ("QB",))
    assert model_sync.load_manifest(s3, "bucket", "models", "QB")["stable"]["key"] == winner[0]
    assert not any("/history/seed-" in key for key in s3.objects)
    assert source.source_key("models", sha) in s3.objects


def test_verify_only_cli_never_registers_or_uploads(monkeypatch):
    import boto3

    s3 = FakeS3()
    for position in model_sync.POSITIONS:
        put_models(s3, position)
    monkeypatch.setattr(boto3, "client", lambda *_, **__: s3)
    monkeypatch.setattr(seed, "_check_models", check_extracted_models)
    monkeypatch.setattr(seed, "seed_models", lambda *_: pytest.fail("verify-only seeded"))
    assert seed.main(["--verify-only"]) == 0


def test_shell_wrapper_runs_module_from_repo_root(tmp_path):
    interpreter = tmp_path / "python"
    interpreter.write_text('#!/bin/bash\nprintf "%s\\n" "$PWD" "$@"\n')
    interpreter.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "infra/aws/seed_s3_models.sh"), "--verify-only"],
        cwd=tmp_path,
        env={**os.environ, "PYTHON": str(interpreter)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        str(ROOT),
        "-m",
        "src.scripts.seed_s3_models",
        "--verify-only",
    ]


def test_bootstrap_failed_preflight_never_calls_aws(tmp_path):
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/bash\nexit 19\n")
    interpreter.chmod(0o755)
    aws = tmp_path / "aws"
    calls = tmp_path / "aws-called"
    aws.write_text('#!/bin/bash\ntouch "$AWS_CALL_LOG"\nexit 99\n')
    aws.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "infra/aws/bootstrap.sh")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHON": str(interpreter),
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "AWS_CALL_LOG": str(calls),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 19
    assert not calls.exists()


def test_task_role_reconciliation_keeps_artifact_only_read_permissions():
    policy = json.loads((ROOT / "infra/aws/task-role-policy.json").read_text())

    def allowed(action, resource):
        for statement in policy["Statement"]:
            resources = statement["Resource"]
            if isinstance(resources, str):
                resources = [resources]
            if (
                statement["Effect"] == "Allow"
                and action in statement["Action"]
                and any(fnmatch.fnmatchcase(resource, pattern) for pattern in resources)
            ):
                return True
        return False

    bucket = "arn:aws:s3:::ff-predictor-training"
    assert allowed("s3:ListBucket", bucket)
    for suffix in (
        "/models/QB/releases/manifest.json",
        "/models/QB/releases/history/release/model.tar.gz",
        "/models/predictions_cache/cache.tar.gz",
    ):
        assert allowed("s3:GetObject", bucket + suffix)
    assert not allowed("s3:PutObject", bucket + "/models/predictions_cache/cache.tar.gz")
    assert not allowed("s3:GetObject", bucket + "/data/raw/schedules.parquet")
    assert not allowed("s3:PutObject", bucket + "/models/QB/releases/manifest.json")
    assert not allowed(
        "s3:DeleteObject", bucket + "/models/QB/releases/history/release/model.tar.gz"
    )
    assert not allowed("s3:GetObject", "arn:aws:s3:::another-bucket/data/splits/train.parquet")
