"""Completion-side history publication, independent of serving state or a live waiter."""

import hashlib
import io
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from botocore.exceptions import ClientError

from src.artifacts import publication, source
from src.batch import benchmark, launch, run_history, train
from src.shared.model_sync import manifest_key
from src.shared.registry import ALL_POSITIONS

pytestmark = pytest.mark.unit


class MemoryS3:
    """Strongly consistent S3 boundary with conditional writes and separate clients."""

    def __init__(self):
        self.objects = {}
        self.lock = threading.Lock()

    def get_object(self, *, Bucket, Key):
        with self.lock:
            if Key not in self.objects:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            return {
                "Body": io.BytesIO(self.objects[Key]),
                "ETag": hashlib.sha256(self.objects[Key]).hexdigest(),
            }

    def put_object(self, *, Bucket, Key, Body, ContentType=None, IfNoneMatch=None, IfMatch=None):
        with self.lock:
            if IfNoneMatch == "*" and Key in self.objects:
                raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
            if IfMatch is not None and (
                Key not in self.objects or hashlib.sha256(self.objects[Key]).hexdigest() != IfMatch
            ):
                raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
            self.objects[Key] = Body

    def upload_file(self, path, bucket, key, **kwargs):
        with open(path, "rb") as file:
            self.put_object(Bucket=bucket, Key=key, Body=file.read())

    def download_file(self, bucket, key, path):
        from pathlib import Path

        Path(path).write_bytes(self.get_object(Bucket=bucket, Key=key)["Body"].read())

    def history(self):
        return [json.loads(v) for k, v in self.objects.items() if "/benchmark_history/" in k]


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "test-history")
    for key in (
        "FF_BUILD_PLAN_ID",
        "FF_DATASET_ID",
        "FF_DATA_RELEASE",
        "FF_DATA_FORMAT",
        "FF_REQUIRE_BUILD_PLAN",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        source, "image_source_sha", lambda: os.environ.get("FF_TRAIN_GIT_SHA", "a" * 40)
    )
    monkeypatch.setattr(publication, "image_source_sha", lambda: source.image_source_sha())
    return MemoryS3()


def metrics(sha="a" * 40, value=1.0):
    return {
        "git_sha": sha,
        "elapsed_sec": 10,
        "ridge_metrics": {"total": {"mae": value, "r2": 0.5, "rmse": value + 1}},
        "nn_metrics": {"total": {"mae": value + 0.1, "r2": 0.4, "rmse": value + 2}},
        "cohorts": {"weekly_reference_top24": {"status": "unavailable"}},
    }


def create(s3, positions=ALL_POSITIONS, run_id="run-a", sha="a" * 40):
    return run_history.create_run(
        s3,
        "bucket",
        positions,
        run_id=run_id,
        git_sha=sha,
        pr_number=100,
    )


def publish(s3, run_id, pos, sha="a" * 40, value=1.0):
    return run_history.publish_position(
        s3,
        "bucket",
        run_id,
        pos,
        metrics(sha, value),
        f"test-history/{pos}/{run_id}.tar.gz",
    )


def test_out_of_order_runs_keep_their_own_six_position_metrics(s3, monkeypatch, tmp_path):
    create(s3)
    create(s3, run_id="run-b", sha="b" * 40)
    # B overtakes A halfway through its six-position fan-out.
    for pos in ALL_POSITIONS[:3]:
        assert publish(s3, "run-a", pos) is None
    for pos in reversed(ALL_POSITIONS):
        publish(s3, "run-b", pos, "b" * 40, 9.0)
    for pos in ALL_POSITIONS[3:]:
        publish(s3, "run-a", pos)
    rows = {row["training_run_id"]: row for row in s3.history()}
    assert len(rows) == 2
    for run_id, value in (("run-a", 1.0), ("run-b", 9.0)):
        assert [r["ridge_mae"] for r in rows[run_id]["results"]] == [value] * 6
        assert rows[run_id]["positions"] == list(ALL_POSITIONS)
        assert rows[run_id]["results"][0]["cohorts"] == metrics()["cohorts"]

    # The CI collector must download the immutable A row, never a serving manifest.
    monkeypatch.setattr(benchmark.boto3, "client", lambda *a, **kw: s3)
    monkeypatch.setattr(benchmark, "HISTORY_DIR", str(tmp_path))
    monkeypatch.setattr(benchmark, "RESULTS_FILE", str(tmp_path / "results.json"))
    monkeypatch.setattr(benchmark, "download_metrics", lambda *a: pytest.fail("mutable manifest"))
    path = benchmark.record_benchmark_run(ALL_POSITIONS, run_id="run-a", git_hash="a" * 40)
    with open(path) as file:
        assert json.load(file) == rows["run-a"]


def test_completion_days_after_waiter_exit_publishes_without_collector(s3, monkeypatch):
    monkeypatch.setattr(run_history, "utc_now_iso", lambda: "2026-09-10T01:00:00")
    create(s3, ["QB", "K"])
    assert publish(s3, "run-a", "QB") is None
    assert s3.history() == []
    monkeypatch.setattr(run_history, "utc_now_iso", lambda: "2026-09-12T10:00:00")
    row = publish(s3, "run-a", "K")
    assert row["timestamp"] == "2026-09-12T10:00:00"
    assert row["pr_number"] == 100
    assert len(s3.history()) == 1


def test_concurrent_completion_and_retries_publish_once(s3):
    create(s3)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda pos: publish(s3, "run-a", pos), ALL_POSITIONS))
    assert any(results)
    original = s3.history()[0]
    with ThreadPoolExecutor(max_workers=6) as pool:
        repeats = list(pool.map(lambda pos: publish(s3, "run-a", pos, value=99), ALL_POSITIONS))
    assert repeats == [original] * 6
    assert s3.history() == [original]


def test_failed_attempt_does_not_poison_accepted_result_slot(s3, monkeypatch):
    monkeypatch.setattr(run_history, "utc_now_iso", lambda: "2026-09-10T01:00:00")
    create(s3, ["QB"])
    failed = run_history.publish_position(
        s3,
        "bucket",
        "run-a",
        "QB",
        metrics(value=9),
        "failed-artifact",
        smoke_passed=False,
        publication_status="validation_failed",
    )
    assert failed["validation_status"] == "validation_failed"
    assert failed["accepted_positions"] == []
    assert "test-history/training_runs/run-a/QB.json" not in s3.objects
    accepted = publish(s3, "run-a", "QB", value=1)
    assert accepted["validation_status"] == "accepted"
    assert accepted["accepted_positions"] == ["QB"]
    assert accepted["results"][0]["ridge_mae"] == 1
    assert any("/attempts/QB/" in key for key in s3.objects)
    assert len(s3.history()) == 2  # Immutable failure evidence and accepted presentation.
    assert run_history.complete_run(s3, "bucket", "run-a") == accepted


def test_history_identity_cannot_rebind_another_publication_request(s3):
    run_history.create_run(
        s3,
        "bucket",
        ["QB"],
        run_id="bound",
        git_sha="a" * 40,
        legacy_run_id="publication-a",
    )
    with pytest.raises(ValueError, match="different work"):
        run_history.create_run(
            s3,
            "bucket",
            ["QB"],
            run_id="bound",
            git_sha="a" * 40,
            legacy_run_id="publication-b",
        )
    wrong = {**metrics(), "publication_intent": {"run_id": "publication-b"}}
    with pytest.raises(ValueError, match="publication run differs"):
        run_history.publish_position(s3, "bucket", "bound", "QB", wrong, "artifact-b")
    assert s3.history() == []


def test_retry_after_canonical_claim_crash_records_accepted_metrics(s3, monkeypatch, tmp_path):
    from src.artifacts.receipts import load_run_receipt

    create(s3, ["QB"])
    sha = "a" * 40
    s3.objects[source.source_key("test-history", sha)] = json.dumps(
        {"source_sha": sha, "source_order": 1, "lineage": [sha]}
    ).encode()
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", sha)
    monkeypatch.setenv("FF_LEGACY_RUN_ID", "run-a")
    monkeypatch.setenv("FF_BENCHMARK_RUN_ID", "run-a")
    context = publication.prepare_training(s3, "bucket", "test-history", "QB")
    monkeypatch.setattr(train.boto3, "client", lambda *_, **__: s3)
    monkeypatch.setattr(train, "_validate_remote_tarball", lambda *_: None)
    monkeypatch.setattr(train, "_try_smoke_test", lambda *_: True)
    directory = tmp_path / "models"
    directory.mkdir()

    def write_metrics(value):
        payload = {
            **metrics(sha, value),
            "position": "QB",
            "publication_intent": context["intent"],
            "publication_revision": context["initial_revision"],
        }
        (directory / "benchmark_metrics.json").write_text(json.dumps(payload))

    write_metrics(1)
    publish_candidate = publication.publish_candidate
    monkeypatch.setattr(
        publication,
        "publish_candidate",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("crash after claim")),
    )
    with pytest.raises(RuntimeError, match="crash after claim"):
        train.upload_artifacts("bucket", "QB", str(directory))
    canonical = load_run_receipt(s3, "bucket", "test-history", sha, "QB", "run-a")
    assert s3.history() == []

    write_metrics(9)
    monkeypatch.setattr(publication, "publish_candidate", publish_candidate)
    train.upload_artifacts("bucket", "QB", str(directory))
    row = s3.history()[0]
    assert row["results"][0]["ridge_mae"] == 1
    assert row["artifacts"]["QB"] == canonical["key"]
    assert row["validation_status"] == "accepted"


@pytest.mark.parametrize("sha", ["b" * 40, None])
def test_wrong_or_missing_commit_cannot_enter_known_run(s3, sha):
    create(s3, ["QB"])
    with pytest.raises(ValueError, match="SHA mismatch"):
        publish(s3, "run-a", "QB", sha)
    assert s3.history() == []


def test_unknown_image_sha_is_not_invented_from_collector_checkout(s3):
    create(s3, ["QB"], sha=None)
    row = publish(s3, "run-a", "QB", sha=None)
    assert row["git_hash"] == "unknown"
    assert "code_fingerprints" not in row


def test_run_id_cannot_be_reused_for_different_work(s3):
    create(s3, ["QB"])
    assert create(s3, ["QB"]) == "run-a"
    with pytest.raises(ValueError, match="different work"):
        create(s3, ["RB"])


def test_collection_rejects_partial_or_wrong_identity(s3):
    create(s3, ["QB", "K"])
    publish(s3, "run-a", "QB")
    assert run_history.complete_run(s3, "bucket", "run-a") is None
    with pytest.raises(ValueError, match="positions"):
        run_history.complete_run(s3, "bucket", "run-a", positions=["QB"])
    with pytest.raises(ValueError, match="SHA mismatch"):
        run_history.complete_run(s3, "bucket", "run-a", git_sha="b" * 40)


def test_history_survives_artifact_pruning_and_uses_isolated_prefix(s3):
    create(s3, ["DST"])
    original = publish(s3, "run-a", "DST")
    assert all(key.startswith("test-history/") for key in s3.objects)
    # No model tarball is needed to collect previously completed history.
    assert run_history.complete_run(s3, "bucket", "run-a") == original


@pytest.mark.parametrize("matching_checkout", [False, True])
def test_collected_run_retains_only_sha_verified_local_fingerprints(
    s3,
    monkeypatch,
    tmp_path,
    matching_checkout,
    capsys,
):
    create(s3, ["QB"])
    original = publish(s3, "run-a", "QB")
    monkeypatch.setattr(benchmark.boto3, "client", lambda *a, **kw: s3)
    monkeypatch.setattr(benchmark, "HISTORY_DIR", str(tmp_path / "history"))
    monkeypatch.setattr(benchmark, "RESULTS_FILE", str(tmp_path / "results.json"))
    monkeypatch.setattr(
        benchmark, "get_git_hash", lambda: "a" * 8 if matching_checkout else "b" * 8
    )
    calls = []

    def fingerprint(positions, *, repo_root, source):
        calls.append((positions, source))
        return {"QB": "f" * 64}

    monkeypatch.setattr(benchmark, "collect_code_fingerprints", fingerprint)
    path = benchmark.record_benchmark_run(["QB"], run_id="run-a")
    with open(path) as file:
        collected = json.load(file)
    assert collected.get("code_fingerprints") == ({"QB": "f" * 64} if matching_checkout else None)
    assert calls == ([(["QB"], "head")] if matching_checkout else [])
    assert s3.history() == [original]
    if matching_checkout:
        from src.scripts import pre_pr_bench_check

        monkeypatch.setattr(pre_pr_bench_check, "HISTORY_DIR", str(tmp_path / "history"))
        monkeypatch.setattr(pre_pr_bench_check, "position_fingerprint", lambda *a, **kw: "f" * 64)
        capsys.readouterr()
        assert pre_pr_bench_check.cmd_evaluate(["src/qb/config.py"]) == 0
        assert capsys.readouterr().out.startswith("PASS\n")


@pytest.mark.parametrize("trained_checkout", ["module", "caller"])
def test_provenance_and_fingerprints_use_the_same_worktree(
    s3, monkeypatch, tmp_path, trained_checkout, capsys
):
    module_root = tmp_path / "module"
    caller_root = tmp_path / "caller"
    module_root.mkdir()

    def git(root, *args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()

    git(module_root, "init", "-q")
    git(module_root, "config", "user.name", "History Test")
    git(module_root, "config", "user.email", "history@example.invalid")
    source = module_root / "src/qb/config.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    git(module_root, "add", ".")
    git(module_root, "commit", "-qm", "module version")
    git(module_root, "worktree", "add", "-qb", "trained", str(caller_root))
    (caller_root / "src/qb/config.py").write_text("VALUE = 2\n")
    git(caller_root, "commit", "-qam", "caller version")
    trained_root = module_root if trained_checkout == "module" else caller_root
    sha = git(trained_root, "rev-parse", "HEAD")
    create(s3, ["QB"], sha=sha)
    publish(s3, "run-a", "QB", sha=sha)
    monkeypatch.setattr(benchmark.boto3, "client", lambda *a, **kw: s3)
    monkeypatch.setattr(benchmark, "_REPO_ROOT", str(module_root))
    monkeypatch.setattr(benchmark, "HISTORY_DIR", str(tmp_path / "history"))
    monkeypatch.setattr(benchmark, "RESULTS_FILE", str(tmp_path / "results.json"))
    monkeypatch.chdir(caller_root)
    path = benchmark.record_benchmark_run(["QB"], run_id="run-a")
    with open(path) as file:
        entry = json.load(file)
    assert bool(entry.get("code_fingerprints")) is (trained_checkout == "module")

    # Exercise the real gate in the module checkout, with old fingerprinted
    # history disabling the legacy mtime fallback. Caller code is not evidence.
    from src.scripts import pre_pr_bench_check

    prior = {**entry, "code_fingerprints": {"QB": "0" * 64}}
    (tmp_path / "history/prior.json").write_text(json.dumps(prior))
    monkeypatch.setattr(pre_pr_bench_check, "HISTORY_DIR", str(tmp_path / "history"))
    monkeypatch.chdir(module_root)
    capsys.readouterr()
    assert pre_pr_bench_check.cmd_evaluate(["src/qb/config.py"]) == 0
    verdict = capsys.readouterr().out.splitlines()[0]
    assert verdict == ("PASS" if trained_checkout == "module" else "FAIL")


def test_artifact_upload_publishes_its_own_metrics_even_when_stable_is_pinned(
    s3,
    monkeypatch,
    tmp_path,
):
    create(s3, ["QB"])
    source_record = {"source_sha": "a" * 40, "source_order": 1, "lineage": ["a" * 40]}
    s3.objects[source.source_key("test-history", "a" * 40)] = json.dumps(source_record).encode()
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", "a" * 40)
    monkeypatch.setattr(publication, "image_source_sha", lambda: "a" * 40)
    monkeypatch.setenv("FF_BENCHMARK_RUN_ID", "run-a")
    monkeypatch.setattr(train.boto3, "client", lambda *a, **kw: s3)
    monkeypatch.setattr(train, "_validate_remote_tarball", lambda *a: None)
    monkeypatch.setattr(train, "_try_smoke_test", lambda *a: False)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setenv("FF_LEGACY_RUN_ID", "run-a")
    context = publication.prepare_training(s3, "bucket", "test-history", "QB")
    payload = {
        **metrics(),
        "position": "QB",
        "publication_intent": context["intent"],
        "publication_revision": context["initial_revision"],
    }
    (model_dir / "benchmark_metrics.json").write_text(json.dumps(payload))
    train.upload_artifacts("bucket", "QB", str(model_dir))
    assert s3.history()[0]["results"][0]["ridge_mae"] == 1.0
    manifest = json.loads(s3.objects[manifest_key("test-history", "QB")])
    assert manifest["stable"] is None
    assert s3.history()[0]["artifacts"]["QB"] == manifest["current"]["key"]


def test_superseded_older_completion_keeps_new_models_and_both_history_rows(
    s3, monkeypatch, tmp_path
):
    old, new = "a" * 40, "b" * 40
    for sha, lineage in ((old, [old]), (new, [new, old])):
        source_record = {"source_sha": sha, "source_order": len(lineage), "lineage": lineage}
        s3.objects[source.source_key("test-history", sha)] = json.dumps(source_record).encode()
    create(s3, run_id="older", sha=old)
    create(s3, run_id="newer", sha=new)
    monkeypatch.setattr(train.boto3, "client", lambda *a, **kw: s3)
    monkeypatch.setattr(publication, "image_source_sha", lambda: os.environ["FF_TRAIN_GIT_SHA"])
    monkeypatch.setattr(train, "_validate_remote_tarball", lambda *a: None)
    monkeypatch.setattr(train, "_try_smoke_test", lambda *a: True)
    active = {}
    for run_id, sha, value in (("newer", new, 9.0), ("older", old, 1.0)):
        monkeypatch.setenv("FF_BENCHMARK_RUN_ID", run_id)
        monkeypatch.setenv("FF_TRAIN_GIT_SHA", sha)
        monkeypatch.setenv("FF_LEGACY_RUN_ID", run_id)
        for pos in ALL_POSITIONS:
            directory = tmp_path / f"{run_id}-{pos}"
            directory.mkdir()
            context = publication.prepare_training(s3, "bucket", "test-history", pos)
            payload = {
                **metrics(sha, value),
                "position": pos,
                "publication_intent": context["intent"],
                "publication_revision": context["initial_revision"],
            }
            (directory / "benchmark_metrics.json").write_text(json.dumps(payload))
            if run_id == "newer":
                train.upload_artifacts("bucket", pos, str(directory))
                active[pos] = json.loads(s3.objects[manifest_key("test-history", pos)])
            else:
                train.upload_artifacts("bucket", pos, str(directory))
                assert json.loads(s3.objects[manifest_key("test-history", pos)]) == active[pos]
    rows = {row["training_run_id"]: row for row in s3.history()}
    assert set(rows) == {"older", "newer"}
    for run_id, value in (("older", 1.0), ("newer", 9.0)):
        assert [r["ridge_mae"] for r in rows[run_id]["results"]] == [value] * 6
        assert rows[run_id]["positions"] == list(ALL_POSITIONS)


@pytest.mark.parametrize("split", [False, True])
def test_launcher_registers_before_submit_and_only_full_or_merge_jobs_publish(
    s3, monkeypatch, split
):
    from tests.batch._submission_fixture import mock_submission_boundaries

    mock_submission_boundaries(monkeypatch, launch)
    submitted = []

    class Batch:
        def submit_job(self, **kwargs):
            # Even an immediately completing job can resolve the full descriptor.
            descriptor = json.loads(s3.objects["test-history/training_runs/ci-run/run.json"])
            assert descriptor["positions"] == list(ALL_POSITIONS)
            assert descriptor["pr_number"] == 1554
            assert descriptor["data_release"] == "f" * 64
            submitted.append(kwargs)
            return {"jobId": str(len(submitted))}

    monkeypatch.setattr(
        launch.boto3, "client", lambda service, **kw: s3 if service == "s3" else Batch()
    )
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", "a" * 40)
    monkeypatch.setenv("FF_LEGACY_RUN_ID", "ci-run")
    monkeypatch.setattr(launch, "JOB_DEFINITION_REVISION", "1")
    monkeypatch.setattr(launch, "JOB_DEFINITION_CPU_REVISION", "1")
    monkeypatch.setattr(source, "register_source", lambda *a, **kw: None)
    monkeypatch.setattr(launch, "JOB_DEFINITION_CPU", "cpu-def")
    monkeypatch.setattr(launch, "JOB_QUEUE_CPU", "cpu-queue")
    monkeypatch.setattr(launch, "JOB_IDS_FILE", None)
    args = [
        "launch.py",
        "--wait",
        "false",
        "--skip-upload",
        "--history-run-id",
        "ci-run",
        "--pr-number",
        "1554",
    ]
    monkeypatch.setattr("sys.argv", args + (["--split"] if split else []))
    launch.main()
    publishers = []
    for job in submitted:
        env = {e["name"]: e["value"] for e in job["containerOverrides"]["environment"]}
        assert env["FF_DATA_RELEASE"] == env["FF_DATASET_ID"] == "f" * 64
        if "FF_BENCHMARK_RUN_ID" in env:
            assert env["FF_BENCHMARK_RUN_ID"] == "ci-run"
            assert env["FF_TRAIN_GIT_SHA"] == "a" * 40
            if split:
                assert "merge" in job["containerOverrides"]["command"]
                assert len(job["dependsOn"]) == 2
            publishers.append(job)
    assert len(publishers) == 6
    assert len(submitted) == (18 if split else 6)
