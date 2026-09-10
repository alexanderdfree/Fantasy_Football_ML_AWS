"""Completion-side history publication, independent of serving state or a live waiter."""

import io
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from botocore.exceptions import ClientError

from src.batch import benchmark, launch, run_history, train
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
            return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket, Key, Body, ContentType=None, IfNoneMatch=None):
        with self.lock:
            if IfNoneMatch == "*" and Key in self.objects:
                raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
            self.objects[Key] = Body

    def upload_file(self, path, bucket, key):
        with open(path, "rb") as file:
            self.put_object(Bucket=bucket, Key=key, Body=file.read())

    def history(self):
        return [json.loads(v) for k, v in self.objects.items() if "/benchmark_history/" in k]


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "test-history")
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
    monkeypatch.setenv("FF_BENCHMARK_RUN_ID", "run-a")
    monkeypatch.setattr(train.boto3, "client", lambda *a, **kw: s3)
    monkeypatch.setattr(train, "_validate_remote_tarball", lambda *a: None)
    monkeypatch.setattr(train, "_try_smoke_test", lambda *a: False)
    monkeypatch.setattr(train, "_gc_prune", lambda *a: [])
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "benchmark_metrics.json").write_text(json.dumps(metrics()))
    train.upload_artifacts("bucket", "QB", str(model_dir))
    assert s3.history()[0]["results"][0]["ridge_mae"] == 1.0
    manifest = json.loads(s3.objects["test-history/QB/manifest.json"])
    assert manifest["stable"] is None
    assert s3.history()[0]["artifacts"]["QB"] == manifest["current"]["key"]


@pytest.mark.parametrize("split", [False, True])
def test_launcher_registers_before_submit_and_only_full_or_merge_jobs_publish(
    s3, monkeypatch, split
):
    submitted = []

    class Batch:
        def submit_job(self, **kwargs):
            # Even an immediately completing job can resolve the full descriptor.
            descriptor = json.loads(s3.objects["test-history/training_runs/ci-run/run.json"])
            assert descriptor["positions"] == list(ALL_POSITIONS)
            assert descriptor["pr_number"] == 1554
            submitted.append(kwargs)
            return {"jobId": str(len(submitted))}

    monkeypatch.setattr(
        launch.boto3, "client", lambda service, **kw: s3 if service == "s3" else Batch()
    )
    monkeypatch.setattr(launch, "TRAIN_GIT_SHA", "a" * 40)
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
        if "FF_BENCHMARK_RUN_ID" in env:
            assert env["FF_BENCHMARK_RUN_ID"] == "ci-run"
            assert env["FF_TRAIN_GIT_SHA"] == "a" * 40
            if split:
                assert "merge" in job["containerOverrides"]["command"]
                assert len(job["dependsOn"]) == 2
            publishers.append(job)
    assert len(publishers) == 6
    assert len(submitted) == (18 if split else 6)
