"""Coverage tests for ``src/batch/launch.py::main`` and ``_print_plan``.

``test_launch.py`` covers the helpers (upload_data, wait_for_jobs, the
CPU-definition dispatch, etc.) but never exercises ``main()`` itself or
the ``--dry-run`` path. These tests fill the gap by driving main() with
mocked boto3 clients + pipeline stubs so the full argparse → submit →
wait → download flow runs in-process.
"""

from __future__ import annotations

from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def registered_source(monkeypatch):
    monkeypatch.setattr("src.shared.artifact_publication.register_source", lambda *a, **k: None)
    monkeypatch.setattr("src.batch.launch.validate_submission_source", lambda *a, **k: None)
    monkeypatch.setattr("src.batch.launch.create_run", lambda *a, **k: "unit-run")


@pytest.fixture(autouse=True)
def _legacy_data_for_launcher_stubs(monkeypatch):
    # These orchestration-only fakes do not provide S3 manifests. Real release
    # pinning and split-job propagation are covered by test_data_release.py.
    monkeypatch.setenv("FF_DATA_RELEASE", "legacy")
    from src.batch import launch

    monkeypatch.setattr(
        launch,
        "resolve_launch_binding",
        lambda *a, **k: {
            "image_sha": "a" * 40,
            "gpu_definition": "gpu:1",
            "cpu_definition": "cpu:1",
        },
    )
    monkeypatch.setattr(launch, "validate_local_publish", lambda *a: None)


@pytest.mark.unit
def test_print_plan_emits_expected_lines(capsys):
    """``_print_plan`` should list every region/bucket/queue/def line + the
    per-position dispatch. Sensitive to the string keys the plan relies on."""
    from src.batch.launch import _print_plan

    _print_plan(["QB", "K"], seed=99)
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "region:" in out
    assert "bucket:" in out
    assert "queue:" in out
    assert "definition:" in out
    assert "seed:         99" in out
    # Per-position lines
    assert "- QB" in out
    assert "- K" in out


@pytest.mark.unit
def test_main_dry_run_makes_no_aws_calls(monkeypatch, capsys):
    """``--dry-run`` must early-return without instantiating boto3 clients."""
    from src.batch import launch as lm

    def _boom(*args, **kwargs):
        raise AssertionError(f"boto3.client called during --dry-run: {args} {kwargs}")

    monkeypatch.setattr(lm.boto3, "client", _boom)
    monkeypatch.setattr("sys.argv", ["launch.py", "--positions", "QB", "--dry-run"])
    lm.main()

    out = capsys.readouterr().out
    assert "DRY RUN" in out


@pytest.mark.unit
def test_main_split_dry_run_prints_branch_plan(monkeypatch, capsys):
    """``--split --dry-run`` should render nn/cpu/merge commands and dependencies."""
    from src.batch import launch as lm

    def _boom(*args, **kwargs):
        raise AssertionError(f"boto3.client called during --dry-run: {args} {kwargs}")

    monkeypatch.setattr(lm.boto3, "client", _boom)
    monkeypatch.setattr(lm, "JOB_DEFINITION_CPU", "cpu-def")
    monkeypatch.setattr(lm, "JOB_QUEUE_CPU", "cpu-queue")
    monkeypatch.setattr("sys.argv", ["launch.py", "--positions", "WR", "--split", "--dry-run"])
    lm.main()

    out = capsys.readouterr().out
    assert "split:        true" in out
    assert "WR   nn" in out
    assert "WR   cpu" in out
    assert "WR   merge" in out
    assert "dependsOn nn+cpu" in out


@pytest.fixture()
def _main_happy_stubs(monkeypatch):
    """Stub every external call in ``main()`` non-dry path. Returns call log."""
    from src.batch import launch as lm

    calls: list[dict] = []

    class _FakeS3:
        def __init__(self):
            self.client_kind = "s3"

    class _FakeBatch:
        def __init__(self):
            self.client_kind = "batch"

    def _fake_boto_client(service, region_name=None):
        return {"s3": _FakeS3(), "batch": _FakeBatch()}[service]

    monkeypatch.setattr(lm.boto3, "client", _fake_boto_client)
    monkeypatch.setattr(lm, "create_run", lambda *a, **kw: "unit-run")

    def _upload(bucket, s3_client=None, force=False):
        calls.append({"upload": bucket, "force": force})

    def _submit(pos, seed, batch_client=None, **kwargs):
        calls.append({"submit": pos, "seed": seed})
        return pos, f"job-{pos}"

    def _wait(job_ids, timeout_seconds=None, batch_client=None):
        calls.append({"wait_for": list(job_ids.keys())})
        return {p: ("SUCCEEDED", 123456789) for p in job_ids}

    def _download(positions, stopped_at_by_pos=None, s3_client=None):
        calls.append({"download": positions})

    monkeypatch.setattr(lm, "upload_data", _upload)
    monkeypatch.setattr(lm, "submit_job", _submit)
    monkeypatch.setattr(lm, "wait_for_jobs", _wait)
    monkeypatch.setattr(lm, "download_artifacts", _download)

    # Stub the benchmark_history auto-append so the wait path doesn't reach S3
    # in unit tests; record the call so tests can assert it fired for the
    # succeeded set.
    def _append(positions, *, note=None, **metadata):
        calls.append({"append": list(positions), "note": note})

    monkeypatch.setattr(lm, "_append_benchmark_history", _append)
    return calls


@pytest.mark.unit
def test_main_default_path_runs_full_flow(_main_happy_stubs, monkeypatch, capsys):
    """Default CLI: upload → submit (parallel) → wait → download artifacts."""
    from src.batch import launch as lm

    monkeypatch.setattr(
        "sys.argv",
        ["launch.py", "--positions", "QB", "RB", "--seed", "11"],
    )
    lm.main()

    kinds = [list(c)[0] for c in _main_happy_stubs]
    assert "upload" in kinds
    # Two submits (one per position)
    submits = [c for c in _main_happy_stubs if "submit" in c]
    assert sorted(c["submit"] for c in submits) == ["QB", "RB"]
    assert all(c["seed"] == 11 for c in submits)
    assert "wait_for" in kinds
    assert "download" in kinds

    out = capsys.readouterr().out
    assert "All done." in out


@pytest.mark.unit
def test_main_wait_false_skips_wait_and_download(_main_happy_stubs, monkeypatch, capsys):
    """``--wait false`` short-circuits after submit — no wait, no download."""
    from src.batch import launch as lm

    monkeypatch.setattr(
        "sys.argv",
        ["launch.py", "--positions", "QB", "--wait", "false"],
    )
    lm.main()

    kinds = [list(c)[0] for c in _main_happy_stubs]
    assert "upload" in kinds
    assert "submit" in kinds
    # No wait/download
    assert "wait_for" not in kinds
    assert "download" not in kinds

    out = capsys.readouterr().out
    assert "aws batch describe-jobs" in out


@pytest.mark.unit
def test_main_failed_jobs_branch(monkeypatch, capsys):
    """When wait_for_jobs flags a position FAILED, main() prints it and skips
    download for that position but downloads the successful ones."""
    from src.batch import launch as lm

    calls: list[dict] = []
    monkeypatch.setattr(lm.boto3, "client", lambda *a, **k: mock.MagicMock())
    monkeypatch.setattr(lm, "upload_data", lambda *a, **k: None)
    monkeypatch.setattr(lm, "submit_job", lambda p, s, c, **kwargs: (p, f"j-{p}"))

    def _wait(job_ids, timeout_seconds=None, batch_client=None):
        out = {p: ("SUCCEEDED", 0) for p in job_ids}
        out["QB"] = ("FAILED", 0)
        return out

    monkeypatch.setattr(lm, "wait_for_jobs", _wait)

    def _download(positions, stopped_at_by_pos=None, s3_client=None):
        calls.append({"download": list(positions)})

    monkeypatch.setattr(lm, "download_artifacts", _download)
    monkeypatch.setattr("sys.argv", ["launch.py", "--positions", "QB", "RB"])
    monkeypatch.setattr(lm, "_append_benchmark_history", lambda *a, **k: None)
    # main() now exits non-zero when any position is FAILED / TIMED_OUT so CI
    # surfaces the regression instead of silently passing; train-batch.yml's
    # post-step comment claims the workflow blocks on non-success.
    with pytest.raises(SystemExit) as exc:
        lm.main()
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert "Failed positions" in out
    assert "QB" in out
    # Download was called but only for succeeded positions (RB) — side effects
    # before the exit still fire.
    assert calls == [{"download": ["RB"]}]


@pytest.mark.unit
def test_main_submit_exception_is_logged(monkeypatch, capsys):
    """Exception from submit_job is caught and printed, other submissions still
    proceed (no global abort mid-loop), and the succeeded position is still
    downloaded — but main() now exits non-zero so the workflow surfaces the
    failed submit instead of false-greening (#758)."""
    from src.batch import launch as lm

    monkeypatch.setattr(lm.boto3, "client", lambda *a, **k: mock.MagicMock())
    monkeypatch.setattr(lm, "upload_data", lambda *a, **k: None)

    def _bad_submit(pos, seed, batch_client=None, **kwargs):
        if pos == "QB":
            raise RuntimeError("transient aws fault")
        return pos, f"j-{pos}"

    waited_for_ids: list[dict] = []

    def _wait(j, timeout_seconds=None, batch_client=None):
        waited_for_ids.append(dict(j))
        return {p: ("SUCCEEDED", 0) for p in j}

    downloaded: list[list[str]] = []

    def _download(positions, stopped_at_by_pos=None, s3_client=None):
        downloaded.append(list(positions))

    monkeypatch.setattr(lm, "submit_job", _bad_submit)
    monkeypatch.setattr(lm, "wait_for_jobs", _wait)
    monkeypatch.setattr(lm, "download_artifacts", _download)
    monkeypatch.setattr(lm, "_append_benchmark_history", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["launch.py", "--positions", "QB", "RB"])
    with pytest.raises(SystemExit) as exc_info:
        lm.main()
    assert exc_info.value.code == 1

    out = capsys.readouterr().out
    # The exception is logged with the failing position and the original message
    # (must include both — a generic "something failed" line wouldn't be enough).
    assert "[QB] FAILED to submit" in out
    assert "transient aws fault" in out
    # Loop did not abort mid-flight: RB was submitted, waited on, and downloaded
    # (the non-zero exit fires only after). QB, whose submit raised, never made
    # it into job_ids and is absent from both.
    assert waited_for_ids == [{"RB": "j-RB"}]
    assert downloaded == [["RB"]]


@pytest.mark.unit
def test_main_wait_timeout_override(monkeypatch, capsys):
    """``--wait-timeout`` must override WAIT_TIMEOUT_SECONDS on the wait call."""
    from src.batch import launch as lm

    monkeypatch.setattr(lm.boto3, "client", lambda *a, **k: mock.MagicMock())
    monkeypatch.setattr(lm, "upload_data", lambda *a, **k: None)
    monkeypatch.setattr(lm, "submit_job", lambda p, s, c, **kwargs: (p, f"j-{p}"))
    monkeypatch.setattr(lm, "download_artifacts", lambda *a, **k: None)

    captured_timeouts: list[int | None] = []

    def _wait(j, timeout_seconds=None, batch_client=None):
        captured_timeouts.append(timeout_seconds)
        return {p: ("SUCCEEDED", 0) for p in j}

    monkeypatch.setattr(lm, "wait_for_jobs", _wait)
    monkeypatch.setattr(lm, "_append_benchmark_history", lambda *a, **k: None)

    override = 1234
    monkeypatch.setattr(
        "sys.argv",
        ["launch.py", "--positions", "QB", "--wait-timeout", str(override)],
    )
    lm.main()

    # main() must forward the CLI override (not the module default
    # WAIT_TIMEOUT_SECONDS) to wait_for_jobs as `timeout_seconds`.
    assert captured_timeouts == [override]


@pytest.mark.unit
def test_main_writes_job_ids_breadcrumb_when_env_set(_main_happy_stubs, monkeypatch, tmp_path):
    """With FF_BATCH_JOB_IDS_FILE set (module global JOB_IDS_FILE), main()
    records the submitted job ids + expected positions so train-batch.yml's
    recovery step can re-check the exact jobs after a failed wait."""
    import json as _json

    from src.batch import launch as lm

    ids_path = tmp_path / "batch_job_ids.json"
    monkeypatch.setattr(lm, "JOB_IDS_FILE", str(ids_path))
    monkeypatch.setattr("sys.argv", ["launch.py", "--positions", "QB", "RB"])
    lm.main()

    payload = _json.loads(ids_path.read_text())
    assert payload["expected_positions"] == ["QB", "RB"]
    assert payload["jobs"] == {"QB": "job-QB", "RB": "job-RB"}


@pytest.mark.unit
def test_write_job_ids_file_labels_split_tuple_keys(tmp_path):
    """Split-mode job_ids use (position, branch) tuple keys; the breadcrumb
    must serialize them as the same "POS/branch" labels wait_for_jobs prints,
    and a write failure must not raise (jobs are already submitted)."""
    import json as _json

    from src.batch.launch import _write_job_ids_file

    ids_path = tmp_path / "ids.json"
    _write_job_ids_file(
        str(ids_path),
        ["WR"],
        {("WR", "nn"): "j-nn", ("WR", "cpu"): "j-cpu", ("WR", "merge"): "j-merge"},
    )
    payload = _json.loads(ids_path.read_text())
    assert payload["expected_positions"] == ["WR"]
    assert payload["jobs"] == {"WR/nn": "j-nn", "WR/cpu": "j-cpu", "WR/merge": "j-merge"}

    # Unwritable path: prints a warning, never raises.
    _write_job_ids_file(str(tmp_path / "no-such-dir" / "ids.json"), ["WR"], {"WR": "j"})


@pytest.mark.unit
def test_main_auto_appends_history_for_succeeded(_main_happy_stubs, monkeypatch):
    """Default wait path rolls the succeeded positions into a benchmark_history
    row via ``_append_benchmark_history`` so a standalone (non-CI) run shows up
    in the serving History tab."""
    from src.batch import launch as lm

    monkeypatch.setattr("sys.argv", ["launch.py", "--positions", "QB", "RB"])
    lm.main()

    appended = [c for c in _main_happy_stubs if "append" in c]
    assert len(appended) == 1
    # job_ids order is non-deterministic (as_completed) — compare sorted.
    assert sorted(appended[0]["append"]) == ["QB", "RB"]


@pytest.mark.unit
def test_main_append_history_false_skips_append(_main_happy_stubs, monkeypatch):
    """The explicit history opt-out disables publication and local collection."""
    from src.batch import launch as lm

    monkeypatch.setattr(
        "sys.argv",
        ["launch.py", "--positions", "QB", "--append-history", "false"],
    )
    lm.main()

    assert not [c for c in _main_happy_stubs if "append" in c]


@pytest.mark.unit
def test_history_registration_binds_resolved_image_and_selected_data(
    _main_happy_stubs, monkeypatch
):
    from src.batch import launch as lm

    release_id = "d" * 64
    events = []

    def pin(_client, *, source_ref):
        assert source_ref == "a" * 40
        events.append("data-pinned")
        monkeypatch.setenv("FF_DATA_RELEASE", release_id)
        return release_id

    def create(_client, _bucket, positions, **kwargs):
        assert events == ["data-pinned"]
        assert not [c for c in _main_happy_stubs if "submit" in c]
        assert positions == ["QB"]
        assert kwargs["git_sha"] == "a" * 40
        assert kwargs["data_release"] == release_id
        events.append("run-registered")
        return "bound-run"

    def submit(pos, seed, batch_client=None, *, binding, history_run_id):
        assert events == ["data-pinned", "run-registered"]
        assert binding["image_sha"] == "a" * 40
        assert history_run_id == "bound-run"
        return pos, "job"

    monkeypatch.setattr(lm, "TRAIN_GIT_SHA", "b" * 40)
    monkeypatch.setattr(lm, "pin_data_release", pin)
    monkeypatch.setattr(lm, "create_run", create)
    monkeypatch.setattr(lm, "submit_job", submit)
    monkeypatch.setattr(
        "sys.argv", ["launch", "--positions", "QB", "--skip-upload", "--wait", "false"]
    )
    lm.main()
    assert events == ["data-pinned", "run-registered"]


@pytest.mark.unit
def test_bad_data_cannot_register_a_history_run(_main_happy_stubs, monkeypatch):
    from src.batch import launch as lm

    create = mock.Mock()
    submit = mock.Mock()
    monkeypatch.setattr(lm, "create_run", create)
    monkeypatch.setattr(lm, "submit_job", submit)
    monkeypatch.setattr(
        lm, "pin_data_release", mock.Mock(side_effect=RuntimeError("data mismatch"))
    )
    monkeypatch.setattr("sys.argv", ["launch", "--positions", "QB", "--skip-upload"])
    with pytest.raises(RuntimeError, match="data mismatch"):
        lm.main()
    create.assert_not_called()
    submit.assert_not_called()


@pytest.mark.unit
def test_partial_success_collects_registered_positions_not_a_subset(_main_happy_stubs, monkeypatch):
    from src.batch import launch as lm

    monkeypatch.setattr(
        lm, "wait_for_jobs", lambda *a, **kw: {"QB": ("SUCCEEDED", 1), "RB": ("FAILED", 2)}
    )
    collect = mock.Mock()
    monkeypatch.setattr(lm, "_append_benchmark_history", collect)
    monkeypatch.setattr("sys.argv", ["launch", "--positions", "QB", "RB"])
    with pytest.raises(SystemExit) as error:
        lm.main()
    assert error.value.code == 1
    collect.assert_called_once_with(
        ["QB", "RB"],
        note="Standalone Batch run",
        git_hash="a" * 40,
        run_id="unit-run",
        data_release="legacy",
    )


@pytest.mark.unit
def test_split_submission_forwards_binding_and_only_merge_publishes_history(monkeypatch):
    from src.batch import launch as lm

    batch = mock.Mock()
    batch.submit_job.side_effect = [{"jobId": "nn"}, {"jobId": "cpu"}, {"jobId": "merge"}]
    binding = {"image_sha": "a" * 40, "gpu_definition": "gpu:7", "cpu_definition": "cpu:8"}
    monkeypatch.setenv("FF_DATA_RELEASE", "d" * 64)
    monkeypatch.setattr(lm, "JOB_DEFINITION_CPU", "cpu")
    monkeypatch.setattr(lm, "JOB_QUEUE_CPU", "cpu-queue")
    lm._submit_split_for_position(
        "QB", 42, "split-attempt-1", batch, binding, history_run_id="history-1"
    )
    calls = [call.kwargs for call in batch.submit_job.call_args_list]
    assert [call["jobDefinition"] for call in calls] == ["gpu:7", "cpu:8", "cpu:8"]
    for index, call in enumerate(calls):
        env = {e["name"]: e["value"] for e in call["containerOverrides"]["environment"]}
        assert env["FF_TRAIN_GIT_SHA"] == binding["image_sha"]
        assert env["FF_DATA_RELEASE"] == "d" * 64
        assert (env.get("FF_BENCHMARK_RUN_ID") == "history-1") is (index == 2)


@pytest.mark.unit
def test_ci_skips_local_collection_but_registers_completion_publication(
    _main_happy_stubs, monkeypatch
):
    from src.batch import launch as lm

    registered = []
    monkeypatch.setattr(lm, "create_run", lambda *a, **kw: registered.append(kw) or "ci-run")
    monkeypatch.setattr(
        "sys.argv",
        [
            "launch.py",
            "--positions",
            "QB",
            "--history-run-id",
            "ci-run",
            "--collect-history",
            "false",
        ],
    )
    lm.main()
    assert registered[0]["run_id"] == "ci-run"
    assert not [c for c in _main_happy_stubs if "append" in c]
