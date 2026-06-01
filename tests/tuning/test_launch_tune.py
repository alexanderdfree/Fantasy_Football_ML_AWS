"""Unit tests for src/tuning/launch_tune.py.

Tests don't hit AWS — `boto3.client` is mocked. We assert the *shape* of the
Batch submission (command, env, retry strategy) rather than verifying that
Batch itself works.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tuning import launch_tune

pytestmark = pytest.mark.unit


def test_submit_tune_job_builds_expected_command():
    """The container command must dispatch into --mode=tune with the right
    forwarded args. This is the contract that src/batch/train.py's --mode=tune
    branch (PR 2) relies on."""
    batch = MagicMock()
    batch.submit_job.return_value = {"jobId": "fake-job-id-123"}

    pos, job_id = launch_tune.submit_tune_job(
        "RB", n_trials=30, timeout=600, seed=42, batch_client=batch
    )

    assert pos == "RB"
    assert job_id == "fake-job-id-123"

    call = batch.submit_job.call_args
    kwargs = call.kwargs
    overrides = kwargs["containerOverrides"]
    assert overrides["command"] == [
        "--position",
        "RB",
        "--mode",
        "tune",
        "--seed",
        "42",
        "--n-trials",
        "30",
        "--parallel-backend",
        "auto",
        "--n-jobs",
        "3",
        "--timeout",
        "600",
    ]
    # S3_BUCKET / S3_DATA_PREFIX must be in env so tune_nn's
    # _ensure_data_from_s3 can populate data/splits + data/raw inside the
    # container.
    env_keys = {e["name"] for e in overrides["environment"]}
    assert {
        "S3_BUCKET",
        "S3_DATA_PREFIX",
        "LOG_EVERY",
        "FF_DEVICE",
        "FF_CUDA_GRAPH",
        "FF_AMP_DTYPE",
        "FF_COMPILE",
        "TUNE_NN_STORAGE_VERSION",
    } <= env_keys
    env = {e["name"]: e["value"] for e in overrides["environment"]}
    assert env["FF_DEVICE"] == "cuda"
    assert env["FF_CUDA_GRAPH"] == "1"
    assert env["FF_COMPILE"] == "0"
    assert env["TUNE_NN_STORAGE_VERSION"] == "scheduler_v2_mps_graph"
    assert kwargs["timeout"] == {"attemptDurationSeconds": 7200}
    # Retry strategy comes from launch.py — Spot interruptions retry, other
    # errors exit fast.
    assert kwargs["retryStrategy"] == launch_tune.RETRY_STRATEGY
    # Job name is unique enough to coexist with concurrent submissions.
    assert kwargs["jobName"].startswith("ff-tune-rb-")


def test_submit_tune_job_omits_timeout_when_none():
    """timeout=None should drop the --timeout flag from the command — Batch
    enforces its own wait-timeout layer via launch.py's WAIT_TIMEOUT_SECONDS."""
    batch = MagicMock()
    batch.submit_job.return_value = {"jobId": "fake"}
    launch_tune.submit_tune_job("QB", n_trials=15, timeout=None, batch_client=batch)
    cmd = batch.submit_job.call_args.kwargs["containerOverrides"]["command"]
    assert "--timeout" not in cmd
    assert "--n-trials" in cmd
    assert cmd[cmd.index("--n-trials") + 1] == "15"
    assert cmd[cmd.index("--parallel-backend") + 1] == "auto"
    assert cmd[cmd.index("--n-jobs") + 1] == "3"


def test_cli_rejects_unknown_position_name(monkeypatch):
    """argparse ``choices`` pins ``--positions`` to ``SUPPORTED_POSITIONS`` so
    a typo'd position fails locally instead of submitting a Spot job that
    will eventually fail on ``get_config()``. All six real positions are now
    supported; only nonsense names should be rejected."""
    monkeypatch.setattr("sys.argv", ["launch_tune", "--positions", "FOO"])
    with pytest.raises(SystemExit):
        launch_tune.main()


@pytest.mark.parametrize("pos", list(launch_tune.SUPPORTED_POSITIONS))
def test_submit_tune_job_works_for_all_supported_positions(pos):
    """Every position in SUPPORTED_POSITIONS — including the K/DST additions
    in PR 3 — must round-trip through submit_tune_job without special-casing.
    Catches a regression if a future PR adds a position-type carve-out
    inside submit_tune_job."""
    batch = MagicMock()
    batch.submit_job.return_value = {"jobId": f"fake-{pos.lower()}"}
    returned_pos, job_id = launch_tune.submit_tune_job(pos, n_trials=5, batch_client=batch)
    assert returned_pos == pos
    assert job_id == f"fake-{pos.lower()}"
    cmd = batch.submit_job.call_args.kwargs["containerOverrides"]["command"]
    assert cmd[:6] == ["--position", pos, "--mode", "tune", "--seed", "42"]
    assert "--parallel-backend" in cmd


def test_dry_run_does_not_call_aws(monkeypatch, capsys):
    """--dry-run must print the plan and exit before any AWS call. Asserted
    by ensuring boto3.client is never invoked."""
    monkeypatch.setattr("sys.argv", ["launch_tune", "--positions", "QB", "--dry-run"])
    with patch("src.tuning.launch_tune.boto3") as mock_boto3:
        launch_tune.main()
        mock_boto3.client.assert_not_called()
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "QB" in out
    assert "backend:      auto" in out
    assert "cuda graph:   True" in out


def test_submit_tune_job_can_disable_cuda_graph_and_change_workers():
    batch = MagicMock()
    batch.submit_job.return_value = {"jobId": "fake"}

    launch_tune.submit_tune_job(
        "QB",
        n_trials=6,
        n_jobs=2,
        parallel_backend="thread",
        cuda_graph=False,
        attempt_timeout=900,
        batch_client=batch,
    )

    kwargs = batch.submit_job.call_args.kwargs
    cmd = kwargs["containerOverrides"]["command"]
    assert cmd[cmd.index("--parallel-backend") + 1] == "thread"
    assert cmd[cmd.index("--n-jobs") + 1] == "2"
    env = {e["name"]: e["value"] for e in kwargs["containerOverrides"]["environment"]}
    assert env["FF_CUDA_GRAPH"] == "0"
    assert env["TUNE_NN_STORAGE_VERSION"] == "scheduler_v2"
    assert kwargs["timeout"] == {"attemptDurationSeconds": 900}
