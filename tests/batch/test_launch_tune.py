"""Unit tests for src/batch/launch_tune.py.

Tests don't hit AWS — `boto3.client` is mocked. We assert the *shape* of the
Batch submission (command, env, retry strategy) rather than verifying that
Batch itself works.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.batch import launch_tune

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
        "--timeout",
        "600",
    ]
    # S3_BUCKET / S3_DATA_PREFIX must be in env so tune_nn's
    # _ensure_data_from_s3 can populate data/splits + data/raw inside the
    # container.
    env_keys = {e["name"] for e in overrides["environment"]}
    assert {"S3_BUCKET", "S3_DATA_PREFIX", "LOG_EVERY"} <= env_keys
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


@pytest.mark.parametrize("bad_pos", ["K", "DST", "k", "Dst"])
def test_cli_rejects_unsupported_positions(monkeypatch, bad_pos):
    """K and DST must be rejected at the CLI boundary (argparse choices),
    mirroring tune_nn's _UNSUPPORTED_POSITIONS guard so the Batch submission
    can't waste a Spot job."""
    monkeypatch.setattr("sys.argv", ["launch_tune", "--positions", bad_pos])
    with pytest.raises(SystemExit):
        launch_tune.main()


def test_dry_run_does_not_call_aws(monkeypatch, capsys):
    """--dry-run must print the plan and exit before any AWS call. Asserted
    by ensuring boto3.client is never invoked."""
    monkeypatch.setattr("sys.argv", ["launch_tune", "--positions", "QB", "--dry-run"])
    with patch("src.batch.launch_tune.boto3") as mock_boto3:
        launch_tune.main()
        mock_boto3.client.assert_not_called()
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "QB" in out
