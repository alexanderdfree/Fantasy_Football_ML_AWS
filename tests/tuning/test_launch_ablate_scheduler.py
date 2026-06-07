"""Unit tests for src/tuning/launch_ablate_scheduler.py.

No AWS — the batch client is a MagicMock. We assert the *shape* of the Batch
submission (command, env, per-job timeout) that src/batch/train.py's
--ablation scheduler-type branch relies on.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.tuning import launch_ablate_scheduler as las

pytestmark = pytest.mark.unit


def test_submit_ablate_job_builds_expected_command():
    batch = MagicMock()
    batch.submit_job.return_value = {"jobId": "job-abc"}

    pos, job_id = las.submit_ablate_job("K", seeds="42,43,44", batch_client=batch)

    assert pos == "K"
    assert job_id == "job-abc"
    overrides = batch.submit_job.call_args.kwargs["containerOverrides"]
    assert overrides["command"] == [
        "--position",
        "K",
        "--ablation",
        "scheduler-type",
        "--seeds",
        "42,43,44",
    ]


def test_submit_ablate_job_forces_eager_cuda_graph():
    """CUDA graphs autodetect ON for sm_80+ but are NOT numerically inert — the
    A/B must be eager (FF_CUDA_GRAPH=0) for bit-comparable scheduler deltas."""
    batch = MagicMock()
    batch.submit_job.return_value = {"jobId": "job-abc"}
    las.submit_ablate_job("RB", seeds="42", batch_client=batch)
    overrides = batch.submit_job.call_args.kwargs["containerOverrides"]
    env = {e["name"]: e["value"] for e in overrides["environment"]}
    assert env["FF_CUDA_GRAPH"] == "0"
    assert env["FF_DEVICE"] == "cuda"
    assert env["S3_BUCKET"]
    assert env["S3_DATA_PREFIX"] == "data"


def test_submit_ablate_job_sets_attempt_timeout():
    batch = MagicMock()
    batch.submit_job.return_value = {"jobId": "x"}
    las.submit_ablate_job("DST", seeds="42", attempt_timeout=5400, batch_client=batch)
    timeout = batch.submit_job.call_args.kwargs["timeout"]
    assert timeout == {"attemptDurationSeconds": 5400}


def test_dry_run_makes_no_aws_calls(capsys):
    # --dry-run must not construct a real client or call submit_job.
    import sys
    from unittest.mock import patch

    argv = ["prog", "--dry-run", "--positions", "K", "DST", "--seeds", "42,43,44"]
    with patch.object(sys, "argv", argv), patch("boto3.client") as mk:
        las.main()
    mk.assert_not_called()
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "FF_CUDA_GRAPH: 0" in out
    assert "--seeds 42,43,44" in out


def test_empty_seeds_rejected():
    import sys
    from unittest.mock import patch

    argv = ["prog", "--dry-run", "--seeds", " , ,"]
    with patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit):
            las.main()
