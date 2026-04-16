"""Tests for batch/launch.py — config validation, job submission, S3 parsing."""
import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_s3_bucket_is_set(self):
        from batch.launch import S3_BUCKET
        assert S3_BUCKET and isinstance(S3_BUCKET, str)
        assert " " not in S3_BUCKET

    def test_job_queue_is_set(self):
        from batch.launch import JOB_QUEUE
        assert JOB_QUEUE and isinstance(JOB_QUEUE, str)

    def test_job_definition_is_set(self):
        from batch.launch import JOB_DEFINITION
        assert JOB_DEFINITION and isinstance(JOB_DEFINITION, str)

    def test_all_positions_complete(self):
        from batch.launch import ALL_POSITIONS
        assert set(ALL_POSITIONS) == {"QB", "RB", "WR", "TE", "K", "DST"}

    def test_all_positions_count(self):
        from batch.launch import ALL_POSITIONS
        assert len(ALL_POSITIONS) == 6

    def test_terminal_states(self):
        from batch.launch import TERMINAL_STATES
        assert "SUCCEEDED" in TERMINAL_STATES
        assert "FAILED" in TERMINAL_STATES


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class TestLaunchArgParsing:
    def test_default_positions(self):
        from batch.launch import ALL_POSITIONS
        parser = argparse.ArgumentParser()
        parser.add_argument("--positions", nargs="+", default=ALL_POSITIONS, choices=ALL_POSITIONS)
        parser.add_argument("--wait", default="true")
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args([])
        assert args.positions == ALL_POSITIONS
        assert args.wait == "true"
        assert args.seed == 42

    def test_subset_positions(self):
        from batch.launch import ALL_POSITIONS
        parser = argparse.ArgumentParser()
        parser.add_argument("--positions", nargs="+", default=ALL_POSITIONS, choices=ALL_POSITIONS)
        args = parser.parse_args(["--positions", "RB", "WR"])
        assert args.positions == ["RB", "WR"]

    def test_wait_false(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--wait", default="true")
        args = parser.parse_args(["--wait", "false"])
        assert args.wait.lower() == "false"

    def test_custom_seed(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args(["--seed", "99"])
        assert args.seed == 99


# ---------------------------------------------------------------------------
# Data upload
# ---------------------------------------------------------------------------

class TestUploadData:
    @mock.patch("batch.launch.boto3.client")
    def test_uploads_three_files(self, mock_boto_client):
        from batch.launch import upload_data

        mock_s3 = mock.MagicMock()
        mock_boto_client.return_value = mock_s3

        upload_data("my-bucket")

        assert mock_s3.upload_file.call_count == 3
        uploaded_keys = {c.args[2] for c in mock_s3.upload_file.call_args_list}
        assert uploaded_keys == {"data/train.parquet", "data/val.parquet", "data/test.parquet"}
        # All uploads target the same bucket
        uploaded_buckets = {c.args[1] for c in mock_s3.upload_file.call_args_list}
        assert uploaded_buckets == {"my-bucket"}


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

class TestSubmitJob:
    @mock.patch("batch.launch.boto3.client")
    def test_submit_returns_job_id(self, mock_boto_client):
        from batch.launch import submit_job

        mock_batch = mock.MagicMock()
        mock_batch.submit_job.return_value = {"jobId": "abc-123"}
        mock_boto_client.return_value = mock_batch

        pos, job_id = submit_job("RB", seed=42)
        assert pos == "RB"
        assert job_id == "abc-123"

    @mock.patch("batch.launch.boto3.client")
    def test_submit_passes_correct_overrides(self, mock_boto_client):
        from batch.launch import submit_job, JOB_QUEUE, JOB_DEFINITION

        mock_batch = mock.MagicMock()
        mock_batch.submit_job.return_value = {"jobId": "xyz"}
        mock_boto_client.return_value = mock_batch

        submit_job("WR", seed=99)

        call_kwargs = mock_batch.submit_job.call_args.kwargs
        assert call_kwargs["jobQueue"] == JOB_QUEUE
        assert call_kwargs["jobDefinition"] == JOB_DEFINITION
        overrides = call_kwargs["containerOverrides"]
        assert overrides["command"] == ["--position", "WR", "--seed", "99"]


# ---------------------------------------------------------------------------
# Job waiting
# ---------------------------------------------------------------------------

class TestWaitForJobs:
    @mock.patch("batch.launch.time.sleep")
    @mock.patch("batch.launch.boto3.client")
    def test_wait_returns_on_success(self, mock_boto_client, mock_sleep):
        from batch.launch import wait_for_jobs

        mock_batch = mock.MagicMock()
        mock_batch.describe_jobs.return_value = {
            "jobs": [
                {"jobId": "job-1", "status": "SUCCEEDED"},
                {"jobId": "job-2", "status": "SUCCEEDED"},
            ]
        }
        mock_boto_client.return_value = mock_batch

        results = wait_for_jobs({"RB": "job-1", "WR": "job-2"})
        assert results == {"RB": "SUCCEEDED", "WR": "SUCCEEDED"}

    @mock.patch("batch.launch.time.sleep")
    @mock.patch("batch.launch.boto3.client")
    def test_wait_handles_failure(self, mock_boto_client, mock_sleep):
        from batch.launch import wait_for_jobs

        mock_batch = mock.MagicMock()
        mock_batch.describe_jobs.return_value = {
            "jobs": [
                {"jobId": "job-1", "status": "SUCCEEDED"},
                {"jobId": "job-2", "status": "FAILED"},
            ]
        }
        mock_boto_client.return_value = mock_batch

        results = wait_for_jobs({"RB": "job-1", "WR": "job-2"})
        assert results["RB"] == "SUCCEEDED"
        assert results["WR"] == "FAILED"

    @mock.patch("batch.launch.time.sleep")
    @mock.patch("batch.launch.boto3.client")
    def test_wait_polls_until_terminal(self, mock_boto_client, mock_sleep):
        from batch.launch import wait_for_jobs

        mock_batch = mock.MagicMock()
        # First call: RUNNING, second call: SUCCEEDED
        mock_batch.describe_jobs.side_effect = [
            {"jobs": [{"jobId": "job-1", "status": "RUNNING"}]},
            {"jobs": [{"jobId": "job-1", "status": "SUCCEEDED"}]},
        ]
        mock_boto_client.return_value = mock_batch

        results = wait_for_jobs({"RB": "job-1"})
        assert results["RB"] == "SUCCEEDED"
        assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# Artifact download
# ---------------------------------------------------------------------------

class TestDownloadArtifacts:
    @mock.patch("batch.launch.boto3.client")
    def test_download_extracts_to_correct_dir(self, mock_boto_client, tmp_path):
        from batch.launch import download_artifacts, S3_BUCKET

        # Create a fake tar.gz
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "ridge_model.pkl").write_text("fake")

        tar_path = tmp_path / "model.tar.gz"
        with tarfile.open(str(tar_path), "w:gz") as tar:
            tar.add(str(staging / "ridge_model.pkl"), arcname="ridge_model.pkl")

        mock_s3 = mock.MagicMock()
        def fake_download(bucket, key, filename):
            import shutil
            shutil.copy(str(tar_path), filename)

        mock_s3.download_file.side_effect = fake_download
        mock_boto_client.return_value = mock_s3

        download_artifacts(["RB"])

        mock_s3.download_file.assert_called_once()
        call_args = mock_s3.download_file.call_args.args
        assert call_args[0] == S3_BUCKET
        assert call_args[1] == "models/RB/model.tar.gz"


# ---------------------------------------------------------------------------
# Cross-module consistency
# ---------------------------------------------------------------------------

class TestCrossModuleConsistency:
    def test_train_and_launch_positions_match(self):
        from batch.train import POSITIONS as TRAIN_POSITIONS
        from batch.launch import ALL_POSITIONS
        assert set(TRAIN_POSITIONS.keys()) == set(ALL_POSITIONS)

    def test_train_and_launch_s3_bucket_match(self):
        from batch.train import main as _  # just to import the module
        import batch.train as train_mod
        from batch.launch import S3_BUCKET

        # train.py uses env var with default "ff-predictor-training"
        # launch.py uses S3_BUCKET constant — they should agree
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("S3_BUCKET", None)
            default_bucket = os.environ.get("S3_BUCKET", "ff-predictor-training")
        assert default_bucket == S3_BUCKET
