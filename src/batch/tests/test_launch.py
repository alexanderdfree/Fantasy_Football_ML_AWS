"""Tests for batch/launch.py — config validation, job submission, S3 parsing."""

import argparse
import datetime
import importlib
import os
import sys
import tarfile
from pathlib import Path
from unittest import mock

from botocore.exceptions import ClientError

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_s3_bucket_is_set(self):
        from src.batch.launch import S3_BUCKET

        assert S3_BUCKET and isinstance(S3_BUCKET, str)
        assert " " not in S3_BUCKET

    def test_job_queue_is_set(self):
        from src.batch.launch import JOB_QUEUE

        assert JOB_QUEUE and isinstance(JOB_QUEUE, str)

    def test_job_definition_is_set(self):
        from src.batch.launch import JOB_DEFINITION

        assert JOB_DEFINITION and isinstance(JOB_DEFINITION, str)

    def test_all_positions_complete(self):
        from src.batch.launch import ALL_POSITIONS

        assert set(ALL_POSITIONS) == {"QB", "RB", "WR", "TE", "K", "DST"}

    def test_all_positions_count(self):
        from src.batch.launch import ALL_POSITIONS

        assert len(ALL_POSITIONS) == 6

    def test_terminal_states(self):
        from src.batch.launch import TERMINAL_STATES

        assert "SUCCEEDED" in TERMINAL_STATES
        assert "FAILED" in TERMINAL_STATES

    def test_cpu_only_positions(self):
        from src.batch.launch import CPU_ONLY_POSITIONS

        assert {"K", "DST"} == CPU_ONLY_POSITIONS


class TestEnvVarOverrides:
    """Config constants should respect FF_* env vars when set."""

    def _reload(self):
        import src.batch.launch as mod

        return importlib.reload(mod)

    def test_bucket_override(self):
        with mock.patch.dict(os.environ, {"FF_S3_BUCKET": "alt-bucket"}):
            mod = self._reload()
            assert mod.S3_BUCKET == "alt-bucket"
        # Reset to defaults for subsequent tests
        self._reload()

    def test_job_queue_override(self):
        with mock.patch.dict(os.environ, {"FF_JOB_QUEUE": "alt-queue"}):
            mod = self._reload()
            assert mod.JOB_QUEUE == "alt-queue"
        self._reload()

    def test_job_definition_override(self):
        with mock.patch.dict(os.environ, {"FF_JOB_DEFINITION": "alt-def"}):
            mod = self._reload()
            assert mod.JOB_DEFINITION == "alt-def"
        self._reload()

    def test_cpu_job_definition_optional(self):
        # Unset -> None
        with mock.patch.dict(os.environ, {"FF_JOB_DEFINITION_CPU": ""}):
            mod = self._reload()
            assert mod.JOB_DEFINITION_CPU is None
        # Set -> str
        with mock.patch.dict(os.environ, {"FF_JOB_DEFINITION_CPU": "cpu-def"}):
            mod = self._reload()
            assert mod.JOB_DEFINITION_CPU == "cpu-def"
        self._reload()

    def test_wait_timeout_override(self):
        with mock.patch.dict(os.environ, {"FF_WAIT_TIMEOUT": "600"}):
            mod = self._reload()
            assert mod.WAIT_TIMEOUT_SECONDS == 600
        self._reload()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestLaunchArgParsing:
    def test_default_positions(self):
        from src.batch.launch import ALL_POSITIONS

        parser = argparse.ArgumentParser()
        parser.add_argument("--positions", nargs="+", default=ALL_POSITIONS, choices=ALL_POSITIONS)
        parser.add_argument("--wait", default="true")
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args([])
        assert args.positions == ALL_POSITIONS
        assert args.wait == "true"
        assert args.seed == 42

    def test_subset_positions(self):
        from src.batch.launch import ALL_POSITIONS

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
# Data upload + ETag dedup
# ---------------------------------------------------------------------------


class TestUploadData:
    def test_uploads_when_missing(self, monkeypatch, tmp_path):
        """No remote object -> upload all three files."""
        from src.batch import launch

        # Build fake local parquets
        data_dir = tmp_path / "data" / "splits"
        data_dir.mkdir(parents=True)
        for name in ("train.parquet", "val.parquet", "test.parquet"):
            (data_dir / name).write_bytes(b"fake-parquet")
        monkeypatch.chdir(tmp_path)

        mock_s3 = mock.MagicMock()
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )

        launch.upload_data("my-bucket", s3_client=mock_s3)

        assert mock_s3.upload_file.call_count == 3
        uploaded_keys = {c.args[2] for c in mock_s3.upload_file.call_args_list}
        assert uploaded_keys == {"data/train.parquet", "data/val.parquet", "data/test.parquet"}

    def test_skips_when_etag_matches(self, monkeypatch, tmp_path):
        """ETag == local MD5 -> skip upload."""
        from src.batch import launch

        data_dir = tmp_path / "data" / "splits"
        data_dir.mkdir(parents=True)
        content = b"fake-parquet"
        import hashlib

        local_md5 = hashlib.md5(content).hexdigest()
        for name in ("train.parquet", "val.parquet", "test.parquet"):
            (data_dir / name).write_bytes(content)
        monkeypatch.chdir(tmp_path)

        mock_s3 = mock.MagicMock()
        mock_s3.head_object.return_value = {"ETag": f'"{local_md5}"'}

        launch.upload_data("my-bucket", s3_client=mock_s3)
        assert mock_s3.upload_file.call_count == 0

    def test_force_upload_bypasses_dedup(self, monkeypatch, tmp_path):
        from src.batch import launch

        data_dir = tmp_path / "data" / "splits"
        data_dir.mkdir(parents=True)
        content = b"fake-parquet"
        import hashlib

        local_md5 = hashlib.md5(content).hexdigest()
        for name in ("train.parquet", "val.parquet", "test.parquet"):
            (data_dir / name).write_bytes(content)
        monkeypatch.chdir(tmp_path)

        mock_s3 = mock.MagicMock()
        mock_s3.head_object.return_value = {"ETag": f'"{local_md5}"'}

        launch.upload_data("my-bucket", s3_client=mock_s3, force=True)
        assert mock_s3.upload_file.call_count == 3


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------


class TestSubmitJob:
    def test_submit_returns_job_id(self):
        from src.batch.launch import submit_job

        mock_batch = mock.MagicMock()
        mock_batch.submit_job.return_value = {"jobId": "abc-123"}

        pos, job_id = submit_job("RB", seed=42, batch_client=mock_batch)
        assert pos == "RB"
        assert job_id == "abc-123"

    def test_submit_passes_correct_overrides(self):
        from src.batch.launch import JOB_DEFINITION, JOB_QUEUE, submit_job

        mock_batch = mock.MagicMock()
        mock_batch.submit_job.return_value = {"jobId": "xyz"}

        submit_job("WR", seed=99, batch_client=mock_batch)

        call_kwargs = mock_batch.submit_job.call_args.kwargs
        assert call_kwargs["jobQueue"] == JOB_QUEUE
        assert call_kwargs["jobDefinition"] == JOB_DEFINITION
        overrides = call_kwargs["containerOverrides"]
        assert overrides["command"] == ["--position", "WR", "--seed", "99"]

    def test_job_names_are_unique_within_same_second(self):
        """Two rapid submissions must not collide on job name."""
        from src.batch.launch import submit_job

        mock_batch = mock.MagicMock()
        mock_batch.submit_job.return_value = {"jobId": "id"}

        with mock.patch("src.batch.launch.time.time", return_value=1_700_000_000):
            submit_job("RB", batch_client=mock_batch)
            submit_job("RB", batch_client=mock_batch)

        names = [c.kwargs["jobName"] for c in mock_batch.submit_job.call_args_list]
        assert len(set(names)) == 2, f"Job names collided: {names}"
        # Both should match the fixed timestamp prefix
        assert all(n.startswith("ff-rb-1700000000-") for n in names), names

    def test_k_routes_to_cpu_def_when_configured(self):
        """If FF_JOB_DEFINITION_CPU is set, K/DST should use it."""
        import src.batch.launch as mod

        mock_batch = mock.MagicMock()
        mock_batch.submit_job.return_value = {"jobId": "id"}

        with mock.patch.object(mod, "JOB_DEFINITION_CPU", "cpu-def"):
            mod.submit_job("K", batch_client=mock_batch)
            mod.submit_job("QB", batch_client=mock_batch)

        defs = [c.kwargs["jobDefinition"] for c in mock_batch.submit_job.call_args_list]
        assert defs[0] == "cpu-def"  # K routed to CPU
        assert defs[1] == mod.JOB_DEFINITION  # QB stays on GPU

    def test_k_falls_back_to_default_when_cpu_def_unset(self):
        """If no CPU def is configured, K/DST still run on the default queue."""
        import src.batch.launch as mod

        mock_batch = mock.MagicMock()
        mock_batch.submit_job.return_value = {"jobId": "id"}

        with mock.patch.object(mod, "JOB_DEFINITION_CPU", None):
            mod.submit_job("K", batch_client=mock_batch)

        assert mock_batch.submit_job.call_args.kwargs["jobDefinition"] == mod.JOB_DEFINITION


# ---------------------------------------------------------------------------
# Job waiting
# ---------------------------------------------------------------------------


class TestCloudWatchUrl:
    """Regression: path separators must be encoded as $252F, not $2F.

    The CloudWatch console expects the fragment path components to be
    double-URL-encoded: `/` -> `%2F` -> `$252F`. Single-encoding yields
    `$2F` and the console renders an empty log stream page.
    """

    def test_group_and_stream_use_double_encoded_slashes(self):
        from src.batch.launch import _cloudwatch_url

        url = _cloudwatch_url("my-stream/part/subpart")
        # Default BATCH_LOG_GROUP is "/aws/batch/job"
        assert "$252Faws$252Fbatch$252Fjob" in url
        assert "my-stream$252Fpart$252Fsubpart" in url
        # And never the single-encoded form that misses the `25`
        assert "$2Faws" not in url
        assert "my-stream$2Fpart" not in url


class TestWaitForJobs:
    @mock.patch("src.batch.launch.time.sleep")
    def test_wait_returns_tuples_on_success(self, mock_sleep):
        from src.batch.launch import wait_for_jobs

        mock_batch = mock.MagicMock()
        mock_batch.describe_jobs.return_value = {
            "jobs": [
                {"jobId": "job-1", "status": "SUCCEEDED", "stoppedAt": 1700000000000},
                {"jobId": "job-2", "status": "SUCCEEDED", "stoppedAt": 1700000001000},
            ]
        }

        results = wait_for_jobs({"RB": "job-1", "WR": "job-2"}, batch_client=mock_batch)
        assert results["RB"] == ("SUCCEEDED", 1700000000000)
        assert results["WR"] == ("SUCCEEDED", 1700000001000)

    @mock.patch("src.batch.launch.time.sleep")
    def test_wait_handles_failure(self, mock_sleep):
        from src.batch.launch import wait_for_jobs

        mock_batch = mock.MagicMock()
        mock_batch.describe_jobs.return_value = {
            "jobs": [
                {"jobId": "job-1", "status": "SUCCEEDED", "stoppedAt": 1},
                {"jobId": "job-2", "status": "FAILED", "statusReason": "boom"},
            ]
        }

        results = wait_for_jobs({"RB": "job-1", "WR": "job-2"}, batch_client=mock_batch)
        assert results["RB"][0] == "SUCCEEDED"
        assert results["WR"][0] == "FAILED"

    @mock.patch("src.batch.launch.time.sleep")
    def test_wait_polls_until_terminal(self, mock_sleep):
        from src.batch.launch import wait_for_jobs

        mock_batch = mock.MagicMock()
        # First call: RUNNING, second call: SUCCEEDED
        mock_batch.describe_jobs.side_effect = [
            {"jobs": [{"jobId": "job-1", "status": "RUNNING"}]},
            {"jobs": [{"jobId": "job-1", "status": "SUCCEEDED", "stoppedAt": 2}]},
        ]

        results = wait_for_jobs({"RB": "job-1"}, batch_client=mock_batch)
        assert results["RB"] == ("SUCCEEDED", 2)
        assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# Artifact download + stale-artifact guard
# ---------------------------------------------------------------------------


class TestDownloadArtifacts:
    def test_download_extracts_to_correct_dir(self, tmp_path, monkeypatch, capsys):
        from src.batch import launch
        from src.batch.launch import S3_BUCKET, download_artifacts

        # Create a fake tar.gz
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "ridge_model.pkl").write_text("fake")

        tar_path = tmp_path / "model.tar.gz"
        with tarfile.open(str(tar_path), "w:gz") as tar:
            tar.add(str(staging / "ridge_model.pkl"), arcname="ridge_model.pkl")

        mock_s3 = mock.MagicMock()
        # Fresh artifact (LastModified after stoppedAt)
        mock_s3.head_object.return_value = {
            "LastModified": datetime.datetime(2023, 11, 14, 22, 13, 30, tzinfo=datetime.UTC),
        }

        def fake_download(bucket, key, filename):
            import shutil

            shutil.copy(str(tar_path), filename)

        mock_s3.download_file.side_effect = fake_download

        monkeypatch.chdir(tmp_path)
        download_artifacts(["RB"], s3_client=mock_s3)

        mock_s3.download_file.assert_called_once()
        call_args = mock_s3.download_file.call_args.args
        assert call_args[0] == S3_BUCKET
        assert call_args[1] == "models/RB/model.tar.gz"

    def test_warns_on_stale_artifact(self, tmp_path, monkeypatch, capsys):
        """If LastModified < stoppedAt, download_artifacts should print a WARNING."""
        from src.batch.launch import download_artifacts

        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "x.pkl").write_text("x")
        tar_path = tmp_path / "m.tar.gz"
        with tarfile.open(str(tar_path), "w:gz") as tar:
            tar.add(str(staging / "x.pkl"), arcname="x.pkl")

        mock_s3 = mock.MagicMock()
        # Remote LastModified is a day before job stoppedAt
        old_modified = datetime.datetime(2023, 1, 1, tzinfo=datetime.UTC)
        mock_s3.head_object.return_value = {"LastModified": old_modified}

        def fake_download(bucket, key, filename):
            import shutil

            shutil.copy(str(tar_path), filename)

        mock_s3.download_file.side_effect = fake_download

        monkeypatch.chdir(tmp_path)
        # stopped_at is much later than LastModified
        stopped_at_ms = int(datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC).timestamp() * 1000)
        download_artifacts(["RB"], stopped_at_by_pos={"RB": stopped_at_ms}, s3_client=mock_s3)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "stale" in captured.out.lower()

    def test_skip_when_no_object(self, tmp_path, monkeypatch, capsys):
        from src.batch.launch import download_artifacts

        mock_s3 = mock.MagicMock()
        mock_s3.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")

        monkeypatch.chdir(tmp_path)
        download_artifacts(["RB"], s3_client=mock_s3)
        mock_s3.download_file.assert_not_called()
        assert "No artifacts" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Cross-module consistency
# ---------------------------------------------------------------------------


class TestCrossModuleConsistency:
    def test_train_and_launch_positions_match(self):
        from src.batch.launch import ALL_POSITIONS
        from src.shared.registry import ALL_POSITIONS as REG_POSITIONS

        assert set(ALL_POSITIONS) == set(REG_POSITIONS)

    def test_cpu_only_positions_match_across_modules(self):
        from src.batch.launch import CPU_ONLY_POSITIONS as LAUNCH_CPU
        from src.shared.registry import CPU_ONLY_POSITIONS as REG_CPU

        assert LAUNCH_CPU == REG_CPU

    def test_train_and_launch_s3_bucket_match(self):
        from src.batch.launch import S3_BUCKET

        # train.py uses env var with default "ff-predictor-training"
        # launch.py uses S3_BUCKET constant — they should agree
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("S3_BUCKET", None)
            os.environ.pop("FF_S3_BUCKET", None)
            default_bucket = os.environ.get("S3_BUCKET", "ff-predictor-training")
        assert default_bucket == S3_BUCKET
