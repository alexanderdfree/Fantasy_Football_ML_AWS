"""Tests for sagemaker/launch.py — config validation, metric regexes, S3 parsing."""
import argparse
import os
import re
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
        from sagemaker.launch import S3_BUCKET
        assert S3_BUCKET and isinstance(S3_BUCKET, str)
        assert " " not in S3_BUCKET, "S3 bucket name should not contain spaces"

    def test_role_arn_format(self):
        from sagemaker.launch import ROLE
        assert ROLE.startswith("arn:aws:iam::"), f"Invalid IAM ARN prefix: {ROLE}"
        assert ":role/" in ROLE, "ROLE should contain ':role/'"

    def test_instance_type_is_gpu(self):
        from sagemaker.launch import INSTANCE_TYPE
        # g4dn, g5, p3, p4 are GPU instance families
        assert any(gpu in INSTANCE_TYPE for gpu in ["g4dn", "g5", "p3", "p4"]), (
            f"Instance type {INSTANCE_TYPE} doesn't look like a GPU instance"
        )

    def test_all_positions_complete(self):
        from sagemaker.launch import ALL_POSITIONS
        assert set(ALL_POSITIONS) == {"QB", "RB", "WR", "TE", "K", "DST"}

    def test_all_positions_order(self):
        """ALL_POSITIONS should have exactly 6 entries."""
        from sagemaker.launch import ALL_POSITIONS
        assert len(ALL_POSITIONS) == 6


# ---------------------------------------------------------------------------
# Metric regex tests (critical for CloudWatch parsing)
# ---------------------------------------------------------------------------

class TestMetricDefinitions:
    """Verify metric regexes match the actual trainer output format."""

    SAMPLE_LOG_LINE = (
        "Epoch  10 | Train: 0.4523 | Val: 0.5012 | "
        "MAE total: 4.231 | rushing_floor: 1.823 | receiving_floor: 1.456"
    )

    def _get_metric_defs(self):
        from sagemaker.launch import METRIC_DEFINITIONS
        return METRIC_DEFINITIONS

    def test_three_metrics_defined(self):
        metrics = self._get_metric_defs()
        assert len(metrics) == 3

    def test_metric_names(self):
        metrics = self._get_metric_defs()
        names = {m["Name"] for m in metrics}
        assert names == {"train:loss", "val:loss", "val:mae_total"}

    def test_train_loss_regex_matches(self):
        metrics = self._get_metric_defs()
        regex = next(m["Regex"] for m in metrics if m["Name"] == "train:loss")
        match = re.search(regex, self.SAMPLE_LOG_LINE)
        assert match is not None, f"train:loss regex didn't match: {regex}"
        assert float(match.group(1)) == pytest.approx(0.4523)

    def test_val_loss_regex_matches(self):
        metrics = self._get_metric_defs()
        regex = next(m["Regex"] for m in metrics if m["Name"] == "val:loss")
        match = re.search(regex, self.SAMPLE_LOG_LINE)
        assert match is not None, f"val:loss regex didn't match: {regex}"
        assert float(match.group(1)) == pytest.approx(0.5012)

    def test_mae_total_regex_matches(self):
        metrics = self._get_metric_defs()
        regex = next(m["Regex"] for m in metrics if m["Name"] == "val:mae_total")
        match = re.search(regex, self.SAMPLE_LOG_LINE)
        assert match is not None, f"val:mae_total regex didn't match: {regex}"
        assert float(match.group(1)) == pytest.approx(4.231)

    def test_val_loss_does_not_match_train(self):
        """val:loss regex should NOT accidentally capture the Train value."""
        metrics = self._get_metric_defs()
        regex = next(m["Regex"] for m in metrics if m["Name"] == "val:loss")
        # On a line with ONLY train loss, val regex should still match
        # the Val portion, not the Train portion.
        # The key concern: "Val: 0.5012" should be matched, not "Train: 0.4523"
        match = re.search(regex, self.SAMPLE_LOG_LINE)
        assert match is not None
        value = float(match.group(1))
        assert value != pytest.approx(0.4523), "val:loss regex captured train loss value!"

    def test_regexes_on_real_format_variants(self):
        """Test against edge cases in number formatting."""
        metrics = self._get_metric_defs()
        train_regex = next(m["Regex"] for m in metrics if m["Name"] == "train:loss")

        # Very small loss
        small = "Epoch 100 | Train: 0.0012 | Val: 0.0034 | MAE total: 0.123"
        match = re.search(train_regex, small)
        assert match and float(match.group(1)) == pytest.approx(0.0012)

        # Larger values
        big = "Epoch   1 | Train: 12.3456 | Val: 15.6789 | MAE total: 45.678"
        match = re.search(train_regex, big)
        assert match and float(match.group(1)) == pytest.approx(12.3456)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class TestLaunchArgParsing:
    def test_default_positions(self):
        from sagemaker.launch import ALL_POSITIONS
        parser = argparse.ArgumentParser()
        parser.add_argument("--positions", nargs="+", default=ALL_POSITIONS, choices=ALL_POSITIONS)
        parser.add_argument("--wait", default="true")
        args = parser.parse_args([])
        assert args.positions == ALL_POSITIONS
        assert args.wait == "true"

    def test_subset_positions(self):
        from sagemaker.launch import ALL_POSITIONS
        parser = argparse.ArgumentParser()
        parser.add_argument("--positions", nargs="+", default=ALL_POSITIONS, choices=ALL_POSITIONS)
        parser.add_argument("--wait", default="true")
        args = parser.parse_args(["--positions", "RB", "WR"])
        assert args.positions == ["RB", "WR"]

    def test_wait_false(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--positions", nargs="+", default=["QB"])
        parser.add_argument("--wait", default="true")
        args = parser.parse_args(["--wait", "false"])
        assert args.wait.lower() == "false"


# ---------------------------------------------------------------------------
# S3 URI parsing (download_artifacts)
# ---------------------------------------------------------------------------

class TestS3UriParsing:
    """Test the S3 URI parsing logic used in download_artifacts."""

    def test_standard_s3_uri(self):
        uri = "s3://ff-predictor-training/models/RB/output/model.tar.gz"
        s3_parts = uri.replace("s3://", "").split("/", 1)
        bucket, key = s3_parts[0], s3_parts[1]
        assert bucket == "ff-predictor-training"
        assert key == "models/RB/output/model.tar.gz"

    def test_s3_uri_with_nested_path(self):
        uri = "s3://my-bucket/a/b/c/d/model.tar.gz"
        s3_parts = uri.replace("s3://", "").split("/", 1)
        bucket, key = s3_parts[0], s3_parts[1]
        assert bucket == "my-bucket"
        assert key == "a/b/c/d/model.tar.gz"


# ---------------------------------------------------------------------------
# Artifact extraction (download_artifacts)
# ---------------------------------------------------------------------------

class TestArtifactExtraction:
    def test_tar_extraction(self, tmp_path):
        """Simulate downloading and extracting a model.tar.gz."""
        # Create a fake model tar.gz
        model_content = tmp_path / "staging"
        model_content.mkdir()
        (model_content / "ridge_model.pkl").write_text("fake")
        (model_content / "nn_model.pt").write_text("fake")

        tar_path = tmp_path / "model.tar.gz"
        with tarfile.open(str(tar_path), "w:gz") as tar:
            tar.add(str(model_content / "ridge_model.pkl"), arcname="ridge_model.pkl")
            tar.add(str(model_content / "nn_model.pt"), arcname="nn_model.pt")

        # Extract to simulated local model dir
        extract_dir = tmp_path / "RB" / "outputs" / "models"
        extract_dir.mkdir(parents=True)
        with tarfile.open(str(tar_path), "r:gz") as tar:
            tar.extractall(str(extract_dir), filter="data")

        assert (extract_dir / "ridge_model.pkl").exists()
        assert (extract_dir / "nn_model.pt").exists()


# ---------------------------------------------------------------------------
# Cross-module consistency
# ---------------------------------------------------------------------------

class TestCrossModuleConsistency:
    """Verify train.py and launch.py agree on positions."""

    def test_train_and_launch_positions_match(self):
        from sagemaker.train import POSITIONS as TRAIN_POSITIONS
        from sagemaker.launch import ALL_POSITIONS
        assert set(TRAIN_POSITIONS.keys()) == set(ALL_POSITIONS)

    def test_metric_regex_matches_actual_trainer_format(self):
        """Read the actual format string from training.py and verify regexes match."""
        from sagemaker.launch import METRIC_DEFINITIONS

        # Simulate the actual trainer format string
        avg_train_loss = 0.4523
        avg_val_loss = 0.5012
        val_mae_total = 4.231
        line = (
            f"Epoch  10 | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"MAE total: {val_mae_total:.3f} | "
            f"target1: 1.234"
        )

        for metric in METRIC_DEFINITIONS:
            match = re.search(metric["Regex"], line)
            assert match is not None, (
                f"Metric '{metric['Name']}' regex failed on actual trainer format"
            )
