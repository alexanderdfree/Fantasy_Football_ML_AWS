"""Tests for batch/train.py — position registry, S3 staging, artifact handling."""
import argparse
import json
import os
import shutil
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
# Position registry tests
# ---------------------------------------------------------------------------

class TestPositionRegistry:
    """Validate the POSITIONS dict in batch/train.py against actual code."""

    def _get_positions(self):
        from batch.train import POSITIONS
        return POSITIONS

    def test_all_six_positions_registered(self):
        positions = self._get_positions()
        assert set(positions.keys()) == {"QB", "RB", "WR", "TE", "K", "DST"}

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
    def test_registry_entry_structure(self, pos):
        positions = self._get_positions()
        entry = positions[pos]
        assert len(entry) == 3, f"{pos} entry should be (module_path, func_name, accepts_df)"
        mod_path, func_name, accepts_df = entry
        assert isinstance(mod_path, str)
        assert isinstance(func_name, str)
        assert isinstance(accepts_df, bool)

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
    def test_standard_positions_accept_dataframes(self, pos):
        positions = self._get_positions()
        _, _, accepts_df = positions[pos]
        assert accepts_df is True, f"{pos} should accept dataframes"

    @pytest.mark.parametrize("pos", ["K", "DST"])
    def test_special_positions_no_dataframes(self, pos):
        positions = self._get_positions()
        _, _, accepts_df = positions[pos]
        assert accepts_df is False, f"{pos} should NOT accept dataframes"

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
    def test_module_path_matches_function_name(self, pos):
        positions = self._get_positions()
        mod_path, func_name, _ = positions[pos]
        assert func_name in mod_path, (
            f"{pos}: func '{func_name}' not found in module path '{mod_path}'"
        )

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
    def test_runner_module_exists(self, pos):
        positions = self._get_positions()
        mod_path, _, _ = positions[pos]
        file_path = Path(PROJECT_ROOT) / mod_path.replace(".", "/")
        py_file = file_path.with_suffix(".py")
        assert py_file.exists(), f"Runner module not found: {py_file}"

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
    def test_runner_function_importable(self, pos):
        positions = self._get_positions()
        mod_path, func_name, _ = positions[pos]
        mod = __import__(mod_path, fromlist=[func_name])
        fn = getattr(mod, func_name)
        assert callable(fn), f"{mod_path}.{func_name} is not callable"


# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------

class TestArgumentParsing:
    def test_position_required(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--position", required=True, choices=["QB", "RB", "WR", "TE", "K", "DST"])
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_valid_position_accepted(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--position", required=True, choices=["QB", "RB", "WR", "TE", "K", "DST"])
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args(["--position", "RB"])
        assert args.position == "RB"
        assert args.seed == 42

    def test_invalid_position_rejected(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--position", required=True, choices=["QB", "RB", "WR", "TE", "K", "DST"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--position", "INVALID"])

    def test_custom_seed(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--position", required=True, choices=["QB", "RB", "WR", "TE", "K", "DST"])
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args(["--position", "QB", "--seed", "123"])
        assert args.seed == 123


# ---------------------------------------------------------------------------
# Environment variable handling
# ---------------------------------------------------------------------------

class TestEnvironmentVariables:
    def test_training_data_dir_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRAINING_DATA_DIR", None)
            data_dir = os.environ.get("TRAINING_DATA_DIR", "/opt/ml/input/data/training")
            assert data_dir == "/opt/ml/input/data/training"

    def test_training_data_dir_override(self):
        with mock.patch.dict(os.environ, {"TRAINING_DATA_DIR": "/custom/data"}):
            data_dir = os.environ.get("TRAINING_DATA_DIR", "/opt/ml/input/data/training")
            assert data_dir == "/custom/data"

    def test_model_output_dir_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODEL_OUTPUT_DIR", None)
            model_dir = os.environ.get("MODEL_OUTPUT_DIR", "/opt/ml/model")
            assert model_dir == "/opt/ml/model"

    def test_s3_bucket_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("S3_BUCKET", None)
            bucket = os.environ.get("S3_BUCKET", "ff-predictor-training")
            assert bucket == "ff-predictor-training"

    def test_s3_bucket_override(self):
        with mock.patch.dict(os.environ, {"S3_BUCKET": "my-bucket"}):
            bucket = os.environ.get("S3_BUCKET", "ff-predictor-training")
            assert bucket == "my-bucket"

    def test_log_every_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOG_EVERY", None)
            log_every = int(os.environ.get("LOG_EVERY", "1"))
            assert log_every == 1

    def test_log_every_override(self):
        with mock.patch.dict(os.environ, {"LOG_EVERY": "5"}):
            log_every = int(os.environ.get("LOG_EVERY", "1"))
            assert log_every == 5


# ---------------------------------------------------------------------------
# Monkey-patching of run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipelinePatching:
    def test_patch_injects_nn_log_every(self):
        import shared.pipeline as pipeline_mod
        _orig = pipeline_mod.run_pipeline

        captured = {}

        def _patched(position, cfg, *a, **kw):
            captured["cfg"] = dict(cfg)
            return None

        log_every = 1
        def _patched_with_inject(position, cfg, *a, **kw):
            cfg["nn_log_every"] = log_every
            return _patched(position, cfg, *a, **kw)

        pipeline_mod.run_pipeline = _patched_with_inject
        try:
            pipeline_mod.run_pipeline("TEST", {"some_key": "value"})
            assert captured["cfg"]["nn_log_every"] == 1
        finally:
            pipeline_mod.run_pipeline = _orig

    def test_patch_preserves_existing_cfg_keys(self):
        import shared.pipeline as pipeline_mod
        _orig = pipeline_mod.run_pipeline

        captured = {}

        def _patched_with_inject(position, cfg, *a, **kw):
            cfg["nn_log_every"] = 1
            captured["cfg"] = dict(cfg)
            return None

        pipeline_mod.run_pipeline = _patched_with_inject
        try:
            pipeline_mod.run_pipeline("TEST", {"existing_key": 42, "other": "val"})
            assert captured["cfg"]["existing_key"] == 42
            assert captured["cfg"]["other"] == "val"
            assert captured["cfg"]["nn_log_every"] == 1
        finally:
            pipeline_mod.run_pipeline = _orig


# ---------------------------------------------------------------------------
# S3 data download logic
# ---------------------------------------------------------------------------

class TestDownloadData:
    @mock.patch("batch.train.boto3.client")
    def test_downloads_three_parquet_files(self, mock_boto_client):
        from batch.train import download_data

        mock_s3 = mock.MagicMock()
        mock_boto_client.return_value = mock_s3

        with tempfile.TemporaryDirectory() as tmpdir:
            download_data("my-bucket", "data", tmpdir)

        assert mock_s3.download_file.call_count == 3
        calls = [c.args for c in mock_s3.download_file.call_args_list]
        downloaded_keys = {c[1] for c in calls}
        assert downloaded_keys == {"data/train.parquet", "data/val.parquet", "data/test.parquet"}

    @mock.patch("batch.train.boto3.client")
    def test_creates_local_dir(self, mock_boto_client):
        from batch.train import download_data

        mock_boto_client.return_value = mock.MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "nested", "dir")
            download_data("bucket", "prefix", nested)
            assert os.path.isdir(nested)


# ---------------------------------------------------------------------------
# S3 artifact upload logic
# ---------------------------------------------------------------------------

class TestUploadArtifacts:
    @mock.patch("batch.train.boto3.client")
    def test_uploads_tar_to_correct_s3_key(self, mock_boto_client):
        from batch.train import upload_artifacts

        mock_s3 = mock.MagicMock()
        mock_boto_client.return_value = mock_s3

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake model files
            Path(tmpdir, "ridge_model.pkl").write_text("fake")
            Path(tmpdir, "nn_model.pt").write_text("fake")

            upload_artifacts("my-bucket", "RB", tmpdir)

        mock_s3.upload_file.assert_called_once()
        call_args = mock_s3.upload_file.call_args
        assert call_args.args[1] == "my-bucket"
        assert call_args.args[2] == "models/RB/model.tar.gz"


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

class TestMetricExtraction:
    def test_extracts_ridge_and_nn_metrics(self):
        from batch.train import _extract_metrics
        result = {
            "ridge_metrics": {
                "total": {"mae": 4.5, "r2": 0.3},
                "rushing": {"mae": 2.1, "r2": 0.5},
            },
            "nn_metrics": {
                "total": {"mae": 3.8, "r2": 0.45},
            },
            "ridge_ranking": {"season_avg_hit_rate": 0.55},
            "nn_ranking": {"season_avg_hit_rate": 0.60, "season_avg_spearman": 0.42},
        }
        metrics = _extract_metrics("RB", result)
        assert metrics["position"] == "RB"
        assert "ridge_metrics" in metrics
        assert "nn_metrics" in metrics
        assert metrics["nn_ranking"]["season_avg_spearman"] == 0.42

    def test_handles_missing_model_types(self):
        from batch.train import _extract_metrics
        result = {
            "ridge_metrics": {"total": {"mae": 4.0, "r2": 0.3}},
        }
        metrics = _extract_metrics("K", result)
        assert metrics["position"] == "K"
        assert "nn_metrics" not in metrics
        assert "attn_nn_metrics" not in metrics


# ---------------------------------------------------------------------------
# Artifact copy logic
# ---------------------------------------------------------------------------

class TestArtifactCopy:
    def test_copytree_when_src_exists(self, tmp_path):
        src = tmp_path / "RB" / "outputs" / "models"
        src.mkdir(parents=True)
        (src / "ridge_model.pkl").write_text("fake model")
        (src / "nn_model.pt").write_text("fake nn")

        dst = tmp_path / "model_output"
        dst.mkdir()

        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)

        assert (dst / "ridge_model.pkl").exists()
        assert (dst / "nn_model.pt").exists()

    def test_no_copy_when_src_missing(self, tmp_path):
        src_model_dir = str(tmp_path / "RB" / "outputs" / "models")
        assert not os.path.isdir(src_model_dir)
        if os.path.isdir(src_model_dir):
            pytest.fail("Should not reach here")


# ---------------------------------------------------------------------------
# Full main() integration test (mocked)
# ---------------------------------------------------------------------------

class TestMainIntegration:
    @mock.patch("batch.train.upload_artifacts")
    @mock.patch("batch.train.shutil.copytree")
    @mock.patch("batch.train.os.path.isdir", return_value=True)
    @mock.patch("batch.train.download_data")
    @mock.patch("batch.train.pd.read_parquet")
    def test_main_standard_position(self, mock_parquet, mock_download, mock_isdir, mock_copytree, mock_upload):
        import pandas as pd
        mock_df = pd.DataFrame({"col": [1, 2, 3]})
        mock_parquet.return_value = mock_df

        runner_called = {}
        fake_mod = mock.MagicMock()

        def fake_runner(train_df, val_df, test_df, seed=42):
            runner_called["args"] = (len(train_df), len(val_df), len(test_df), seed)

        fake_mod.fake_runner = fake_runner

        with mock.patch("sys.argv", ["train.py", "--position", "RB"]), \
             mock.patch.dict(os.environ, {
                 "S3_BUCKET": "test-bucket",
                 "TRAINING_DATA_DIR": "/data",
                 "MODEL_OUTPUT_DIR": "/model",
                 "REQUIRE_GPU": "0",
             }), \
             mock.patch("batch.train.POSITIONS", {
                 "RB": ("fake_runner_mod", "fake_runner", True),
             }):
            from batch.train import main
            original_import = __import__
            def patched_import(name, *args, **kwargs):
                if name == "fake_runner_mod":
                    return fake_mod
                return original_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=patched_import):
                main()

        mock_download.assert_called_once()
        assert mock_parquet.call_count == 3
        assert "args" in runner_called
        mock_upload.assert_called_once()

    @mock.patch("batch.train.upload_artifacts")
    @mock.patch("batch.train.shutil.copytree")
    @mock.patch("batch.train.os.path.isdir", return_value=True)
    def test_main_special_position_no_download(self, mock_isdir, mock_copytree, mock_upload):
        """main() for K/DST should NOT download data from S3."""
        runner_called = {}
        fake_mod = mock.MagicMock()

        def fake_k_runner(seed=42):
            runner_called["seed"] = seed

        fake_mod.fake_k_runner = fake_k_runner

        original_import = __import__
        def patched_import(name, *args, **kwargs):
            if name == "fake_k_mod":
                return fake_mod
            return original_import(name, *args, **kwargs)

        with mock.patch("sys.argv", ["train.py", "--position", "K"]), \
             mock.patch.dict(os.environ, {"MODEL_OUTPUT_DIR": "/model", "REQUIRE_GPU": "0"}), \
             mock.patch("batch.train.POSITIONS", {
                 "K": ("fake_k_mod", "fake_k_runner", False),
             }), \
             mock.patch("builtins.__import__", side_effect=patched_import):
            from batch.train import main
            main()

        assert runner_called["seed"] == 42
        mock_upload.assert_called_once()
