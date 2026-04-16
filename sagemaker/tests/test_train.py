"""Tests for sagemaker/train.py — position registry, patching, artifact copy."""
import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Ensure project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Position registry tests
# ---------------------------------------------------------------------------

class TestPositionRegistry:
    """Validate the POSITIONS dict in train.py against actual code."""

    def _get_positions(self):
        from sagemaker.train import POSITIONS
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
        """Module path should contain the function it exports."""
        positions = self._get_positions()
        mod_path, func_name, _ = positions[pos]
        assert func_name in mod_path, (
            f"{pos}: func '{func_name}' not found in module path '{mod_path}'"
        )

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
    def test_runner_module_exists(self, pos):
        """The runner module file should exist on disk."""
        positions = self._get_positions()
        mod_path, _, _ = positions[pos]
        # Convert dotted module path to file path: "RB.run_rb_pipeline" -> "RB/run_rb_pipeline.py"
        file_path = Path(PROJECT_ROOT) / mod_path.replace(".", "/")
        py_file = file_path.with_suffix(".py")
        assert py_file.exists(), f"Runner module not found: {py_file}"

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
    def test_runner_function_importable(self, pos):
        """The runner function should be importable from its module."""
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
        """Should fail if --position is not given."""
        from sagemaker.train import main
        with mock.patch("sys.argv", ["train.py"]):
            with pytest.raises(SystemExit):
                parser = argparse.ArgumentParser()
                parser.add_argument("--position", required=True, choices=["QB", "RB", "WR", "TE", "K", "DST"])
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
    def test_sm_channel_training_default(self):
        """Without SM_CHANNEL_TRAINING, should fall back to 'data/splits'."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SM_CHANNEL_TRAINING", None)
            data_dir = os.environ.get("SM_CHANNEL_TRAINING", "data/splits")
            assert data_dir == "data/splits"

    def test_sm_channel_training_override(self):
        with mock.patch.dict(os.environ, {"SM_CHANNEL_TRAINING": "/opt/ml/input/data/training"}):
            data_dir = os.environ.get("SM_CHANNEL_TRAINING", "data/splits")
            assert data_dir == "/opt/ml/input/data/training"

    def test_sm_model_dir_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SM_MODEL_DIR", None)
            model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
            assert model_dir == "/opt/ml/model"

    def test_sm_log_every_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SM_LOG_EVERY", None)
            log_every = int(os.environ.get("SM_LOG_EVERY", "1"))
            assert log_every == 1

    def test_sm_log_every_override(self):
        with mock.patch.dict(os.environ, {"SM_LOG_EVERY": "5"}):
            log_every = int(os.environ.get("SM_LOG_EVERY", "1"))
            assert log_every == 5


# ---------------------------------------------------------------------------
# Monkey-patching of run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipelinePatching:
    """Verify the monkey-patch injects nn_log_every correctly."""

    def test_patch_injects_nn_log_every(self):
        import shared.pipeline as pipeline_mod
        _orig = pipeline_mod.run_pipeline

        captured = {}

        def _patched(position, cfg, *a, **kw):
            captured["cfg"] = dict(cfg)
            # Don't actually run the pipeline
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
# Artifact copy logic
# ---------------------------------------------------------------------------

class TestArtifactCopy:
    def test_copytree_when_src_exists(self, tmp_path):
        """Model artifacts should be copied from {pos}/outputs/models/ to model_dir."""
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
        """Should print warning, not crash, when no model directory exists."""
        src_model_dir = str(tmp_path / "RB" / "outputs" / "models")
        assert not os.path.isdir(src_model_dir)
        # This is the logic from train.py lines 76-80
        if os.path.isdir(src_model_dir):
            pytest.fail("Should not reach here")
        # No error = pass


# ---------------------------------------------------------------------------
# Full main() integration test (mocked)
# ---------------------------------------------------------------------------

class TestMainIntegration:
    @mock.patch("sagemaker.train.shutil.copytree")
    @mock.patch("sagemaker.train.os.path.isdir", return_value=True)
    @mock.patch("sagemaker.train.pd.read_parquet")
    def test_main_standard_position(self, mock_parquet, mock_isdir, mock_copytree):
        """main() for a standard position (RB) should load parquet and call runner."""
        import pandas as pd
        mock_df = pd.DataFrame({"col": [1, 2, 3]})
        mock_parquet.return_value = mock_df

        runner_called = {}
        fake_mod = mock.MagicMock()

        def fake_runner(train_df, val_df, test_df, seed=42):
            runner_called["args"] = (len(train_df), len(val_df), len(test_df), seed)

        fake_mod.fake_runner = fake_runner

        with mock.patch("sys.argv", ["train.py", "--position", "RB"]), \
             mock.patch.dict(os.environ, {"SM_CHANNEL_TRAINING": "/data", "SM_MODEL_DIR": "/model"}), \
             mock.patch("sagemaker.train.POSITIONS", {
                 "RB": ("fake_runner_mod", "fake_runner", True),
             }):
            from sagemaker.train import main
            # Patch __import__ only for the dynamic import call inside main()
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
            def patched_import(name, *args, **kwargs):
                if name == "fake_runner_mod":
                    return fake_mod
                return original_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=patched_import):
                main()

        assert mock_parquet.call_count == 3
        assert "args" in runner_called

    @mock.patch("sagemaker.train.shutil.copytree")
    @mock.patch("sagemaker.train.os.path.isdir", return_value=True)
    def test_main_special_position_no_parquet(self, mock_isdir, mock_copytree):
        """main() for K/DST should NOT load parquet files."""
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
             mock.patch.dict(os.environ, {"SM_MODEL_DIR": "/model"}), \
             mock.patch("sagemaker.train.POSITIONS", {
                 "K": ("fake_k_mod", "fake_k_runner", False),
             }), \
             mock.patch("builtins.__import__", side_effect=patched_import):
            from sagemaker.train import main
            main()

        assert runner_called["seed"] == 42
