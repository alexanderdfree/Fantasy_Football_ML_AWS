"""Tests for batch/train.py — position registry, S3 staging, artifact handling."""

import argparse
import json
import os
import shutil
import sys
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
    """Validate the shared position registry against actual code."""

    def test_all_six_positions_registered(self):
        from shared.registry import ALL_POSITIONS

        assert set(ALL_POSITIONS) == {"QB", "RB", "WR", "TE", "K", "DST"}

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
    def test_standard_positions_accept_dataframes(self, pos):
        from shared.registry import accepts_dataframes

        assert accepts_dataframes(pos) is True

    @pytest.mark.parametrize("pos", ["K", "DST"])
    def test_special_positions_no_dataframes(self, pos):
        from shared.registry import accepts_dataframes

        assert accepts_dataframes(pos) is False

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
    def test_runner_function_importable(self, pos):
        from shared.registry import get_runner

        fn = get_runner(pos)
        assert callable(fn), f"{pos} runner is not callable"


# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    def test_position_required(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--position", required=True, choices=["QB", "RB", "WR", "TE", "K", "DST"]
        )
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_valid_position_accepted(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--position", required=True, choices=["QB", "RB", "WR", "TE", "K", "DST"]
        )
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args(["--position", "RB"])
        assert args.position == "RB"
        assert args.seed == 42

    def test_invalid_position_rejected(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--position", required=True, choices=["QB", "RB", "WR", "TE", "K", "DST"]
        )
        with pytest.raises(SystemExit):
            parser.parse_args(["--position", "INVALID"])

    def test_custom_seed(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--position", required=True, choices=["QB", "RB", "WR", "TE", "K", "DST"]
        )
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args(["--position", "QB", "--seed", "123"])
        assert args.seed == 123

    def test_ablation_flag_requires_known_name(self):
        """--ablation accepts 'rb-gate' but not arbitrary strings."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--position", required=True, choices=["QB", "RB", "WR", "TE", "K", "DST"]
        )
        parser.add_argument("--ablation", choices=["rb-gate"], default=None)
        args = parser.parse_args(["--position", "RB", "--ablation", "rb-gate"])
        assert args.ablation == "rb-gate"
        with pytest.raises(SystemExit):
            parser.parse_args(["--position", "RB", "--ablation", "nope"])

    def test_ablation_rb_gate_rejects_non_rb_position(self, capsys):
        """batch.train.main() must error when --ablation rb-gate is paired with
        a non-RB position — otherwise the ablation would clobber another
        position's run with bogus RB overrides."""
        from batch import train

        with (
            mock.patch.object(
                sys, "argv", ["train.py", "--position", "WR", "--ablation", "rb-gate"]
            ),
            pytest.raises(SystemExit),
        ):
            train.main()


# ---------------------------------------------------------------------------
# _assert_gpu: CPU-only bypass
# ---------------------------------------------------------------------------


class TestAssertGpu:
    def test_cpu_only_position_skips_require_gpu(self):
        """K/DST should never fail _assert_gpu even when REQUIRE_GPU=1."""
        from batch.train import _assert_gpu

        with (
            mock.patch.dict(os.environ, {"REQUIRE_GPU": "1"}),
            mock.patch("batch.train.torch.cuda.is_available", return_value=False),
        ):
            # Should NOT raise for K or DST
            _assert_gpu("K")
            _assert_gpu("DST")

    def test_gpu_position_raises_when_require_gpu_and_no_cuda(self):
        from batch.train import _assert_gpu

        with (
            mock.patch.dict(os.environ, {"REQUIRE_GPU": "1"}),
            mock.patch("batch.train.torch.cuda.is_available", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="REQUIRE_GPU=1"):
                _assert_gpu("RB")

    def test_gpu_position_passes_when_require_gpu_off(self):
        from batch.train import _assert_gpu

        with (
            mock.patch.dict(os.environ, {"REQUIRE_GPU": "0"}),
            mock.patch("batch.train.torch.cuda.is_available", return_value=False),
        ):
            _assert_gpu("RB")  # should not raise


# ---------------------------------------------------------------------------
# LOG_EVERY env-var plumbing (replaces old monkey-patch tests)
# ---------------------------------------------------------------------------


class TestResolveNnLogEvery:
    """shared.pipeline._resolve_nn_log_every is the new injection point."""

    def test_cfg_wins(self):
        from shared.pipeline import _resolve_nn_log_every

        with mock.patch.dict(os.environ, {"LOG_EVERY": "99"}):
            assert _resolve_nn_log_every({"nn_log_every": 3}) == 3

    def test_env_var_used_when_cfg_missing(self):
        from shared.pipeline import _resolve_nn_log_every

        with mock.patch.dict(os.environ, {"LOG_EVERY": "1"}):
            assert _resolve_nn_log_every({}) == 1

    def test_default_when_neither_set(self):
        from shared.pipeline import _resolve_nn_log_every

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOG_EVERY", None)
            assert _resolve_nn_log_every({}) == 10

    def test_non_int_env_var_falls_back_to_default(self):
        from shared.pipeline import _resolve_nn_log_every

        with mock.patch.dict(os.environ, {"LOG_EVERY": "not-a-number"}):
            assert _resolve_nn_log_every({}) == 10

    def test_null_cfg_value_treated_as_missing(self):
        from shared.pipeline import _resolve_nn_log_every

        with mock.patch.dict(os.environ, {"LOG_EVERY": "7"}):
            assert _resolve_nn_log_every({"nn_log_every": None}) == 7


# ---------------------------------------------------------------------------
# S3 data download logic
# ---------------------------------------------------------------------------


class TestDownloadData:
    @mock.patch("batch.train.boto3.client")
    def test_downloads_three_parquet_files(self, mock_boto_client):
        from batch.train import download_data

        mock_s3 = mock.MagicMock()
        mock_s3.head_object.return_value = {"ETag": '"abc123"'}
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

        mock_s3 = mock.MagicMock()
        mock_s3.head_object.return_value = {"ETag": '"abc123"'}
        mock_boto_client.return_value = mock_s3

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "nested", "dir")
            download_data("bucket", "prefix", nested)
            assert os.path.isdir(nested)


class TestDownloadIfStale:
    def test_skips_download_on_etag_match(self, tmp_path):
        from batch.train import _download_if_stale

        mock_s3 = mock.MagicMock()
        mock_s3.head_object.return_value = {"ETag": '"abc123"'}

        local = tmp_path / "train.parquet"
        local.write_text("cached")
        (tmp_path / "train.parquet.etag").write_text('"abc123"')

        _download_if_stale(mock_s3, "bucket", "prefix/train.parquet", str(local))

        mock_s3.download_file.assert_not_called()
        assert local.read_text() == "cached"

    def test_downloads_on_etag_mismatch(self, tmp_path):
        from batch.train import _download_if_stale

        mock_s3 = mock.MagicMock()
        mock_s3.head_object.return_value = {"ETag": '"newver"'}

        local = tmp_path / "train.parquet"
        local.write_text("stale")
        (tmp_path / "train.parquet.etag").write_text('"oldver"')

        _download_if_stale(mock_s3, "bucket", "prefix/train.parquet", str(local))

        mock_s3.download_file.assert_called_once_with("bucket", "prefix/train.parquet", str(local))
        assert (tmp_path / "train.parquet.etag").read_text() == '"newver"'

    def test_force_refresh_bypasses_cache(self, tmp_path, monkeypatch):
        from batch.train import _download_if_stale

        mock_s3 = mock.MagicMock()
        mock_s3.head_object.return_value = {"ETag": '"abc123"'}

        local = tmp_path / "train.parquet"
        local.write_text("cached")
        (tmp_path / "train.parquet.etag").write_text('"abc123"')

        monkeypatch.setenv("FF_FORCE_REFRESH", "1")
        _download_if_stale(mock_s3, "bucket", "prefix/train.parquet", str(local))

        mock_s3.download_file.assert_called_once()


# ---------------------------------------------------------------------------
# S3 artifact upload logic — empty-dir and missing-metrics guards
# ---------------------------------------------------------------------------


class TestUploadArtifacts:
    @mock.patch("batch.train.boto3.client")
    def test_uploads_tar_to_correct_s3_key(self, mock_boto_client):
        from batch.train import upload_artifacts

        mock_s3 = mock.MagicMock()
        mock_boto_client.return_value = mock_s3

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake model files + required metrics file
            Path(tmpdir, "ridge_model.pkl").write_text("fake")
            Path(tmpdir, "nn_model.pt").write_text("fake")
            Path(tmpdir, "benchmark_metrics.json").write_text("{}")

            upload_artifacts("my-bucket", "RB", tmpdir)

        mock_s3.upload_file.assert_called_once()
        call_args = mock_s3.upload_file.call_args
        assert call_args.args[1] == "my-bucket"
        assert call_args.args[2] == "models/RB/model.tar.gz"

    def test_raises_on_empty_model_dir(self, tmp_path):
        from batch.train import upload_artifacts

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RuntimeError, match="empty"):
            upload_artifacts("bucket", "RB", str(empty))

    def test_raises_when_model_dir_missing(self, tmp_path):
        from batch.train import upload_artifacts

        missing = tmp_path / "not-there"
        with pytest.raises(RuntimeError, match="does not exist"):
            upload_artifacts("bucket", "RB", str(missing))

    def test_raises_when_metrics_missing(self, tmp_path):
        from batch.train import upload_artifacts

        d = tmp_path / "m"
        d.mkdir()
        (d / "ridge_model.pkl").write_text("x")
        with pytest.raises(RuntimeError, match="benchmark_metrics.json"):
            upload_artifacts("bucket", "RB", str(d))


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

    def test_replace_model_dir_contents_clears_stale_and_copies_new(self, tmp_path):
        """dst pre-populated with a prior position's artifacts ends up
        containing only the new src contents (no accumulation).
        """
        from batch.train import _replace_model_dir_contents

        src = tmp_path / "RB" / "outputs" / "models"
        src.mkdir(parents=True)
        (src / "rb_ridge.pkl").write_text("new rb model")
        (src / "subdir").mkdir()
        (src / "subdir" / "leaf.bin").write_text("leaf")

        # Prior run left QB's artifacts behind in the mount dir.
        dst = tmp_path / "mount"
        dst.mkdir()
        (dst / "qb_ridge.pkl").write_text("stale qb model")
        stale_dir = dst / "stale_subdir"
        stale_dir.mkdir()
        (stale_dir / "stale_leaf.bin").write_text("stale")

        _replace_model_dir_contents(str(src), str(dst))

        assert dst.exists() and dst.is_dir()
        assert not (dst / "qb_ridge.pkl").exists()
        assert not stale_dir.exists()
        assert (dst / "rb_ridge.pkl").read_text() == "new rb model"
        assert (dst / "subdir" / "leaf.bin").read_text() == "leaf"

    def test_replace_model_dir_contents_does_not_rmtree_the_root(self, tmp_path, monkeypatch):
        """Regression test: on EC2 dst is a bind-mount that cannot be removed.
        The implementation must clear contents without passing dst itself to
        shutil.rmtree — otherwise the copytree fails with FileExistsError
        (the mount-point failure mode observed in run 24651387974).
        """
        import batch.train

        src = tmp_path / "src"
        src.mkdir()
        (src / "new.bin").write_text("new")

        dst = tmp_path / "mount"
        dst.mkdir()
        (dst / "stale.bin").write_text("stale")
        (dst / "stale_subdir").mkdir()

        real_rmtree = shutil.rmtree
        dst_str = str(dst)

        def guarded_rmtree(path, *args, **kwargs):
            assert str(path) != dst_str, (
                f"rmtree was called on the mount root {path!r} — on EC2 this "
                "silently leaves the dir in place and the next copytree raises "
                "FileExistsError. Clear children individually instead."
            )
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(batch.train.shutil, "rmtree", guarded_rmtree)

        batch.train._replace_model_dir_contents(str(src), str(dst))

        assert dst.exists()
        assert not (dst / "stale.bin").exists()
        assert not (dst / "stale_subdir").exists()
        assert (dst / "new.bin").read_text() == "new"


# ---------------------------------------------------------------------------
# Full main() integration test (mocked)
# ---------------------------------------------------------------------------


class TestMainIntegration:
    @mock.patch("batch.train.sync_raw_data")
    @mock.patch("batch.train.upload_artifacts")
    @mock.patch("batch.train.shutil.copytree")
    @mock.patch("batch.train.download_data")
    @mock.patch("batch.train.pd.read_parquet")
    def test_main_standard_position(
        self, mock_parquet, mock_download, mock_copytree, mock_upload, mock_sync, tmp_path
    ):
        import pandas as pd

        mock_df = pd.DataFrame({"col": [1, 2, 3]})
        mock_parquet.return_value = mock_df

        # main() clears model_dir's contents then copytree's into it before
        # writing metrics, so the mock must recreate the destination dir.
        mock_copytree.side_effect = lambda src, dst, **kw: Path(dst).mkdir(
            parents=True, exist_ok=True
        )

        runner_called = {}
        fake_mod = mock.MagicMock()

        # Pipeline must return a non-None result now — return a minimal metrics dict
        def fake_runner(train_df, val_df, test_df, seed=42):
            runner_called["args"] = (len(train_df), len(val_df), len(test_df), seed)
            return {"ridge_metrics": {"total": {"mae": 1.0, "r2": 0.5}}}

        fake_mod.fake_runner = fake_runner

        model_dir = tmp_path / "model"
        data_dir = tmp_path / "data"

        with (
            mock.patch("sys.argv", ["train.py", "--position", "RB"]),
            mock.patch.dict(
                os.environ,
                {
                    "S3_BUCKET": "test-bucket",
                    "TRAINING_DATA_DIR": str(data_dir),
                    "MODEL_OUTPUT_DIR": str(model_dir),
                    "REQUIRE_GPU": "0",
                },
            ),
            mock.patch("batch.train.get_runner", return_value=fake_runner),
            mock.patch("batch.train.accepts_dataframes", return_value=True),
        ):
            from batch.train import main

            main()

        mock_sync.assert_called_once_with("test-bucket")
        mock_download.assert_called_once()
        assert mock_parquet.call_count == 3
        assert "args" in runner_called
        mock_upload.assert_called_once()
        # Metrics file must have been written before upload
        metrics_path = model_dir / "benchmark_metrics.json"
        assert metrics_path.exists()
        # Timing must be threaded into the metrics so the EC2 history row
        # picks it up via summarize_pipeline_result().
        saved = json.loads(metrics_path.read_text())
        assert isinstance(saved["elapsed_sec"], (int, float))
        assert isinstance(saved["phase_seconds"], dict)
        assert "run_pipeline" in saved["phase_seconds"]

    @mock.patch("batch.train.sync_raw_data")
    @mock.patch("batch.train.upload_artifacts")
    @mock.patch("batch.train.shutil.copytree")
    def test_main_special_position_no_download(
        self, mock_copytree, mock_upload, mock_sync, tmp_path
    ):
        """main() for K/DST should skip download_data() (train/val/test splits) and
        REQUIRE_GPU. sync_raw_data() still runs for all positions — K/DST's
        self-contained loaders (and weather features) read from data/raw/.
        """
        # main() clears model_dir's contents then copytree's into it before
        # writing metrics, so the mock must recreate the destination dir.
        mock_copytree.side_effect = lambda src, dst, **kw: Path(dst).mkdir(
            parents=True, exist_ok=True
        )

        runner_called = {}

        def fake_k_runner(seed=42):
            runner_called["seed"] = seed
            return {"ridge_metrics": {"total": {"mae": 1.0, "r2": 0.5}}}

        model_dir = tmp_path / "model"

        with (
            mock.patch("sys.argv", ["train.py", "--position", "K"]),
            mock.patch.dict(os.environ, {"MODEL_OUTPUT_DIR": str(model_dir), "REQUIRE_GPU": "1"}),
            mock.patch("batch.train.get_runner", return_value=fake_k_runner),
            mock.patch("batch.train.accepts_dataframes", return_value=False),
            mock.patch("batch.train.torch.cuda.is_available", return_value=False),
        ):
            from batch.train import main

            # REQUIRE_GPU=1 with no CUDA would normally raise, but K's CPU-only
            # flag in the registry tells _assert_gpu to skip the check.
            main()

        assert runner_called["seed"] == 42
        mock_sync.assert_called_once_with("ff-predictor-training")
        mock_upload.assert_called_once()
        metrics_path = model_dir / "benchmark_metrics.json"
        assert metrics_path.exists()
        saved = json.loads(metrics_path.read_text())
        assert isinstance(saved["elapsed_sec"], (int, float))
        assert saved["phase_seconds"].keys() >= {"run_pipeline"}
