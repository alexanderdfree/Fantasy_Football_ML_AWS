"""Tests for src/batch/train.py — position registry, S3 staging, artifact handling."""

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from botocore.exceptions import ClientError

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _nosuchkey_error(key: str) -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": "NoSuchKey", "Message": f"{key} not found"}},
        operation_name="GetObject",
    )


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakePaginator:
    def __init__(self, objects: dict):
        self._objects = objects

    def paginate(self, Bucket: str, Prefix: str):  # noqa: N803
        contents = [{"Key": k} for k in self._objects if k.startswith(Prefix)]
        yield {"Contents": contents}


class _FakeS3Producer:
    """In-memory S3 fake supporting the operations ``upload_artifacts`` uses:
    ``upload_file`` (local → key), ``put_object`` (bytes → key), ``get_object``,
    ``list_objects_v2`` (via paginator), and ``delete_objects``.

    The ``ops`` list records every mutating call in order so tests can assert
    that the legacy mirror upload happens **after** the manifest put — the
    atomic-promotion invariant documented in src/batch/train.py::upload_artifacts.
    Missing keys raise ``ClientError`` with code ``NoSuchKey`` to mirror real
    boto3 semantics.
    """

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.ops: list[tuple[str, str]] = []

    def upload_file(self, local_path, Bucket, Key):  # noqa: N803
        with open(local_path, "rb") as f:
            self.objects[Key] = f.read()
        self.ops.append(("upload_file", Key))

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        if hasattr(Body, "read"):
            Body = Body.read()
        self.objects[Key] = Body
        self.ops.append(("put_object", Key))

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise _nosuchkey_error(Key)
        return {"Body": _FakeBody(self.objects[Key])}

    def get_paginator(self, op):
        assert op == "list_objects_v2"
        return _FakePaginator(self.objects)

    def delete_objects(self, Bucket, Delete):  # noqa: N803
        for obj in Delete["Objects"]:
            self.objects.pop(obj["Key"], None)
            self.ops.append(("delete", obj["Key"]))


def _write_fake_model_dir(d: Path, pos: str) -> None:
    """Populate ``d`` with the exact set of files the inference registry will
    expect for ``pos`` plus ``benchmark_metrics.json``. Keeps validation
    tests honest — if a future position adds a required file, the registry
    change flows straight through here via ``INFERENCE_REGISTRY[pos]``.
    """
    from src.shared.registry import INFERENCE_REGISTRY

    reg = INFERENCE_REGISTRY[pos]
    files = {
        reg["nn_file"]: b"fake-nn-weights",
        "nn_scaler.pkl": b"fake-scaler",
        "nn_scaler_meta.json": b"{}",
        "benchmark_metrics.json": b'{"position":"' + pos.encode() + b'"}',
    }
    if reg.get("train_attention_nn") and reg.get("attn_nn_file"):
        files[reg["attn_nn_file"]] = b"fake-attn-weights"
        files["attention_nn_scaler.pkl"] = b"fake-attn-scaler"
        files["attention_nn_scaler_meta.json"] = b"{}"
    # A single ridge file is enough to make the dir non-empty beyond the
    # required set; validation doesn't enforce Ridge's per-target layout.
    files["ridge_model.pkl"] = b"fake-ridge"
    for name, data in files.items():
        (d / name).write_bytes(data)


# ---------------------------------------------------------------------------
# Position registry tests
# ---------------------------------------------------------------------------


class TestPositionRegistry:
    """Validate the shared position registry against actual code."""

    def test_all_six_positions_registered(self):
        from src.shared.registry import ALL_POSITIONS

        assert set(ALL_POSITIONS) == {"QB", "RB", "WR", "TE", "K", "DST"}

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
    def test_standard_positions_accept_dataframes(self, pos):
        from src.shared.registry import accepts_dataframes

        assert accepts_dataframes(pos) is True

    @pytest.mark.parametrize("pos", ["K", "DST"])
    def test_special_positions_no_dataframes(self, pos):
        from src.shared.registry import accepts_dataframes

        assert accepts_dataframes(pos) is False

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
    def test_runner_function_importable(self, pos):
        from src.shared.registry import get_runner

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
        """src.batch.train.main() must error when --ablation rb-gate is paired with
        a non-RB position — otherwise the ablation would clobber another
        position's run with bogus RB overrides."""
        from src.batch import train

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
        from src.batch.train import _assert_gpu

        with (
            mock.patch.dict(os.environ, {"REQUIRE_GPU": "1"}),
            mock.patch("src.batch.train.torch.cuda.is_available", return_value=False),
        ):
            # Should NOT raise for K or DST
            _assert_gpu("K")
            _assert_gpu("DST")

    def test_gpu_position_raises_when_require_gpu_and_no_cuda(self):
        from src.batch.train import _assert_gpu

        with (
            mock.patch.dict(os.environ, {"REQUIRE_GPU": "1"}),
            mock.patch("src.batch.train.torch.cuda.is_available", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="REQUIRE_GPU=1"):
                _assert_gpu("RB")

    def test_gpu_position_passes_when_require_gpu_off(self):
        from src.batch.train import _assert_gpu

        with (
            mock.patch.dict(os.environ, {"REQUIRE_GPU": "0"}),
            mock.patch("src.batch.train.torch.cuda.is_available", return_value=False),
        ):
            _assert_gpu("RB")  # should not raise


# ---------------------------------------------------------------------------
# LOG_EVERY env-var plumbing (replaces old monkey-patch tests)
# ---------------------------------------------------------------------------


class TestResolveNnLogEvery:
    """src.shared.pipeline._resolve_nn_log_every is the new injection point."""

    def test_cfg_wins(self):
        from src.shared.pipeline import _resolve_nn_log_every

        with mock.patch.dict(os.environ, {"LOG_EVERY": "99"}):
            assert _resolve_nn_log_every({"nn_log_every": 3}) == 3

    def test_env_var_used_when_cfg_missing(self):
        from src.shared.pipeline import _resolve_nn_log_every

        with mock.patch.dict(os.environ, {"LOG_EVERY": "1"}):
            assert _resolve_nn_log_every({}) == 1

    def test_default_when_neither_set(self):
        from src.shared.pipeline import _resolve_nn_log_every

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOG_EVERY", None)
            assert _resolve_nn_log_every({}) == 10

    def test_non_int_env_var_falls_back_to_default(self):
        from src.shared.pipeline import _resolve_nn_log_every

        with mock.patch.dict(os.environ, {"LOG_EVERY": "not-a-number"}):
            assert _resolve_nn_log_every({}) == 10

    def test_null_cfg_value_treated_as_missing(self):
        from src.shared.pipeline import _resolve_nn_log_every

        with mock.patch.dict(os.environ, {"LOG_EVERY": "7"}):
            assert _resolve_nn_log_every({"nn_log_every": None}) == 7


# ---------------------------------------------------------------------------
# S3 data download logic
# ---------------------------------------------------------------------------


class TestDownloadData:
    @mock.patch("src.batch.train.boto3.client")
    def test_downloads_three_parquet_files(self, mock_boto_client):
        from src.batch.train import download_data

        mock_s3 = mock.MagicMock()
        mock_s3.head_object.return_value = {"ETag": '"abc123"'}
        mock_boto_client.return_value = mock_s3

        with tempfile.TemporaryDirectory() as tmpdir:
            download_data("my-bucket", "data", tmpdir)

        assert mock_s3.download_file.call_count == 3
        calls = [c.args for c in mock_s3.download_file.call_args_list]
        downloaded_keys = {c[1] for c in calls}
        assert downloaded_keys == {"data/train.parquet", "data/val.parquet", "data/test.parquet"}

    @mock.patch("src.batch.train.boto3.client")
    def test_creates_local_dir(self, mock_boto_client):
        from src.batch.train import download_data

        mock_s3 = mock.MagicMock()
        mock_s3.head_object.return_value = {"ETag": '"abc123"'}
        mock_boto_client.return_value = mock_s3

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "nested", "dir")
            download_data("bucket", "prefix", nested)
            assert os.path.isdir(nested)


class TestDownloadIfStale:
    def test_skips_download_on_etag_match(self, tmp_path):
        from src.batch.train import _download_if_stale

        mock_s3 = mock.MagicMock()
        mock_s3.head_object.return_value = {"ETag": '"abc123"'}

        local = tmp_path / "train.parquet"
        local.write_text("cached")
        (tmp_path / "train.parquet.etag").write_text('"abc123"')

        _download_if_stale(mock_s3, "bucket", "prefix/train.parquet", str(local))

        mock_s3.download_file.assert_not_called()
        assert local.read_text() == "cached"

    def test_downloads_on_etag_mismatch(self, tmp_path):
        from src.batch.train import _download_if_stale

        mock_s3 = mock.MagicMock()
        mock_s3.head_object.return_value = {"ETag": '"newver"'}

        local = tmp_path / "train.parquet"
        local.write_text("stale")
        (tmp_path / "train.parquet.etag").write_text('"oldver"')

        _download_if_stale(mock_s3, "bucket", "prefix/train.parquet", str(local))

        mock_s3.download_file.assert_called_once_with("bucket", "prefix/train.parquet", str(local))
        assert (tmp_path / "train.parquet.etag").read_text() == '"newver"'

    def test_force_refresh_bypasses_cache(self, tmp_path, monkeypatch):
        from src.batch.train import _download_if_stale

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
    """upload_artifacts ships to ``models/{POS}/history/{ts}-{sha}/model.tar.gz``,
    structurally validates the uploaded bytes, then atomically promotes via a
    manifest.json write. Legacy ``model.tar.gz`` is mirrored last for consumers
    running pre-manifest code. See the docstring on upload_artifacts."""

    @mock.patch("src.batch.train.boto3.client")
    def test_uploads_versioned_key_and_writes_manifest(self, mock_boto_client, tmp_path):
        from src.batch.train import upload_artifacts

        fake_s3 = _FakeS3Producer()
        mock_boto_client.return_value = fake_s3

        d = tmp_path / "model"
        d.mkdir()
        _write_fake_model_dir(d, "RB")

        upload_artifacts("my-bucket", "RB", str(d))

        # Exactly one versioned history key was written.
        history_keys = [k for k in fake_s3.objects if k.startswith("models/RB/history/")]
        assert len(history_keys) == 1
        history_key = history_keys[0]
        assert history_key.endswith("/model.tar.gz")

        # Manifest is present and points current at the versioned key.
        manifest = json.loads(fake_s3.objects["models/RB/manifest.json"])
        assert manifest["schema_version"] == 2
        assert manifest["current"]["key"] == history_key
        assert manifest["previous"] is None  # first write
        assert history_key in manifest["history"]
        # The fixture writes fake bytes that can't be deserialized → smoke
        # test fails → ``stable`` stays unset on this first upload.
        assert manifest["stable"] is None

        # Legacy mirror was written too (pre-manifest consumer compat).
        assert "models/RB/model.tar.gz" in fake_s3.objects
        # All three tarball bytes agree — otherwise pre-manifest consumers
        # would see different bytes than manifest-aware consumers.
        assert fake_s3.objects[history_key] == fake_s3.objects["models/RB/model.tar.gz"]

    @mock.patch("src.batch.train.boto3.client")
    def test_legacy_mirror_upload_happens_after_manifest_write(self, mock_boto_client, tmp_path):
        """Ordering invariant: the manifest put MUST be committed before the
        legacy mirror upload. A crash between the two leaves consumers on the
        freshly-promoted current; the reverse would briefly expose new bytes
        to old consumers without manifest-aware fallback being available."""
        from src.batch.train import upload_artifacts

        fake_s3 = _FakeS3Producer()
        mock_boto_client.return_value = fake_s3

        d = tmp_path / "model"
        d.mkdir()
        _write_fake_model_dir(d, "RB")

        upload_artifacts("my-bucket", "RB", str(d))

        manifest_idx = next(
            i for i, (_, k) in enumerate(fake_s3.ops) if k == "models/RB/manifest.json"
        )
        legacy_idx = next(
            i
            for i, (op, k) in enumerate(fake_s3.ops)
            if op == "upload_file" and k == "models/RB/model.tar.gz"
        )
        assert manifest_idx < legacy_idx, (
            f"manifest put must precede legacy mirror, got ops: {fake_s3.ops}"
        )

    @mock.patch("src.batch.train.boto3.client")
    def test_validation_rejects_missing_required_file(self, mock_boto_client, tmp_path):
        """Validation re-downloads the uploaded tarball and checks for
        required files. A missing nn_scaler.pkl must raise BEFORE the
        manifest write — otherwise a promoted bad artifact sticks."""
        from src.batch.train import upload_artifacts

        fake_s3 = _FakeS3Producer()
        mock_boto_client.return_value = fake_s3

        d = tmp_path / "model"
        d.mkdir()
        _write_fake_model_dir(d, "RB")
        # Remove a required file AFTER the dir was populated.
        (d / "nn_scaler.pkl").unlink()

        with pytest.raises(RuntimeError, match="missing required files"):
            upload_artifacts("my-bucket", "RB", str(d))

        # Manifest must NOT have been written — the promotion didn't happen.
        assert "models/RB/manifest.json" not in fake_s3.objects
        # Legacy mirror must NOT have been overwritten.
        assert "models/RB/model.tar.gz" not in fake_s3.objects

    @mock.patch("src.batch.train.boto3.client")
    def test_validation_detects_truncation(self, mock_boto_client, tmp_path):
        """If the uploaded bytes get truncated (replication lag, network blip),
        validation's tarfile reopen fails and the manifest stays on the
        previous good pointer. We simulate by intercepting the first
        upload_file to store only the first 32 bytes."""
        from src.batch.train import upload_artifacts

        fake_s3 = _FakeS3Producer()
        original_upload = fake_s3.upload_file

        def truncated_upload(local_path, Bucket, Key):  # noqa: N803
            with open(local_path, "rb") as f:
                fake_s3.objects[Key] = f.read()[:32]  # deliberately truncated
            fake_s3.ops.append(("upload_file", Key))

        mock_boto_client.return_value = fake_s3
        fake_s3.upload_file = truncated_upload  # type: ignore[method-assign]

        d = tmp_path / "model"
        d.mkdir()
        _write_fake_model_dir(d, "RB")

        try:
            with pytest.raises((RuntimeError, tarfile.TarError, OSError, EOFError)):
                upload_artifacts("my-bucket", "RB", str(d))
        finally:
            fake_s3.upload_file = original_upload  # type: ignore[method-assign]

        assert "models/RB/manifest.json" not in fake_s3.objects

    @mock.patch("src.batch.train.boto3.client")
    def test_second_upload_promotes_old_current_to_previous(self, mock_boto_client, tmp_path):
        """After two back-to-back uploads, manifest.previous must equal the
        first upload's current. This is the rollback path: if upload #2's
        artifact later fails to load, ``src.shared.model_sync._sync_one`` falls
        back to #1's versioned key via manifest.previous."""
        from src.batch.train import upload_artifacts

        fake_s3 = _FakeS3Producer()
        mock_boto_client.return_value = fake_s3

        d = tmp_path / "model"
        d.mkdir()
        _write_fake_model_dir(d, "RB")

        upload_artifacts("my-bucket", "RB", str(d))
        first_manifest = json.loads(fake_s3.objects["models/RB/manifest.json"])
        first_current_key = first_manifest["current"]["key"]

        # Second upload with slightly different bytes so sha7 differs.
        (d / "benchmark_metrics.json").write_bytes(b'{"position":"RB","round":2}')
        upload_artifacts("my-bucket", "RB", str(d))

        second_manifest = json.loads(fake_s3.objects["models/RB/manifest.json"])
        assert second_manifest["previous"] is not None
        assert second_manifest["previous"]["key"] == first_current_key
        assert second_manifest["current"]["key"] != first_current_key
        # Both versioned artifacts remain in S3 — the fallback has bytes to
        # serve from.
        assert first_current_key in fake_s3.objects
        assert second_manifest["current"]["key"] in fake_s3.objects

    @mock.patch("src.batch.train.boto3.client")
    def test_manifest_validates_end_to_end_with_consumer(
        self, mock_boto_client, tmp_path, monkeypatch
    ):
        """Contract test: what upload_artifacts writes, src.shared.model_sync can
        read. Uses a real RB tarball layout (via _write_fake_model_dir) and
        src.shared.model_sync._sync_one against the same fake S3. If the producer
        ever changes the manifest schema without updating the consumer, this
        test breaks."""
        from src.batch.train import upload_artifacts
        from src.shared import model_sync

        fake_s3 = _FakeS3Producer()
        mock_boto_client.return_value = fake_s3

        d = tmp_path / "model"
        d.mkdir()
        _write_fake_model_dir(d, "RB")
        upload_artifacts("my-bucket", "RB", str(d))

        dest_root = tmp_path / "consumer_root"
        result = model_sync._sync_one(fake_s3, "my-bucket", "models", "RB", dest_root)

        assert result["source"] == "current"
        assert (dest_root / "RB" / "outputs" / "models" / "nn_scaler.pkl").is_file()
        assert (dest_root / "RB" / "outputs" / "models" / "rb_multihead_nn.pt").is_file()

    def test_raises_on_empty_model_dir(self, tmp_path):
        from src.batch.train import upload_artifacts

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RuntimeError, match="empty"):
            upload_artifacts("bucket", "RB", str(empty))

    def test_raises_when_model_dir_missing(self, tmp_path):
        from src.batch.train import upload_artifacts

        missing = tmp_path / "not-there"
        with pytest.raises(RuntimeError, match="does not exist"):
            upload_artifacts("bucket", "RB", str(missing))

    def test_raises_when_metrics_missing(self, tmp_path):
        from src.batch.train import upload_artifacts

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
        from src.batch.train import _extract_metrics

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
        from src.batch.train import _extract_metrics

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
        from src.batch.train import _replace_model_dir_contents

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
        import src.batch.train as _batch_train

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

        monkeypatch.setattr(_batch_train.shutil, "rmtree", guarded_rmtree)

        _batch_train._replace_model_dir_contents(str(src), str(dst))

        assert dst.exists()
        assert not (dst / "stale.bin").exists()
        assert not (dst / "stale_subdir").exists()
        assert (dst / "new.bin").read_text() == "new"


# ---------------------------------------------------------------------------
# Full main() integration test (mocked)
# ---------------------------------------------------------------------------


class TestMainIntegration:
    @mock.patch("src.batch.train.sync_raw_data")
    @mock.patch("src.batch.train.upload_artifacts")
    @mock.patch("src.batch.train.shutil.copytree")
    @mock.patch("src.batch.train.download_data")
    @mock.patch("src.batch.train.pd.read_parquet")
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
            mock.patch("src.batch.train.get_runner", return_value=fake_runner),
            mock.patch("src.batch.train.accepts_dataframes", return_value=True),
        ):
            from src.batch.train import main

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

    @mock.patch("src.batch.train.sync_raw_data")
    @mock.patch("src.batch.train.upload_artifacts")
    @mock.patch("src.batch.train.shutil.copytree")
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
            mock.patch("src.batch.train.get_runner", return_value=fake_k_runner),
            mock.patch("src.batch.train.accepts_dataframes", return_value=False),
            mock.patch("src.batch.train.torch.cuda.is_available", return_value=False),
        ):
            from src.batch.train import main

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
