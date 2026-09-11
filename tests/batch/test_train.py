"""Tests for src/batch/train.py — position registry, S3 staging, artifact handling."""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
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

    The ``ops`` list records writes so tests can verify output claiming precedes
    promotion and protected publication never rewrites legacy objects.
    Missing keys raise ``ClientError`` with code ``NoSuchKey`` to mirror real
    boto3 semantics.
    """

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.ops: list[tuple[str, str]] = []
        self.metadata: dict[str, dict] = {}

    def upload_file(self, local_path, Bucket, Key, ExtraArgs=None):  # noqa: N803
        with open(local_path, "rb") as f:
            self.objects[Key] = f.read()
        self.ops.append(("upload_file", Key))
        self.metadata[Key] = (ExtraArgs or {}).get("Metadata", {})

    def put_object(self, Bucket, Key, Body, ContentType=None, IfMatch=None, IfNoneMatch=None):  # noqa: N803
        existing = self.objects.get(Key)
        etag = None if existing is None else hashlib.sha256(existing).hexdigest()
        if (IfMatch is not None and IfMatch != etag) or (
            IfNoneMatch == "*" and existing is not None
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if hasattr(Body, "read"):
            Body = Body.read()
        self.objects[Key] = Body
        self.ops.append(("put_object", Key))

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise _nosuchkey_error(Key)
        return {
            "Body": _FakeBody(self.objects[Key]),
            "ETag": hashlib.sha256(self.objects[Key]).hexdigest(),
        }

    def download_file(self, Bucket, Key, Filename):
        Path(Filename).write_bytes(self.get_object(Bucket, Key)["Body"].read())

    def get_paginator(self, op):
        assert op == "list_objects_v2"
        return _FakePaginator(self.objects)

    def delete_objects(self, Bucket, Delete):  # noqa: N803
        for obj in Delete["Objects"]:
            self.objects.pop(obj["Key"], None)
            self.ops.append(("delete", obj["Key"]))


def _write_fake_model_dir(d: Path, pos: str, *, metrics=None) -> None:
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
        "benchmark_metrics.json": json.dumps({"position": pos, **(metrics or {})}).encode(),
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


def _publication_metrics(
    store,
    monkeypatch,
    image_root,
    *,
    position="RB",
    bucket="bucket",
    run_id="run",
    source_sha="d" * 40,
    ancestors=(),
):
    """Model an authenticated image and real source/plan/intent/receipt contracts."""
    from src.artifacts import publication, source
    from src.data.release import producer_fingerprint
    from src.orchestration.build_plan import create_plan
    from src.scripts import wait_data_release
    from tests.orchestration.test_build_plan import Batch as PlanBatch
    from tests.orchestration.test_datasets import write_release

    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "models")
    image = image_root / source_sha
    image.mkdir(parents=True, exist_ok=True)
    (image / ".training-source-sha").write_text(source_sha + "\n")
    read_image = source.image_source_sha

    def image_source(*, root=None):
        return read_image(root=image if root is None else root)

    monkeypatch.setattr(source, "image_source_sha", image_source)
    monkeypatch.setattr(publication, "image_source_sha", image_source)
    lineage = [source_sha, *ancestors]
    store.objects[source.source_key("models", source_sha)] = json.dumps(
        {"source_sha": source_sha, "source_order": len(lineage), "lineage": lineage}
    ).encode()
    producer_hashes = {"src/config.py": hashlib.sha256(b"fixture").hexdigest()}
    monkeypatch.setattr(
        wait_data_release, "producer_hashes_at_revision", lambda *_, **__: producer_hashes
    )
    source_id = producer_fingerprint(producer_hashes)
    dataset_id = write_release(store, producer_hashes)

    class Batch(PlanBatch):
        def describe_job_definitions(self, **kwargs):
            return {
                "jobDefinitions": [
                    {
                        "revision": 1,
                        "status": "ACTIVE",
                        "jobDefinitionName": "train",
                        "jobDefinitionArn": "arn:aws:batch:definition/train:1",
                        "containerProperties": {"image": "registry/train:" + source_sha},
                    }
                ]
            }

    plan_id, plan = create_plan(
        store,
        Batch(),
        bucket,
        dataset_id=dataset_id,
        source_id=source_id,
        code_sha=source_sha,
        gpu_definition="train",
        positions=[position],
        seed=42,
        run_id=run_id,
    )
    return {
        "position": position,
        "git_sha": source_sha,
        "dataset_id": dataset_id,
        "data_release": dataset_id,
        "data_format": "data-release-v1",
        "build_plan_id": plan_id,
        "image_id": "registry/train:" + source_sha,
        "publication_intent": plan["intents"][position],
        "publication_revision": plan["publication_revisions"][position],
    }


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
    def test_cpu_only_position_skips_require_gpu(self, capsys):
        """K/DST should never fail _assert_gpu even when REQUIRE_GPU=1, and the
        skip must be logged so an EC2 run reading container logs can confirm
        the bypass was deliberate (not a missed REQUIRE_GPU check)."""
        from src.batch.train import _assert_gpu

        with (
            mock.patch.dict(os.environ, {"REQUIRE_GPU": "1"}),
            mock.patch("src.batch.train.torch.cuda.is_available", return_value=False),
        ):
            _assert_gpu("K")
            _assert_gpu("DST")

        out = capsys.readouterr().out
        assert "K is CPU-only; skipping REQUIRE_GPU assertion" in out
        assert "DST is CPU-only; skipping REQUIRE_GPU assertion" in out

    def test_gpu_position_raises_when_require_gpu_and_no_cuda(self):
        from src.batch.train import _assert_gpu

        with (
            mock.patch.dict(os.environ, {"REQUIRE_GPU": "1"}),
            mock.patch("src.batch.train.torch.cuda.is_available", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="REQUIRE_GPU=1"):
                _assert_gpu("RB")

    def test_force_gpu_assertion_applies_to_cpu_only_positions(self):
        from src.batch.train import _assert_gpu

        with (
            mock.patch.dict(os.environ, {"REQUIRE_GPU": "1"}),
            mock.patch("src.batch.train.torch.cuda.is_available", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="REQUIRE_GPU=1"):
                _assert_gpu("K", force=True)

    def test_gpu_position_passes_when_require_gpu_off(self, capsys):
        """REQUIRE_GPU=0 lets GPU positions run on CPU without raising — local
        sanity checks on non-GPU boxes rely on this. The "not CPU-only skip"
        assertion guards against a regression where is_cpu_only("RB") flipped
        to True and silently bypassed the env check."""
        from src.batch.train import _assert_gpu

        with (
            mock.patch.dict(os.environ, {"REQUIRE_GPU": "0"}),
            mock.patch("src.batch.train.torch.cuda.is_available", return_value=False),
        ):
            _assert_gpu("RB")

        out = capsys.readouterr().out
        assert "torch.cuda.is_available()" in out
        assert "is CPU-only; skipping REQUIRE_GPU assertion" not in out


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
    @pytest.fixture(autouse=True)
    def explicit_legacy_download(self, monkeypatch):
        monkeypatch.setenv("FF_DATA_RELEASE", "legacy")

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


@pytest.mark.unit
class TestUploadArtifacts:
    """upload_artifacts ships to protected v3 history with authenticated receipts,
    structurally validates the uploaded bytes, then atomically promotes via a
    manifest.json write. The legacy ``models/{POS}/model.tar.gz`` mirror was
    removed in Layer C of the parallel-train-batch race fix — see the
    docstring on upload_artifacts."""

    def test_identified_upload_writes_exact_receipt_and_plan_ownership(self, monkeypatch, tmp_path):
        from src.batch import train

        fake = _FakeS3Producer()
        monkeypatch.setattr(train.boto3, "client", lambda *_: fake)
        monkeypatch.setattr(train, "_try_smoke_test", lambda *_: True)
        metrics = _publication_metrics(fake, monkeypatch, tmp_path / "image")
        _write_fake_model_dir(tmp_path, "RB", metrics=metrics)
        train.upload_artifacts("bucket", "RB", str(tmp_path))
        receipt_key = f"build-plans/{metrics['build_plan_id']}/artifacts/RB.json"
        receipt = json.loads(fake.objects[receipt_key])
        manifest = json.loads(fake.objects["models/releases/v3/RB/manifest.json"])
        assert receipt["key"] == manifest["stable"]["key"]
        assert receipt["dataset_id"] == metrics["dataset_id"]
        assert receipt["publication_intent"] == metrics["publication_intent"]
        assert "promoted" not in receipt
        assert receipt["sha256"] == hashlib.sha256(fake.objects[receipt["key"]]).hexdigest()
        assert fake.metadata[receipt["key"]] == {
            "build-plan-id": metrics["build_plan_id"],
            "publication-intent": hashlib.sha256(
                json.dumps(receipt["publication_intent"], sort_keys=True).encode()
            ).hexdigest(),
        }
        assert fake.ops.index(("put_object", receipt_key)) < fake.ops.index(
            ("put_object", "models/releases/v3/RB/manifest.json")
        )

    def test_held_collection_lock_aborts_before_upload(self, monkeypatch, tmp_path):
        from src.artifacts.model_sync import ManifestLockedError
        from src.batch import train

        fake = _FakeS3Producer()
        fake.objects["models/releases/v3/RB/manifest.json"] = b'{"gc_lock":{"owner":"operator"}}'
        monkeypatch.setattr(train.boto3, "client", lambda *_: fake)
        _write_fake_model_dir(tmp_path, "RB")
        with pytest.raises(ManifestLockedError, match="lock is held"):
            train.upload_artifacts("bucket", "RB", str(tmp_path))
        assert fake.ops == []

    @pytest.mark.parametrize("existing_manifest", [False, True])
    def test_same_intent_concurrent_uploads_share_first_successful_canonical_output(
        self, monkeypatch, tmp_path, existing_manifest
    ):
        from src.batch import train
        from src.shared.model_sync import build_manifest

        fake = _FakeS3Producer()
        monkeypatch.setattr(train.boto3, "client", lambda *_: fake)
        manifest_key = "models/releases/v3/RB/manifest.json"
        old_key = "models/releases/v3/RB/history/old/model.tar.gz"
        if existing_manifest:
            old = build_manifest(old_key, "old", 3, "t0", smoke_passed=True)
            fake.objects[manifest_key] = json.dumps(old).encode()
            fake.objects[old_key] = b"old"

        metrics = _publication_metrics(fake, monkeypatch, tmp_path / "image")
        a, b = tmp_path / "a", tmp_path / "b"
        for directory in (a, b):
            directory.mkdir()
            _write_fake_model_dir(directory, "RB", metrics={**metrics, "publisher": directory.name})

        def smoke(pos, directory):
            if directory == str(a):
                # A has captured its generation and uploaded; B now completes
                # an entire competing publication before A resumes its PUT.
                train.upload_artifacts("bucket", pos, str(b))
            return True

        monkeypatch.setattr(train, "_try_smoke_test", smoke)
        train.upload_artifacts("bucket", "RB", str(a))
        live = json.loads(fake.objects[manifest_key])
        uploads = [key for op, key in fake.ops if op == "upload_file"]
        assert len(uploads) == 2
        assert live["stable"]["key"] == uploads[1]
        receipt = json.loads(
            fake.objects[f"build-plans/{metrics['build_plan_id']}/artifacts/RB.json"]
        )
        assert receipt["key"] == live["stable"]["key"]
        assert "promoted" not in receipt
        assert all(key in fake.objects for key in uploads)
        assert not any(op == "delete" for op, _ in fake.ops)
        if existing_manifest:
            assert live["previous_stable"]["key"] == old_key

    @pytest.mark.parametrize("code", ["AccessDenied", "SlowDown", "InternalError"])
    def test_manifest_read_error_aborts_before_upload(self, monkeypatch, tmp_path, code):
        from src.batch import train

        fake = _FakeS3Producer()
        monkeypatch.setattr(train.boto3, "client", lambda *_: fake)
        monkeypatch.setattr(
            fake,
            "get_object",
            mock.Mock(side_effect=ClientError({"Error": {"Code": code}}, "GetObject")),
        )
        _write_fake_model_dir(tmp_path, "RB")
        with pytest.raises(ClientError, match=code):
            train.upload_artifacts("bucket", "RB", str(tmp_path))
        assert fake.ops == []

    def test_failed_first_smoke_cannot_be_served(self, monkeypatch, tmp_path):
        from src.batch import train
        from src.shared import model_sync

        fake = _FakeS3Producer()
        monkeypatch.setattr(train.boto3, "client", lambda *_: fake)
        monkeypatch.setattr(train, "_try_smoke_test", lambda *_: False)
        metrics = _publication_metrics(fake, monkeypatch, tmp_path / "image")
        _write_fake_model_dir(tmp_path, "RB", metrics=metrics)
        train.upload_artifacts("bucket", "RB", str(tmp_path))
        assert f"build-plans/{metrics['build_plan_id']}/artifacts/RB.json" not in fake.objects
        with pytest.raises(RuntimeError, match="all manifest entries failed"):
            model_sync._sync_one(fake, "bucket", "models", "RB", tmp_path / "serving")

    def test_failed_smoke_does_not_poison_successful_retry_for_same_intent(
        self, monkeypatch, tmp_path
    ):
        from src.batch import train

        fake = _FakeS3Producer()
        monkeypatch.setattr(train.boto3, "client", lambda *_: fake)
        monkeypatch.setattr(train, "_try_smoke_test", mock.Mock(side_effect=[False, True]))
        metrics = _publication_metrics(fake, monkeypatch, tmp_path / "image")
        _write_fake_model_dir(tmp_path, "RB", metrics=metrics)
        key = f"build-plans/{metrics['build_plan_id']}/artifacts/RB.json"
        train.upload_artifacts("bucket", "RB", str(tmp_path))
        assert key not in fake.objects
        train.upload_artifacts("bucket", "RB", str(tmp_path))
        receipt = json.loads(fake.objects[key])
        manifest = json.loads(fake.objects["models/releases/v3/RB/manifest.json"])
        assert receipt["smoke_passed"] is True
        assert receipt["key"] == manifest["stable"]["key"]
        assert "promoted" not in receipt

    @mock.patch("src.batch.train.boto3.client")
    def test_uploads_versioned_key_and_writes_manifest(
        self, mock_boto_client, tmp_path, monkeypatch
    ):
        from src.batch.train import upload_artifacts

        fake_s3 = _FakeS3Producer()
        mock_boto_client.return_value = fake_s3

        d = tmp_path / "model"
        d.mkdir()
        metrics = _publication_metrics(fake_s3, monkeypatch, tmp_path / "image", bucket="my-bucket")
        _write_fake_model_dir(d, "RB", metrics=metrics)

        upload_artifacts("my-bucket", "RB", str(d))

        # Exactly one versioned history key was written.
        history_keys = [
            k for k in fake_s3.objects if k.startswith("models/releases/v3/RB/history/")
        ]
        assert len(history_keys) == 1
        history_key = history_keys[0]
        assert history_key.endswith("/model.tar.gz")

        # Manifest is present and points current at the versioned key.
        manifest = json.loads(fake_s3.objects["models/releases/v3/RB/manifest.json"])
        assert manifest["schema_version"] == 3
        assert manifest["current"]["key"] == history_key
        assert manifest["previous"] is None  # first write
        assert history_key in manifest["history"]
        # The fixture writes fake bytes that can't be deserialized → smoke
        # test fails → ``stable`` stays unset on this first upload.
        assert manifest["stable"] is None

        # Legacy mirror is NOT written — Layer C removed the producer-side
        # legacy ``models/{POS}/model.tar.gz`` upload. Two parallel train-batch
        # runs writing the same legacy key were last-write-wins; the manifest
        # is the only artifact pointer now.
        assert "models/RB/model.tar.gz" not in fake_s3.objects

    @mock.patch("src.batch.train.boto3.client")
    def test_validation_rejects_missing_required_file(
        self, mock_boto_client, tmp_path, monkeypatch
    ):
        """Validation re-downloads the uploaded tarball and checks for
        required files. A missing nn_scaler.pkl must raise BEFORE the
        manifest write — otherwise a promoted bad artifact sticks."""
        from src.batch.train import upload_artifacts

        fake_s3 = _FakeS3Producer()
        mock_boto_client.return_value = fake_s3

        d = tmp_path / "model"
        d.mkdir()
        metrics = _publication_metrics(fake_s3, monkeypatch, tmp_path / "image", bucket="my-bucket")
        _write_fake_model_dir(d, "RB", metrics=metrics)
        # Remove a required file AFTER the dir was populated.
        (d / "nn_scaler.pkl").unlink()

        with pytest.raises(RuntimeError, match="missing required files"):
            upload_artifacts("my-bucket", "RB", str(d))

        # Manifest must NOT have been written — the promotion didn't happen.
        assert "models/releases/v3/RB/manifest.json" not in fake_s3.objects
        # Legacy mirror must NOT have been overwritten.
        assert "models/RB/model.tar.gz" not in fake_s3.objects

    @mock.patch("src.batch.train.boto3.client")
    def test_validation_detects_truncation(self, mock_boto_client, tmp_path, monkeypatch):
        """If the uploaded bytes get truncated (replication lag, network blip),
        validation's tarfile reopen fails and the manifest stays on the
        previous good pointer. We simulate by intercepting the first
        upload_file to store only the first 32 bytes."""
        from src.batch.train import upload_artifacts

        fake_s3 = _FakeS3Producer()
        original_upload = fake_s3.upload_file

        def truncated_upload(local_path, Bucket, Key, ExtraArgs=None):  # noqa: N803
            with open(local_path, "rb") as f:
                fake_s3.objects[Key] = f.read()[:32]  # deliberately truncated
            fake_s3.ops.append(("upload_file", Key))

        mock_boto_client.return_value = fake_s3
        fake_s3.upload_file = truncated_upload  # type: ignore[method-assign]

        d = tmp_path / "model"
        d.mkdir()
        metrics = _publication_metrics(fake_s3, monkeypatch, tmp_path / "image", bucket="my-bucket")
        _write_fake_model_dir(d, "RB", metrics=metrics)

        try:
            with pytest.raises((RuntimeError, tarfile.TarError, OSError, EOFError)):
                upload_artifacts("my-bucket", "RB", str(d))
        finally:
            fake_s3.upload_file = original_upload  # type: ignore[method-assign]

        assert "models/releases/v3/RB/manifest.json" not in fake_s3.objects
        assert len([op for op in fake_s3.ops if op[0] == "upload_file"]) == 1

    @mock.patch("src.batch.train.boto3.client")
    def test_new_training_intent_retains_previous_approved_output(
        self, mock_boto_client, tmp_path, monkeypatch
    ):
        """After two back-to-back uploads, manifest.previous must equal the
        first upload's current. This is the rollback path: if upload #2's
        artifact later fails to load, ``src.shared.model_sync._sync_one`` falls
        back to #1's versioned key via manifest.previous."""
        from src.batch.train import upload_artifacts

        fake_s3 = _FakeS3Producer()
        mock_boto_client.return_value = fake_s3

        d = tmp_path / "model"
        d.mkdir()
        first = _publication_metrics(
            fake_s3, monkeypatch, tmp_path / "image", bucket="my-bucket", run_id="first"
        )
        _write_fake_model_dir(d, "RB", metrics=first)
        monkeypatch.setattr("src.batch.train._try_smoke_test", lambda *_: True)

        upload_artifacts("my-bucket", "RB", str(d))
        first_manifest = json.loads(fake_s3.objects["models/releases/v3/RB/manifest.json"])
        first_current_key = first_manifest["current"]["key"]

        # Second upload with slightly different bytes so sha7 differs.
        second = _publication_metrics(
            fake_s3, monkeypatch, tmp_path / "image", bucket="my-bucket", run_id="second"
        )
        (d / "benchmark_metrics.json").write_text(json.dumps(second))
        upload_artifacts("my-bucket", "RB", str(d))

        second_manifest = json.loads(fake_s3.objects["models/releases/v3/RB/manifest.json"])
        assert second_manifest["previous"] is not None
        assert second_manifest["previous"]["key"] == first_current_key
        assert second_manifest["previous_stable"]["key"] == first_current_key
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
        metrics = _publication_metrics(fake_s3, monkeypatch, tmp_path / "image", bucket="my-bucket")
        _write_fake_model_dir(d, "RB", metrics=metrics)
        monkeypatch.setattr("src.batch.train._try_smoke_test", lambda *_: True)
        upload_artifacts("my-bucket", "RB", str(d))

        dest_root = tmp_path / "consumer_root"
        result = model_sync._sync_one(fake_s3, "my-bucket", "models", "RB", dest_root)

        assert result["source"] == "stable"
        assert (dest_root / "src" / "rb" / "outputs" / "models" / "nn_scaler.pkl").is_file()
        assert (dest_root / "src" / "rb" / "outputs" / "models" / "rb_multihead_nn.pt").is_file()

    def test_source_superseded_upload_keeps_own_receipt_without_replacing_newer_release(
        self, monkeypatch, tmp_path
    ):
        from src.batch import train

        fake = _FakeS3Producer()
        monkeypatch.setattr(train.boto3, "client", lambda *_: fake)
        older = _publication_metrics(
            fake, monkeypatch, tmp_path / "images", run_id="older", source_sha="d" * 40
        )
        a, b = tmp_path / "older", tmp_path / "newer"
        a.mkdir()
        b.mkdir()
        _write_fake_model_dir(a, "RB", metrics=older)
        newer = {}

        def smoke(position, directory):
            if directory == str(a):
                newer.update(
                    _publication_metrics(
                        fake,
                        monkeypatch,
                        tmp_path / "images",
                        run_id="newer",
                        source_sha="e" * 40,
                        ancestors=("d" * 40,),
                    )
                )
                _write_fake_model_dir(b, "RB", metrics=newer)
                train.upload_artifacts("bucket", position, str(b))
            return True

        monkeypatch.setattr(train, "_try_smoke_test", smoke)
        train.upload_artifacts("bucket", "RB", str(a))
        uploads = [key for operation, key in fake.ops if operation == "upload_file"]
        manifest = json.loads(fake.objects["models/releases/v3/RB/manifest.json"])
        old_receipt = json.loads(
            fake.objects[f"build-plans/{older['build_plan_id']}/artifacts/RB.json"]
        )
        new_receipt = json.loads(
            fake.objects[f"build-plans/{newer['build_plan_id']}/artifacts/RB.json"]
        )
        assert old_receipt["key"] == uploads[0]
        assert new_receipt["key"] == uploads[1] == manifest["stable"]["key"]
        assert all(key in fake.objects for key in uploads)

    def test_upload_authenticates_actual_baked_source(self, monkeypatch, tmp_path):
        from src.batch import train

        fake = _FakeS3Producer()
        monkeypatch.setattr(train.boto3, "client", lambda *_: fake)
        metrics = _publication_metrics(fake, monkeypatch, tmp_path / "image")
        _write_fake_model_dir(tmp_path, "RB", metrics={**metrics, "git_sha": "e" * 40})
        fake.ops.clear()
        with pytest.raises(RuntimeError, match="actual image"):
            train.upload_artifacts("bucket", "RB", str(tmp_path))
        assert fake.ops == []

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

    def test_extracts_elasticnet_metrics(self):
        from src.batch.train import _extract_metrics

        result = {
            "elasticnet_metrics": {"total": {"mae": 4.25, "r2": 0.35}},
            "elasticnet_ranking": {"season_avg_hit_rate": 0.51},
        }
        metrics = _extract_metrics("RB", result)
        assert metrics["elasticnet_metrics"]["total"]["mae"] == 4.25
        assert metrics["elasticnet_ranking"]["season_avg_hit_rate"] == 0.51

    def test_stamps_git_sha_when_env_set(self, monkeypatch):
        """FF_TRAIN_GIT_SHA env var (passed by launch.py via containerOverrides)
        lands in benchmark_metrics.json so benchmark.py can verify per-position
        SHA coherency in the aggregation step."""
        from src.batch.train import _extract_metrics

        monkeypatch.setenv("FF_TRAIN_GIT_SHA", "abcdef1234567890")
        metrics = _extract_metrics("QB", {"ridge_metrics": {"total": {"mae": 6.0}}})
        assert metrics["git_sha"] == "abcdef1234567890"

    def test_git_sha_absent_when_env_unset(self, monkeypatch):
        """Empty FF_TRAIN_GIT_SHA → no git_sha key (workflow_dispatch / local
        runs); benchmark.py's coherency check skips for runs without it."""
        from src.batch.train import _extract_metrics

        monkeypatch.setenv("FF_TRAIN_GIT_SHA", "")
        metrics = _extract_metrics("QB", {"ridge_metrics": {"total": {"mae": 6.0}}})
        assert "git_sha" not in metrics

    def test_skips_none_metrics_from_partial_split_result(self):
        from src.batch.train import _extract_metrics

        metrics = _extract_metrics(
            "WR",
            {
                "ridge_metrics": None,
                "nn_metrics": {"total": {"mae": 5.5, "r2": 0.1}},
            },
        )
        assert "ridge_metrics" not in metrics
        assert metrics["nn_metrics"]["total"]["mae"] == 5.5


class TestMergedSplitMetrics:
    """The merge job's metrics contract: branch benchmark_metrics.json pairs →
    the merged payload src/batch/benchmark.py summarizes into history rows.

    Regression pins for the 2026-06-11..2026-07 silent-zero window: every
    Batch-split history row recorded ``{model}_top12 = 0`` because the split
    branches returned pipeline results without ``*_ranking`` and the summary
    defaulted to 0. The branch fixtures here mirror the post-fix artifact
    shape (``_extract_metrics`` output + split/hardware stamps).
    """

    def _nn_branch_metrics(self) -> dict:
        return {
            "position": "WR",
            "split_branch": "nn",
            "split_run_id": "run-1",
            "seed": 42,
            "git_sha": "abc1234def567",
            "nn_metrics": {"total": {"mae": 5.5, "r2": 0.31, "rmse": 7.0}},
            "attn_nn_metrics": {"total": {"mae": 5.3, "r2": 0.35, "rmse": 6.8}},
            "nn_ranking": {"season_avg_hit_rate": 0.44, "season_avg_spearman": 0.41},
            "attn_nn_ranking": {"season_avg_hit_rate": 0.48, "season_avg_spearman": 0.45},
            "elapsed_sec": 120.0,
            "phase_seconds": {"run_pipeline": 100.0},
            "gpu_name": "NVIDIA L4",
            "sm": 89,
            "cuda_graph_active": True,
            "cuda_graph_full_active": True,
        }

    def _cpu_branch_metrics(self) -> dict:
        return {
            "position": "WR",
            "split_branch": "cpu",
            "split_run_id": "run-1",
            "seed": 42,
            "git_sha": "abc1234def567",
            "ridge_metrics": {"total": {"mae": 5.6, "r2": 0.29, "rmse": 7.1}},
            "lgbm_metrics": {"total": {"mae": 5.4, "r2": 0.33, "rmse": 6.9}},
            "ridge_ranking": {"season_avg_hit_rate": 0.41, "season_avg_spearman": 0.4},
            "lgbm_ranking": {"season_avg_hit_rate": 0.46, "season_avg_spearman": 0.44},
            "elapsed_sec": 90.0,
            "phase_seconds": {"run_pipeline": 80.0},
        }

    def test_merged_metrics_carry_all_four_rankings(self):
        from src.batch.train import _merged_split_metrics

        merged = _merged_split_metrics(
            "WR",
            "run-1",
            self._nn_branch_metrics(),
            self._cpu_branch_metrics(),
            {"merge_split_artifacts": 5.0},
            time.monotonic(),
        )
        for key in ("ridge_ranking", "nn_ranking", "attn_nn_ranking", "lgbm_ranking"):
            assert key in merged, key
        assert merged["nn_ranking"]["season_avg_hit_rate"] == 0.44
        assert merged["split_merged"] is True
        assert merged["git_sha"] == "abc1234def567"
        assert merged["gpu_name"] == "NVIDIA L4"
        assert merged["cuda_graph_full_active"] is True
        assert merged["phase_seconds"]["split.nn.run_pipeline"] == 100.0
        assert merged["phase_seconds"]["split.cpu.elapsed_sec"] == 90.0

    def test_merged_split_result_summarizes_to_nonzero_top12(self):
        """The full lost chain, end to end: branch metrics → merge → summary
        row. Every ``{model}_top12`` must be the branch's real nonzero hit
        rate — this is exactly the path that wrote 0 for a month."""
        from src.batch.train import _merged_split_metrics
        from src.shared.benchmark_utils import summarize_pipeline_result

        merged = _merged_split_metrics(
            "WR",
            "run-1",
            self._nn_branch_metrics(),
            self._cpu_branch_metrics(),
            {},
            time.monotonic(),
        )
        s = summarize_pipeline_result("WR", merged)
        assert s["ridge_top12"] == 0.41
        assert s["nn_top12"] == 0.44
        assert s["attn_nn_top12"] == 0.48
        assert s["lgbm_top12"] == 0.46
        for key in ("ridge_top12", "nn_top12", "attn_nn_top12", "lgbm_top12"):
            assert s[key] is not None and s[key] > 0, key

    def test_merged_result_without_rankings_summarizes_to_none(self):
        """Pre-fix branch artifacts (no ``*_ranking``) must surface as None in
        the summary — the silent-0 default can never come back."""
        from src.batch.train import _merged_split_metrics
        from src.shared.benchmark_utils import summarize_pipeline_result

        nn = self._nn_branch_metrics()
        cpu = self._cpu_branch_metrics()
        for branch_metrics in (nn, cpu):
            for key in [k for k in branch_metrics if k.endswith("_ranking")]:
                del branch_metrics[key]
        merged = _merged_split_metrics("WR", "run-1", nn, cpu, {}, time.monotonic())
        s = summarize_pipeline_result("WR", merged)
        for key in ("ridge_top12", "nn_top12", "attn_nn_top12", "lgbm_top12"):
            assert key in s, key
            assert s[key] is None, f"{key} must be None, got {s[key]!r}"

    def test_duplicate_metric_key_across_branches_raises(self):
        from src.batch.train import _merged_split_metrics

        nn = self._nn_branch_metrics()
        cpu = self._cpu_branch_metrics()
        cpu["nn_metrics"] = {"total": {"mae": 1.0}}
        with pytest.raises(RuntimeError, match="Duplicate metric key"):
            _merged_split_metrics("WR", "run-1", nn, cpu, {}, time.monotonic())


class TestSplitBranchHelpers:
    def test_branch_config_cpu_disables_nn_and_keeps_ridge_lgbm(self):
        from src.batch.train import _branch_config

        cfg = _branch_config("WR", "cpu")
        assert cfg["_artifact_branch"] == "cpu"
        assert cfg["train_ridge"] is True
        assert cfg["train_lightgbm"] is True
        assert cfg["train_base_nn"] is False
        assert cfg["train_attention_nn"] is False
        assert cfg["train_elasticnet"] is False
        assert cfg["train_tabpfn"] is False

    def test_branch_config_nn_disables_cpu_models(self):
        from src.batch.train import _branch_config

        cfg = _branch_config("WR", "nn")
        assert cfg["_artifact_branch"] == "nn"
        assert cfg["train_base_nn"] is True
        assert cfg["train_attention_nn"] is True
        assert cfg["train_ridge"] is False
        assert cfg["train_lightgbm"] is False

    def test_download_split_branch_rejects_wrong_git_sha(self, tmp_path):
        from src.batch.train import _download_split_branch_artifacts

        manifest = {
            "schema_version": 1,
            "split_run_id": "run-1",
            "position": "WR",
            "branch": "nn",
            "git_sha": "old-sha",
            "key": "split-runs/run-1/WR/nn/model.tar.gz",
        }

        class _FakeS3:
            def get_object(self, Bucket, Key):  # noqa: N803
                return {"Body": _FakeBody(json.dumps(manifest).encode())}

        with pytest.raises(RuntimeError, match="SHA mismatch"):
            _download_split_branch_artifacts(
                _FakeS3(),
                "bucket",
                "run-1",
                "WR",
                "nn",
                "new-sha",
                str(tmp_path),
            )


# ---------------------------------------------------------------------------
# Hardware metadata stamping (drives benchmark.py's History-tab label)
# ---------------------------------------------------------------------------


class TestHardwareMetadata:
    """``_hardware_metadata`` stamps the runtime GPU facts benchmark.py turns
    into the History-tab instance label (see ``_derive_instance_label``)."""

    def test_stamps_gpu_name_sm_and_graph_active(self, monkeypatch):
        """L4/sm_89 with capture on → fields benchmark.py renders as a
        ``g6.xlarge (L4, Spot, CUDA-graph)`` label."""
        from types import SimpleNamespace

        import src.batch.train as train

        monkeypatch.setattr(
            train, "detect_platform", lambda: SimpleNamespace(gpu_name="NVIDIA L4", sm="sm_89")
        )
        monkeypatch.setattr(train, "cuda_graph_enabled", lambda: True)
        monkeypatch.setattr(train, "cuda_graph_full_enabled", lambda: True)
        assert train._hardware_metadata() == {
            "gpu_name": "NVIDIA L4",
            "sm": "sm_89",
            "cuda_graph_active": True,
            "cuda_graph_full_active": True,  # prod default since 2026-06-15
        }

    def test_t4_reports_eager(self, monkeypatch):
        """T4/sm_75 < sm_80 → capture off; benchmark.py omits the CUDA-graph
        suffix (current production reality)."""
        from types import SimpleNamespace

        import src.batch.train as train

        monkeypatch.setattr(
            train, "detect_platform", lambda: SimpleNamespace(gpu_name="Tesla T4", sm="sm_75")
        )
        monkeypatch.setattr(train, "cuda_graph_enabled", lambda: False)
        # Full-step requires the base gate, so sm_75 reports it off too.
        monkeypatch.setattr(train, "cuda_graph_full_enabled", lambda: False)
        assert train._hardware_metadata() == {
            "gpu_name": "Tesla T4",
            "sm": "sm_75",
            "cuda_graph_active": False,
            "cuda_graph_full_active": False,
        }

    def test_cpu_box_reports_none(self, monkeypatch):
        """Non-CUDA box (CPU-only position / dev / CI) → no GPU identity, so
        benchmark.py falls back to its --instance-type arg."""
        from types import SimpleNamespace

        import src.batch.train as train

        monkeypatch.setattr(
            train, "detect_platform", lambda: SimpleNamespace(gpu_name=None, sm=None)
        )
        monkeypatch.setattr(train, "cuda_graph_enabled", lambda: False)
        monkeypatch.setattr(train, "cuda_graph_full_enabled", lambda: False)
        assert train._hardware_metadata() == {
            "gpu_name": None,
            "sm": None,
            "cuda_graph_active": False,
            "cuda_graph_full_active": False,
        }


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
        self,
        mock_parquet,
        mock_download,
        mock_copytree,
        mock_upload,
        mock_sync,
        tmp_path,
        monkeypatch,
    ):
        import pandas as pd

        fake = _FakeS3Producer()
        _publication_metrics(fake, monkeypatch, tmp_path / "image", bucket="test-bucket")
        monkeypatch.setenv("FF_TRAIN_GIT_SHA", "d" * 40)
        monkeypatch.setattr("src.batch.train.boto3.client", lambda *_: fake)
        monkeypatch.delenv("FF_BUILD_PLAN_ID", raising=False)
        monkeypatch.delenv("FF_DATASET_ID", raising=False)
        monkeypatch.chdir(tmp_path)
        src_model_dir = tmp_path / "rb" / "outputs" / "models"
        src_model_dir.mkdir(parents=True)
        (src_model_dir / "ridge_model.pkl").write_text("fake rb model")

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
            # CUDA-visible Windows: skip the nvidia-smi sidecar's POSIX /tmp open.
            mock.patch("src.batch.train._start_nvidia_smi_sidecar", return_value=None),
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
        self, mock_copytree, mock_upload, mock_sync, tmp_path, monkeypatch
    ):
        """main() for K/DST should skip download_data() (train/val/test splits) and
        REQUIRE_GPU. sync_raw_data() still runs for all positions — K/DST's
        self-contained loaders (and weather features) read from data/raw/.
        """
        fake = _FakeS3Producer()
        _publication_metrics(
            fake, monkeypatch, tmp_path / "image", position="K", bucket="ff-predictor-training"
        )
        monkeypatch.setenv("FF_TRAIN_GIT_SHA", "d" * 40)
        monkeypatch.setattr("src.batch.train.boto3.client", lambda *_: fake)
        monkeypatch.delenv("FF_BUILD_PLAN_ID", raising=False)
        monkeypatch.delenv("FF_DATASET_ID", raising=False)
        monkeypatch.chdir(tmp_path)
        src_model_dir = tmp_path / "k" / "outputs" / "models"
        src_model_dir.mkdir(parents=True)
        (src_model_dir / "ridge_model.pkl").write_text("fake k model")

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
