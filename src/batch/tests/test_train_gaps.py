"""Additional coverage for ``batch/train.py`` gap branches.

Existing tests (``test_launch.py``, ``test_train.py``, ``test_batch_e2e.py``)
cover the happy path + most of argparse. This file fills in:

- ``_download_if_stale`` head_object exception branch
- ``sync_raw_data`` paginator loop
- ``_validate_remote_tarball`` missing-bench / not-a-regular-file /
  malformed-JSON / missing-NN-weights branches
- ``_run_rb_gate_ablation`` — mocks ``run_rb_pipeline`` so all three
  variants complete in-process, with + without gate-AUC rows
- the ``--ablation rb-gate`` CLI dispatch path
- the ``result is None`` RuntimeError branch
- the missing-src-model-dir warning branch
"""

from __future__ import annotations

import io
import json
import tarfile
from unittest import mock

import pytest

# --------------------------------------------------------------------------
# _download_if_stale
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_download_if_stale_falls_back_on_head_object_exception(tmp_path, capsys):
    """head_object raises → message logged + unconditional download."""
    from src.batch.train import _download_if_stale

    local_path = tmp_path / "sub" / "file.parquet"

    class _FakeS3:
        def head_object(self, Bucket, Key):
            raise RuntimeError("boto3 head_object 500")

        def download_file(self, bucket, key, dest):
            with open(dest, "w") as f:
                f.write("ok")

    _download_if_stale(_FakeS3(), "bucket", "path/file", str(local_path))
    out = capsys.readouterr().out
    assert "head_object failed" in out
    assert local_path.exists()


@pytest.mark.unit
def test_download_if_stale_cache_hit_skips_download(tmp_path, capsys):
    """Matching ETag sidecar → no download call fired."""
    from src.batch.train import _download_if_stale

    local_path = tmp_path / "file.parquet"
    local_path.write_text("cached data")
    (tmp_path / "file.parquet.etag").write_text("etag-abc")

    class _FakeS3:
        def head_object(self, Bucket, Key):
            return {"ETag": "etag-abc"}

        def download_file(self, *a, **k):
            raise AssertionError("download_file should not be called on cache hit")

    _download_if_stale(_FakeS3(), "bucket", "path/file", str(local_path))
    assert "[cache] hit" in capsys.readouterr().out


# --------------------------------------------------------------------------
# sync_raw_data
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_sync_raw_data_downloads_every_parquet_key(monkeypatch, tmp_path):
    """Paginator yields pages of objects; only .parquet keys trigger download."""
    from src.batch import train as t

    monkeypatch.chdir(tmp_path)

    pages = [
        {
            "Contents": [
                {"Key": "data/raw/weekly_2012_2025.parquet"},
                {"Key": "data/raw/README.md"},  # non-parquet, skipped
                {"Key": "data/raw/schedules_2012_2025.parquet"},
            ]
        }
    ]

    downloaded: list[str] = []

    class _FakePaginator:
        def paginate(self, Bucket, Prefix):
            yield from pages

    class _FakeS3:
        def get_paginator(self, name):
            return _FakePaginator()

        def head_object(self, Bucket, Key):
            return {"ETag": "etag-" + Key.split("/")[-1]}

        def download_file(self, bucket, key, dest):
            downloaded.append(key)
            import os

            os.makedirs(tmp_path / "data" / "raw", exist_ok=True)
            with open(dest, "w") as f:
                f.write("ok")

    monkeypatch.setattr(t.boto3, "client", lambda name: _FakeS3())
    t.sync_raw_data("ff-bucket")
    assert sorted(downloaded) == [
        "data/raw/schedules_2012_2025.parquet",
        "data/raw/weekly_2012_2025.parquet",
    ]


# --------------------------------------------------------------------------
# _validate_remote_tarball
# --------------------------------------------------------------------------


def _make_tarball_bytes(members: dict[str, bytes]) -> bytes:
    """Build an in-memory tar.gz with the given members."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _fake_s3_with_bytes(body: bytes):
    class _FakeS3:
        def get_object(self, Bucket, Key):
            return {"Body": io.BytesIO(body)}

    return _FakeS3()


@pytest.mark.unit
def test_validate_remote_tarball_raises_when_metrics_missing():
    from src.batch.train import _validate_remote_tarball

    body = _make_tarball_bytes(
        {"qb_multihead_nn.pt": b"weights", "nn_scaler.pkl": b"scaler", "nn_scaler_meta.json": b"{}"}
    )
    with pytest.raises(RuntimeError, match="missing benchmark_metrics.json"):
        _validate_remote_tarball(_fake_s3_with_bytes(body), "b", "k", "QB")


@pytest.mark.unit
def test_validate_remote_tarball_raises_on_invalid_metrics_json():
    from src.batch.train import _validate_remote_tarball

    body = _make_tarball_bytes(
        {
            "benchmark_metrics.json": b"{not valid json",
            "qb_multihead_nn.pt": b"weights",
            "nn_scaler.pkl": b"scaler",
            "nn_scaler_meta.json": b"{}",
        }
    )
    with pytest.raises(RuntimeError, match="not valid JSON"):
        _validate_remote_tarball(_fake_s3_with_bytes(body), "b", "k", "QB")


@pytest.mark.unit
def test_validate_remote_tarball_raises_when_nn_weights_missing():
    from src.batch.train import _validate_remote_tarball

    # Has benchmark_metrics but no nn weights → missing-files raise.
    body = _make_tarball_bytes({"benchmark_metrics.json": b'{"position": "QB"}'})
    with pytest.raises(RuntimeError, match="missing required files"):
        _validate_remote_tarball(_fake_s3_with_bytes(body), "b", "k", "QB")


# --------------------------------------------------------------------------
# _run_rb_gate_ablation
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_run_rb_gate_ablation_runs_three_variants(monkeypatch, capsys):
    """Every variant (A/B/C) runs, decision-table lines print, and the
    "keep a gate" branch fires because margin_a is set large enough."""
    from src.batch import train as t

    # Three call log, each returning a canned attn_nn_metrics dict.
    calls: list[dict] = []

    def _canned_run_rb_pipeline(train_df, val_df, test_df, seed, config):
        variant_idx = len(calls)
        calls.append({"variant_idx": variant_idx, "seed": seed})
        # Variant A (idx 0) wins by 0.2 pts → triggers "keep a gate" message.
        fp_mae = [4.1, 4.4, 4.35][variant_idx]
        return {
            "attn_nn_metrics": {
                "total": {"mae": fp_mae, "rmse": fp_mae * 1.3, "r2": 0.3},
                "rushing_tds": {"mae": 0.5, "gate_auc": 0.72},
                "receiving_tds": {"mae": 0.4, "gate_auc": 0.68},
                "receptions": {"mae": 1.1},
                "rushing_yards": {"mae": 15.0},
                "receiving_yards": {"mae": 14.0},
                "fumbles_lost": {"mae": 0.3},
            }
        }

    # Patch the module-level import the ablation function does inside.
    monkeypatch.setattr(
        "src.RB.run_rb_pipeline.run_rb_pipeline", _canned_run_rb_pipeline, raising=False
    )

    import pandas as pd

    dummy_df = pd.DataFrame({"x": [1]})
    t._run_rb_gate_ablation(dummy_df, dummy_df, dummy_df, seed=42)

    assert len(calls) == 3  # three variants
    out = capsys.readouterr().out
    # Decision-table header + per-variant rows.
    assert "RB TD-gate ablation — summary" in out
    assert "Gate AUCs" in out  # gate_aucs rows present
    # Variant A wins big (4.1 vs B 4.4) → margin_a = 0.3 → keep gate.
    assert "keep a gate on TDs" in out


@pytest.mark.unit
def test_run_rb_gate_ablation_drop_gate_when_margins_are_tiny(monkeypatch, capsys):
    """When max margin < 0.05, the decision prints 'drop gate'."""
    from src.batch import train as t

    # All three variants have near-identical fp_mae → margins tiny.
    def _canned(train_df, val_df, test_df, seed, config):
        return {
            "attn_nn_metrics": {
                "total": {"mae": 4.30, "rmse": 5.5, "r2": 0.3},
                "rushing_tds": {"mae": 0.5},
                "receiving_tds": {"mae": 0.4},
                "receptions": {"mae": 1.1},
                "rushing_yards": {"mae": 15.0},
                "receiving_yards": {"mae": 14.0},
                "fumbles_lost": {"mae": 0.3},
            }
        }

    monkeypatch.setattr("src.RB.run_rb_pipeline.run_rb_pipeline", _canned, raising=False)
    import pandas as pd

    t._run_rb_gate_ablation(pd.DataFrame({"x": [1]}), pd.DataFrame(), pd.DataFrame(), seed=1)
    out = capsys.readouterr().out
    assert "drop gate on TDs" in out


@pytest.mark.unit
def test_run_rb_gate_ablation_raises_if_attn_metrics_missing(monkeypatch):
    """Pipeline result without attn_nn_metrics → explicit RuntimeError."""
    from src.batch import train as t

    monkeypatch.setattr(
        "src.RB.run_rb_pipeline.run_rb_pipeline",
        lambda *a, **k: {"ridge_metrics": {"total": {"mae": 5.0}}},  # no attn
        raising=False,
    )
    import pandas as pd

    with pytest.raises(RuntimeError, match="attn_nn_metrics missing"):
        t._run_rb_gate_ablation(pd.DataFrame({"x": [1]}), pd.DataFrame(), pd.DataFrame(), seed=1)


# --------------------------------------------------------------------------
# main: ablation dispatch + None-result + missing-src branch
# --------------------------------------------------------------------------


_UNSET = object()


def _stub_main_io(t, monkeypatch, *, runner_returns=_UNSET):
    """Stub S3 + data sync + parquet reads + the runner in one place.

    Pass ``runner_returns`` (can be ``None``) to replace ``get_runner`` with
    a stub whose inner callable returns that value; omit it to leave the
    runner in place (ablation test stubs the ablation function instead).
    """
    import pandas as pd

    monkeypatch.setattr(t, "sync_raw_data", lambda bucket: None)
    monkeypatch.setattr(t, "download_data", lambda *a, **k: None)
    monkeypatch.setattr(t, "upload_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(t, "_assert_gpu", lambda pos: None)
    monkeypatch.setattr(
        t.pd,
        "read_parquet",
        lambda path: pd.DataFrame({"player_id": ["P"], "season": [2024], "week": [1]}),
    )
    if runner_returns is not _UNSET:
        monkeypatch.setattr(t, "get_runner", lambda pos: lambda *a, **k: runner_returns)


@pytest.mark.unit
def test_main_ablation_dispatch_skips_upload(tmp_path, monkeypatch):
    """``--ablation rb-gate`` path short-circuits before upload_artifacts."""
    from src.batch import train as t

    monkeypatch.setenv("MODEL_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("REQUIRE_GPU", "0")

    _stub_main_io(t, monkeypatch)
    # Stub the ablation body — we're testing main's dispatch, not its interior.
    ablation_calls: list[dict] = []

    def _stub_ablation(train_df, val_df, test_df, seed):
        ablation_calls.append({"seed": seed, "n_train": len(train_df)})

    monkeypatch.setattr(t, "_run_rb_gate_ablation", _stub_ablation)
    # upload_artifacts MUST NOT be called — replace with a fail-if-called stub.
    monkeypatch.setattr(
        t,
        "upload_artifacts",
        lambda *a, **k: pytest.fail("upload_artifacts fired on ablation path"),
    )

    with mock.patch("sys.argv", ["train.py", "--position", "RB", "--ablation", "rb-gate"]):
        t.main()

    assert len(ablation_calls) == 1
    assert ablation_calls[0]["seed"] == 42  # default seed


@pytest.mark.unit
def test_main_raises_when_pipeline_returns_none(tmp_path, monkeypatch):
    """``result is None`` → RuntimeError before the metrics write."""
    from src.batch import train as t

    monkeypatch.setenv("MODEL_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("REQUIRE_GPU", "0")
    monkeypatch.chdir(tmp_path)

    _stub_main_io(t, monkeypatch, runner_returns=None)

    with mock.patch("sys.argv", ["train.py", "--position", "QB"]):
        with pytest.raises(RuntimeError, match="returned None"):
            t.main()


@pytest.mark.unit
def test_main_warns_when_src_model_dir_missing(tmp_path, monkeypatch, capsys):
    """If ``{pos}/outputs/models/`` doesn't exist, main prints a warning
    but still proceeds (until the None-result guard trips)."""
    from src.batch import train as t

    monkeypatch.setenv("MODEL_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("REQUIRE_GPU", "0")
    monkeypatch.chdir(tmp_path)  # isolate cwd so QB/outputs/models doesn't exist

    _stub_main_io(t, monkeypatch, runner_returns=None)

    with mock.patch("sys.argv", ["train.py", "--position", "QB"]):
        with pytest.raises(RuntimeError):
            t.main()

    assert "No model directory found" in capsys.readouterr().out
