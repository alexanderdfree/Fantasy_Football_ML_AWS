"""Coverage tests for ``src/benchmarking/benchmark.py::_maybe_upload_to_s3``.

The local benchmark entrypoint mirrors each run's JSON to S3 so it reaches the
website's History tab (the serving container pulls
``s3://{bucket}/{prefix}/benchmark_history/*.json`` at boot). These tests pin
the env-gate, the S3 key layout, the best-effort error handling (the deliberate
divergence from ``src/batch/benchmark.py``'s propagating copy), and the region
fallback. boto3 is always mocked — no test touches real S3.

Lives under ``tests/shared/`` rather than a new ``tests/benchmarking/`` so it
runs only on the ``shared`` CI test shard (where ``src/benchmarking/`` changes
already route via ``scope_positions._TEST_SHARED_REGEX``); a ``tests/benchmarking/``
dir would fan out to all seven shards.
"""

from __future__ import annotations

from unittest import mock

import pytest


@pytest.mark.unit
def test_maybe_upload_to_s3_noop_when_bucket_unset(monkeypatch, tmp_path, capsys):
    """Without ``FF_MODEL_S3_BUCKET``, the helper must not call boto3 and must
    print a visible skip notice so the no-op is never silent."""
    import src.benchmarking.benchmark as lb

    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    sentinel = mock.Mock(side_effect=AssertionError("boto3.client should NOT be called"))
    with mock.patch("boto3.client", sentinel):
        lb._maybe_upload_to_s3(str(tmp_path / "anything.json"))
    sentinel.assert_not_called()
    assert "skipping cloud sync" in capsys.readouterr().out


@pytest.mark.unit
def test_maybe_upload_to_s3_uploads_with_expected_key(monkeypatch, tmp_path, capsys):
    """With env set, calls ``s3.upload_file(local, bucket, key)`` where
    ``key = {prefix}/benchmark_history/{basename(local_path)}`` — the same key the
    serving container reads from."""
    import src.benchmarking.benchmark as lb

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "my-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "models")
    fake_s3 = mock.Mock()
    with mock.patch("boto3.client", return_value=fake_s3):
        lb._maybe_upload_to_s3(str(tmp_path / "2026-05-29T02-10-55_ce7f535.json"))
    fake_s3.upload_file.assert_called_once_with(
        str(tmp_path / "2026-05-29T02-10-55_ce7f535.json"),
        "my-bucket",
        "models/benchmark_history/2026-05-29T02-10-55_ce7f535.json",
    )
    assert "s3://my-bucket/models/benchmark_history/" in capsys.readouterr().out


@pytest.mark.unit
def test_maybe_upload_to_s3_strips_prefix_slashes(monkeypatch, tmp_path):
    """A trailing/leading slash on FF_MODEL_S3_PREFIX must not produce a
    double-slash in the resulting S3 key (S3 treats // as literal segments)."""
    import src.benchmarking.benchmark as lb

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "my-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "/nightly/v2/")
    fake_s3 = mock.Mock()
    with mock.patch("boto3.client", return_value=fake_s3):
        lb._maybe_upload_to_s3(str(tmp_path / "x.json"))
    args = fake_s3.upload_file.call_args.args
    assert args[2] == "nightly/v2/benchmark_history/x.json"


@pytest.mark.unit
def test_maybe_upload_to_s3_swallows_upload_error(monkeypatch, tmp_path, capsys):
    """A network/credential failure during upload must NOT raise — the run's
    local JSON is already durably written. This is the deliberate divergence
    from src/batch/benchmark.py's propagating copy; lock it in."""
    import src.benchmarking.benchmark as lb

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "my-bucket")
    fake_s3 = mock.Mock()
    fake_s3.upload_file.side_effect = RuntimeError("boom")
    local = str(tmp_path / "run.json")
    with mock.patch("boto3.client", return_value=fake_s3):
        lb._maybe_upload_to_s3(local)  # must not raise
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert local in out


@pytest.mark.unit
def test_maybe_upload_to_s3_region_fallback(monkeypatch, tmp_path):
    """Region resolves AWS_REGION -> AWS_DEFAULT_REGION -> us-east-1 inline (not
    imported from src/batch/launch.py, which would drag batch orchestration into
    the local tool's import graph)."""
    import src.benchmarking.benchmark as lb

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "my-bucket")
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    fake_s3 = mock.Mock()
    with mock.patch("boto3.client", return_value=fake_s3) as factory:
        lb._maybe_upload_to_s3(str(tmp_path / "x.json"))
    factory.assert_called_once_with("s3", region_name="us-east-1")

    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    with mock.patch("boto3.client", return_value=fake_s3) as factory2:
        lb._maybe_upload_to_s3(str(tmp_path / "y.json"))
    factory2.assert_called_once_with("s3", region_name="eu-west-1")
