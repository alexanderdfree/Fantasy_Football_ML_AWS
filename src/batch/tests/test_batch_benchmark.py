"""Coverage tests for ``src/batch/benchmark.py``.

The CLI launches AWS Batch jobs and aggregates benchmark_metrics.json from
each position's model artifacts. These tests stub every external call
(``boto3``, ``submit_job``, ``upload_data``, ``wait_for_jobs``, history
serialization) so the full orchestration + main() run in-process.

We exercise both the job-launch path and the ``--download-only`` path, plus
the S3-404 fallback inside ``download_metrics``.
"""

from __future__ import annotations

import io
import json
import tarfile
from unittest import mock

import pytest

# --------------------------------------------------------------------------
# download_metrics — boto3 + tarfile mocked
# --------------------------------------------------------------------------


def _make_tarfile_with_metrics(tmp_path, payload: dict | None) -> str:
    """Write a tar.gz containing benchmark_metrics.json (or empty) to tmp_path."""
    tar_path = tmp_path / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        if payload is not None:
            buf = io.BytesIO(json.dumps(payload).encode())
            info = tarfile.TarInfo(name="benchmark_metrics.json")
            info.size = len(buf.getvalue())
            tar.addfile(info, buf)
        else:
            # Tar with some other file but NOT benchmark_metrics.json
            buf = io.BytesIO(b"decoy")
            info = tarfile.TarInfo(name="other.json")
            info.size = len(buf.getvalue())
            tar.addfile(info, buf)
    return str(tar_path)


@pytest.mark.unit
def test_download_metrics_happy_path(tmp_path, monkeypatch):
    """Good tar → metrics dict keyed by position."""
    import src.batch.benchmark as bb

    tar_path = _make_tarfile_with_metrics(tmp_path, {"position": "QB", "mae": 6.2})

    class _FakeS3:
        def download_file(self, bucket, key, dest):
            # Copy our pre-built tar to wherever NamedTemporaryFile landed.
            with open(tar_path, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())

    monkeypatch.setattr(bb.boto3, "client", lambda *a, **k: _FakeS3())

    result = bb.download_metrics(["QB"])
    assert result == {"QB": {"position": "QB", "mae": 6.2}}


@pytest.mark.unit
def test_download_metrics_missing_metrics_file(tmp_path, monkeypatch):
    """tar without benchmark_metrics.json → position omitted (KeyError branch)."""
    import src.batch.benchmark as bb

    tar_path = _make_tarfile_with_metrics(tmp_path, None)

    class _FakeS3:
        def download_file(self, bucket, key, dest):
            with open(tar_path, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())

    monkeypatch.setattr(bb.boto3, "client", lambda *a, **k: _FakeS3())

    result = bb.download_metrics(["RB"])
    assert result == {}  # no metrics → nothing in the dict


@pytest.mark.unit
def test_download_metrics_s3_error_swallowed(monkeypatch):
    """S3 download exception → ``no artifacts found`` branch returns None."""
    import src.batch.benchmark as bb

    class _FakeS3:
        def download_file(self, *a, **k):
            raise RuntimeError("NoSuchKey")

    monkeypatch.setattr(bb.boto3, "client", lambda *a, **k: _FakeS3())

    result = bb.download_metrics(["WR"])
    assert result == {}


# --------------------------------------------------------------------------
# main — drive both --download-only and the full launch path
# --------------------------------------------------------------------------


@pytest.fixture()
def _main_stubs(tmp_path, monkeypatch):
    """Stub every external call main() makes. Returns the tmp project root."""
    import src.batch.benchmark as bb

    # Redirect writes into tmp_path. main() chdirs to project root (``..``)
    # — we make that safe by pointing ``__file__``'s parent to tmp_path.
    monkeypatch.chdir(tmp_path)
    # main() still calls os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    # which resolves relative to src/batch/benchmark.py — safe because we don't
    # write anything there (RESULTS_FILE/HISTORY_FILE land in project root,
    # which our append_to_history stub won't touch).

    launched: list[dict] = []

    def _submit_job(pos, seed):
        launched.append({"pos": pos, "seed": seed})
        return pos, f"job-{pos}"

    def _wait_for_jobs(job_ids):
        return {p: ("SUCCEEDED", 0) for p in job_ids}

    def _upload_data(bucket):
        launched.append({"upload": bucket})

    monkeypatch.setattr(bb, "submit_job", _submit_job)
    monkeypatch.setattr(bb, "upload_data", _upload_data)
    monkeypatch.setattr(bb, "wait_for_jobs", _wait_for_jobs)

    # Metrics are taken from download_metrics — stub it.
    fake_metrics = {
        "QB": {"position": "QB", "mae": 6.2},
        "RB": {"position": "RB", "mae": 4.3},
    }
    monkeypatch.setattr(bb, "download_metrics", lambda positions: fake_metrics)

    # summary + table + history — src.shared.benchmark_utils functions re-bound
    # on bench at import time. summarize returns a dict, the other three
    # are no-ops.
    monkeypatch.setattr(
        bb,
        "summarize_pipeline_result",
        lambda pos, metrics: {"position": pos, **metrics},
    )
    printed: list[dict] = []
    monkeypatch.setattr(
        bb,
        "print_comparison_table",
        lambda summaries, header, show_time: printed.extend(summaries),
    )
    appended: list[dict] = []
    monkeypatch.setattr(
        bb,
        "append_to_history",
        lambda path, entry: appended.append(entry),
    )
    monkeypatch.setattr(bb, "get_git_hash", lambda: "abc1234")

    # RESULTS_FILE writes are fine — they go to cwd which we monkeypatch.chdir'd
    # to tmp_path. Actually main() does os.chdir(project_root) FIRST, overriding
    # that. We patch RESULTS_FILE/HISTORY_DIR to point under tmp_path.
    monkeypatch.setattr(bb, "RESULTS_FILE", str(tmp_path / "results.json"))
    monkeypatch.setattr(bb, "HISTORY_DIR", str(tmp_path / "history"))

    return launched, printed, appended


@pytest.mark.unit
def test_main_download_only_skips_submit(_main_stubs, monkeypatch):
    """``--download-only`` must NOT call submit_job / upload_data / wait_for_jobs."""
    import src.batch.benchmark as bb

    launched, printed, appended = _main_stubs
    monkeypatch.setattr("sys.argv", ["src/batch/benchmark.py", "--download-only"])
    bb.main()
    # No submit/upload/wait calls — launched must be empty.
    assert launched == []
    # Metrics were printed and history was written.
    assert len(printed) == 2  # QB + RB
    assert len(appended) == 1
    assert appended[0]["git_hash"] == "abc1234"


@pytest.mark.unit
def test_main_full_launch_path(_main_stubs, monkeypatch):
    """Default invocation runs the upload → submit → wait → download flow."""
    import src.batch.benchmark as bb

    launched, printed, appended = _main_stubs
    monkeypatch.setattr(
        "sys.argv",
        ["src/batch/benchmark.py", "--positions", "QB", "RB", "--note", "unit test", "--seed", "7"],
    )
    bb.main()

    # upload_data(bucket) was called, plus submit_job for each of 2 positions.
    assert any("upload" in entry for entry in launched)
    submitted = [e["pos"] for e in launched if "pos" in e]
    assert sorted(submitted) == ["QB", "RB"]
    # Seed plumbed through.
    assert all(e["seed"] == 7 for e in launched if "seed" in e)
    # History entry has our note.
    assert appended[0]["note"] == "unit test"


@pytest.mark.unit
def test_main_empty_metrics_early_returns(monkeypatch, tmp_path):
    """If download_metrics returns nothing, main() prints and exits — no writes."""
    import src.batch.benchmark as bb

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bb, "download_metrics", lambda positions: {})
    monkeypatch.setattr(bb, "submit_job", lambda pos, seed: (pos, "id"))
    monkeypatch.setattr(bb, "upload_data", lambda bucket: None)
    monkeypatch.setattr(bb, "wait_for_jobs", lambda job_ids: {p: ("SUCCEEDED", 0) for p in job_ids})

    appended: list[dict] = []
    monkeypatch.setattr(bb, "append_to_history", lambda p, e: appended.append(e))
    monkeypatch.setattr(bb, "print_comparison_table", lambda *a, **k: None)
    monkeypatch.setattr(bb, "summarize_pipeline_result", lambda *a, **k: {})

    monkeypatch.setattr("sys.argv", ["src/batch/benchmark.py", "--positions", "QB"])
    bb.main()
    # Early-return branch: no history writes.
    assert appended == []


@pytest.mark.unit
def test_main_reports_failed_jobs(monkeypatch, tmp_path, capsys):
    """When wait_for_jobs returns FAILED for a position, main() prints it."""
    import src.batch.benchmark as bb

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bb, "upload_data", lambda bucket: None)
    monkeypatch.setattr(bb, "submit_job", lambda pos, seed: (pos, f"j-{pos}"))

    def _wait(job_ids):
        out = {p: ("SUCCEEDED", 0) for p in job_ids}
        out["QB"] = ("FAILED", 0)
        return out

    monkeypatch.setattr(bb, "wait_for_jobs", _wait)
    monkeypatch.setattr(bb, "download_metrics", lambda positions: {})  # early-exit after
    monkeypatch.setattr("sys.argv", ["src/batch/benchmark.py", "--positions", "QB", "RB"])
    bb.main()
    out = capsys.readouterr().out
    assert "Failed positions" in out
    assert "QB" in out


@pytest.mark.unit
def test_main_reports_submit_exception(monkeypatch, tmp_path, capsys):
    """If submit_job raises, main() logs the failure and keeps going."""
    import src.batch.benchmark as bb

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bb, "upload_data", lambda bucket: None)

    def _bad_submit(pos, seed):
        raise RuntimeError(f"{pos} submit boom")

    monkeypatch.setattr(bb, "submit_job", _bad_submit)
    monkeypatch.setattr(bb, "wait_for_jobs", lambda job_ids: {})
    monkeypatch.setattr(bb, "download_metrics", lambda positions: {})
    monkeypatch.setattr("sys.argv", ["src/batch/benchmark.py", "--positions", "QB"])
    bb.main()
    out = capsys.readouterr().out
    assert "FAILED to submit" in out
