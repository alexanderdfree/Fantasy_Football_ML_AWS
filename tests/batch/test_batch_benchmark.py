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


def _fake_manifest(pos: str, current_key: str | None = None, stable_key: str | None = None):
    """Build a fake manifest with the slots populated for testing."""
    base_key = current_key or f"models/{pos}/history/2026-05-21T05-00-00Z-abc1234/model.tar.gz"
    manifest = {
        "schema_version": 2,
        "current": {"key": base_key, "sha7": "abc1234", "bytes": 1000, "uploaded_at": "now"},
    }
    if stable_key is not None:
        manifest["stable"] = {
            "key": stable_key,
            "sha7": "def5678",
            "bytes": 999,
            "uploaded_at": "earlier",
        }
    return manifest


@pytest.mark.unit
def test_download_metrics_happy_path(tmp_path, monkeypatch):
    """Manifest's ``current`` resolves → tar fetched from history key → metrics dict."""
    import src.batch.benchmark as bb

    tar_path = _make_tarfile_with_metrics(tmp_path, {"position": "QB", "mae": 6.2})

    class _FakeS3:
        def download_file(self, bucket, key, dest):
            # Confirm the manifest's history key (NOT the legacy mirror) is what
            # gets requested — the whole point of Layer B is to stop reading
            # ``models/{POS}/model.tar.gz``.
            assert key == "models/QB/history/2026-05-21T05-00-00Z-abc1234/model.tar.gz"
            with open(tar_path, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())

    monkeypatch.setattr(bb.boto3, "client", lambda *a, **k: _FakeS3())
    monkeypatch.setattr(bb, "load_manifest", lambda s3, bucket, prefix, pos: _fake_manifest(pos))

    result = bb.download_metrics(["QB"])
    assert result == {"QB": {"position": "QB", "mae": 6.2}}


@pytest.mark.unit
def test_download_metrics_missing_metrics_file(tmp_path, monkeypatch):
    """Manifest resolves but tar has no benchmark_metrics.json → position omitted."""
    import src.batch.benchmark as bb

    tar_path = _make_tarfile_with_metrics(tmp_path, None)

    class _FakeS3:
        def download_file(self, bucket, key, dest):
            with open(tar_path, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())

    monkeypatch.setattr(bb.boto3, "client", lambda *a, **k: _FakeS3())
    monkeypatch.setattr(bb, "load_manifest", lambda s3, bucket, prefix, pos: _fake_manifest(pos))

    result = bb.download_metrics(["RB"])
    assert result == {}  # no metrics → nothing in the dict


@pytest.mark.unit
def test_download_metrics_s3_error_swallowed(monkeypatch):
    """``current`` download throws and no fallback → position omitted (not raised)."""
    import src.batch.benchmark as bb

    class _FakeS3:
        def download_file(self, *a, **k):
            raise RuntimeError("NoSuchKey")

    monkeypatch.setattr(bb.boto3, "client", lambda *a, **k: _FakeS3())
    monkeypatch.setattr(bb, "load_manifest", lambda s3, bucket, prefix, pos: _fake_manifest(pos))

    result = bb.download_metrics(["WR"])
    assert result == {}


@pytest.mark.unit
def test_download_metrics_absent_manifest_returns_none(monkeypatch):
    """No manifest at all → position omitted with a WARNING (not silently
    fallen back to the legacy key)."""
    import src.batch.benchmark as bb

    monkeypatch.setattr(bb.boto3, "client", lambda *a, **k: object())
    monkeypatch.setattr(bb, "load_manifest", lambda *a, **k: None)

    result = bb.download_metrics(["WR"])
    assert result == {}


@pytest.mark.unit
def test_download_metrics_falls_through_to_stable_on_current_failure(tmp_path, monkeypatch):
    """``current`` download fails → falls through to ``stable``."""
    import src.batch.benchmark as bb

    tar_path = _make_tarfile_with_metrics(tmp_path, {"position": "TE", "mae": 3.5})
    current_key = "models/TE/history/2026-05-21T06-00-00Z-bad9999/model.tar.gz"
    stable_key = "models/TE/history/2026-05-21T05-30-00Z-good777/model.tar.gz"

    class _FakeS3:
        def download_file(self, bucket, key, dest):
            if key == current_key:
                raise RuntimeError("simulated current-key failure")
            assert key == stable_key
            with open(tar_path, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())

    monkeypatch.setattr(bb.boto3, "client", lambda *a, **k: _FakeS3())
    monkeypatch.setattr(
        bb,
        "load_manifest",
        lambda *a, **k: _fake_manifest("TE", current_key=current_key, stable_key=stable_key),
    )

    result = bb.download_metrics(["TE"])
    assert result == {"TE": {"position": "TE", "mae": 3.5}}


# --------------------------------------------------------------------------
# find_git_sha_divergence — coherency check
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_find_git_sha_divergence_empty_when_no_expected():
    """No expected SHA (workflow_dispatch / local) → skip the check entirely."""
    import src.batch.benchmark as bb

    all_metrics = {"QB": {"git_sha": "abc1234deadbeef"}, "RB": {"git_sha": "ffffeee0000aaaa"}}
    assert bb.find_git_sha_divergence(all_metrics, None) == []
    assert bb.find_git_sha_divergence(all_metrics, "") == []


@pytest.mark.unit
def test_find_git_sha_divergence_coherent_all_match():
    """Every position's git_sha matches expected → no divergence."""
    import src.batch.benchmark as bb

    expected = "abc1234"
    all_metrics = {
        "QB": {"git_sha": "abc1234deadbeef"},
        "RB": {"git_sha": "abc1234ffffeeed"},  # same 7-char prefix
    }
    assert bb.find_git_sha_divergence(all_metrics, expected) == []


@pytest.mark.unit
def test_find_git_sha_divergence_flags_diverged_positions():
    """A position whose recorded SHA disagrees with the run's expected SHA
    surfaces as (pos, recorded_short_sha)."""
    import src.batch.benchmark as bb

    expected = "abc1234"
    all_metrics = {
        "QB": {"git_sha": "abc1234deadbeef"},
        "RB": {"git_sha": "fff5678cafe"},  # ≠ expected
        "WR": {"git_sha": "abc1234aaaaaaa"},
    }
    diverged = bb.find_git_sha_divergence(all_metrics, expected)
    assert diverged == [("RB", "fff5678")]


@pytest.mark.unit
def test_find_git_sha_divergence_skips_positions_without_sha():
    """Positions whose metrics lack ``git_sha`` (pre-PR artifacts) are
    silently skipped — absence is not divergence."""
    import src.batch.benchmark as bb

    expected = "abc1234"
    all_metrics = {
        "QB": {"git_sha": "abc1234deadbeef"},
        "RB": {},  # no git_sha — pre-PR artifact
    }
    assert bb.find_git_sha_divergence(all_metrics, expected) == []


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

    def _stub_append(path, entry, **kwargs):
        # Mirror the real signature: ``pr_number`` is recorded into the
        # entry so tests can assert against it. Returns a dummy file path
        # so the caller's downstream S3 upload helper has something to
        # work with (it's env-gated and no-ops in tests).
        out = dict(entry)
        if kwargs.get("pr_number") is not None:
            out["pr_number"] = int(kwargs["pr_number"])
        appended.append(out)
        return str(tmp_path / "history" / "fake.json")

    monkeypatch.setattr(bb, "append_to_history", _stub_append)
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
def test_main_git_hash_arg_overrides_workspace_head(_main_stubs, monkeypatch):
    """``--git-hash`` must override ``get_git_hash()`` in both the JSON's
    ``git_hash`` field and the ``run_id`` filename suffix, truncated to 7
    chars to match the existing run_id convention.

    Guards against the failure mode in TODO.md's "Benchmark ``git_hash``
    recorded the wrong SHA on rapid back-to-back merges" archive entry: the
    workflow workspace HEAD can advance past the image-build SHA when a
    follow-up PR lands while a Batch run is queued. Passing
    ``--git-hash $GITHUB_SHA`` from CI pins the recorded SHA to the SHA the
    docker image was actually built from. The unset/fallback path is
    covered by ``test_main_download_only_skips_submit``'s assertion that
    ``appended[0]["git_hash"] == "abc1234"`` (the stubbed get_git_hash).
    """
    import src.batch.benchmark as bb

    _, _, appended = _main_stubs
    monkeypatch.setattr(
        "sys.argv",
        ["src/batch/benchmark.py", "--download-only", "--git-hash", "d34db33f00ba12"],
    )
    bb.main()
    assert appended[0]["git_hash"] == "d34db33"
    assert appended[0]["run_id"].endswith("_d34db33")


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
    monkeypatch.setattr(
        bb, "append_to_history", lambda p, e, **kw: appended.append(e) or "/tmp/fake.json"
    )
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
    """If submit_job raises, main() logs the failure (and keeps going for other positions)."""
    import src.batch.benchmark as bb

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bb, "upload_data", lambda bucket: None)

    def _bad_submit(pos, seed):
        raise RuntimeError(f"{pos} submit boom")

    waited_for: list[dict] = []

    def _wait(job_ids):
        waited_for.append(dict(job_ids))
        return {}

    monkeypatch.setattr(bb, "submit_job", _bad_submit)
    monkeypatch.setattr(bb, "wait_for_jobs", _wait)
    monkeypatch.setattr(bb, "download_metrics", lambda positions: {})
    monkeypatch.setattr("sys.argv", ["src/batch/benchmark.py", "--positions", "QB", "RB"])
    bb.main()
    out = capsys.readouterr().out
    # Both submit failures must be logged with the position name + the
    # underlying error message, so a one-line CI tail still pinpoints which
    # position blew up and why.
    assert "[QB] FAILED to submit" in out
    assert "[RB] FAILED to submit" in out
    assert "QB submit boom" in out
    assert "RB submit boom" in out
    # Both submits raised → wait_for_jobs gets an empty job_ids dict (no
    # jobs to poll). The exception was caught, not propagated.
    assert waited_for == [{}]


# --------------------------------------------------------------------------
# _maybe_upload_to_s3 — env-gated S3 mirror of each new benchmark JSON
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_maybe_upload_to_s3_noop_when_bucket_unset(monkeypatch, tmp_path):
    """Without ``FF_MODEL_S3_BUCKET`` set, the helper must not call boto3 —
    matches the ``sync_*`` env-gate pattern so local dev / CI tests don't
    accidentally touch S3."""
    import src.batch.benchmark as bb

    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    sentinel = mock.Mock(side_effect=AssertionError("boto3.client should NOT be called"))
    with mock.patch("boto3.client", sentinel):
        bb._maybe_upload_to_s3(str(tmp_path / "anything.json"))
    sentinel.assert_not_called()


@pytest.mark.unit
def test_maybe_upload_to_s3_uploads_with_expected_key(monkeypatch, tmp_path, capsys):
    """With env set, calls ``s3.upload_file(local, bucket, key)`` where
    ``key = {prefix}/benchmark_history/{basename(local_path)}``."""
    import src.batch.benchmark as bb

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "my-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "models")
    fake_s3 = mock.Mock()
    with mock.patch("boto3.client", return_value=fake_s3) as boto_factory:
        bb._maybe_upload_to_s3(str(tmp_path / "2026-05-19T22-47-20_dff43fb.json"))
    boto_factory.assert_called_once_with("s3", region_name=bb.AWS_REGION)
    fake_s3.upload_file.assert_called_once_with(
        str(tmp_path / "2026-05-19T22-47-20_dff43fb.json"),
        "my-bucket",
        "models/benchmark_history/2026-05-19T22-47-20_dff43fb.json",
    )
    assert "s3://my-bucket/models/benchmark_history/" in capsys.readouterr().out


@pytest.mark.unit
def test_maybe_upload_to_s3_strips_prefix_slashes(monkeypatch, tmp_path):
    """A trailing/leading slash on FF_MODEL_S3_PREFIX must not produce a
    double-slash in the resulting S3 key (S3 treats // as literal segments)."""
    import src.batch.benchmark as bb

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "my-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "/nightly/v2/")
    fake_s3 = mock.Mock()
    with mock.patch("boto3.client", return_value=fake_s3):
        bb._maybe_upload_to_s3(str(tmp_path / "x.json"))
    args = fake_s3.upload_file.call_args.args
    assert args[2] == "nightly/v2/benchmark_history/x.json"


# --------------------------------------------------------------------------
# record_benchmark_run — shared by main() (CLI/CI) and launch.py auto-append
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_record_benchmark_run_writes_row(_main_stubs):
    """Aggregates the resolved metrics into one history row and returns the
    written path. ``_main_stubs`` stubs download_metrics → {QB, RB} + the
    benchmark_utils helpers and points HISTORY_DIR/RESULTS_FILE at tmp_path."""
    import src.batch.benchmark as bb

    _, printed, appended = _main_stubs
    path = bb.record_benchmark_run(["QB", "RB"], note="Standalone Batch run")

    assert path is not None
    assert len(appended) == 1
    row = appended[0]
    assert row["note"] == "Standalone Batch run"
    assert row["backend"] == "batch"
    assert row["positions"] == ["QB", "RB"]
    assert row["git_hash"] == "abc1234"  # stubbed get_git_hash
    assert len(printed) == 2  # QB + RB summaries


@pytest.mark.unit
def test_record_benchmark_run_returns_none_when_no_metrics(monkeypatch):
    """No resolvable metrics → prints, writes nothing, returns None (so the
    launch.py auto-append is a clean no-op)."""
    import src.batch.benchmark as bb

    monkeypatch.setattr(bb, "download_metrics", lambda positions: {})
    appended: list[dict] = []
    monkeypatch.setattr(
        bb, "append_to_history", lambda p, e, **kw: appended.append(e) or "/tmp/x.json"
    )
    monkeypatch.setattr(bb, "print_comparison_table", lambda *a, **k: None)

    assert bb.record_benchmark_run(["QB"]) is None
    assert appended == []
