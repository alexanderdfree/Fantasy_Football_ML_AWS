"""Tests for the in-flight model refresh path in src.shared.model_sync.

Covers refresh_position (head-then-fall-through-and-swap) and
start_refresh_poller (daemon thread driving the above for all positions).
The boot-time sync path is exercised by test_model_sync.py.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest import mock

import pytest
from botocore.exceptions import ClientError

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.shared import model_sync
from tests.shared._helpers import FakeBody as _FakeBody
from tests.shared._helpers import make_tarball as _make_tarball


def _manifest_body(history_key: str) -> bytes:
    return json.dumps(
        {
            "schema_version": 3,
            "current": {
                "key": history_key,
                "sha7": "abc1234",
                "bytes": 1024,
                "uploaded_at": "2026-05-21T00-00-00Z",
            },
            "stable": {"key": history_key},
            "previous": None,
            "previous_stable": None,
            "history": [history_key],
        }
    ).encode("utf-8")


def _nosuchkey() -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": "NoSuchKey", "Message": "missing"}},
        operation_name="GetObject",
    )


class _FakeS3:
    """Minimal stub: head_object returns a configurable etag per key;
    get_object returns the stored bytes. Both raise NoSuchKey on missing keys.
    """

    def __init__(self, objects: dict[str, bytes], etags: dict[str, str] | None = None):
        self._objects = objects
        self._etags = etags or {}
        self.head_calls: list[str] = []
        self.get_calls: list[str] = []

    def set_etag(self, key: str, etag: str) -> None:
        self._etags[key] = etag

    def head_object(self, Bucket: str, Key: str):  # noqa: N803
        self.head_calls.append(Key)
        if Key not in self._objects:
            raise _nosuchkey()
        return {"ETag": self._etags.get(Key, '"default-etag"')}

    def get_object(self, Bucket: str, Key: str):  # noqa: N803
        self.get_calls.append(Key)
        if Key not in self._objects:
            raise _nosuchkey()
        return {
            "Body": _FakeBody(self._objects[Key]),
            "ETag": self._etags.get(Key, '"default-etag"'),
        }


@pytest.fixture
def env_bucket(monkeypatch):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "models")


@pytest.fixture
def fake_root(monkeypatch, tmp_path):
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    # Pre-create the models dir so the swap has something to rename out of.
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        (tmp_path / "src" / pos.lower() / "outputs" / "models").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.mark.unit
def test_refresh_position_noop_when_bucket_unset(monkeypatch):
    """No FF_MODEL_S3_BUCKET → return (None, False), no S3 calls."""
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    etag, did_refresh = model_sync.refresh_position("QB", last_etag="something")
    assert etag is None
    assert did_refresh is False


@pytest.mark.unit
def test_refresh_position_bootstrap_records_etag_without_download(env_bucket, fake_root):
    """First call with last_etag=None records the etag without re-downloading
    (the boot-time sync already populated models/)."""
    history_key = "models/releases/v3/QB/history/2026-05-21T00-00-00Z-abc1234/model.tar.gz"
    objects = {
        "models/releases/v3/QB/manifest.json": _manifest_body(history_key),
        history_key: _make_tarball({"model.pkl": b"x"}),
    }
    fake_s3 = _FakeS3(objects, etags={"models/releases/v3/QB/manifest.json": '"v1"'})
    # A real protected boot records which artifact is already on disk.
    model_sync._sync_one(fake_s3, "test-bucket", "models", "QB", fake_root)
    fake_s3.head_calls.clear()
    fake_s3.get_calls.clear()

    new_etag, did_refresh = model_sync.refresh_position("QB", last_etag=None, s3_client=fake_s3)

    assert new_etag == '"v1"'
    assert did_refresh is False
    assert fake_s3.head_calls == ["models/releases/v3/QB/manifest.json"]
    assert "models/releases/v3/QB/manifest.json" in fake_s3.get_calls
    assert history_key not in fake_s3.get_calls
    # Sentinel was NOT created on bootstrap.
    assert not (fake_root / "src" / "qb" / "outputs" / ".refreshed_at").exists()


@pytest.mark.unit
def test_legacy_boot_then_first_protected_publication_refreshes_local_models(env_bucket, fake_root):
    legacy = "models/QB/history/legacy/model.tar.gz"
    protected = "models/releases/v3/QB/history/new/model.tar.gz"
    legacy_manifest = json.loads(_manifest_body(legacy))
    legacy_manifest["schema_version"] = 2
    fake = _FakeS3(
        {
            model_sync.legacy_manifest_key("models", "QB"): json.dumps(legacy_manifest).encode(),
            legacy: _make_tarball({"marker": b"legacy"}),
        }
    )
    model_sync._sync_one(fake, "test-bucket", "models", "QB", fake_root)
    previous, refreshed = model_sync.refresh_position("QB", last_etag=None, s3_client=fake)
    assert previous == '"default-etag"' and not refreshed
    fake._objects[model_sync.manifest_key("models", "QB")] = _manifest_body(protected)
    fake._objects[protected] = _make_tarball({"marker": b"protected"})
    fake.set_etag(model_sync.manifest_key("models", "QB"), '"protected-v1"')
    etag, refreshed = model_sync.refresh_position("QB", last_etag=previous, s3_client=fake)
    assert refreshed and etag == '"protected-v1"'
    assert (fake_root / "src/qb/outputs/models/marker").read_bytes() == b"protected"


@pytest.mark.unit
def test_refresh_tracks_predecessor_version_consumed_after_head_race(
    env_bucket, fake_root, monkeypatch
):
    pointer = model_sync.previous_protocol_manifest_key("models", "QB")
    first = "models/QB/releases/history/first/model.tar.gz"
    second = "models/QB/releases/history/second/model.tar.gz"
    fake = _FakeS3(
        {pointer: _manifest_body(first), first: _make_tarball({"marker": b"first"})},
        etags={pointer: '"first"'},
    )
    model_sync._sync_one(fake, "test-bucket", "models", "QB", fake_root)
    original = fake.head_object

    def head(Bucket, Key):
        result = original(Bucket, Key)
        if Key == pointer:
            fake._objects[pointer] = _manifest_body(second)
            fake._objects[second] = _make_tarball({"marker": b"second"})
            fake.set_etag(pointer, '"second"')
        return result

    monkeypatch.setattr(fake, "head_object", head)
    etag, refreshed = model_sync.refresh_position("QB", '"older"', s3_client=fake)
    assert refreshed and etag == '"second"'
    assert (fake_root / "src/qb/outputs/models/marker").read_bytes() == b"second"
    assert (fake_root / "src/qb/outputs/.manifest-etag").read_text() == '"second"'


@pytest.mark.unit
def test_refresh_position_unchanged_etag_skips_download(env_bucket, fake_root):
    """Second call with same etag → (etag, False), no get_object."""
    objects = {"models/releases/v3/QB/manifest.json": _manifest_body("ignored")}
    fake_s3 = _FakeS3(objects, etags={"models/releases/v3/QB/manifest.json": '"v1"'})

    new_etag, did_refresh = model_sync.refresh_position("QB", last_etag='"v1"', s3_client=fake_s3)

    assert new_etag == '"v1"'
    assert did_refresh is False
    assert fake_s3.get_calls == []


@pytest.mark.unit
def test_refresh_follows_protected_head_and_ignores_newer_legacy_writer(env_bucket, fake_root):
    protected = "models/releases/v3/QB/history/protected/model.tar.gz"
    legacy = "models/QB/history/legacy/model.tar.gz"
    protected_manifest = model_sync.manifest_key("models", "QB")
    legacy_manifest = model_sync.legacy_manifest_key("models", "QB")
    fake = _FakeS3(
        {
            protected_manifest: _manifest_body(protected),
            legacy_manifest: _manifest_body(legacy),
            protected: _make_tarball({"marker": b"protected"}),
            legacy: _make_tarball({"marker": b"legacy"}),
        },
        etags={protected_manifest: '"protected-v2"', legacy_manifest: '"legacy-v999"'},
    )
    etag, refreshed = model_sync.refresh_position("QB", last_etag='"protected-v1"', s3_client=fake)
    assert refreshed and etag == '"protected-v2"'
    assert (fake_root / "src/qb/outputs/models/marker").read_bytes() == b"protected"
    assert legacy_manifest not in fake.head_calls + fake.get_calls
    assert legacy not in fake.get_calls


@pytest.mark.unit
def test_refresh_position_changed_etag_swaps_and_touches_sentinel(env_bucket, fake_root):
    """When etag advances past a known last_etag, refresh_position downloads
    the new tarball, swaps it into models/, and touches the sentinel."""
    history_key = "models/releases/v3/RB/history/2026-05-21T01-00-00Z-def5678/model.tar.gz"
    new_tar = _make_tarball({"new_model.pkl": b"new content"})
    objects = {
        "models/releases/v3/RB/manifest.json": _manifest_body(history_key),
        history_key: new_tar,
    }
    fake_s3 = _FakeS3(objects, etags={"models/releases/v3/RB/manifest.json": '"v2"'})

    # Pre-populate the old models dir with a sentinel file so we can verify
    # the rename actually replaced the dir contents.
    models_dir = fake_root / "src" / "rb" / "outputs" / "models"
    (models_dir / "old_model.pkl").write_bytes(b"old content")

    new_etag, did_refresh = model_sync.refresh_position(
        "RB", last_etag='"v1-stale"', s3_client=fake_s3
    )

    assert new_etag == '"v2"'
    assert did_refresh is True
    # New model is in place, old model is gone.
    assert (models_dir / "new_model.pkl").read_bytes() == b"new content"
    assert not (models_dir / "old_model.pkl").exists()
    # Staging dirs cleaned up.
    assert not (fake_root / "src" / "rb" / "outputs" / "models.new").exists()
    assert not (fake_root / "src" / "rb" / "outputs" / "models.bak").exists()
    # Sentinel was touched.
    sentinel = fake_root / "src" / "rb" / "outputs" / ".refreshed_at"
    assert sentinel.exists()


@pytest.mark.unit
def test_refresh_position_head_failure_keeps_old_etag(env_bucket, fake_root):
    """head_object raises (network blip / IAM hiccup) → return (last_etag, False),
    no swap attempted. The next poll retries."""

    class _FakeS3Headfail:
        def head_object(self, Bucket, Key):  # noqa: N803
            raise _nosuchkey()

    new_etag, did_refresh = model_sync.refresh_position(
        "WR", last_etag='"v1"', s3_client=_FakeS3Headfail()
    )
    assert new_etag == '"v1"'
    assert did_refresh is False


@pytest.mark.unit
def test_refresh_position_extract_failure_keeps_old_models(env_bucket, fake_root):
    """If the tarball download fails for every entry in the manifest, the live
    models/ dir is left untouched and (last_etag, False) is returned."""
    history_key = "models/releases/v3/TE/history/missing.tar.gz"
    # Manifest references a key that doesn't exist in the bucket.
    objects = {"models/releases/v3/TE/manifest.json": _manifest_body(history_key)}
    fake_s3 = _FakeS3(objects, etags={"models/releases/v3/TE/manifest.json": '"v2"'})

    models_dir = fake_root / "src" / "te" / "outputs" / "models"
    (models_dir / "kept.pkl").write_bytes(b"untouched")

    new_etag, did_refresh = model_sync.refresh_position("TE", last_etag='"v1"', s3_client=fake_s3)

    assert new_etag == '"v1"'
    assert did_refresh is False
    # Old models still there.
    assert (models_dir / "kept.pkl").read_bytes() == b"untouched"
    # Staging cleaned up.
    assert not (fake_root / "src" / "te" / "outputs" / "models.new").exists()


@pytest.mark.unit
def test_refresh_unapproved_candidate_keeps_old_models(env_bucket, fake_root):
    key = "models/releases/v3/TE/history/failed-smoke/model.tar.gz"
    manifest = json.loads(_manifest_body(key))
    manifest["stable"] = None
    manifest["current"]["smoke_passed"] = False
    fake = _FakeS3(
        {
            "models/releases/v3/TE/manifest.json": json.dumps(manifest).encode(),
            key: _make_tarball({"failed.pkl": b"unapproved"}),
        },
        etags={"models/releases/v3/TE/manifest.json": '"v2"'},
    )
    models = fake_root / "src/te/outputs/models"
    (models / "kept.pkl").write_bytes(b"last working")
    assert model_sync.refresh_position("TE", '"v1"', s3_client=fake) == ('"v1"', False)
    assert (models / "kept.pkl").read_bytes() == b"last working"
    assert not (models / "failed.pkl").exists()
    assert key not in fake.get_calls


@pytest.mark.unit
def test_refresh_position_reaps_stale_staging_dirs(env_bucket, fake_root):
    """A prior crashed refresh might have left models.new/ or models.bak/
    behind. refresh_position must remove them before extracting."""
    history_key = "models/releases/v3/K/history/fresh.tar.gz"
    objects = {
        "models/releases/v3/K/manifest.json": _manifest_body(history_key),
        history_key: _make_tarball({"k_model.pkl": b"y"}),
    }
    fake_s3 = _FakeS3(objects, etags={"models/releases/v3/K/manifest.json": '"v2"'})

    # Leave a stale staging dir behind to simulate a prior crash.
    stale_new = fake_root / "src" / "k" / "outputs" / "models.new"
    stale_new.mkdir(parents=True, exist_ok=True)
    (stale_new / "crashed_remnant.pkl").write_bytes(b"junk")

    new_etag, did_refresh = model_sync.refresh_position("K", last_etag='"v1"', s3_client=fake_s3)

    assert did_refresh is True
    # Staging dir is gone (rename moved the new content into models/).
    assert not (fake_root / "src" / "k" / "outputs" / "models.new").exists()
    # The stale remnant should NOT have ended up in the live dir.
    assert not (fake_root / "src" / "k" / "outputs" / "models" / "crashed_remnant.pkl").exists()


@pytest.mark.unit
@pytest.mark.parametrize("bad_pos", ["", "../../etc", "INVALID", None, 42, "qb/../wr"])
def test_refresh_position_invalid_pos_returns_unchanged_etag(bad_pos):
    """The allowlist guard at the top of refresh_position must reject anything
    that isn't a known position, returning (last_etag, False) so the poller's
    state doesn't get corrupted and no filesystem operations fire. This is
    the defense the CodeQL path-injection alert wanted; the test pins it."""
    new_etag, did_refresh = model_sync.refresh_position(bad_pos, last_etag='"sentinel"')
    assert new_etag == '"sentinel"'
    assert did_refresh is False


@pytest.mark.unit
@pytest.mark.parametrize("bad_pos", ["", "../../etc", "INVALID", None, 42, "qb/../wr"])
def test_refresh_sentinel_mtime_invalid_pos_returns_zero(bad_pos):
    """refresh_sentinel_mtime must return 0.0 (the 'no refresh pending'
    sentinel) for any pos outside the POSITIONS allowlist, without touching
    the filesystem. Defends against accidental path traversal via the
    `pos.lower()` used in the path construction."""
    assert model_sync.refresh_sentinel_mtime(bad_pos) == 0.0


@pytest.mark.unit
def test_refresh_sentinel_mtime_returns_zero_when_absent(monkeypatch, tmp_path):
    """refresh_sentinel_mtime is the public read-side of the poller's touch.
    Returns 0.0 when the sentinel doesn't exist so the serving fast-path
    treats absence as 'no refresh pending'."""
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    assert model_sync.refresh_sentinel_mtime("QB") == 0.0


@pytest.mark.unit
def test_refresh_sentinel_mtime_returns_mtime_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    sentinel = tmp_path / "src" / "qb" / "outputs" / ".refreshed_at"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.touch()
    mt = model_sync.refresh_sentinel_mtime("QB")
    assert mt > 0.0


@pytest.mark.unit
def test_start_refresh_poller_calls_refresh_position_for_all_positions(monkeypatch):
    """The poller thread must iterate every position. We replace refresh_position
    with a stub that records calls, then verify each POSITION appears at least
    once before stopping the thread."""
    seen: list[str] = []
    barrier = threading.Event()

    def fake_refresh(pos, last_etag, s3_client=None):
        seen.append(pos)
        if set(seen) >= set(model_sync.POSITIONS):
            barrier.set()
        return f"etag-{pos}", False

    monkeypatch.setattr(model_sync, "refresh_position", fake_refresh)

    stop = threading.Event()
    thread = model_sync.start_refresh_poller(interval_s=0, stop_event=stop)
    try:
        assert thread.daemon is True
        # Wait up to 2s for the poller to cycle through all 6 positions once.
        assert barrier.wait(timeout=2.0), f"poller did not visit all positions; seen={seen}"
    finally:
        # Stop + join BEFORE monkeypatch restores the real refresh_position, so
        # the daemon never calls real boto3. A leaked spinning poller pollutes
        # later tests' global boto3.client mocks — the cross-test flake this fixes
        # (tests/shared/test_local_benchmark_sync.py saw boto3.client "called 7 times").
        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "refresh poller thread did not stop"


@pytest.mark.unit
def test_start_refresh_poller_survives_refresh_exception(monkeypatch):
    """If refresh_position raises (unexpected bug), the poller thread must not
    die — it logs and continues with the next position."""
    seen: list[str] = []
    raised_once: list[bool] = [False]

    def fake_refresh(pos, last_etag, s3_client=None):
        seen.append(pos)
        if not raised_once[0]:
            raised_once[0] = True
            raise RuntimeError("simulated bug")
        return f"etag-{pos}", False

    monkeypatch.setattr(model_sync, "refresh_position", fake_refresh)

    stop = threading.Event()
    thread = model_sync.start_refresh_poller(interval_s=0, stop_event=stop)
    try:
        # Give the daemon time to recover from the exception and visit at least 2
        # more positions, proving it didn't die.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if len(seen) >= 3:
                break
            time.sleep(0.05)
        assert len(seen) >= 3, f"poller died after exception; seen={seen}"
    finally:
        # Stop + join before teardown so the daemon doesn't leak and call real
        # boto3 (see the sibling test + start_refresh_poller docstring).
        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "refresh poller thread did not stop"
