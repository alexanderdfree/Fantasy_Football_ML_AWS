"""Tests for shared.artifact_gc.prune — retention for versioned artifacts."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
from botocore.exceptions import ClientError

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.artifact_gc import prune


class _FakeS3Gc:
    """Minimal S3 fake exposing just the two calls prune needs."""

    def __init__(self, keys: list[str]):
        self._keys = set(keys)
        self.deletes: list[str] = []

    def get_paginator(self, op):
        assert op == "list_objects_v2"
        keys = self._keys

        class _Paginator:
            def paginate(self, Bucket: str, Prefix: str):  # noqa: N803
                yield {"Contents": [{"Key": k} for k in keys if k.startswith(Prefix)]}

        return _Paginator()

    def delete_objects(self, Bucket, Delete):  # noqa: N803
        for obj in Delete["Objects"]:
            self._keys.discard(obj["Key"])
            self.deletes.append(obj["Key"])


def _history_key(n: int) -> str:
    return f"models/QB/history/2026-04-{n:02d}T00-00-00Z-aaa{n:04d}/model.tar.gz"


def test_prune_keeps_current_previous_and_history():
    """Any key referenced by current/previous/history[:keep_n] must survive."""
    keys = [_history_key(n) for n in range(1, 11)]  # 10 entries
    fake = _FakeS3Gc(keys)

    manifest = {
        "current": {"key": _history_key(10)},
        "previous": {"key": _history_key(9)},
        "history": [_history_key(n) for n in range(10, 5, -1)],  # newest-first, 5 items
    }

    deleted = prune(fake, "bucket", "models", "QB", manifest, keep_n=5)

    # Entries 1..5 are beyond the 5-newest window AND not pointed at.
    assert set(deleted) == {_history_key(n) for n in range(1, 6)}
    # Entries 6..10 survive (all referenced by current/previous/history).
    for n in range(6, 11):
        assert _history_key(n) in fake._keys


def test_prune_is_idempotent_on_rerun():
    """A second run with the same manifest deletes nothing new."""
    keys = [_history_key(n) for n in range(1, 8)]  # 7 entries
    fake = _FakeS3Gc(keys)
    manifest = {
        "current": {"key": _history_key(7)},
        "previous": {"key": _history_key(6)},
        "history": [_history_key(n) for n in range(7, 2, -1)],  # 5 entries
    }

    first = prune(fake, "bucket", "models", "QB", manifest, keep_n=5)
    second = prune(fake, "bucket", "models", "QB", manifest, keep_n=5)
    # Set compare: the paginator iterates keys in insertion / set order,
    # not lexical order — only the membership matters for correctness.
    assert set(first) == {_history_key(1), _history_key(2)}
    assert second == []


def test_prune_sweeps_orphans_not_in_manifest():
    """Keys under history/ that the manifest never knew about (e.g. abandoned
    uploads from a crashed producer) get cleaned up too — source of truth is
    the S3 listing, not the manifest's history list."""
    orphans = [
        "models/QB/history/abandoned-1/model.tar.gz",
        "models/QB/history/abandoned-2/model.tar.gz",
    ]
    tracked = [_history_key(n) for n in range(1, 4)]
    fake = _FakeS3Gc(tracked + orphans)

    manifest = {
        "current": {"key": _history_key(3)},
        "previous": {"key": _history_key(2)},
        "history": tracked,
    }

    deleted = prune(fake, "bucket", "models", "QB", manifest, keep_n=5)
    assert set(deleted) == set(orphans)
    for k in tracked:
        assert k in fake._keys


def test_prune_handles_missing_manifest_entries_gracefully():
    """previous may be None on the first post-migration write; current's key
    may be absent if called with a weirdly shaped manifest. Prune must not
    crash."""
    fake = _FakeS3Gc([_history_key(1)])
    manifest = {
        "current": None,
        "previous": None,
        "history": [],
    }
    # No current/previous/history → everything under history/ gets deleted.
    deleted = prune(fake, "bucket", "models", "QB", manifest, keep_n=5)
    assert deleted == [_history_key(1)]


def test_prune_respects_custom_position_prefix():
    """Only the requested position's history/ is listed and deleted from —
    prune('QB', ...) must not touch RB's artifacts even if the manifest
    passed in is RB's (which it wouldn't be in normal use, but the isolation
    matters)."""
    fake = _FakeS3Gc(
        [
            "models/QB/history/q1/model.tar.gz",
            "models/RB/history/r1/model.tar.gz",
        ]
    )
    manifest = {"current": None, "previous": None, "history": []}
    prune(fake, "bucket", "models", "QB", manifest)
    assert "models/RB/history/r1/model.tar.gz" in fake._keys
    assert "models/QB/history/q1/model.tar.gz" not in fake._keys


def test_prune_propagates_real_s3_errors():
    """Not a retention concern: if the listing call itself errors (auth,
    throttle), prune should raise so the caller's best-effort try/except
    in upload_artifacts can log and move on."""

    class _ErroringS3:
        def get_paginator(self, op):
            raise ClientError(
                error_response={"Error": {"Code": "AccessDenied", "Message": "no"}},
                operation_name="ListObjectsV2",
            )

    manifest = {"current": {"key": "k"}, "previous": None, "history": ["k"]}
    with pytest.raises(ClientError, match="AccessDenied"):
        prune(_ErroringS3(), "bucket", "models", "QB", manifest)


def test_prune_no_op_when_history_empty():
    """No keys under history/ → no delete call is made (S3 charges per
    request; a cleanup run shouldn't hit delete_objects at all)."""
    fake = _FakeS3Gc([])
    # patch delete_objects to detect if it gets called
    fake.delete_objects = mock.MagicMock(wraps=fake.delete_objects)
    manifest = {"current": {"key": "x"}, "previous": None, "history": ["x"]}
    deleted = prune(fake, "bucket", "models", "QB", manifest)
    assert deleted == []
    fake.delete_objects.assert_not_called()


def test_prune_preserves_stable_key_outside_history_window():
    """The artifact pointed to by ``stable`` must survive prune even when
    it's older than the ``keep_n`` newest history entries. This is the
    conservation guarantee the consumer's stable-first fallback depends on:
    if a stretch of failing-smoke-test retrains pushes ``stable`` off the
    history window, the pointer still resolves to a real key.
    """
    keys = [_history_key(n) for n in range(1, 11)]  # 10 entries, oldest n=1
    fake = _FakeS3Gc(keys)

    # Manifest's history is the newest 5 (keys 6..10). Stable points at the
    # OLDEST key (n=1), which would otherwise be pruned.
    manifest = {
        "current": {"key": _history_key(10)},
        "stable": {"key": _history_key(1)},
        "previous": {"key": _history_key(9)},
        "history": [_history_key(n) for n in range(10, 5, -1)],
    }
    deleted = prune(fake, "bucket", "models", "QB", manifest, keep_n=5)

    assert _history_key(1) not in deleted
    assert _history_key(1) in fake._keys
    # The other out-of-window entries (n=2..5) are still pruned.
    assert set(deleted) == {_history_key(n) for n in range(2, 6)}


def test_prune_handles_missing_stable_field():
    """Old (v1) manifests have no ``stable`` field. Prune must read absence
    as 'no stable pin' and behave exactly as before — keep current, previous,
    and the top-N history."""
    keys = [_history_key(n) for n in range(1, 6)]
    fake = _FakeS3Gc(keys)
    manifest = {
        "current": {"key": _history_key(5)},
        "previous": {"key": _history_key(4)},
        "history": [_history_key(n) for n in range(5, 0, -1)],
    }
    deleted = prune(fake, "bucket", "models", "QB", manifest, keep_n=5)
    # All 5 keys are in history[:5] → nothing to delete.
    assert deleted == []
