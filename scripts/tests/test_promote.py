"""Tests for scripts/promote.py — manual rollback CLI."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import promote
from shared.model_sync import (  # noqa: E402
    legacy_model_key,
    manifest_key,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Fake S3 — subset of the boto3 surface promote.py touches.
# --------------------------------------------------------------------------


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


class _FakeS3:
    """In-memory S3 stand-in with just the operations promote.py needs:
    get_object, put_object, head_object, copy_object.
    """

    def __init__(self, objects: dict[str, bytes]):
        self.objects = dict(objects)
        self.ops: list[tuple[str, str]] = []

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise _nosuchkey_error(Key)
        self.ops.append(("get", Key))
        return {"Body": _FakeBody(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        if hasattr(Body, "read"):
            Body = Body.read()
        self.objects[Key] = Body
        self.ops.append(("put", Key))

    def head_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise ClientError(
                error_response={"Error": {"Code": "NoSuchKey", "Message": f"{Key} not found"}},
                operation_name="HeadObject",
            )
        self.ops.append(("head", Key))
        return {"ContentLength": len(self.objects[Key])}

    def copy_object(self, Bucket, Key, CopySource):  # noqa: N803
        src_key = CopySource["Key"] if isinstance(CopySource, dict) else CopySource
        if src_key not in self.objects:
            raise _nosuchkey_error(src_key)
        self.objects[Key] = self.objects[src_key]
        self.ops.append(("copy", f"{src_key}->{Key}"))


# --------------------------------------------------------------------------
# Manifest fixture factories.
# --------------------------------------------------------------------------


def _hist_key(n: int) -> str:
    """Produce a ``shared.model_sync.new_history_key``-shaped path."""
    return f"models/WR/history/2026-04-{n:02d}T00-00-00Z-aaaa{n:03d}/model.tar.gz"


def _make_manifest(current_key: str, previous_key: str | None, history: list[str]) -> dict:
    cur = {
        "key": current_key,
        "sha7": "current1",
        "bytes": 4096,
        "uploaded_at": "2026-04-23T00-00-00Z",
    }
    prev = None
    if previous_key is not None:
        prev = {
            "key": previous_key,
            "sha7": "previou",
            "bytes": 4096,
            "uploaded_at": "2026-04-22T00-00-00Z",
        }
    return {
        "schema_version": 1,
        "current": cur,
        "previous": prev,
        "history": history,
    }


def _bucket_with_manifest(
    prefix: str,
    pos: str,
    current_key: str,
    previous_key: str | None,
    history: list[str],
    *,
    history_objects: bool = True,
) -> _FakeS3:
    """Prime a FakeS3 with a manifest at ``{prefix}/{pos}/manifest.json`` plus
    a distinct tarball body at every key in ``history`` (so copy/head_object
    succeed). Set ``history_objects=False`` to test the GC-orphaned case.
    """
    manifest = _make_manifest(current_key, previous_key, history)
    objects: dict[str, bytes] = {
        manifest_key(prefix, pos): json.dumps(manifest).encode("utf-8"),
    }
    if history_objects:
        for k in history:
            objects[k] = f"BYTES-FOR-{k}".encode()
    return _FakeS3(objects)


# --------------------------------------------------------------------------
# list_history — pure function
# --------------------------------------------------------------------------


class TestListHistory:
    def test_annotates_current_and_previous(self):
        m = _make_manifest(
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        out = promote.list_history(m)
        assert "[0]" in out and _hist_key(5) in out and "← current" in out
        assert "[1]" in out and _hist_key(4) in out and "← previous" in out
        # Entry [2] has no flags.
        assert "[2]" in out
        assert out.count("← current") == 1
        assert out.count("← previous") == 1

    def test_empty_history_is_handled(self):
        m = _make_manifest(current_key=_hist_key(1), previous_key=None, history=[])
        assert "(empty)" in promote.list_history(m)


# --------------------------------------------------------------------------
# _parse_version_from_key
# --------------------------------------------------------------------------


class TestParseVersionFromKey:
    def test_well_formed_key(self):
        ts, sha7 = promote._parse_version_from_key(
            "models/WR/history/2026-04-23T00-00-00Z-abc1234/model.tar.gz"
        )
        assert ts == "2026-04-23T00-00-00Z"
        assert sha7 == "abc1234"

    def test_legacy_key_returns_blanks(self):
        ts, sha7 = promote._parse_version_from_key("models/WR/model.tar.gz")
        assert ts == "" and sha7 == ""

    def test_malformed_dir_returns_blanks(self):
        ts, sha7 = promote._parse_version_from_key("models/WR/history/no-dashes-here/model.tar.gz")
        assert ts == "no-dashes"  # rsplit("-", 1) still splits; accept whatever lands in ts
        assert sha7 == "here"


# --------------------------------------------------------------------------
# promote() — core CLI logic
# --------------------------------------------------------------------------


class TestPromote:
    def test_happy_path_promotes_history_entry(self):
        """current=A, previous=B, history=[A, B, C]; promote --to C
        → current=C, previous=A, history unchanged. Legacy mirror copied."""
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        target = _hist_key(3)

        new = promote.promote(fake, "b", "models", "WR", target)

        assert new["current"]["key"] == target
        assert new["previous"]["key"] == _hist_key(5)
        assert new["history"] == [_hist_key(5), _hist_key(4), _hist_key(3)]

        # Manifest was actually written.
        on_disk = json.loads(fake.objects[manifest_key("models", "WR")])
        assert on_disk == new

        # Legacy mirror bytes == target's bytes.
        assert fake.objects[legacy_model_key("models", "WR")] == fake.objects[target]

    def test_rejects_key_not_in_history(self):
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        bad = _hist_key(99)  # not in history
        with pytest.raises(promote.PromotionError, match="not in manifest.history"):
            promote.promote(fake, "b", "models", "WR", bad)

        # Manifest must be untouched.
        on_disk = json.loads(fake.objects[manifest_key("models", "WR")])
        assert on_disk["current"]["key"] == _hist_key(5)  # no change
        # Legacy mirror was NOT overwritten.
        assert legacy_model_key("models", "WR") not in fake.objects

    def test_rejects_key_present_in_history_but_missing_in_s3(self):
        """Defensive: if GC deleted a tracked key (shouldn't happen, but it
        has been a source of subtle bugs in the past), refuse to promote
        rather than write a manifest pointing at a ghost."""
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
            history_objects=False,  # no tarball bodies in S3
        )
        with pytest.raises(promote.PromotionError, match="missing from S3"):
            promote.promote(fake, "b", "models", "WR", _hist_key(3))

        # Manifest unchanged, legacy not created.
        on_disk = json.loads(fake.objects[manifest_key("models", "WR")])
        assert on_disk["current"]["key"] == _hist_key(5)
        assert legacy_model_key("models", "WR") not in fake.objects

    def test_dry_run_does_not_write(self):
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        orig_manifest_bytes = fake.objects[manifest_key("models", "WR")]

        new = promote.promote(fake, "b", "models", "WR", _hist_key(3), dry_run=True)
        assert new["current"]["key"] == _hist_key(3)

        # No put/copy was issued.
        assert not any(op[0] in ("put", "copy") for op in fake.ops)
        # Manifest bytes still the original.
        assert fake.objects[manifest_key("models", "WR")] == orig_manifest_bytes
        # No legacy mirror was created in dry-run.
        assert legacy_model_key("models", "WR") not in fake.objects

    def test_promote_with_no_previous(self):
        """Starting from a fresh bucket where previous=None, promoting still
        works; the new manifest's previous becomes old.current as normal."""
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=None,
            history=[_hist_key(5), _hist_key(4)],
        )
        new = promote.promote(fake, "b", "models", "WR", _hist_key(4))
        assert new["current"]["key"] == _hist_key(4)
        assert new["previous"]["key"] == _hist_key(5)

    def test_raises_when_no_manifest_exists(self):
        """Fresh bucket with only a legacy model.tar.gz (pre-migration) and
        no manifest — promotion has nothing to rewrite; fail clearly."""
        fake = _FakeS3(objects={legacy_model_key("models", "WR"): b"legacy"})
        with pytest.raises(promote.PromotionError, match="No manifest"):
            promote.promote(fake, "b", "models", "WR", _hist_key(1))

    def test_schema_version_preserved(self):
        """Defensive: if a future manifest bumps schema_version, promotion
        shouldn't silently downgrade it."""
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4)],
        )
        # Patch the stored manifest to schema_version=2.
        raw = json.loads(fake.objects[manifest_key("models", "WR")])
        raw["schema_version"] = 2
        fake.objects[manifest_key("models", "WR")] = json.dumps(raw).encode()

        new = promote.promote(fake, "b", "models", "WR", _hist_key(4))
        assert new["schema_version"] == 2


# --------------------------------------------------------------------------
# main() — end-to-end CLI coverage
# --------------------------------------------------------------------------


class TestMainCLI:
    @pytest.fixture
    def stub_boto3(self, monkeypatch):
        """Swap ``sys.modules["boto3"]`` so ``import boto3`` inside
        ``promote.main()`` resolves to a stub whose ``.client(...)`` returns
        the given ``_FakeS3``. Monkeypatch's auto-unwind restores the real
        module at teardown."""
        import sys as _sys
        import types

        def _factory(fake_s3: _FakeS3):
            fake_mod = types.ModuleType("boto3")
            fake_mod.client = lambda *a, **kw: fake_s3  # type: ignore[attr-defined]
            monkeypatch.setitem(_sys.modules, "boto3", fake_mod)

        return _factory

    def test_list_prints_history(self, stub_boto3, capsys):
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        stub_boto3(fake)
        rc = promote.main(["--position", "WR", "--list"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "← current" in out
        assert "← previous" in out

    def test_to_promotes_and_logs(self, stub_boto3, capsys):
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        stub_boto3(fake)
        rc = promote.main(["--position", "WR", "--to", _hist_key(3)])
        out = capsys.readouterr().out
        assert rc == 0
        assert f"Promoted WR: current → {_hist_key(3)}" in out
        # Manifest actually changed.
        m = json.loads(fake.objects[manifest_key("models", "WR")])
        assert m["current"]["key"] == _hist_key(3)

    def test_to_with_dry_run_prints_json_does_not_write(self, stub_boto3, capsys):
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        orig = fake.objects[manifest_key("models", "WR")]
        stub_boto3(fake)
        rc = promote.main(["--position", "WR", "--to", _hist_key(3), "--dry-run"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "[dry-run]" in out
        # JSON is emitted and parseable.
        planned = json.loads(out.split("\n", 1)[1])
        assert planned["current"]["key"] == _hist_key(3)
        # Manifest bytes unchanged.
        assert fake.objects[manifest_key("models", "WR")] == orig

    def test_error_exit_code_on_bad_key(self, stub_boto3, capsys):
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4)],
        )
        stub_boto3(fake)
        rc = promote.main(["--position", "WR", "--to", _hist_key(99)])
        err = capsys.readouterr().err
        assert rc == 2
        assert "ERROR" in err
        assert "not in manifest.history" in err
