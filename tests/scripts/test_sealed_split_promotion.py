"""Merge housekeeping must not splice new splits into an immutable data release."""

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.data import release
from tests.scripts.test_provider_guard_integrity import _git, repos  # noqa: F401
from tests.test_data_release import FakeS3

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def _promote(provider, parent, worktree):
    commit = _git(worktree, "rev-parse", "HEAD")
    args = [str(worktree), commit] if provider == "codex" else [commit]
    return subprocess.run(
        [
            "bash",
            "-c",
            f'source "$1"; shift; {provider}_promote_worktree_splits "$@"',
            "fixture",
            str(ROOT / f".{provider}/hooks/lib.sh"),
            *args,
        ],
        cwd=worktree,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _populate(parent, worktree):
    for root, value in ((parent, 1), (worktree, 2)):
        raw, splits = root / "data/raw", root / "data/splits"
        raw.mkdir()
        for path in [
            raw / "weekly.parquet",
            *[splits / f"{name}.parquet" for name in ("train", "val", "test")],
        ]:
            pd.DataFrame({"season": [2025], "week": [1], "value": [value]}).to_parquet(
                path, index=False
            )


@pytest.mark.parametrize("provider", ["claude", "codex"])
@pytest.mark.parametrize("side", ["parent", "worktree"])
@pytest.mark.parametrize("marker", ["data/raw/.release.json", f"data/splits/{release.SEAL_NAME}"])
def test_either_side_seal_prevents_partial_promotion(repos, provider, side, marker):
    parent, worktree = repos
    _populate(parent, worktree)
    source = parent if side == "parent" else worktree
    (source / marker).write_text('{"release_id":"fixture"}')
    before = {path.name: path.read_bytes() for path in (parent / "data/splits").iterdir()}
    result = _promote(provider, parent, worktree)
    assert result.returncode == 0, result.stderr
    assert "sealed" in result.stderr and "refresh raw and splits together" in result.stderr
    assert {path.name: path.read_bytes() for path in (parent / "data/splits").iterdir()} == before


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_unsealed_legacy_promotion_still_copies_verified_merge(repos, provider):
    parent, worktree = repos
    _populate(parent, worktree)
    result = _promote(provider, parent, worktree)
    assert result.returncode == 0, result.stderr
    assert "copied 3 parquet(s)" in result.stdout
    for name in ("train", "val", "test"):
        assert (parent / f"data/splits/{name}.parquet").read_bytes() == (
            worktree / f"data/splits/{name}.parquet"
        ).read_bytes()


def _publish_fixture(s3, root):
    raw, splits = root / "data/raw", root / "data/splits"
    manifest = {
        "schema_version": 1,
        "producer": release._producer_hashes(root),
        "data_producer_sha256": release.producer_fingerprint(release.data_producer_hashes(root)),
        "files": {
            name: release._record(path) for name, path in release._input_files(raw, splits).items()
        },
    }
    (splits / release.SEAL_NAME).write_text(json.dumps(manifest))
    selected = release.publish_release(
        s3, "fixture", raw_dir=raw, splits_dir=splits, repo_root=root
    )
    (raw / ".release.json").write_text(json.dumps({"release_id": selected}))
    return selected


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_actual_release_validation_stays_coherent_after_merge_hook(repos, provider):
    parent, worktree = repos
    _populate(parent, worktree)
    s3 = FakeS3()
    original = _publish_fixture(s3, parent)
    assert _publish_fixture(s3, worktree) != original
    result = _promote(provider, parent, worktree)
    assert result.returncode == 0, result.stderr
    # This exact production checksum gate failed after the old parquet-only copy.
    assert (
        release.publish_release(
            s3,
            "fixture",
            raw_dir=parent / "data/raw",
            splits_dir=parent / "data/splits",
            repo_root=parent,
        )
        == original
    )
