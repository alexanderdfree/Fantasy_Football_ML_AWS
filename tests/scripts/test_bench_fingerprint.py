"""Unit tests for src/scripts/bench_fingerprint.py — the content-fingerprint
identity the pre-PR benchmark gate (B2) accepts evidence on.

Runs against a synthetic git repo (never this checkout) so head-vs-worktree
divergence, dirty files, and add/remove effects are all controlled. Fixture
pins ``git init -b main`` (CI git defaults to master) and disables gpgsign
(a global signing config makes ``git commit`` exit 128 in some sandboxes).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.scripts.bench_fingerprint import (
    GLOBAL_PATHS,
    collect_code_fingerprints,
    fingerprint_from_manifest,
    head_manifest,
    position_fingerprint,
    position_paths,
)
from src.scripts.scope_positions import _BENCH_SHARED_REGEX, ALL_POSITIONS

pytestmark = pytest.mark.unit


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(r)], check=True, capture_output=True)
    _git(r, "config", "user.email", "fp@example.test")
    _git(r, "config", "user.name", "FP Test")
    _git(r, "config", "commit.gpgsign", "false")
    files = {
        "src/__init__.py": "",
        "src/config.py": "SEASONS = [2024]\n",
        "src/shared/pipeline.py": "def run():\n    return 1\n",
        "src/data/loader.py": "def load():\n    return []\n",
        "src/features/engineer.py": "def build():\n    return {}\n",
        "src/qb/config.py": "ALPHA = 1.0\n",
        "src/qb/features.py": "def f():\n    return 0\n",
        "src/rb/config.py": "ALPHA = 2.0\n",
    }
    for rel, content in files.items():
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(r, "add", "-A")
    _git(r, "commit", "-m", "init")
    return r


def test_head_equals_worktree_on_clean_tree(repo):
    for pos in ("QB", "RB"):
        head = position_fingerprint(pos, str(repo), source="head")
        wt = position_fingerprint(pos, str(repo), source="worktree")
        assert head == wt, pos


def test_dirty_tracked_file_diverges_worktree_only(repo):
    head_before = position_fingerprint("QB", str(repo), source="head")
    (repo / "src/qb/config.py").write_text("ALPHA = 99.0\n")
    assert position_fingerprint("QB", str(repo), source="head") == head_before
    assert position_fingerprint("QB", str(repo), source="worktree") != head_before


def test_per_position_edit_changes_only_that_position(repo):
    before = {p: position_fingerprint(p, str(repo), source="head") for p in ("QB", "RB")}
    (repo / "src/qb/config.py").write_text("ALPHA = 3.0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "qb edit")
    assert position_fingerprint("QB", str(repo), source="head") != before["QB"]
    assert position_fingerprint("RB", str(repo), source="head") == before["RB"]


def test_global_edit_changes_every_position(repo):
    before = {p: position_fingerprint(p, str(repo), source="head") for p in ("QB", "RB")}
    (repo / "src/shared/pipeline.py").write_text("def run():\n    return 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "shared edit")
    for p in ("QB", "RB"):
        assert position_fingerprint(p, str(repo), source="head") != before[p], p


def test_added_and_removed_files_change_fingerprint(repo):
    base = position_fingerprint("QB", str(repo), source="head")
    (repo / "src/qb/heads.py").write_text("X = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add file")
    with_file = position_fingerprint("QB", str(repo), source="head")
    assert with_file != base
    _git(repo, "rm", "-q", "src/qb/heads.py")
    _git(repo, "commit", "-m", "remove file")
    assert position_fingerprint("QB", str(repo), source="head") == base


def test_untracked_file_absent_from_both_modes(repo):
    before_head = position_fingerprint("QB", str(repo), source="head")
    before_wt = position_fingerprint("QB", str(repo), source="worktree")
    (repo / "src/qb/new_untracked.py").write_text("Y = 2\n")
    assert position_fingerprint("QB", str(repo), source="head") == before_head
    assert position_fingerprint("QB", str(repo), source="worktree") == before_wt


def test_pinned_digest_is_version_stable():
    """The manifest→digest format is a persisted contract (recorded in
    benchmark_history entries) — pin an exact value so a refactor that would
    silently orphan all recorded fingerprints fails here instead."""
    manifest = [("src/config.py", "a" * 40), ("src/qb/config.py", "b" * 40)]
    assert (
        fingerprint_from_manifest(manifest)
        == "0e8894180afb16b9631d159b6cb7edf586e6a4b059b61692a2ad659b3cfc68e5"
    )


def test_head_manifest_lists_blobs_sorted(repo):
    manifest = head_manifest(position_paths("QB"), str(repo))
    paths = [p for p, _ in manifest]
    assert paths == sorted(paths)
    assert "src/qb/config.py" in paths
    assert "src/shared/pipeline.py" in paths
    assert "src/rb/config.py" not in paths


def test_collect_fingerprints_fails_open_outside_repo(tmp_path, capsys):
    assert collect_code_fingerprints(["QB"], str(tmp_path / "not-a-repo")) is None
    assert "fingerprints unavailable" in capsys.readouterr().out


def test_manifest_covers_everything_the_gate_can_scope():
    """Soundness invariant: every path that can scope a position into the B2
    gate must be inside that position's fingerprint manifest — otherwise a
    scoped change outside the manifest lets stale evidence match silently."""
    # Shared-trigger paths are all under GLOBAL_PATHS.
    for probe in (
        "src/shared/deep/nested.py",
        "src/data/loader.py",
        "src/features/engineer.py",
        "src/config.py",
        "src/__init__.py",
    ):
        assert _BENCH_SHARED_REGEX.match(probe), probe
        assert any(probe == gp or probe.startswith(gp.rstrip("/") + "/") for gp in GLOBAL_PATHS), (
            probe
        )
    # Per-position prefix rule maps to the position dir, which is in its manifest.
    for pos in ALL_POSITIONS:
        assert f"src/{pos.lower()}" in position_paths(pos)
