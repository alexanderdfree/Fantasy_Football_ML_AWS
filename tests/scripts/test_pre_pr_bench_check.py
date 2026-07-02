"""Unit tests for src/scripts/pre_pr_bench_check.py — the B2 gate's brain.

Everything runs chdir'd into a synthetic git repo (``git init -b main`` — CI
git defaults to master; gpgsign disabled), so the three evidence tiers, the
self-retiring legacy fallback, the shared arm, and the AST-inert detector are
all exercised against controlled history entries — none of this checkout's
real ``benchmark_history``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from src.scripts.bench_fingerprint import position_fingerprint
from src.scripts.pre_pr_bench_check import cmd_evaluate, cmd_inert, is_inert

pytestmark = pytest.mark.unit


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(r)], check=True, capture_output=True)
    _git(r, "config", "user.email", "b2@example.test")
    _git(r, "config", "user.name", "B2 Test")
    _git(r, "config", "commit.gpgsign", "false")
    files = {
        "src/__init__.py": "",
        "src/config.py": "SEASONS = [2024]\n",
        "src/shared/pipeline.py": "def run():\n    return 1\n",
        "src/data/loader.py": "def load():\n    return []\n",
        "src/features/engineer.py": "def build():\n    return {}\n",
        "src/te/features.py": "def f():\n    return 0\n",
        "src/qb/config.py": "ALPHA = 1.0\n",
    }
    for rel, content in files.items():
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    (r / "benchmark_history").mkdir()
    _git(r, "add", "-A")
    _git(r, "commit", "-m", "init")
    monkeypatch.chdir(r)
    return r


def _write_entry(repo: Path, name: str, positions, fingerprints=None, subdir=""):
    d = repo / "benchmark_history" / subdir if subdir else repo / "benchmark_history"
    d.mkdir(parents=True, exist_ok=True)
    entry: dict = {"run_id": name, "positions": list(positions)}
    if fingerprints is not None:
        entry["code_fingerprints"] = fingerprints
    (d / f"{name}.json").write_text(json.dumps(entry))


def _age(path: Path, seconds: float = 3600.0) -> None:
    old = time.time() - seconds
    os.utime(path, (old, old))


def _evaluate(files: list[str], capsys) -> tuple[str, str]:
    rc = cmd_evaluate(files)
    assert rc == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    return lines[0], "\n".join(lines[1:])


# --------------------------------------------------------------------------
# evaluate — verdicts and tiers
# --------------------------------------------------------------------------


def test_nothing_scoped_passes(repo, capsys):
    verdict, _ = _evaluate(["src/serving/app.py", "docs/foo.md"], capsys)
    assert verdict == "PASS"


def test_scoped_position_with_no_evidence_fails_with_scoped_command(repo, capsys):
    verdict, detail = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "FAIL"
    assert "TE" in detail
    assert "python -m src.benchmarking.benchmark TE --no-sync" in detail


def test_fingerprint_match_passes(repo, capsys):
    fp = position_fingerprint("TE", source="head")
    _write_entry(repo, "run1", ["TE"], {"TE": fp})
    verdict, detail = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "PASS"
    assert "accepted via" not in detail  # primary tier needs no nudge


def test_fingerprint_mismatch_fails_even_with_fresh_mtime(repo, capsys):
    """A fingerprinted entry for the position retires the legacy mtime tier:
    stale-content evidence with a fresh file mtime must FAIL."""
    _write_entry(repo, "run1", ["TE"], {"TE": "0" * 64})  # wrong code
    verdict, _ = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "FAIL"


def test_fingerprint_entry_must_list_the_position(repo, capsys):
    """code_fingerprints[P] without P in positions proves nothing — the run
    never trained P."""
    fp = position_fingerprint("TE", source="head")
    _write_entry(repo, "run1", ["QB"], {"TE": fp, "QB": "1" * 64})
    verdict, _ = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "FAIL"


def test_legacy_entry_accepts_while_no_fingerprint_era(repo, capsys):
    _write_entry(repo, "legacy", ["TE"])  # no code_fingerprints key
    # the entry file is newer than the changed source file (both just written,
    # so age the source to be safe)
    _age(repo / "src/te/features.py")
    verdict, detail = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "PASS"
    assert "legacy-mtime" in detail  # nudge toward fingerprinted evidence


def test_legacy_entry_stale_mtime_fails(repo, capsys):
    _write_entry(repo, "legacy", ["TE"])
    _age(repo / "benchmark_history" / "legacy.json")  # older than the source edit
    verdict, _ = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "FAIL"


def test_outputs_models_mtime_accepts_with_nudge(repo, capsys):
    d = repo / "te/outputs/models"
    d.mkdir(parents=True)
    (d / "model.pt").write_text("weights")  # a real fresh ARTIFACT is the evidence
    _age(repo / "src/te/features.py")
    verdict, detail = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "PASS"
    assert "outputs-mtime" in detail


def test_outputs_models_empty_dir_is_not_evidence(repo, capsys):
    """A freshly-mkdir'd EMPTY artifact dir (e.g. a run that crashed before
    saving) must not count — evidence comes from artifact FILES."""
    (repo / "te/outputs/models").mkdir(parents=True)
    _age(repo / "src/te/features.py")
    verdict, _ = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "FAIL"


def test_outputs_models_stale_fails(repo, capsys):
    d = repo / "te/outputs/models"
    d.mkdir(parents=True)
    _age(d)
    verdict, _ = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "FAIL"


def test_shared_arm_needs_one_position(repo, capsys):
    verdict, detail = _evaluate(["src/shared/pipeline.py"], capsys)
    assert verdict == "FAIL"
    assert "any one position suffices" in detail
    fp = position_fingerprint("QB", source="head")
    _write_entry(repo, "run1", ["QB"], {"QB": fp})
    verdict, _ = _evaluate(["src/shared/pipeline.py"], capsys)
    assert verdict == "PASS"


def test_exempt_paths_pass_with_note(repo, capsys):
    verdict, detail = _evaluate(["src/batch/train.py", "requirements.txt"], capsys)
    assert verdict == "PASS"
    assert "exempt" in detail
    assert "src/batch/train.py" in detail
    assert "requirements.txt" in detail


def test_tuning_and_ablation_subdirs_are_ignored(repo, capsys):
    fp = position_fingerprint("TE", source="head")
    _write_entry(repo, "tuned", ["TE"], {"TE": fp}, subdir="tuning")
    _write_entry(repo, "ablated", ["TE"], {"TE": fp}, subdir="ablations")
    verdict, _ = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "FAIL"


def test_corrupt_history_entry_is_skipped(repo, capsys):
    (repo / "benchmark_history" / "corrupt.json").write_text("{not json")
    fp = position_fingerprint("TE", source="head")
    _write_entry(repo, "good", ["TE"], {"TE": fp})
    verdict, _ = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "PASS"


def test_dirty_gated_path_warns_without_blocking(repo, capsys):
    fp = position_fingerprint("TE", source="head")
    _write_entry(repo, "run1", ["TE"], {"TE": fp})
    (repo / "src/qb/config.py").write_text("ALPHA = 42.0\n")  # uncommitted
    verdict, detail = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "PASS"
    assert "uncommitted changes" in detail


# --------------------------------------------------------------------------
# inert — AST equivalence
# --------------------------------------------------------------------------


def test_comment_docstring_and_format_changes_are_inert():
    base = 'def f(x):\n    """Doc."""\n    return x + 1\n'
    assert is_inert(base, 'def f(x):\n    # a comment\n    """New doc."""\n    return x + 1\n')
    assert is_inert(base, "def f(x):\n    return (\n        x + 1\n    )\n")
    assert not is_inert(base, "def f(x):\n    return x + 2\n")
    assert not is_inert(
        base,
        "def f(x)\n    return x",
    )  # syntax error side
    # string LITERAL changes (non-docstring) are behavior — never inert
    assert not is_inert('MSG = "a"\n', 'MSG = "b"\n')


def test_cmd_inert_prints_only_inert_committed_files(repo, capsys):
    base = _git(repo, "rev-parse", "HEAD")
    # inert edit: comment added
    (repo / "src/te/features.py").write_text("# comment\ndef f():\n    return 0\n")
    # behavioral edit
    (repo / "src/qb/config.py").write_text("ALPHA = 9.0\n")
    # new file (absent at base -> not inert)
    (repo / "src/data/newmod.py").write_text("Z = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "edits")

    rc = cmd_inert(base, ["src/te/features.py", "src/qb/config.py", "src/data/newmod.py"])
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["src/te/features.py"]


def test_deletion_only_change_requires_fingerprint_evidence(repo, capsys):
    """A deletion-only change has no on-disk mtime anchor — the mtime tiers
    must NOT accept (a 0.0 anchor would let arbitrarily stale evidence pass);
    only the fingerprint tier (which sees the deletion exactly) can."""
    _git(repo, "rm", "-q", "src/te/features.py")
    _git(repo, "commit", "-m", "delete te features")
    _write_entry(repo, "legacy", ["TE"])  # fresh-mtime legacy entry
    (repo / "te/outputs/models").mkdir(parents=True)  # fresh outputs dir too
    verdict, _ = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "FAIL"
    # A fingerprinted entry for the post-deletion HEAD is the valid evidence.
    fp = position_fingerprint("TE", source="head")
    _write_entry(repo, "fp", ["TE"], {"TE": fp})
    verdict, _ = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "PASS"


def test_outputs_models_accepts_in_place_overwrite(repo, capsys):
    """run_pipeline overwrites fixed artifact names in place, which never bumps
    the DIRECTORY mtime — the tier must scan file mtimes, or a warm box wedges
    into always-FAIL with a fix message that can't work."""
    d = repo / "te/outputs/models"
    d.mkdir(parents=True)
    artifact = d / "model.pt"
    artifact.write_text("weights-v1")
    _age(d)  # dir mtime stale
    _age(artifact)  # artifact stale too
    _age(repo / "src/te/features.py", seconds=1800)  # edit between the two
    verdict, _ = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "FAIL"
    artifact.write_text("weights-v2")  # in-place overwrite: dir mtime unchanged
    verdict, detail = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "PASS"
    assert "outputs-mtime" in detail


def test_worktree_mode_evidence_matches_head_when_clean(repo, capsys):
    """Production writers record WORKTREE-mode fingerprints; the gate matches
    HEAD-mode. On a clean committed tree they must be identical — the real
    recovery loop is edit -> commit -> benchmark -> gh pr create."""
    wt_fp = position_fingerprint("TE", source="worktree")
    assert wt_fp == position_fingerprint("TE", source="head")
    _write_entry(repo, "run1", ["TE"], {"TE": wt_fp})
    verdict, _ = _evaluate(["src/te/features.py"], capsys)
    assert verdict == "PASS"
