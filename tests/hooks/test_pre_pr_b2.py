"""Characterization tests for .claude/hooks/pre-pr.sh's B2 benchmark gate.

The hook is run as a real subprocess against a synthetic fixture repo:
the REAL hook scripts and the REAL src/scripts gate modules are copied in
(so the bash↔Python protocol is exercised end-to-end), while ``ruff`` and
``pytest`` are stubbed with exit-0 fakes on PATH so B1 is instant and the
verdicts under test are B2's. Prior to this suite the entire B2 body was
untested (the #894 wedge-class lesson: a hook that can block `gh pr create`
must be pinned by regression tests, not manual review).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def _jq_available() -> bool:
    for candidate in (
        "jq",
        "/usr/bin/jq",
        "/usr/local/bin/jq",
        "/opt/homebrew/bin/jq",
        "/home/linuxbrew/.linuxbrew/bin/jq",
    ):
        if shutil.which(candidate):
            return True
    return False


requires_jq = pytest.mark.skipif(not _jq_available(), reason="jq not installed")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def gate_repo(tmp_path: Path) -> Path:
    """Fixture repo with the real hook + real gate modules + stub B1 tools."""
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(r)], check=True, capture_output=True)
    _git(r, "config", "user.email", "hook@example.test")
    _git(r, "config", "user.name", "Hook Test")
    _git(r, "config", "commit.gpgsign", "false")

    # Real hook machinery.
    (r / ".claude/hooks").mkdir(parents=True)
    for rel in (".claude/hooks/pre-pr.sh", ".claude/hooks/lib.sh"):
        shutil.copy(PROJECT_ROOT / rel, r / rel)
    (r / "scripts").mkdir()
    shutil.copy(PROJECT_ROOT / "scripts/agent-hooks-lib.sh", r / "scripts/agent-hooks-lib.sh")

    # Real gate modules (bash↔Python protocol under test, not a fake).
    (r / "src/scripts").mkdir(parents=True)
    for rel in (
        "src/__init__.py",
        "src/scripts/__init__.py",
        "src/scripts/scope_positions.py",
        "src/scripts/bench_fingerprint.py",
        "src/scripts/pre_pr_bench_check.py",
    ):
        shutil.copy(PROJECT_ROOT / rel, r / rel)

    # Minimal position/pipeline tree.
    for rel, content in {
        "src/config.py": "SEASONS = [2024]\n",
        "src/shared/pipeline.py": "def run():\n    return 1\n",
        "src/data/loader.py": "def load():\n    return []\n",
        "src/features/engineer.py": "def build():\n    return {}\n",
        "src/te/features.py": "def f():\n    return 0\n",
        "src/qb/config.py": "ALPHA = 1.0\n",
    }.items():
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    (r / "benchmark_history").mkdir()

    # Stub B1 tools: instant exit-0 ruff/pytest so B2 is the verdict under test.
    stub = r / "stubbin"
    stub.mkdir()
    for tool in ("ruff", "pytest"):
        t = stub / tool
        t.write_text("#!/bin/sh\nexit 0\n")
        t.chmod(0o755)

    _git(r, "add", "-A")
    _git(r, "commit", "-m", "init")
    return r


def _run_hook(repo: Path, command: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(repo)
    env["PATH"] = f"{repo / 'stubbin'}{os.pathsep}{env['PATH']}"
    for key in ("GIT_DIR", "GIT_WORK_TREE", "VIRTUAL_ENV"):
        env.pop(key, None)
    payload = json.dumps({"tool_input": {"command": command}})
    return subprocess.run(
        ["bash", str(repo / ".claude/hooks/pre-pr.sh")],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
        cwd=repo,
        timeout=120,
    )


def _commit_on_feature(repo: Path, rel: str, content: str, message: str = "edit") -> None:
    if _git(repo, "branch", "--show-current") != "feature":
        _git(repo, "checkout", "-b", "feature")
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)


def _te_head_fingerprint(repo: Path) -> str:
    out = subprocess.run(
        ["python3", "-m", "src.scripts.bench_fingerprint", "TE", "--head"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo,
    ).stdout
    return out.split()[-1]


@requires_jq
def test_non_pr_create_commands_pass_through(gate_repo):
    for cmd in ("git status", "gh pr list", "echo 'gh pr create'"):
        res = _run_hook(gate_repo, cmd)
        assert res.returncode == 0, (cmd, res.stderr)


@requires_jq
def test_docs_only_subject_bypasses_gates(gate_repo):
    _commit_on_feature(
        gate_repo, "src/te/features.py", "def f():\n    return 99\n", "x [docs-only]"
    )
    res = _run_hook(gate_repo, "gh pr create --title t")
    assert res.returncode == 0
    assert "[docs-only]" in res.stderr


@requires_jq
def test_unbenchmarked_position_change_blocks_naming_position(gate_repo):
    _commit_on_feature(gate_repo, "src/te/features.py", "nn_lr = 0.01\ndef f():\n    return 0\n")
    res = _run_hook(gate_repo, "gh pr create --title t")
    assert res.returncode == 2
    assert "TE" in res.stderr
    assert "benchmark TE --no-sync" in res.stderr


@requires_jq
def test_fingerprint_evidence_unblocks(gate_repo):
    _commit_on_feature(gate_repo, "src/te/features.py", "nn_lr = 0.01\ndef f():\n    return 0\n")
    fp = _te_head_fingerprint(gate_repo)
    (gate_repo / "benchmark_history/run1.json").write_text(
        json.dumps({"positions": ["TE"], "code_fingerprints": {"TE": fp}})
    )
    res = _run_hook(gate_repo, "gh pr create --title t")
    assert res.returncode == 0, res.stderr


@requires_jq
def test_inert_comment_only_change_passes(gate_repo):
    _commit_on_feature(gate_repo, "src/te/features.py", "# note\ndef f():\n    return 0\n")
    res = _run_hook(gate_repo, "gh pr create --title t")
    assert res.returncode == 0, res.stderr
    assert "AST-inert" in res.stderr


@requires_jq
def test_additive_safe_change_passes(gate_repo):
    _commit_on_feature(
        gate_repo,
        "src/te/features.py",
        "def f():\n    return 0\n\n\ndef helper():\n    return 5\n",
    )
    res = _run_hook(gate_repo, "gh pr create --title t")
    assert res.returncode == 0, res.stderr
    assert "additive-only" in res.stderr


@requires_jq
def test_shared_change_accepts_any_one_position_evidence(gate_repo):
    _commit_on_feature(
        gate_repo, "src/shared/pipeline.py", 'optimizer = "sgd"\ndef run():\n    return 1\n'
    )
    res = _run_hook(gate_repo, "gh pr create --title t")
    assert res.returncode == 2
    assert "shared pipeline files changed" in res.stderr

    out = subprocess.run(
        ["python3", "-m", "src.scripts.bench_fingerprint", "QB", "--head"],
        check=True,
        capture_output=True,
        text=True,
        cwd=gate_repo,
    ).stdout
    qb_fp = out.split()[-1]
    (gate_repo / "benchmark_history/run1.json").write_text(
        json.dumps({"positions": ["QB"], "code_fingerprints": {"QB": qb_fp}})
    )
    res = _run_hook(gate_repo, "gh pr create --title t")
    assert res.returncode == 0, res.stderr


@requires_jq
def test_evaluator_crash_fails_open_with_warning(gate_repo):
    _commit_on_feature(gate_repo, "src/te/features.py", "nn_lr = 0.01\ndef f():\n    return 0\n")
    # Break the evaluator module: a real crash must warn + fail open, never block.
    (gate_repo / "src/scripts/pre_pr_bench_check.py").write_text("raise RuntimeError('boom')\n")
    res = _run_hook(gate_repo, "gh pr create --title t")
    assert res.returncode == 0
    assert "failing open" in res.stderr
