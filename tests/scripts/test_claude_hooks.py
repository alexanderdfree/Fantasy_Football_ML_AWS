"""Tests for the Claude Code hook shared library (`.claude/hooks/lib.sh`).

Exercises `claude_command_invokes_gh_pr_create` — the shell-parser-aware matcher
that both `pre-pr.sh` (PreToolUse gate) and `post-pr-create.sh` (PostToolUse
workflow injector) use to decide whether a Bash command actually invokes
`gh pr create`. The previous flat `=~` regex also matched the literal token
sequence inside quoted strings / heredocs / comments / other commands' args, so
a log grep or echoed release notes would wedge the gate / fire the workflow with
no PR opened (audit #893 / #894). The parity twin is tested for Codex in
``test_codex_hooks.py``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIB = PROJECT_ROOT / ".claude/hooks/lib.sh"
pytestmark = pytest.mark.unit


def _bash() -> str:
    return shutil.which("bash") or "/bin/bash"


def _matcher_result(command: str) -> bool:
    script = f'. "{LIB}"; claude_command_invokes_gh_pr_create "$1"'
    result = subprocess.run(
        [_bash(), "-c", script, "claude-hook-test", command],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "command",
    [
        "gh pr create --fill",
        "GH_TOKEN=example gh pr create --fill",
        "env GH_TOKEN=example gh pr create --fill",
        "git status --short && gh pr create --fill",
        "/opt/homebrew/bin/gh pr create --fill",
        "git add -A; gh pr create --draft",
    ],
)
def test_pr_create_matcher_accepts_real_top_level_invocations(command: str):
    assert _matcher_result(command)


@pytest.mark.parametrize(
    "command",
    [
        "echo gh pr create",
        'grep -rn "gh pr create" .claude',
        "grep -rn 'gh pr create' .claude",
        'echo "run gh pr create to open a PR"',
        "# gh pr create\n git status --short",
        "git status --short",
        "bash -lc 'gh pr create --fill'",
        "gh pr list  # was: gh pr create",
    ],
)
def test_pr_create_matcher_rejects_quoted_or_argument_text(command: str):
    assert not _matcher_result(command)


def test_both_hooks_source_the_shared_lib():
    """pre-pr.sh and post-pr-create.sh must both wire the parser, not re-inline
    the flawed flat regex."""
    for name in ("pre-pr.sh", "post-pr-create.sh"):
        text = (PROJECT_ROOT / ".claude/hooks" / name).read_text()
        assert "lib.sh" in text, f"{name} does not source lib.sh"
        assert "claude_command_invokes_gh_pr_create" in text, f"{name} does not call the matcher"
        assert "=~ (^|[[:space:]" not in text, f"{name} still has the flat regex"
