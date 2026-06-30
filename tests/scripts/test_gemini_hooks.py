"""Gemini/Antigravity (.gemini/) hook contract tests.

Parity twin of tests/scripts/test_codex_hooks.py (audit P2).
Antigravity hooks receive JSON on stdin
with `tool_name` + `tool_input` and block via exit code 2 (stderr = reason).
These exercise the deterministic guardrails purely through the stdin/exit-code
contract — no Antigravity install needed.
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


def _bash() -> str:
    return shutil.which("bash") or "/bin/bash"


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


@pytest.fixture
def git_worktree_pair(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "main"
    worktree = tmp_path / "feature"

    subprocess.run(["git", "init", "-b", "main", str(main)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(main), "config", "user.email", "gemini-hooks@example.test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(main), "config", "user.name", "Gemini Hooks"], check=True)
    # Temp repos must never sign commits — a global commit.gpgsign / signing
    # server makes ``git commit`` fail (exit 128) in some CI/sandbox envs.
    subprocess.run(["git", "-C", str(main), "config", "commit.gpgsign", "false"], check=True)
    (main / "README.md").write_text("test repo\n")
    subprocess.run(["git", "-C", str(main), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(main), "commit", "-m", "init"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-b", "feature", str(worktree)],
        check=True,
        capture_output=True,
    )
    return main, worktree


def _run_hook(
    script: str,
    payload: dict[str, object],
    cwd: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in (
        "GEMINI_PROJECT_DIR",
        "CODEX_PROJECT_DIR",
        "CLAUDE_PROJECT_DIR",
        "GIT_CEILING_DIRECTORIES",
        "GIT_DIR",
        "GIT_WORK_TREE",
    ):
        env.pop(key, None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(PROJECT_ROOT / script)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=cwd,
        env=env,
        check=False,
    )


def _call_gemini_lib(func_call: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Source .gemini/hooks/lib.sh and run one function call, passing args as bash
    positionals ($1=lib path, $2.. = args) so payloads are never re-quoted."""
    lib = PROJECT_ROOT / ".gemini/hooks/lib.sh"
    return subprocess.run(
        [_bash(), "-c", f'source "$1"; {func_call}', "_", str(lib), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def _matcher_result(command: str) -> bool:
    result = _call_gemini_lib('gemini_command_invokes_gh_pr_create "$2"', command)
    return result.returncode == 0


# --- gh pr create matcher (parity with the Codex tokenizer) -------------------


@pytest.mark.parametrize(
    "command",
    [
        "gh pr create --fill",
        "GH_TOKEN=example gh pr create --fill",
        "env GH_TOKEN=example gh pr create --fill",
        "git status --short && gh pr create --fill",
        "/opt/homebrew/bin/gh pr create --fill",
    ],
)
def test_pr_create_matcher_accepts_real_top_level_invocations(command: str):
    assert _matcher_result(command)


@pytest.mark.parametrize(
    "command",
    [
        "echo gh pr create",
        'rg -n "post-pr|gh pr create" .gemini',
        "rg -n 'gh pr create' .gemini",
        "# gh pr create\n git status --short",
        "git status --short",
        "bash -lc 'gh pr create --fill'",
    ],
)
def test_pr_create_matcher_rejects_quoted_or_argument_text(command: str):
    assert not _matcher_result(command)


# --- no-jq python3 fallbacks (guard stays armed without jq, #1232 parity) -----


@pytest.mark.skipif(shutil.which("python3") is None, reason="no-jq fallback needs python3")
def test_gemini_tool_paths_file_path_python3_fallback_without_jq():
    # Test TargetFile (Antigravity format)
    payload_tg = json.dumps(
        {"tool_input": {"TargetFile": "/repo/src/x.py", "CodeContent": '{"TargetFile": "/evil"}'}}
    )
    result = _call_gemini_lib('gemini_tool_paths "$2" ""', payload_tg)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/repo/src/x.py"

    # Test file_path (legacy/Claude format)
    payload_fp = json.dumps(
        {"tool_input": {"file_path": "/repo/src/x.py", "content": '{"file_path": "/evil"}'}}
    )
    result = _call_gemini_lib('gemini_tool_paths "$2" ""', payload_fp)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/repo/src/x.py"


@pytest.mark.skipif(shutil.which("python3") is None, reason="no-jq fallback needs python3")
def test_gemini_tool_command_python3_fallback_without_jq():
    # Test CommandLine (Antigravity format)
    payload_cl = json.dumps({"tool_input": {"CommandLine": "gh pr create --fill"}})
    result = _call_gemini_lib('gemini_tool_command "$2" ""', payload_cl)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "gh pr create --fill"

    # Test command (legacy/Codex format)
    payload_cmd = json.dumps({"tool_input": {"command": "gh pr create --fill"}})
    result = _call_gemini_lib('gemini_tool_command "$2" ""', payload_cmd)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "gh pr create --fill"


# --- full hook scripts (need jq to parse hook JSON like the running agent) -----


@pytest.mark.skipif(not _jq_available(), reason="Gemini hooks need jq to parse hook JSON")
class TestGeminiHooks:
    def test_guard_blocks_parent_checkout_file_path(self, git_worktree_pair: tuple[Path, Path]):
        main, worktree = git_worktree_pair
        parent_path = main / "src/qb/config.py"
        corrected_path = worktree / "src/qb/config.py"

        # Test Antigravity TargetFile tools and the legacy file_path tool.
        for tool_name, param_name in [
            ("write_to_file", "TargetFile"),
            ("edit_file", "TargetFile"),
            ("write_file", "file_path"),
        ]:
            result = _run_hook(
                ".gemini/hooks/guard-worktree-path.sh",
                {
                    "cwd": str(worktree),
                    "tool_name": tool_name,
                    "tool_input": {param_name: str(parent_path)},
                },
                worktree,
            )

            assert result.returncode == 2
            assert "main checkout" in result.stderr
            assert str(parent_path) in result.stderr
            assert str(corrected_path) in result.stderr

    def test_guard_allows_worktree_file_path(self, git_worktree_pair: tuple[Path, Path]):
        _, worktree = git_worktree_pair

        # Test both replace_file_content (Antigravity format) and replace (legacy format)
        for tool_name, param_name in [
            ("replace_file_content", "TargetFile"),
            ("replace", "file_path"),
        ]:
            result = _run_hook(
                ".gemini/hooks/guard-worktree-path.sh",
                {
                    "cwd": str(worktree),
                    "tool_name": tool_name,
                    "tool_input": {param_name: str(worktree / "src/qb/config.py")},
                },
                worktree,
            )

            assert result.returncode == 0
            assert result.stderr == ""

    def test_pre_pr_ignores_non_gh_pr_create_command(self, git_worktree_pair: tuple[Path, Path]):
        _, worktree = git_worktree_pair
        # Test run_command (Antigravity format)
        result = _run_hook(
            ".gemini/hooks/pre-pr.sh",
            {
                "cwd": str(worktree),
                "tool_name": "run_command",
                "tool_input": {"CommandLine": "git status --short"},
            },
            worktree,
        )
        assert result.returncode == 0

        # Test run_shell_command (legacy format)
        result = _run_hook(
            ".gemini/hooks/pre-pr.sh",
            {
                "cwd": str(worktree),
                "tool_name": "run_shell_command",
                "tool_input": {"command": "git status --short"},
            },
            worktree,
        )
        assert result.returncode == 0

    def test_pre_pr_routes_gh_pr_create_to_the_gate(self, git_worktree_pair: tuple[Path, Path]):
        # The temp worktree has no .claude/hooks/pre-pr.sh, so a matched gh pr
        # create reaches the delegation branch and blocks (exit 2) with the
        # fallback message — proving the matcher fired and routing is wired.
        _, worktree = git_worktree_pair

        # Test run_command (Antigravity format)
        result = _run_hook(
            ".gemini/hooks/pre-pr.sh",
            {
                "cwd": str(worktree),
                "tool_name": "run_command",
                "tool_input": {"CommandLine": "gh pr create --fill"},
            },
            worktree,
        )
        assert result.returncode == 2
        assert ".claude/hooks/pre-pr.sh" in result.stderr

        # Test run_shell_command (legacy format)
        result = _run_hook(
            ".gemini/hooks/pre-pr.sh",
            {
                "cwd": str(worktree),
                "tool_name": "run_shell_command",
                "tool_input": {"command": "gh pr create --fill"},
            },
            worktree,
        )
        assert result.returncode == 2
        assert ".claude/hooks/pre-pr.sh" in result.stderr

    @pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff-format hook needs ruff")
    def test_ruff_format_formats_written_python(self, git_worktree_pair: tuple[Path, Path]):
        _, worktree = git_worktree_pair
        target = worktree / "messy.py"

        # Test Antigravity TargetFile tools.
        for tool_name in ["write_to_file", "edit_file"]:
            target.write_text("x=1\n")
            result = _run_hook(
                ".gemini/hooks/ruff-format.sh",
                {
                    "cwd": str(worktree),
                    "tool_name": tool_name,
                    "tool_input": {"TargetFile": str(target)},
                },
                worktree,
            )
            assert result.returncode == 0
            assert target.read_text() == "x = 1\n"

        # Test write_file (legacy format)
        target.write_text("x=1\n")
        result = _run_hook(
            ".gemini/hooks/ruff-format.sh",
            {
                "cwd": str(worktree),
                "tool_name": "write_file",
                "tool_input": {"file_path": str(target)},
            },
            worktree,
        )
        assert result.returncode == 0
        assert target.read_text() == "x = 1\n"
