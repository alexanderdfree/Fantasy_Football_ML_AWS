from __future__ import annotations

import json
import os
import re
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
        ["git", "-C", str(main), "config", "user.email", "codex-hooks@example.test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(main), "config", "user.name", "Codex Hooks"], check=True)
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


def _run_fresh_worktree(
    args: list[str],
    cwd: Path,
    codex_home: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    return subprocess.run(
        [str(PROJECT_ROOT / "scripts/codex-fresh-worktree.sh"), *args],
        text=True,
        capture_output=True,
        cwd=cwd,
        env=env,
        check=False,
    )


@pytest.fixture
def launcher_repo(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "Final-Project"
    remote = tmp_path / "remote.git"

    subprocess.run(["git", "init", "-b", "main", str(main)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(main), "config", "user.email", "codex-hooks@example.test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(main), "config", "user.name", "Codex Hooks"], check=True)
    (main / ".gitignore").write_text("data/raw/\ndata/splits/\n.venv/\n")
    (main / "README.md").write_text("test repo\n")
    subprocess.run(["git", "-C", str(main), "add", ".gitignore", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(main), "commit", "-m", "init"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(main), "remote", "add", "origin", str(remote)], check=True)
    subprocess.run(
        ["git", "-C", str(main), "push", "-u", "origin", "main"],
        check=True,
        capture_output=True,
    )
    return main, remote


def _add_codex_worktree(main: Path, codex_home: Path, short_id: str, branch: str) -> Path:
    worktree = codex_home / "worktrees" / short_id / "Final-Project"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-b", branch, str(worktree), "main"],
        check=True,
        capture_output=True,
    )
    return worktree


def _current_branch(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "branch", "--show-current"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def test_codex_json_context_uses_resolved_jq_path(tmp_path: Path):
    fake_jq = tmp_path / "jq-not-on-path"
    fake_jq.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' "
        '\'{"hookSpecificOutput":{"hookEventName":"SessionStart",'
        '"additionalContext":"ok"}}\'\n'
    )
    fake_jq.chmod(0o755)

    script = (
        f'. "{PROJECT_ROOT / ".codex/hooks/lib.sh"}"; '
        f'codex_json_context "SessionStart" "ok" "{fake_jq}"'
    )
    result = subprocess.run(
        [_bash(), "-c", script],
        text=True,
        capture_output=True,
        env={**os.environ, "PATH": ""},
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "ok",
        }
    }


@pytest.mark.skipif(not _jq_available(), reason="Codex hooks need jq to parse hook JSON")
class TestCodexHooks:
    def test_guard_blocks_parent_checkout_file_path(self, git_worktree_pair: tuple[Path, Path]):
        main, worktree = git_worktree_pair
        parent_path = main / "src/qb/config.py"
        corrected_path = worktree / "src/qb/config.py"

        result = _run_hook(
            ".codex/hooks/guard-worktree-path.sh",
            {"cwd": str(worktree), "tool_input": {"file_path": str(parent_path)}},
            worktree,
        )

        assert result.returncode == 2
        assert "main checkout" in result.stderr
        assert str(parent_path) in result.stderr
        assert str(corrected_path) in result.stderr

    def test_guard_blocks_parent_checkout_apply_patch_header(
        self, git_worktree_pair: tuple[Path, Path]
    ):
        main, worktree = git_worktree_pair
        parent_path = main / "src/qb/config.py"
        patch = f"*** Begin Patch\n*** Update File: {parent_path}\n@@\n test\n*** End Patch\n"

        result = _run_hook(
            ".codex/hooks/guard-worktree-path.sh",
            {"cwd": str(worktree), "tool_input": {"command": patch}},
            worktree,
        )

        assert result.returncode == 2
        assert str(parent_path) in result.stderr

    def test_guard_allows_worktree_file_path(self, git_worktree_pair: tuple[Path, Path]):
        _, worktree = git_worktree_pair

        result = _run_hook(
            ".codex/hooks/guard-worktree-path.sh",
            {"cwd": str(worktree), "tool_input": {"file_path": str(worktree / "src/qb/config.py")}},
            worktree,
        )

        assert result.returncode == 0
        assert result.stderr == ""

    def test_session_start_emits_codex_context(self, git_worktree_pair: tuple[Path, Path]):
        _, worktree = git_worktree_pair

        result = _run_hook(".codex/hooks/session-start.sh", {"cwd": str(worktree)}, worktree)

        assert result.returncode == 0
        context = json.loads(result.stdout)["hookSpecificOutput"]
        assert context["hookEventName"] == "SessionStart"
        assert "read AGENTS.md" in context["additionalContext"]

    def test_session_start_mentions_launcher_for_non_reusable_cwd(
        self, git_worktree_pair: tuple[Path, Path], tmp_path: Path
    ):
        main, _ = git_worktree_pair

        result = _run_hook(
            ".codex/hooks/session-start.sh",
            {"cwd": str(main)},
            main,
            {"CODEX_HOME": str(tmp_path / "codex-home")},
        )

        assert result.returncode == 0
        context = json.loads(result.stdout)["hookSpecificOutput"]
        assert "scripts/codex-fresh-worktree.sh" in context["additionalContext"]

    def test_session_start_omits_launcher_for_clean_codex_worktree(
        self, launcher_repo: tuple[Path, Path], tmp_path: Path
    ):
        main, _ = launcher_repo
        codex_home = tmp_path / "codex-home"
        worktree = _add_codex_worktree(main, codex_home, "abcd", "codex/existing")

        result = _run_hook(
            ".codex/hooks/session-start.sh",
            {"cwd": str(worktree)},
            worktree,
            {"CODEX_HOME": str(codex_home)},
        )

        assert result.returncode == 0
        context = json.loads(result.stdout)["hookSpecificOutput"]
        assert "scripts/codex-fresh-worktree.sh" not in context["additionalContext"]

    def test_fresh_worktree_reuses_clean_codex_worktree(
        self, launcher_repo: tuple[Path, Path], tmp_path: Path
    ):
        main, _ = launcher_repo
        codex_home = tmp_path / "codex-home"
        worktree = _add_codex_worktree(main, codex_home, "abcd", "codex/existing")

        result = _run_fresh_worktree(["--print-path"], worktree, codex_home)

        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()) == worktree

    def test_fresh_worktree_creates_from_main_and_links_data(
        self, launcher_repo: tuple[Path, Path], tmp_path: Path
    ):
        main, _ = launcher_repo
        codex_home = tmp_path / "codex-home"
        (main / "data/raw").mkdir(parents=True)
        (main / "data/splits").mkdir(parents=True)

        result = _run_fresh_worktree(["--print-path"], main, codex_home)

        assert result.returncode == 0, result.stderr
        target = Path(result.stdout.strip())
        assert target.parent.parent == codex_home / "worktrees"
        assert target.name == "Final-Project"
        assert re.fullmatch(r"codex/session-[0-9a-f]{4}", _current_branch(target))
        assert (target / "data/raw").is_symlink()
        assert (target / "data/raw").resolve() == main / "data/raw"
        assert (target / "data/splits").is_symlink()
        assert (target / "data/splits").resolve() == main / "data/splits"
        assert not (target / ".venv").exists()

    def test_fresh_worktree_creates_when_codex_worktree_is_dirty(
        self, launcher_repo: tuple[Path, Path], tmp_path: Path
    ):
        main, _ = launcher_repo
        codex_home = tmp_path / "codex-home"
        dirty_worktree = _add_codex_worktree(main, codex_home, "abcd", "codex/existing")
        (dirty_worktree / "README.md").write_text("dirty\n")

        result = _run_fresh_worktree(["--print-path"], dirty_worktree, codex_home)

        assert result.returncode == 0, result.stderr
        target = Path(result.stdout.strip())
        assert target != dirty_worktree
        assert target.parent.parent == codex_home / "worktrees"
        assert re.fullmatch(r"codex/session-[0-9a-f]{4}", _current_branch(target))

    def test_fresh_worktree_force_new_base_branch_no_fetch_print_path(
        self, launcher_repo: tuple[Path, Path], tmp_path: Path
    ):
        main, _ = launcher_repo
        codex_home = tmp_path / "codex-home"
        clean_worktree = _add_codex_worktree(main, codex_home, "abcd", "codex/existing")

        result = _run_fresh_worktree(
            [
                "--force-new",
                "--no-fetch",
                "--base",
                "main",
                "--branch",
                "codex/session-custom",
                "--print-path",
            ],
            clean_worktree,
            codex_home,
        )

        assert result.returncode == 0, result.stderr
        target = Path(result.stdout.strip())
        assert target == codex_home / "worktrees" / "session-custom" / "Final-Project"
        assert target != clean_worktree
        assert _current_branch(target) == "codex/session-custom"

    def test_pre_pr_hook_ignores_non_pr_create_commands(self):
        result = _run_hook(
            ".codex/hooks/pre-pr.sh",
            {"cwd": str(PROJECT_ROOT), "tool_input": {"command": "git status --short"}},
            PROJECT_ROOT,
        )

        assert result.returncode == 0
        assert result.stdout == ""
        assert result.stderr == ""

    def test_post_pr_hook_injects_codex_review_workflow(self):
        result = _run_hook(
            ".codex/hooks/post-pr-create.sh",
            {"cwd": str(PROJECT_ROOT), "tool_input": {"command": "gh pr create --fill"}},
            PROJECT_ROOT,
        )

        assert result.returncode == 0
        context = json.loads(result.stdout)["hookSpecificOutput"]
        assert context["hookEventName"] == "PostToolUse"
        assert "codex review --base origin/main" in context["additionalContext"]
        assert "Do not use `--delete-branch`" in context["additionalContext"]
