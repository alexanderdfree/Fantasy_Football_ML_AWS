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
    # Temp repos must never sign commits — a global commit.gpgsign / signing
    # server makes ``git commit`` fail (exit 128) in some CI/sandbox envs.
    subprocess.run(["git", "-C", str(main), "config", "commit.gpgsign", "false"], check=True)
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


def _matcher_result(command: str) -> bool:
    script = f'. "{PROJECT_ROOT / ".codex/hooks/lib.sh"}"; codex_command_invokes_gh_pr_create "$1"'
    result = subprocess.run(
        [_bash(), "-c", script, "codex-hook-test", command],
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
    ],
)
def test_pr_create_matcher_accepts_real_top_level_invocations(command: str):
    assert _matcher_result(command)


@pytest.mark.parametrize(
    "command",
    [
        "echo gh pr create",
        'rg -n "post-pr|gh pr create|codex review" .codex',
        "rg -n 'gh pr create' .codex",
        "# gh pr create\n git status --short",
        "git status --short",
        "bash -lc 'gh pr create --fill'",
    ],
)
def test_pr_create_matcher_rejects_quoted_or_argument_text(command: str):
    assert not _matcher_result(command)


def test_codex_review_quiet_filters_known_loader_noise(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_codex = fake_bin / "codex"
    fake_codex.write_text(
        """#!/bin/sh
printf '%s\\n' 'review finding'
printf '%s\\n' "2026-06-01T07:56:46Z  WARN codex_core_skills::loader: ignoring interface.icon_large: icon path with '..' must resolve under plugin assets/" >&2
printf '%s\\n' "2026-06-01T07:56:46Z  WARN codex_core_skills::loader: ignoring interface.icon_small: icon path with '..' must resolve under plugin assets/" >&2
printf '%s\\n' "2026-06-01T07:56:46Z ERROR codex_core::session::session: failed to load skill /tmp/SKILL.md: invalid name: exceeds maximum length of 64 characters" >&2
printf '%s\\n' 'real stderr problem' >&2
exit 7
"""
    )
    fake_codex.chmod(0o755)

    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}
    result = subprocess.run(
        [str(PROJECT_ROOT / "scripts/codex-review-quiet.sh"), "--base", "origin/main"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 7
    assert result.stdout == "review finding\n"
    assert "real stderr problem" in result.stderr
    assert "codex_core_skills::loader" not in result.stderr
    assert "failed to load skill" not in result.stderr


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
        for command in (
            "git status --short",
            "echo gh pr create",
            'rg -n "post-pr|gh pr create|codex review" .codex',
            "# gh pr create\n git status --short",
            "bash -lc 'gh pr create --fill'",
        ):
            result = _run_hook(
                ".codex/hooks/pre-pr.sh",
                {"cwd": str(PROJECT_ROOT), "tool_input": {"command": command}},
                PROJECT_ROOT,
            )

            assert result.returncode == 0
            assert result.stdout == ""
            assert result.stderr == ""

    @pytest.mark.parametrize(
        "command",
        [
            "/opt/homebrew/bin/gh pr create --fill",
            "env GH_TOKEN=example gh pr create --fill",
        ],
    )
    def test_pre_pr_hook_normalizes_delegated_pr_create_command(
        self, git_worktree_pair: tuple[Path, Path], tmp_path: Path, command: str
    ):
        _, worktree = git_worktree_pair
        marker = tmp_path / "delegated-input.json"
        fake_hook = worktree / ".claude/hooks/pre-pr.sh"
        fake_hook.parent.mkdir(parents=True)
        fake_hook.write_text(f"#!/bin/sh\ncat > {marker}\nexit 43\n")
        fake_hook.chmod(0o755)

        result = _run_hook(
            ".codex/hooks/pre-pr.sh",
            {"cwd": str(worktree), "tool_input": {"command": command}},
            worktree,
        )

        assert result.returncode == 43
        delegated_payload = json.loads(marker.read_text())
        assert delegated_payload["tool_input"]["command"] == "gh pr create"

    def test_post_pr_hook_ignores_non_pr_create_commands(self):
        for command in (
            "git status --short",
            "echo gh pr create",
            'rg -n "post-pr|gh pr create|codex review" .codex',
            "# gh pr create\n git status --short",
            "bash -lc 'gh pr create --fill'",
        ):
            result = _run_hook(
                ".codex/hooks/post-pr-create.sh",
                {"cwd": str(PROJECT_ROOT), "tool_input": {"command": command}},
                PROJECT_ROOT,
            )

            assert result.returncode == 0
            assert result.stdout == ""
            assert result.stderr == ""

    @pytest.mark.parametrize(
        "command",
        [
            "gh pr create --fill",
            "env GH_TOKEN=example gh pr create --fill",
        ],
    )
    def test_post_pr_hook_injects_compact_codex_review_workflow(self, command: str):
        result = _run_hook(
            ".codex/hooks/post-pr-create.sh",
            {"cwd": str(PROJECT_ROOT), "tool_input": {"command": command}},
            PROJECT_ROOT,
        )

        assert result.returncode == 0
        context = json.loads(result.stdout)["hookSpecificOutput"]
        assert context["hookEventName"] == "PostToolUse"
        additional_context = context["additionalContext"]
        assert "post-pr-followup" in additional_context
        assert "scripts/codex-review-quiet.sh --base origin/main" in additional_context
        assert "audit/tier explicit merge sign-off" in additional_context
        assert "post-session-critique" in additional_context
        assert "Run this Codex post-create workflow now, in order" not in additional_context
        assert "1. Rebase onto latest main" not in additional_context
