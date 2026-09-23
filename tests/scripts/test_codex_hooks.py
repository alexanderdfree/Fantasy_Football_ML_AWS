from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
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
    if script in (".codex/hooks/post-pr-create.sh", ".codex/hooks/post-pr-merge.sh"):
        # Codex 0.153.4 ExecCommandToolOutput sends raw stdout, without status.
        payload = {"tool_response": "", **payload}
    env = os.environ.copy()
    for key in (
        "CODEX_PROJECT_DIR",
        "CLAUDE_PROJECT_DIR",
        "GIT_CEILING_DIRECTORIES",
        "GIT_DIR",
        "GIT_WORK_TREE",
    ):
        env.pop(key, None)
    if (cwd / ".test-gh/gh").exists():
        env["PATH"] = str(cwd / ".test-gh") + os.pathsep + env.get("PATH", "")
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


def _head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


@pytest.fixture
def merge_scenario_codex(launcher_repo: tuple[Path, Path], tmp_path: Path) -> tuple[Path, Path]:
    """launcher_repo (main + bare origin) + a feature worktree, with origin/main
    advanced one commit beyond the parent — the post-pr-merge fast-forward case."""
    main, remote = launcher_repo
    worktree = tmp_path / "feature"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-b", "feature", str(worktree), "main"],
        check=True,
        capture_output=True,
    )
    other = tmp_path / "other"
    # -b main: origin/main exists (launcher_repo pushed it), but the bare's HEAD
    # may be `master` in CI, which would otherwise leave `other` off `main`.
    subprocess.run(
        ["git", "clone", "-b", "main", str(remote), str(other)], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(other), "config", "user.email", "codex-hooks@example.test"], check=True
    )
    subprocess.run(["git", "-C", str(other), "config", "user.name", "Codex Hooks"], check=True)
    subprocess.run(["git", "-C", str(other), "config", "commit.gpgsign", "false"], check=True)
    (other / "NEW.md").write_text("more\n")
    subprocess.run(["git", "-C", str(other), "add", "NEW.md"], check=True)
    subprocess.run(
        ["git", "-C", str(other), "commit", "-m", "advance"], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(other), "push", "origin", "main"], check=True, capture_output=True
    )
    _stub_merged_pr(worktree, _head(other))
    return main, worktree


def _stub_merged_pr(worktree: Path, merge_commit: str, *, state: str = "MERGED") -> Path:
    tools_dir = worktree / ".test-gh"
    tools_dir.mkdir()
    gh = tools_dir / "gh"
    gh.write_text('#!/bin/sh\ncat "$(dirname "$0")/pr.json"\n')
    gh.chmod(0o755)
    metadata = tools_dir / "pr.json"
    metadata.write_text(
        json.dumps(
            {
                "state": state,
                "baseRefName": "main",
                "headRefOid": _head(worktree),
                "mergeCommit": {"oid": merge_commit},
            }
        )
    )
    return metadata


def _matcher_result(command: str, fn: str = "codex_command_invokes_gh_pr_create") -> bool:
    script = f'. "{PROJECT_ROOT / ".codex/hooks/lib.sh"}"; {fn} "$1"'
    result = subprocess.run(
        [_bash(), "-c", script, "codex-hook-test", command],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _merge_matcher_result(command: str) -> bool:
    return _matcher_result(command, "codex_command_invokes_gh_pr_merge")


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


@pytest.mark.parametrize(
    "command",
    [
        "gh pr merge 5 --squash",
        "gh pr merge 12 --squash --admin",
        "git status --short && gh pr merge 7 --squash",
        "env GH_TOKEN=example gh pr merge 7 --squash",
        "/opt/homebrew/bin/gh pr merge 7 --squash",
    ],
)
def test_pr_merge_matcher_accepts_real_top_level_invocations(command: str):
    assert _merge_matcher_result(command)


@pytest.mark.parametrize(
    "command",
    [
        "echo gh pr merge",
        'rg -n "gh pr merge" .codex',
        "# gh pr merge\n git status --short",
        "git status --short",
        "bash -lc 'gh pr merge 5 --squash'",
        "gh pr create --fill",
    ],
)
def test_pr_merge_matcher_rejects_quoted_or_argument_text(command: str):
    assert not _merge_matcher_result(command)


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


def _call_codex_lib(func_call: str, *args: str, env=None) -> subprocess.CompletedProcess[str]:
    """Source .codex/hooks/lib.sh and run one function call, passing args as bash
    positionals ($1=lib path, $2.. = args) so payloads are never re-quoted."""
    lib = PROJECT_ROOT / ".codex/hooks/lib.sh"
    return subprocess.run(
        [_bash(), "-c", f'source "$1"; {func_call}', "_", str(lib), *args],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


# The Codex worktree-guard must stay armed on a box without jq (parity with the
# Claude guard's python3 fallback, #1232). These exercise the no-jq branch of the
# extraction helpers directly (empty jq_bin -> python3), so they run even where jq
# IS installed — codex_find_jq probes absolute paths that a PATH strip can't hide.
@pytest.mark.skipif(shutil.which("python3") is None, reason="no-jq fallback needs python3")
def test_codex_tool_paths_file_path_python3_fallback_without_jq():
    payload = json.dumps(
        {"tool_input": {"file_path": "/repo/src/x.py", "new_string": '{"file_path": "/evil"}'}}
    )
    # "" = empty jq_bin -> python3 branch; the JSON parse also resists a new_string
    # that itself contains the text "file_path".
    result = _call_codex_lib('codex_tool_paths "$2" ""', payload)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/repo/src/x.py"


@pytest.mark.skipif(shutil.which("python3") is None, reason="no-jq fallback needs python3")
def test_codex_tool_paths_apply_patch_python3_fallback_without_jq():
    patch = "*** Begin Patch\n*** Update File: /repo/src/x.py\n@@\n test\n*** End Patch\n"
    payload = json.dumps({"tool_input": {"command": patch}})
    result = _call_codex_lib('codex_tool_paths "$2" ""', payload)
    assert result.returncode == 0, result.stderr
    assert "/repo/src/x.py" in result.stdout.splitlines()


@pytest.mark.skipif(shutil.which("python3") is None, reason="no-jq fallback needs python3")
def test_codex_hook_command_python3_fallback_without_jq():
    payload = json.dumps({"tool_input": {"command": "gh pr create --fill"}})
    result = _call_codex_lib('codex_hook_command "$2" ""', payload)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "gh pr create --fill"


@pytest.mark.parametrize(
    ("response", "success"),
    [
        ({"exit_code": 0}, True),
        ({"exit_code": 1}, False),
        ({"exit_code": None, "session_id": 42}, False),
        ({"exit_code": False}, False),
        (None, False),
        ('{"exit_code": 0}', True),
        ("Wall time: 0.1 seconds\nProcess exited with code 0\nOutput:\n", True),
        ("Wall time: 0.1 seconds\nProcess exited with code 1\nOutput:\nExit code: 0", True),
        ("Output:\nProcess exited with code 0", True),
        ("https://github.com/example/repo/pull/123\n", True),
        ("", True),
    ],
)
def test_codex_hook_allows_state_lookup_for_raw_stdout(response, success: bool):
    result = _call_codex_lib(
        'codex_hook_can_verify_pr "$2"', json.dumps({"tool_response": response})
    )
    assert (result.returncode == 0) is success


def test_codex_normalizes_native_windows_paths_for_shell_comparison(tmp_path: Path):
    # Run the actual normalization code with ntpath's Windows semantics. The
    # machine running pytest need not have a native Windows Python installed.
    driver = tmp_path / "native_python.py"
    driver.write_text(
        'import ntpath, os, sys\nprogram = sys.argv[2]\nsys.argv = ["-c", *sys.argv[3:]]\n'
        'sys.stdout.reconfigure(newline="\\r\\n")\n'
        'if "os.path.realpath" in program:\n    os.path = ntpath\n    os.sep = "\\\\"\n'
        "exec(program)\n"
    )
    python = tmp_path / "python3"
    python.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"\n'
    )
    python.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", "")}
    result = _call_codex_lib(
        'root=$(codex_abs_path "$2" .); target=$(codex_abs_path "$2" "src/new.py"); '
        'printf "%s\\n" "$root" "$target"; case "$target" in "$root"/*) exit 0;; *) exit 1;; esac',
        "C:/WorkTree",
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["c:/worktree", "c:/worktree/src/new.py"]


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

    @pytest.mark.parametrize(
        "path_kind", ["relative", "absolute_dotdot", "symlink", "alias", "nested"]
    )
    @pytest.mark.parametrize("tool", ["file_path", "apply_patch_move"])
    def test_guard_blocks_resolved_parent_paths(
        self, git_worktree_pair: tuple[Path, Path], path_kind: str, tool: str
    ):
        main, worktree = git_worktree_pair
        cwd = worktree
        if path_kind == "relative":
            target = "../main/new.py"
        elif path_kind == "absolute_dotdot":
            target = str(worktree / "../main/new.py")
        elif path_kind == "symlink":
            (worktree / "linked").symlink_to(main, target_is_directory=True)
            target = "linked/new.py"
        elif path_kind == "alias":
            alias = worktree.parent / "parent-alias"
            alias.symlink_to(main, target_is_directory=True)
            target = str(alias / "new.py")
        else:
            cwd = worktree / "subdir"
            cwd.mkdir()
            target = "../../main/new.py"
        tool_input = {"file_path": target}
        if tool == "apply_patch_move":
            tool_input = {
                "command": f"*** Begin Patch\n*** Update File: old.py\n*** Move to: {target}\n*** End Patch\n"
            }
        result = _run_hook(
            ".codex/hooks/guard-worktree-path.sh",
            {"cwd": str(cwd), "tool_input": tool_input},
            cwd,
        )
        assert result.returncode == 2, result.stderr
        assert "main checkout" in result.stderr

    def test_guard_uses_event_cwd_over_inherited_parent_environment(
        self, git_worktree_pair: tuple[Path, Path]
    ):
        main, worktree = git_worktree_pair
        result = _run_hook(
            ".codex/hooks/guard-worktree-path.sh",
            {"cwd": str(worktree), "tool_input": {"file_path": str(main / "new.py")}},
            worktree,
            {"CLAUDE_PROJECT_DIR": str(main)},
        )
        assert result.returncode == 2

    def test_guard_allows_alias_into_own_worktree(self, git_worktree_pair: tuple[Path, Path]):
        _, worktree = git_worktree_pair
        alias = worktree.parent / "worktree-alias"
        alias.symlink_to(worktree, target_is_directory=True)
        result = _run_hook(
            ".codex/hooks/guard-worktree-path.sh",
            {"cwd": str(alias), "tool_input": {"file_path": str(alias / "new.py")}},
            worktree,
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("parent_edit", [False, True])
    def test_guard_works_with_python_but_no_python3_command(
        self, git_worktree_pair: tuple[Path, Path], tmp_path: Path, parent_edit: bool
    ):
        main, worktree = git_worktree_pair
        tools = tmp_path / "python-only-bin"
        tools.mkdir()
        for name in ("bash", "dirname", "git", "cat", "sed", "awk", "tr", "jq"):
            executable = shutil.which(name)
            if executable:
                (tools / name).symlink_to(executable)
        driver = tools / "native_python.py"
        driver.write_text(
            'import sys\nprogram = sys.argv[2]\nsys.argv = ["-c", *sys.argv[3:]]\n'
            'sys.stdout.reconfigure(newline="\\r\\n")\nexec(program)\n'
        )
        python = tools / "python"
        python.write_text(
            f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"\n'
        )
        python.chmod(0o755)
        target = (main if parent_edit else worktree) / "new.py"
        result = _run_hook(
            ".codex/hooks/guard-worktree-path.sh",
            {"cwd": str(worktree), "tool_input": {"file_path": str(target)}},
            worktree,
            {"PATH": str(tools)},
        )
        assert result.returncode == (2 if parent_edit else 0), result.stderr

    @pytest.mark.parametrize("escape", [False, True])
    @pytest.mark.parametrize("nested", [False, True])
    def test_formatter_resolves_relative_paths_from_event_cwd(
        self, git_worktree_pair: tuple[Path, Path], escape: bool, nested: bool
    ):
        main, worktree = git_worktree_pair
        cwd = worktree / "subdir" if nested else worktree
        cwd.mkdir(exist_ok=True)
        (cwd / "local.py").write_text("x=1\n")
        (main / "parent.py").write_text("x=1\n")
        (cwd / "linked").symlink_to(main, target_is_directory=True)
        ruff = worktree / ".venv/bin/ruff"
        ruff.parent.mkdir(parents=True)
        calls = worktree / "ruff-calls"
        ruff.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$RUFF_CALLS"\n')
        ruff.chmod(0o755)
        path = "linked/parent.py" if escape else "local.py"
        result = _run_hook(
            ".codex/hooks/ruff-format.sh",
            {"cwd": str(cwd), "tool_input": {"file_path": path}},
            cwd,
            {"RUFF_CALLS": str(calls)},
        )
        assert result.returncode == 0, result.stderr
        if escape:
            assert not calls.exists()
        else:
            assert str((cwd / "local.py").resolve()) in calls.read_text().splitlines()

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
    @pytest.mark.parametrize("response", ["", "https://github.com/example/repo/pull/123\n"])
    def test_post_pr_hook_injects_compact_codex_review_workflow(
        self, command: str, response: str, git_worktree_pair: tuple[Path, Path]
    ):
        _, worktree = git_worktree_pair
        _stub_merged_pr(worktree, _head(worktree), state="OPEN")
        result = _run_hook(
            ".codex/hooks/post-pr-create.sh",
            {"cwd": str(worktree), "tool_input": {"command": command}, "tool_response": response},
            worktree,
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

    @pytest.mark.parametrize("metadata", [{}, {"state": "CLOSED"}, {"headRefOid": "0" * 40}])
    def test_post_pr_hook_requires_a_verified_open_pr(
        self, git_worktree_pair: tuple[Path, Path], metadata
    ):
        _, worktree = git_worktree_pair
        path = _stub_merged_pr(worktree, _head(worktree), state="OPEN")
        data = json.loads(path.read_text()) | metadata if metadata else {}
        path.write_text(json.dumps(data))
        result = _run_hook(
            ".codex/hooks/post-pr-create.sh",
            {
                "cwd": str(worktree),
                "tool_input": {"command": "gh pr create --fill"},
                "tool_response": "Exit code: 0\n",
            },
            worktree,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == ""

    def test_stop_hook_emits_valid_stop_output_when_memory_sync_is_noop(self, tmp_path: Path):
        result = _run_hook(
            ".codex/hooks/memory-sync-stop.sh",
            {"cwd": str(tmp_path)},
            tmp_path,
        )

        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"continue": True, "suppressOutput": True}

    def test_stop_hook_stdout_stays_pure_json_when_memory_sync_prints(self, tmp_path: Path):
        # aws s3 sync prints `upload: ...` lines to STDOUT on a real push
        # (--no-progress only hides the progress meter); the hook must route
        # them to stderr or the Stop-hook JSON is invalid exactly when memory
        # actually changed — the common case the no-op test above can't catch.
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        sync = scripts_dir / "agent-memory-sync.sh"
        sync.write_text("#!/usr/bin/env bash\necho 'upload: memory/foo.md to s3://bucket/foo.md'\n")
        sync.chmod(0o755)

        result = _run_hook(
            ".codex/hooks/memory-sync-stop.sh",
            {"cwd": str(tmp_path)},
            tmp_path,
        )

        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"continue": True, "suppressOutput": True}
        assert "upload:" in result.stderr

    def test_post_pr_merge_hook_ignores_non_merge_commands(self):
        for command in (
            "git status --short",
            "echo gh pr merge",
            'rg -n "gh pr merge" .codex',
            "# gh pr merge\n git status --short",
            "bash -lc 'gh pr merge 1 --squash'",
        ):
            result = _run_hook(
                ".codex/hooks/post-pr-merge.sh",
                {"cwd": str(PROJECT_ROOT), "tool_input": {"command": command}},
                PROJECT_ROOT,
            )

            assert result.returncode == 0
            assert result.stdout == ""
            assert result.stderr == ""

    def test_post_pr_merge_fast_forwards_clean_main_parent(
        self, merge_scenario_codex: tuple[Path, Path]
    ):
        main, worktree = merge_scenario_codex
        before = _head(main)
        result = _run_hook(
            ".codex/hooks/post-pr-merge.sh",
            {"cwd": str(worktree), "tool_input": {"command": "gh pr merge 1 --squash"}},
            worktree,
        )

        assert result.returncode == 0, result.stderr
        context = json.loads(result.stdout)["hookSpecificOutput"]
        assert context["hookEventName"] == "PostToolUse"
        assert "fast-forwarded" in context["additionalContext"]
        after = _head(main)
        assert after != before
        assert (
            after
            == subprocess.run(
                ["git", "-C", str(main), "rev-parse", "origin/main"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        )

    @pytest.mark.parametrize("response", [{"exit_code": 1}, None, {"session_id": 42}])
    def test_post_pr_merge_does_not_mutate_parent_after_failed_or_unknown_command(
        self, merge_scenario_codex: tuple[Path, Path], response
    ):
        main, worktree = merge_scenario_codex
        before = _head(main)
        result = _run_hook(
            ".codex/hooks/post-pr-merge.sh",
            {
                "cwd": str(worktree),
                "tool_input": {"command": "gh pr merge 1 --squash"},
                "tool_response": response,
            },
            worktree,
        )
        assert result.returncode == 0, result.stderr
        assert _head(main) == before
        assert result.stdout == ""

    def test_post_pr_create_does_not_report_failed_creation(self):
        result = _run_hook(
            ".codex/hooks/post-pr-create.sh",
            {
                "cwd": str(PROJECT_ROOT),
                "tool_input": {"command": "gh pr create --fill"},
                "tool_response": {"exit_code": 1},
            },
            PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert result.stdout == ""

    @pytest.mark.parametrize(
        "change", [{"state": "OPEN"}, {"headRefOid": "0" * 40}, {"baseRefName": "other"}]
    )
    def test_post_pr_merge_requires_this_head_merged_into_main(
        self, merge_scenario_codex: tuple[Path, Path], change
    ):
        main, worktree = merge_scenario_codex
        metadata = worktree / ".test-gh/pr.json"
        metadata.write_text(json.dumps(json.loads(metadata.read_text()) | change))
        before = _head(main)
        result = _run_hook(
            ".codex/hooks/post-pr-merge.sh",
            {"cwd": str(worktree), "tool_input": {"command": "gh pr merge 1 --squash --auto"}},
            worktree,
        )
        assert result.returncode == 0, result.stderr
        assert _head(main) == before
        assert result.stdout == ""

    def test_post_pr_merge_skips_dirty_parent(self, merge_scenario_codex: tuple[Path, Path]):
        main, worktree = merge_scenario_codex
        (main / "README.md").write_text("dirty\n")
        before = _head(main)
        result = _run_hook(
            ".codex/hooks/post-pr-merge.sh",
            {"cwd": str(worktree), "tool_input": {"command": "gh pr merge 2 --squash"}},
            worktree,
        )

        assert result.returncode == 0
        context = json.loads(result.stdout)["hookSpecificOutput"]
        assert "uncommitted changes" in context["additionalContext"]
        assert _head(main) == before

    def test_post_pr_merge_skips_non_main_parent(self, merge_scenario_codex: tuple[Path, Path]):
        main, worktree = merge_scenario_codex
        subprocess.run(
            ["git", "-C", str(main), "checkout", "-b", "codex/wip"],
            check=True,
            capture_output=True,
        )
        result = _run_hook(
            ".codex/hooks/post-pr-merge.sh",
            {"cwd": str(worktree), "tool_input": {"command": "gh pr merge 3 --squash"}},
            worktree,
        )

        assert result.returncode == 0
        context = json.loads(result.stdout)["hookSpecificOutput"]
        assert "not main" in context["additionalContext"]


def _python3_available() -> bool:
    return shutil.which("python3") is not None or shutil.which("python") is not None


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def _vendor_scope_positions(repo_root: Path) -> None:
    """Copy the real (pure-stdlib) scope_positions into a temp repo so the promote
    hook's scope_positions gate runs against the real path→positions mapping."""
    scripts = repo_root / "src" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (repo_root / "src" / "__init__.py").write_text("")
    (scripts / "__init__.py").write_text("")
    shutil.copy(PROJECT_ROOT / "src/scripts/scope_positions.py", scripts / "scope_positions.py")


def _write_splits(splits_dir: Path, content: str) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        (splits_dir / f"{name}.parquet").write_text(content)


def _setup_promote_repo(tmp_path: Path, *, splits_affecting: bool) -> tuple[Path, Path]:
    """main checkout + a feature worktree. The merged commit (origin/main tip)
    touches splits-affecting code (src/features) or just docs. The parent holds
    its own (STALE) data/splits; the worktree its own (FRESH) local one."""
    remote = tmp_path / "remote.git"
    main = tmp_path / "main"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(main)], check=True, capture_output=True)
    _git(main, "config", "user.email", "codex-hooks@example.test")
    _git(main, "config", "user.name", "Codex Hooks")
    _git(main, "config", "commit.gpgsign", "false")
    (main / ".gitignore").write_text("data/\n")
    _vendor_scope_positions(main)
    (main / "src" / "features").mkdir(parents=True, exist_ok=True)
    (main / "src" / "features" / "foo.py").write_text("x = 1\n")
    _git(main, "add", "-A")
    _git(main, "commit", "-m", "init")
    _git(main, "remote", "add", "origin", str(remote))
    _git(main, "push", "-u", "origin", "main")
    if splits_affecting:
        (main / "src" / "features" / "foo.py").write_text("x = 2\n")
        _git(main, "add", "src/features/foo.py")
    else:
        (main / "README.md").write_text("docs\n")
        _git(main, "add", "README.md")
    _git(main, "commit", "-m", "merge")
    _git(main, "push", "origin", "main")
    worktree = tmp_path / "feature"
    _git(main, "worktree", "add", "-b", "feature", str(worktree), "main")
    _write_splits(main / "data" / "splits", "STALE")
    _write_splits(worktree / "data" / "splits", "FRESH")
    _stub_merged_pr(worktree, _head(main))
    return main, worktree


@pytest.mark.skipif(not _jq_available(), reason="post-pr-merge hook needs jq to emit context")
@pytest.mark.skipif(not _python3_available(), reason="splits-promote gate needs python3")
class TestCodexPromoteSplits:
    def _parent_splits(self, main: Path) -> set[str]:
        return {
            (main / "data/splits" / f"{n}.parquet").read_text() for n in ("train", "val", "test")
        }

    def test_promotes_worktree_splits_on_splits_affecting_merge(self, tmp_path: Path):
        main, worktree = _setup_promote_repo(tmp_path, splits_affecting=True)
        result = _run_hook(
            ".codex/hooks/post-pr-merge.sh",
            {"cwd": str(worktree), "tool_input": {"command": "gh pr merge 1 --squash"}},
            worktree,
        )
        assert result.returncode == 0, result.stderr
        assert self._parent_splits(main) == {"FRESH"}
        context = json.loads(result.stdout)["hookSpecificOutput"]
        assert "splits promote: copied" in context["additionalContext"]

    def test_skips_when_worktree_splits_is_symlink(self, tmp_path: Path):
        main, worktree = _setup_promote_repo(tmp_path, splits_affecting=True)
        shutil.rmtree(worktree / "data/splits")
        (worktree / "data/splits").symlink_to(main / "data/splits")
        result = _run_hook(
            ".codex/hooks/post-pr-merge.sh",
            {"cwd": str(worktree), "tool_input": {"command": "gh pr merge 1 --squash"}},
            worktree,
        )
        assert result.returncode == 0
        assert self._parent_splits(main) == {"STALE"}
        assert "splits promote: copied" not in result.stdout

    def test_skips_when_merge_not_splits_affecting(self, tmp_path: Path):
        main, worktree = _setup_promote_repo(tmp_path, splits_affecting=False)
        result = _run_hook(
            ".codex/hooks/post-pr-merge.sh",
            {"cwd": str(worktree), "tool_input": {"command": "gh pr merge 1 --squash"}},
            worktree,
        )
        assert result.returncode == 0
        assert self._parent_splits(main) == {"STALE"}
        assert "splits promote: copied" not in result.stdout

    def test_does_not_promote_splits_for_a_different_main_tip(self, tmp_path: Path):
        main, worktree = _setup_promote_repo(tmp_path, splits_affecting=True)
        (main / "src/features/foo.py").write_text("x = 3\n")
        _git(main, "add", "src/features/foo.py")
        _git(main, "commit", "-m", "another PR")
        _git(main, "push", "origin", "main")
        result = _run_hook(
            ".codex/hooks/post-pr-merge.sh",
            {"cwd": str(worktree), "tool_input": {"command": "gh pr merge 1 --squash"}},
            worktree,
        )
        assert result.returncode == 0, result.stderr
        assert self._parent_splits(main) == {"STALE"}
        assert "verified merge" in result.stderr

    def test_does_not_promote_splits_when_main_cannot_be_refreshed(self, tmp_path: Path):
        main, worktree = _setup_promote_repo(tmp_path, splits_affecting=True)
        _git(main, "remote", "set-url", "origin", str(tmp_path / "missing-remote.git"))
        result = _run_hook(
            ".codex/hooks/post-pr-merge.sh",
            {"cwd": str(worktree), "tool_input": {"command": "gh pr merge 1 --squash"}},
            worktree,
        )
        assert result.returncode == 0, result.stderr
        assert self._parent_splits(main) == {"STALE"}
        assert "could not refresh main" in result.stderr


class TestMainWorktreePipefailSafe:
    """Regression pin for #1369: agent_hooks_main_worktree must not abort under
    the launcher's `set -euo pipefail`. The old `git ... | awk '...exit'` pipeline
    made awk close git's stdout after line 1, so git took SIGPIPE (141) and, with
    pipefail, propagated it to `set -e` — killing codex-fresh-worktree.sh before a
    worktree was created."""

    def _call(self, root: Path) -> subprocess.CompletedProcess[str]:
        lib = PROJECT_ROOT / "scripts/agent-hooks-lib.sh"
        script = f'set -euo pipefail; . "{lib}"; agent_hooks_main_worktree "$1"'
        return subprocess.run(
            [_bash(), "-c", script, "agent-hooks-test", str(root)],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_returns_primary_worktree_under_pipefail(self, tmp_path: Path):
        main = tmp_path / "main"
        main.mkdir()
        subprocess.run(["git", "init", "-b", "main", str(main)], check=True, capture_output=True)
        result = self._call(main)
        assert result.returncode == 0, result.stderr  # 141 (SIGPIPE) was the bug
        assert Path(result.stdout.strip()).resolve() == main.resolve()

    def test_no_abort_when_not_a_git_repo(self, tmp_path: Path):
        # git worktree list fails here; the helper must swallow it, not abort (141/128).
        result = self._call(tmp_path)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == ""
