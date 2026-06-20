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

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIB = PROJECT_ROOT / ".claude/hooks/lib.sh"
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


def _matcher_result(command: str, fn: str = "claude_command_invokes_gh_pr_create") -> bool:
    script = f'. "{LIB}"; {fn} "$1"'
    result = subprocess.run(
        [_bash(), "-c", script, "claude-hook-test", command],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _merge_matcher_result(command: str) -> bool:
    return _matcher_result(command, "claude_command_invokes_gh_pr_merge")


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


def test_all_pr_hooks_source_the_shared_lib():
    """pre-pr.sh / post-pr-create.sh / post-pr-merge.sh must all wire the shared
    parser, not re-inline the flawed flat regex."""
    expected_matcher = {
        "pre-pr.sh": "claude_command_invokes_gh_pr_create",
        "post-pr-create.sh": "claude_command_invokes_gh_pr_create",
        "post-pr-merge.sh": "claude_command_invokes_gh_pr_merge",
    }
    for name, matcher in expected_matcher.items():
        text = (PROJECT_ROOT / ".claude/hooks" / name).read_text()
        assert "lib.sh" in text, f"{name} does not source lib.sh"
        assert matcher in text, f"{name} does not call {matcher}"
        assert "=~ (^|[[:space:]" not in text, f"{name} still has the flat regex"


@pytest.mark.parametrize(
    "command",
    [
        "gh pr merge 5 --squash",
        "gh pr merge 12 --squash --admin",
        "git status --short && gh pr merge 7 --squash",
        "GH_TOKEN=example gh pr merge 7 --squash",
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
        'grep -rn "gh pr merge" .claude',
        "# gh pr merge\n git status --short",
        "git status --short",
        "bash -lc 'gh pr merge 5 --squash'",
        "gh pr list  # was: gh pr merge",
        "gh pr create --fill",
    ],
)
def test_pr_merge_matcher_rejects_quoted_or_argument_text(command: str):
    assert not _merge_matcher_result(command)


def _git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True, text=True, capture_output=True
    )


def _run_merge_hook(
    payload: dict[str, object], cwd: Path, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in ("CLAUDE_PROJECT_DIR", "GIT_CEILING_DIRECTORIES", "GIT_DIR", "GIT_WORK_TREE"):
        env.pop(key, None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(PROJECT_ROOT / ".claude/hooks/post-pr-merge.sh")],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=cwd,
        env=env,
        check=False,
    )


@pytest.fixture
def merge_scenario(tmp_path: Path) -> tuple[Path, Path]:
    """A main checkout on `main` + a feature worktree, with origin/main advanced
    one commit beyond the parent — the post-pr-merge fast-forward scenario."""
    remote = tmp_path / "remote.git"
    main = tmp_path / "main"
    other = tmp_path / "other"
    # Pin `main` explicitly — CI's git defaults init/clone to `master`, so a
    # clone-of-empty + `push origin main` fails ("src refspec main does not
    # match any").
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(main)], check=True, capture_output=True)
    _git(main, "config", "user.email", "claude-hooks@example.test")
    _git(main, "config", "user.name", "Claude Hooks")
    _git(main, "config", "commit.gpgsign", "false")
    (main / "README.md").write_text("init\n")
    _git(main, "add", "README.md")
    _git(main, "commit", "-m", "init")
    _git(main, "remote", "add", "origin", str(remote))
    _git(main, "push", "-u", "origin", "main")
    worktree = tmp_path / "feature"
    _git(main, "worktree", "add", "-b", "feature", str(worktree))
    # advance origin/main from an independent clone so the parent is behind
    # (-b main: origin/main exists, but the bare's HEAD may be `master` in CI)
    subprocess.run(
        ["git", "clone", "-b", "main", str(remote), str(other)], check=True, capture_output=True
    )
    _git(other, "config", "user.email", "claude-hooks@example.test")
    _git(other, "config", "user.name", "Claude Hooks")
    _git(other, "config", "commit.gpgsign", "false")
    (other / "NEW.md").write_text("more\n")
    _git(other, "add", "NEW.md")
    _git(other, "commit", "-m", "advance main")
    _git(other, "push", "origin", "main")
    return main, worktree


@pytest.mark.skipif(not _jq_available(), reason="post-pr-merge hook needs jq to emit context")
class TestClaudePostPrMerge:
    def _head(self, path: Path) -> str:
        return _git(path, "rev-parse", "HEAD").stdout.strip()

    def test_fast_forwards_clean_main_parent(self, merge_scenario: tuple[Path, Path]):
        main, worktree = merge_scenario
        before = self._head(main)
        result = _run_merge_hook(
            {"tool_input": {"command": "gh pr merge 1 --squash"}},
            worktree,
            {"CLAUDE_PROJECT_DIR": str(worktree)},
        )
        assert result.returncode == 0, result.stderr
        context = json.loads(result.stdout)["hookSpecificOutput"]
        assert context["hookEventName"] == "PostToolUse"
        assert "fast-forwarded" in context["additionalContext"]
        after = self._head(main)
        assert after != before
        assert after == _git(main, "rev-parse", "origin/main").stdout.strip()

    def test_skips_non_merge_command(self, merge_scenario: tuple[Path, Path]):
        main, worktree = merge_scenario
        before = self._head(main)
        result = _run_merge_hook(
            {"tool_input": {"command": "git status --short"}},
            worktree,
            {"CLAUDE_PROJECT_DIR": str(worktree)},
        )
        assert result.returncode == 0
        assert result.stdout == ""
        assert self._head(main) == before

    def test_skips_dirty_parent(self, merge_scenario: tuple[Path, Path]):
        main, worktree = merge_scenario
        (main / "README.md").write_text("dirty\n")
        before = self._head(main)
        result = _run_merge_hook(
            {"tool_input": {"command": "gh pr merge 2 --squash"}},
            worktree,
            {"CLAUDE_PROJECT_DIR": str(worktree)},
        )
        assert result.returncode == 0
        context = json.loads(result.stdout)["hookSpecificOutput"]
        assert "uncommitted changes" in context["additionalContext"]
        assert self._head(main) == before  # not clobbered

    def test_skips_non_main_parent(self, merge_scenario: tuple[Path, Path]):
        main, worktree = merge_scenario
        _git(main, "checkout", "-b", "codex/wip")
        result = _run_merge_hook(
            {"tool_input": {"command": "gh pr merge 3 --squash"}},
            worktree,
            {"CLAUDE_PROJECT_DIR": str(worktree)},
        )
        assert result.returncode == 0
        context = json.loads(result.stdout)["hookSpecificOutput"]
        assert "not main" in context["additionalContext"]


def _python3_available() -> bool:
    return shutil.which("python3") is not None or shutil.which("python") is not None


def _vendor_scope_positions(repo_root: Path) -> None:
    """Copy the real (pure-stdlib) scope_positions into a temp repo so the promote
    hook's `python3 -m src.scripts.scope_positions` gate runs against the real
    path→positions mapping."""
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
    _git(main, "config", "user.email", "claude-hooks@example.test")
    _git(main, "config", "user.name", "Claude Hooks")
    _git(main, "config", "commit.gpgsign", "false")
    (main / ".gitignore").write_text("data/\n")
    _vendor_scope_positions(main)
    (main / "src" / "features").mkdir(parents=True, exist_ok=True)
    (main / "src" / "features" / "foo.py").write_text("x = 1\n")
    _git(main, "add", "-A")
    _git(main, "commit", "-m", "init")
    _git(main, "remote", "add", "origin", str(remote))
    _git(main, "push", "-u", "origin", "main")
    # the "merged PR" == origin/main tip; touches splits code (or just docs)
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
    return main, worktree


@pytest.mark.skipif(not _jq_available(), reason="post-pr-merge hook needs jq to emit context")
@pytest.mark.skipif(not _python3_available(), reason="splits-promote gate needs python3")
class TestClaudePromoteSplits:
    def _parent_splits(self, main: Path) -> set[str]:
        return {
            (main / "data/splits" / f"{n}.parquet").read_text() for n in ("train", "val", "test")
        }

    def test_promotes_worktree_splits_on_splits_affecting_merge(self, tmp_path: Path):
        main, worktree = _setup_promote_repo(tmp_path, splits_affecting=True)
        result = _run_merge_hook(
            {"tool_input": {"command": "gh pr merge 1 --squash"}},
            worktree,
            {"CLAUDE_PROJECT_DIR": str(worktree)},
        )
        assert result.returncode == 0, result.stderr
        assert self._parent_splits(main) == {"FRESH"}  # parent now matches the worktree
        context = json.loads(result.stdout)["hookSpecificOutput"]
        assert "splits promote: copied" in context["additionalContext"]

    def test_skips_when_worktree_splits_is_symlink(self, tmp_path: Path):
        main, worktree = _setup_promote_repo(tmp_path, splits_affecting=True)
        shutil.rmtree(worktree / "data/splits")
        (worktree / "data/splits").symlink_to(main / "data/splits")
        result = _run_merge_hook(
            {"tool_input": {"command": "gh pr merge 1 --squash"}},
            worktree,
            {"CLAUDE_PROJECT_DIR": str(worktree)},
        )
        assert result.returncode == 0
        assert self._parent_splits(main) == {"STALE"}  # untouched
        assert "splits promote: copied" not in result.stdout

    def test_skips_when_merge_not_splits_affecting(self, tmp_path: Path):
        main, worktree = _setup_promote_repo(tmp_path, splits_affecting=False)
        result = _run_merge_hook(
            {"tool_input": {"command": "gh pr merge 1 --squash"}},
            worktree,
            {"CLAUDE_PROJECT_DIR": str(worktree)},
        )
        assert result.returncode == 0
        assert self._parent_splits(main) == {"STALE"}  # untouched
        assert "splits promote: copied" not in result.stdout


def _run_link_worktree_data(worktree: Path) -> subprocess.CompletedProcess[str]:
    """Source lib.sh and call claude_link_worktree_data <worktree> directly."""
    script = f'. "{LIB}"; claude_link_worktree_data "$1"'
    return subprocess.run(
        [_bash(), "-c", script, "claude-hook-test", str(worktree)],
        text=True,
        capture_output=True,
        check=False,
    )


def _setup_link_repo(tmp_path: Path) -> tuple[Path, Path]:
    """A main checkout holding prebuilt (gitignored) data/{raw,splits} + a fresh
    feature worktree whose data/ has only the tracked README (no raw/splits) — the
    exact fresh-worktree state session-start.sh's auto-link targets."""
    main = tmp_path / "main"
    subprocess.run(["git", "init", "-b", "main", str(main)], check=True, capture_output=True)
    _git(main, "config", "user.email", "claude-hooks@example.test")
    _git(main, "config", "user.name", "Claude Hooks")
    _git(main, "config", "commit.gpgsign", "false")
    (main / ".gitignore").write_text("data/raw/\ndata/splits/\n")
    (main / "data").mkdir()
    (main / "data" / "README.md").write_text("data dir\n")
    _git(main, "add", "-A")
    _git(main, "commit", "-m", "init")
    # parent's prebuilt gitignored data (what we link into worktrees)
    (main / "data" / "raw").mkdir()
    (main / "data" / "raw" / "weekly.parquet").write_text("RAW")
    (main / "data" / "splits").mkdir()
    for name in ("train", "val", "test"):
        (main / "data" / "splits" / f"{name}.parquet").write_text("SPLIT")
    worktree = tmp_path / "feature"
    _git(main, "worktree", "add", "-b", "feature", str(worktree), "main")
    return main, worktree


class TestClaudeLinkWorktreeData:
    def test_links_both_data_dirs(self, tmp_path: Path):
        main, worktree = _setup_link_repo(tmp_path)
        # fresh worktree starts without raw/splits (gitignored, not checked out)
        assert not (worktree / "data" / "raw").exists()
        assert not (worktree / "data" / "splits").exists()
        result = _run_link_worktree_data(worktree)
        assert result.returncode == 0, result.stderr
        for d in ("raw", "splits"):
            link = worktree / "data" / d
            assert link.is_symlink(), f"{d} not symlinked"
            assert link.resolve() == (main / "data" / d).resolve()
        assert "linked raw splits" in result.stdout

    def test_idempotent_second_call(self, tmp_path: Path):
        main, worktree = _setup_link_repo(tmp_path)
        _run_link_worktree_data(worktree)
        result = _run_link_worktree_data(worktree)
        assert result.returncode == 0
        assert result.stdout == ""  # already linked → nothing new, no output
        for d in ("raw", "splits"):
            assert (worktree / "data" / d).is_symlink()

    def test_noop_in_main_checkout(self, tmp_path: Path):
        main, _ = _setup_link_repo(tmp_path)
        result = _run_link_worktree_data(main)
        assert result.returncode == 0
        assert result.stdout == ""
        # main's real data dirs must stay real dirs (never self-linked)
        assert (main / "data" / "raw").is_dir()
        assert not (main / "data" / "raw").is_symlink()
        assert not (main / "data" / "splits").is_symlink()

    def test_leaves_local_real_dir_alone(self, tmp_path: Path):
        main, worktree = _setup_link_repo(tmp_path)
        # worktree built its OWN splits locally (a real dir) — must not be shadowed
        (worktree / "data" / "splits").mkdir(parents=True)
        (worktree / "data" / "splits" / "train.parquet").write_text("LOCAL")
        result = _run_link_worktree_data(worktree)
        assert result.returncode == 0
        assert not (worktree / "data" / "splits").is_symlink()  # local real dir kept
        assert (worktree / "data" / "splits" / "train.parquet").read_text() == "LOCAL"
        assert (worktree / "data" / "raw").is_symlink()  # raw still linked
        assert "linked raw from" in result.stdout

    def test_replaces_dangling_symlink(self, tmp_path: Path):
        main, worktree = _setup_link_repo(tmp_path)
        dangling = worktree / "data" / "raw"
        dangling.symlink_to(tmp_path / "does-not-exist")
        assert dangling.is_symlink() and not dangling.exists()  # dangling
        result = _run_link_worktree_data(worktree)
        assert result.returncode == 0
        assert (worktree / "data" / "raw").is_symlink()
        assert (worktree / "data" / "raw").resolve() == (main / "data" / "raw").resolve()
