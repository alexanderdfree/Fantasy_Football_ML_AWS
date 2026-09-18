import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, text=True, capture_output=True
    ).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    main = tmp_path / "main"
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-b", "main", str(main)], check=True, capture_output=True)
    _git(main, "config", "user.name", "Hook Fixture")
    _git(main, "config", "user.email", "hooks@example.invalid")
    _git(main, "config", "commit.gpgsign", "false")
    (main / ".gitignore").write_text("data/\n.claude/worktrees/\n.test-gh/\n")
    (main / "src/scripts").mkdir(parents=True)
    shutil.copyfile(
        ROOT / "src/scripts/scope_positions.py", main / "src/scripts/scope_positions.py"
    )
    _git(main, "add", ".")
    _git(main, "commit", "-m", "initial fixture")
    (main / "src/features").mkdir()
    (main / "src/features/change.py").write_text("VALUE = 1\n")
    _git(main, "add", ".")
    _git(main, "commit", "-m", "merged fixture feature")
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    _git(main, "remote", "add", "origin", str(remote))
    _git(main, "push", "-u", "origin", "main")
    worktree = main / ".claude/worktrees/feature"
    _git(main, "worktree", "add", "-b", "feature", str(worktree))
    for root, text in [(main, "approved"), (worktree, "candidate")]:
        (root / "data/splits").mkdir(parents=True)
        for split in ["train", "val", "test"]:
            (root / f"data/splits/{split}.parquet").write_text(text)
    return main.resolve(), worktree.resolve()


def _guard(provider, tmp_path):
    if provider != "bootstrap":
        return ROOT / f".{provider}/hooks/guard-worktree-path.sh"
    template = (ROOT / "scripts/bootstrap-claude-wsl.sh").read_text()
    script = template.split("cat > \"$GUARD\" <<'GUARD_EOF'\n", 1)[1].split("\nGUARD_EOF", 1)[0]
    path = tmp_path / "installed-parent-guard.sh"
    path.write_text(script)
    return path


@pytest.mark.parametrize("provider", ["claude", "gemini", "bootstrap"])
@pytest.mark.parametrize(
    "case", ["direct", "dotdot", "relative", "symlink", "own_alias", "healthy", "outside"]
)
def test_guards_compare_resolved_paths(repos, tmp_path, provider, case):
    main, worktree = repos
    if case == "direct":
        path = main / "src/target.py"
    elif case == "dotdot":
        path = worktree / "../../../src/target.py"
    elif case == "relative":
        path = Path("../../../src/target.py")
    elif case == "symlink":
        (worktree / "parent-alias").symlink_to(main, target_is_directory=True)
        path = worktree / "parent-alias/src/target.py"
    elif case == "own_alias":
        (main / "worktree-alias").symlink_to(worktree, target_is_directory=True)
        path = main / "worktree-alias/src/target.py"
    elif case == "healthy":
        path = worktree / "src/target.py"
    else:
        path = tmp_path / "outside.py"
    key = "TargetFile" if provider == "gemini" else "file_path"
    event = {"cwd": str(worktree), "tool_input": {key: str(path)}}
    result = subprocess.run(
        ["bash", str(_guard(provider, tmp_path))],
        input=json.dumps(event),
        text=True,
        capture_output=True,
        cwd=worktree,
        env={
            **os.environ,
            "CLAUDE_PROJECT_DIR": str(worktree),
            "GEMINI_PROJECT_DIR": str(worktree),
        },
    )
    expected = 0 if case in {"own_alias", "healthy", "outside"} else 2
    assert result.returncode == expected, result.stderr


def _merge_hook(worktree, metadata, response=None):
    stub = worktree / ".test-gh"
    stub.mkdir(exist_ok=True)
    (stub / "pr.json").write_text(json.dumps(metadata))
    (stub / "gh").write_text('#!/bin/sh\ncat "$(dirname "$0")/pr.json"\n')
    (stub / "gh").chmod(0o755)
    event = {
        "tool_input": {"command": "gh pr merge 1 --auto --squash"},
        "tool_response": response
        if response is not None
        else {"stdout": "merge queued", "stderr": "", "interrupted": False},
    }
    return subprocess.run(
        ["bash", str(ROOT / ".claude/hooks/post-pr-merge.sh")],
        input=json.dumps(event),
        text=True,
        capture_output=True,
        cwd=worktree,
        env={
            **os.environ,
            "CLAUDE_PROJECT_DIR": str(worktree),
            "PATH": f"{stub}{os.pathsep}{os.environ['PATH']}",
        },
    )


@pytest.mark.parametrize(
    "case", ["queued", "wrong_head", "wrong_base", "no_merge", "failed", "newer_main", "matching"]
)
def test_only_matching_completed_merge_promotes_splits(repos, case):
    main, worktree = repos
    head = _git(worktree, "rev-parse", "HEAD")
    metadata = {
        "state": "MERGED",
        "baseRefName": "main",
        "headRefOid": head,
        "mergeCommit": {"oid": head},
    }
    response = None
    if case == "queued":
        (worktree / "unmerged.py").write_text("VALUE = 2\n")
        _git(worktree, "add", "unmerged.py")
        _git(worktree, "commit", "-m", "unmerged fixture")
        metadata.update(
            state="OPEN", headRefOid=_git(worktree, "rev-parse", "HEAD"), mergeCommit=None
        )
    elif case == "wrong_head":
        metadata["headRefOid"] = "0" * 40
    elif case == "wrong_base":
        metadata["baseRefName"] = "another-branch"
    elif case == "no_merge":
        metadata["mergeCommit"] = None
    elif case == "failed":
        response = {"exit_code": 1, "stdout": "merge failed"}
    elif case == "newer_main":
        (main / "src/features/later.py").write_text("VALUE = 3\n")
        _git(main, "add", ".")
        _git(main, "commit", "-m", "later fixture")
        _git(main, "push", "origin", "main")
    result = _merge_hook(worktree, metadata, response)
    assert result.returncode == 0, result.stderr
    expected = "candidate" if case == "matching" else "approved"
    assert {
        (main / f"data/splits/{split}.parquet").read_text() for split in ["train", "val", "test"]
    } == {expected}
