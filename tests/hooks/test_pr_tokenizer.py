"""Regression tests for the ``gh pr <subcommand>`` matcher in
``.claude/hooks/lib.sh`` (parity twin of ``.codex/hooks/lib.sh``).

``claude_command_invokes_gh_pr_subcommand`` is the most safety-critical shell in
the hook suite: ``pre-pr.sh`` blocks ``gh pr create`` on it (a false positive
exits 2 and *wedges the session*, #894), and ``post-pr-create.sh`` /
``post-pr-merge.sh`` fire their autonomous workflows on it (a false positive on a
quoted/echoed/comment occurrence injected the post-PR workflow with no PR opened,
#893). The tokenizer strips quotes/escapes/comments and splits on
``; | & && || newline`` so only a real top-level invocation matches. These cases
pin that contract so a future edit can't silently reintroduce the bug class.

The tests shell out to ``bash`` (the helpers use bash arrays, ``[[ =~ ]]`` and
``${cmd:i:1}`` slicing — they are not POSIX sh) and assert the function's exit
code, the same way the hooks consume it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB = REPO_ROOT / ".claude" / "hooks" / "lib.sh"

# Source lib.sh and call the matcher with positional args so the command string
# is passed verbatim (no shell re-quoting of the payload under test).
_SCRIPT = 'source "$1"; claude_command_invokes_gh_pr_subcommand "$2" "$3"'


def _invokes(subcmd: str, cmd: str) -> int:
    """Return the matcher's exit code (0 = matches, 1 = does not)."""
    result = subprocess.run(
        ["bash", "-c", _SCRIPT, "_", str(LIB), subcmd, cmd],
        capture_output=True,
        text=True,
    )
    # The function only ever returns 0/1; anything else means the harness broke
    # (lib.sh not found, function renamed, bash missing) — surface it loudly.
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode} for {subcmd!r} / {cmd!r}; stderr={result.stderr!r}"
    )
    return result.returncode


def test_lib_exists() -> None:
    assert LIB.is_file(), f"hook lib not found at {LIB}"


# --- Real top-level `gh pr create` invocations MUST match (exit 0). ---------
_CREATE_MATCH = [
    "gh pr create",
    "gh pr create --fill",
    "gh pr create --title x --body y",
    "foo && gh pr create",
    "gh pr create | cat",
    "gh pr create; echo done",
    "GH_TOKEN=x gh pr create",  # leading VAR=val assignment is skipped
    "/usr/bin/gh pr create",  # absolute path to gh
    "env GH_PAGER= gh pr create",  # env wrapper is unwrapped
]


@pytest.mark.parametrize("cmd", _CREATE_MATCH)
def test_create_matches_real_invocations(cmd: str) -> None:
    assert _invokes("create", cmd) == 0, f"expected MATCH for: {cmd!r}"


# --- Non-invocations MUST NOT match (exit 1) — the #893/#894 bug class. -----
_CREATE_NO_MATCH = [
    'echo "gh pr create"',  # quoted string, not a command
    "echo 'gh pr create'",
    "git log --grep='gh pr create'",  # quoted flag value
    "# gh pr create",  # whole-line comment
    "echo hi  # gh pr create",  # trailing comment
    "gh pr list",  # different subcommand
    "gh pr merge",  # different subcommand
    "gh issue create",  # not `pr`
    "ls -la",  # no gh at all
    "grep -r 'gh pr create' .",
]


@pytest.mark.parametrize("cmd", _CREATE_NO_MATCH)
def test_create_ignores_non_invocations(cmd: str) -> None:
    assert _invokes("create", cmd) == 1, f"expected NO-MATCH for: {cmd!r}"


# --- The `merge` matcher is the same tokenizer with a different subcommand. --
_MERGE_MATCH = [
    "gh pr merge",
    "gh pr merge 123 --squash",
    "gh pr checks 1 && gh pr merge 1 --squash",
]
_MERGE_NO_MATCH = [
    'grep "gh pr merge" file',
    "# gh pr merge 123",
    "gh pr create",  # create is not merge
    "gh pr view 1",
]


@pytest.mark.parametrize("cmd", _MERGE_MATCH)
def test_merge_matches_real_invocations(cmd: str) -> None:
    assert _invokes("merge", cmd) == 0, f"expected MATCH for: {cmd!r}"


@pytest.mark.parametrize("cmd", _MERGE_NO_MATCH)
def test_merge_ignores_non_invocations(cmd: str) -> None:
    assert _invokes("merge", cmd) == 1, f"expected NO-MATCH for: {cmd!r}"


# --- Public wrappers the hooks actually call. -------------------------------
@pytest.mark.parametrize(
    ("fn", "cmd", "expected"),
    [
        ("claude_command_invokes_gh_pr_create", "gh pr create", 0),
        ("claude_command_invokes_gh_pr_create", 'echo "gh pr create"', 1),
        ("claude_command_invokes_gh_pr_merge", "gh pr merge 1 --squash", 0),
        ("claude_command_invokes_gh_pr_merge", "gh pr create", 1),
    ],
)
def test_public_wrappers(fn: str, cmd: str, expected: int) -> None:
    result = subprocess.run(
        ["bash", "-c", f'source "$1"; {fn} "$2"', "_", str(LIB), cmd],
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected, (
        f"{fn}({cmd!r}) -> {result.returncode}, expected {expected}; stderr={result.stderr!r}"
    )


def test_heredoc_body_is_known_limitation() -> None:
    """The tokenizer does NOT track heredoc state, so an unquoted heredoc *body*
    line that is exactly ``gh pr create`` is treated as a top-level segment and
    matches. This pins the CURRENT behavior (not an endorsement). It's a benign
    theoretical gap — such a construction essentially never appears in practice,
    and the failure mode (an extra workflow nudge, or a blocked heredoc command)
    is recoverable. If heredoc handling is ever added, flip this assertion and
    drop the note."""
    cmd = "cat <<EOF > note.txt\ngh pr create\nEOF"
    assert _invokes("create", cmd) == 0
