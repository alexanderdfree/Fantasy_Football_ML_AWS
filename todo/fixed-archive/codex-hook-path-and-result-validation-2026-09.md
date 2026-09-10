### [FIXED] Codex hooks compared unresolved paths and treated failed PR commands as successful

**File(s):** `.codex/hooks/lib.sh`, `.codex/hooks/guard-worktree-path.sh`,
`.codex/hooks/ruff-format.sh`, `.codex/hooks/post-pr-create.sh`,
`.codex/hooks/post-pr-merge.sh`, `tests/scripts/test_codex_hooks.py`.
Observed on base `1b0ef72e6d311b05dc328436b15558f45ca303e2`.

**What:** The worktree guard compared raw strings, allowing parent-checkout
destinations expressed with `..`, symlinks, or macOS filesystem aliases. Relative
paths were resolved from the repository root rather than the hook event cwd,
so formatting missed edits from subdirectories. An inherited parent project
environment variable could also override the event cwd. The PR follow-up hooks
ran after failed commands: Codex emits `PostToolUse` for nonzero Bash exits too.

**Fix:** Resolve filesystem destinations, including nonexistent new-file leaves,
and use the event cwd ahead of inherited project hints. Restrict formatting to
resolved paths inside the active worktree, with shell-compatible Windows path
separators, LF-only helper output, and Python 3 discovery under either `python3` or `python`. Missing,
pending, and explicitly failed structured responses skip PR follow-ups.
Codex 0.153.4 unified exec sends only stdout to hooks, so raw stdout is never
treated as an exit status: verify an open PR for the worktree's HEAD before
emitting the post-create workflow. Path validation blocks edits if Python 3
is unavailable.
The merge follow-up also verifies that GitHub reports the worktree's exact HEAD
merged into main: enabling auto-merge is insufficient. Dataset promotion checks
the verified squash commit and skips when main has advanced beyond it.

**Evidence (2026-09-10):** The original 107 focused infrastructure checks passed.
Against an isolated copy of the base hook implementations, 15 of 16 added
behavior checks failed, including a failed merge command changing a temporary
parent checkout's HEAD. The repaired hooks passed the expanded regression set.
Tests use temporary repositories, stub formatters, and local bare remotes.

**Lesson:** Verify hook behavior using actual event cwd and completion status,
with filesystem aliases and failed-command controls. Parsing valid hook JSON
does not establish that a guard blocks the intended operation. Hook source
repairs do not imply activation: keep user-level disabled/trust choices intact.
