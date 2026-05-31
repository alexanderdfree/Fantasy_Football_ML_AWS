#!/bin/bash
# PreToolUse hook (Edit|Write|MultiEdit|NotebookEdit): block a write that targets
# the PARENT checkout from inside a worktree session.
#
# Why: this repo is regularly worked from `.claude/worktrees/<name>` clones where
# the parent holds `main`. Explore/Plan sub-agents and plan files report
# *parent*-absolute paths (`/…/Final-Project/src/foo.py`); using those verbatim for
# Edit/Write silently writes to the parent's branch instead of the feature branch —
# `git status` in the worktree stays clean and benchmarks re-run on the unchanged
# code. Prose guidance in CLAUDE.md ("Worktree workflow") + auto-memory
# (feedback_edit_tool_worktree_path) failed to prevent this 4× (PRs #284/#354/#370/#378),
# so this is the deterministic backstop that fires at write time.
#
# Contract: no-op unless CLAUDE_PROJECT_DIR is a worktree. Block (exit 2) only when
# the target path is under the parent root but NOT under the worktree — i.e. the
# exact parent-checkout-write mistake. Paths outside the project entirely (/tmp,
# ~/.claude/.../memory, …) are allowed so legitimate out-of-tree writes still work.
set -u

# Resolve jq: prefer PATH, fall back to common absolute install locations so the
# hook works whether or not jq lives at /usr/bin (WSL/dev boxes differ from CI).
jq_bin=""
for _c in jq /usr/bin/jq /usr/local/bin/jq /opt/homebrew/bin/jq /home/linuxbrew/.linuxbrew/bin/jq; do
  if command -v "$_c" >/dev/null 2>&1; then jq_bin="$_c"; break; fi
done
[ -n "$jq_bin" ] || exit 0  # no jq → cannot parse path; fail open (matches prior behavior)

input=$(cat)

proj="${CLAUDE_PROJECT_DIR:-}"
case "$proj" in
  */.claude/worktrees/*) ;;   # worktree session — guard is active
  *) exit 0 ;;                # parent/normal checkout — nothing to guard
esac
parent="${proj%%/.claude/worktrees/*}"

# Edit/Write/MultiEdit carry file_path; NotebookEdit carries notebook_path.
fp=$(printf '%s' "$input" | "$jq_bin" -r '.tool_input.file_path // .tool_input.notebook_path // empty' 2>/dev/null || true)
[ -n "$fp" ] || exit 0

case "$fp" in
  "$proj"/*) exit 0 ;;        # inside this worktree — allow
  "$parent"/*)               # under parent root but NOT this worktree — block
    corrected="$proj/${fp#"$parent"/}"
    {
      echo "BLOCK: file_path is in the PARENT checkout, not this worktree."
      echo "  file_path:          $fp"
      echo "  CLAUDE_PROJECT_DIR: $proj"
      echo "Edits here land on the parent's branch (usually main), not your feature branch."
      echo "Re-prefix the path to the worktree and retry:"
      echo "  $corrected"
    } >&2
    exit 2 ;;
  *) exit 0 ;;               # outside the project entirely (/tmp, ~/.claude, …) — allow
esac
