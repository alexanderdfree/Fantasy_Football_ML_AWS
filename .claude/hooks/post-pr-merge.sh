#!/bin/bash
# PostToolUse hook: after `gh pr merge`, fast-forward the main/parent checkout's
# `main` branch to origin/main so it tracks the squash-merge that just landed.
# The parent never self-updates — worktrees branch from origin/main and
# `gh pr merge` lands on the remote, so the parent's local `main` drifts stale.
#
# GUARDED in claude_refresh_parent_main: skips when the parent holds a non-main
# branch or has uncommitted WIP (it can be another agent's `codex/*` checkout),
# and uses `pull --ff-only` so it can never create a merge commit or clobber
# work. Best-effort: a failed/skipped refresh never blocks anything.
set -eu

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.claude/hooks/lib.sh
. "$script_dir/lib.sh"

jq_bin="$(claude_find_jq)" || exit 0  # no jq → cannot emit context; skip

input=$(cat)
cmd=$(printf '%s' "$input" | "$jq_bin" -r '.tool_input.command // empty')

# Cheap pre-filter: this hook fires on EVERY Bash call, and the per-character
# tokenizer below is the expensive part. Any real `gh pr ...` invocation contains
# the substring `gh`, so skip the O(n) tokenizer for the ~all commands that don't.
# Behavior is unchanged — the tokenizer still decides every command that survives.
case "$cmd" in *gh*) ;; *) exit 0 ;; esac

# Only fire on an ACTUAL `gh pr merge` invocation. The shell-parser-aware matcher
# strips quotes/heredocs/comments and splits on ; | & before testing, so quoting
# the literal text 'gh pr merge' (a log grep, an echo, a `#` comment) no longer
# triggers a parent refresh.
if ! claude_command_invokes_gh_pr_merge "$cmd"; then
  exit 0
fi

# Run from a path inside the repo so `git worktree list` resolves the parent.
cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || true
# (1) fast-forward the parent's main; (2) promote the worktree's locally-built
# data/splits to the parent if this merge changed splits-affecting code.
main_status=$(claude_refresh_parent_main || true)
splits_status=$(claude_promote_worktree_splits || true)
status=$(printf '%s\n%s\n' "$main_status" "$splits_status" | sed '/^[[:space:]]*$/d')
[ -n "$status" ] || exit 0

# Surface the outcome to the turn that ran the merge.
"$jq_bin" -n --arg ctx "$status" '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: $ctx
  }
}'

exit 0
