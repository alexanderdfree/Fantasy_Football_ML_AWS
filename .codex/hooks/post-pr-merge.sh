#!/usr/bin/env bash
# PostToolUse hook: after `gh pr merge`, fast-forward the main/parent checkout's
# `main` to origin/main (the squash-merge just landed). Parity twin of
# .claude/hooks/post-pr-merge.sh. GUARDED in codex_refresh_parent_main: skips a
# non-main / dirty parent and uses `pull --ff-only`, so it never clobbers work.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.codex/hooks/lib.sh
. "$script_dir/lib.sh"

jq_bin="$(codex_find_jq)" || exit 0
input="$(cat)"
cmd="$(codex_hook_command "$input" "$jq_bin")"

# Cheap pre-filter: this hook fires on every Bash call; any real `gh pr ...`
# invocation contains `gh`, so skip the O(n) tokenizer for the ~all commands that
# don't. Behavior-preserving (parity with .claude/hooks/post-pr-merge.sh).
case "$cmd" in *gh*) ;; *) exit 0 ;; esac

if ! codex_command_invokes_gh_pr_merge "$cmd"; then
  exit 0
fi

root="$(codex_project_root "$input" "$jq_bin")"
# (1) fast-forward the parent's main; (2) promote the worktree's locally-built
# data/splits to the parent if this merge changed splits-affecting code.
main_status="$(codex_refresh_parent_main "$root")"
splits_status="$(codex_promote_worktree_splits "$root")"
status="$(printf '%s\n%s\n' "$main_status" "$splits_status" | sed '/^[[:space:]]*$/d')"
[ -n "$status" ] || exit 0

codex_json_context "PostToolUse" "$status" "$jq_bin"
exit 0
