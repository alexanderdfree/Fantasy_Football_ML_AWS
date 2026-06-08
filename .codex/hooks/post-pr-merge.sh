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

if ! codex_command_invokes_gh_pr_merge "$cmd"; then
  exit 0
fi

root="$(codex_project_root "$input" "$jq_bin")"
status="$(codex_refresh_parent_main "$root")"
[ -n "$status" ] || exit 0

codex_json_context "PostToolUse" "$status" "$jq_bin"
exit 0
