#!/usr/bin/env bash
# PostToolUse hook: after gh pr create, inject the Codex post-PR workflow.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.codex/hooks/lib.sh
. "$script_dir/lib.sh"

jq_bin="$(codex_find_jq)" || exit 0
input="$(cat)"
cmd="$(codex_hook_command "$input" "$jq_bin")"

if ! codex_command_invokes_gh_pr_create "$cmd"; then
  exit 0
fi

ctx='PR opened. Continue with `/prompts:post-pr-followup` (template: `.codex/prompts/post-pr-followup.md`): rebase onto origin/main, run `scripts/codex-review-quiet.sh --base origin/main`, apply localized review fixes, wait for CI, then merge/delete the remote branch when allowed. The prompt preserves audit/tier explicit merge sign-off and the post-session-critique nudge.'

codex_json_context "PostToolUse" "$ctx" "$jq_bin"
exit 0
