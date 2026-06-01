#!/usr/bin/env bash
# PostToolUse hook: after gh pr create, inject the Codex post-PR workflow.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.codex/hooks/lib.sh
. "$script_dir/lib.sh"

jq_bin="$(codex_find_jq)" || exit 0
input="$(cat)"
cmd="$(codex_hook_command "$input" "$jq_bin")"

if ! [[ "$cmd" =~ (^|[[:space:]&|;\(])gh[[:space:]]+pr[[:space:]]+create([[:space:]]|$|[&|;\)]) ]]; then
  exit 0
fi

root="$(codex_project_root "$input" "$jq_bin")"
cd "$root" 2>/dev/null || true
branch="$(git branch --show-current 2>/dev/null || true)"

ctx_head='PR opened. Run this Codex post-create workflow now, in order:

1. Rebase onto latest main: `git fetch origin main --quiet` then `git rebase origin/main`. If the rebase changes the branch, push with `git push --force-with-lease`. If it conflicts, run `git rebase --abort` and surface the conflict to the user.

2. Run a local Codex review against the branch, for example `codex review --base origin/main` from the repo root. Treat findings in code-review stance: apply localized nits, naming, formatting, dead imports, missing small tests, local logic fixes, and localized security fixes; surface architectural or design-level findings to the user instead of auto-applying them.

3. If you applied review fixes, commit with subject `review: address codex review nits` and push.

4. If any architectural or design-level findings remain, stop and list them for the user. Do not merge.'

if [[ "$branch" =~ ^audit-.*/tier- ]]; then
  step5='

5. Do not auto-merge this solve-issues PR. Wait for green CI with `gh pr checks <N> --watch`, show the final `gh pr diff <N>` and any Tier C benchmark deltas, then ask for explicit merge sign-off. Only after approval: `gh pr merge <N> --squash`, then `git push origin --delete <branch-name>`.'
else
  step5='

5. Otherwise, wait for green CI with `gh pr checks <N> --watch`. If checks fail, surface the failure. If they pass, merge with `gh pr merge <N> --squash`, then delete the remote branch with `git push origin --delete <branch-name>`. Do not use `--delete-branch` from a worktree.'
fi

ctx_tail='

6. End-of-turn summary: report review fixes applied, anything surfaced for user judgment, and whether the PR was merged.

Separately, if this session had a non-routine lesson, run `/prompts:post-session-critique` before finishing.'

codex_json_context "PostToolUse" "${ctx_head}${step5}${ctx_tail}" "$jq_bin"
exit 0
