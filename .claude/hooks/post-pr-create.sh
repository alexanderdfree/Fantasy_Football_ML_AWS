#!/bin/bash
# PostToolUse hook: after `gh pr create`, inject two system reminders.
# (1) Post-PR-create review + auto-merge workflow — rebase onto latest
#     origin/main, invoke /review on the PR, auto-apply every finding
#     except architectural / design-level ones, push fixes, wait for green
#     CI, squash-merge, and delete the remote branch. Surface to user only
#     on architectural findings or CI failure.
# (2) post-session-critique nudge — captures *prompt* lessons (CLAUDE.md or
#     memory edits) the way TODO.md's Fixed archive captures *code* lessons.
# Both nudges are best-effort: the skills/workflow encode their own skip
# guards, so unconditional firing on `gh pr create` is fine.
set -eu

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.claude/hooks/lib.sh
. "$script_dir/lib.sh"

# Resolve jq: prefer PATH, fall back to common absolute install locations so the
# hook works whether or not jq lives at /usr/bin (WSL/dev boxes differ from CI).
jq_bin=""
for _c in jq /usr/bin/jq /usr/local/bin/jq /opt/homebrew/bin/jq /home/linuxbrew/.linuxbrew/bin/jq; do
  if command -v "$_c" >/dev/null 2>&1; then jq_bin="$_c"; break; fi
done
[ -n "$jq_bin" ] || exit 0  # no jq → cannot parse/emit; skip nudge injection

input=$(cat)
cmd=$(printf '%s' "$input" | "$jq_bin" -r '.tool_input.command // empty')

# Only fire on an ACTUAL `gh pr create` invocation. The shell-parser-aware
# matcher strips quotes/heredocs/comments and splits on ; | & before testing,
# so quoting the literal text 'gh pr create' (a log grep, echoed release notes,
# a `#` comment) no longer injects the post-PR workflow with no PR opened (#893).
if ! claude_command_invokes_gh_pr_create "$cmd"; then
  exit 0
fi

# Detect a solve-issues autonomous-fix PR by its staging branch (`audit-NNN/tier-X`,
# cut by the solve-issues skill). Those PRs must NOT auto-merge — they stop for
# explicit user merge sign-off (the user reviews the diff + Tier C benchmark deltas
# before anything lands on main). Every other PR keeps the auto-merge workflow.
cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || true
branch=$(git branch --show-current 2>/dev/null || true)

ctx_head="PR opened. Run this post-create workflow now, in order:

1. Rebase onto latest main: \`git fetch origin main --quiet && git rebase origin/main\`. If the rebase is a no-op (branch already up to date), skip the push. If it succeeds with changes, \`git push --force-with-lease\`. If it conflicts, run \`git rebase --abort\` and surface the conflict to the user — do not attempt unilateral resolution.

2. Invoke the /review skill on this PR. Resolve the PR number with \`gh pr view --json number,url\` on the current branch.

3. Categorize each /review finding by whether it requires architectural / design-level judgment:
   - Architectural / design-level (the finding asks you to rethink the approach, do a multi-file refactor outside the PR's scope, introduce a new abstraction, or otherwise raises a question of design judgment a senior engineer should weigh in on): list for the user, do NOT auto-apply, do NOT merge.
   - Everything else (nit, minor, style, naming, redundant code, formatting, dead import, local logic fix, security fix with a localized edit, missing-test addition, missing edge-case handler, doc/comment tweak): apply via Edit/Write.

4. If you applied any fixes, commit with subject \"review: address /review nits\" and \`git push\`.

5. If any architectural findings were surfaced in step 3: stop here. Summarize each one in one line for the user and DO NOT merge. The user decides whether to address them."

if [[ "$branch" =~ ^audit-.*/tier- ]]; then
  step6="

6. Do NOT auto-merge — this is a solve-issues autonomous-fix PR. After steps 1-5:
   a. Wait for green CI: \`gh pr checks <N> --watch\`. If any check fails, surface the failure to the user — do not retry or use \`--admin\`.
   b. Show the user the final diff (\`gh pr diff <N>\`) and — for Tier C — the benchmark deltas from the PR body, and ask for EXPLICIT merge sign-off (AskUserQuestion). Do NOT merge until the user approves.
   c. Only after the user approves: \`gh pr merge <N> --squash\`, then \`git push origin --delete <branch-name>\` (do NOT use \`--delete-branch\` — it fails in worktrees)."
else
  step6="

6. Otherwise, auto-merge:
   a. Resolve the PR number with \`gh pr view --json number\` and the branch name with \`git branch --show-current\`. Hold onto both before the merge step in case the worktree HEAD changes.
   b. Wait for green CI: \`gh pr checks <N> --watch\`. If any check fails, surface the failure to the user — do not retry the merge or use \`--admin\`. (Exception: the documented \`Run Tests\` silent-stop anomaly — see CLAUDE.md's CI section; fall back to local \`pytest\` + manual \`gh pr merge --admin\` only if the user confirms.)
   c. Merge: \`gh pr merge <N> --squash\`. Do NOT use \`--delete-branch\` — it fails in worktrees (see CLAUDE.md's worktree section).
   d. Delete the remote branch: \`git push origin --delete <branch-name>\`."
fi

ctx_tail="

7. End-of-turn summary: report what was applied (one line for the commit), what (if anything) was surfaced for user judgment, and whether the PR was merged.

Separately: if this session had a non-routine moment — user corrected your approach mid-flight, a CLAUDE.md stop-rule bit you, or something went unusually well because of a specific rule — invoke the post-session-critique skill before moving on, to capture the prompt lesson. Skip if the session was routine (do not run the skill just because this hook fired). See .claude/skills/post-session-critique/SKILL.md."

ctx="${ctx_head}${step6}${ctx_tail}"

# Inject context via the PostToolUse hook protocol. Using --arg keeps the
# multi-line payload readable here and lets jq handle JSON escaping.
"$jq_bin" -n --arg ctx "$ctx" '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: $ctx
  }
}'

exit 0
