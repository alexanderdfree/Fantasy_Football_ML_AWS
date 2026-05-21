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

input=$(cat)
cmd=$(printf '%s' "$input" | /usr/bin/jq -r '.tool_input.command // empty')

# Match `gh pr create` at a word boundary; skip otherwise.
if ! [[ "$cmd" =~ (^|[[:space:]&|;\(])gh[[:space:]]+pr[[:space:]]+create([[:space:]]|$|[&|;\)]) ]]; then
  exit 0
fi

ctx="PR opened. Run this post-create workflow now, in order:

1. Rebase onto latest main: \`git fetch origin main --quiet && git rebase origin/main\`. If the rebase is a no-op (branch already up to date), skip the push. If it succeeds with changes, \`git push --force-with-lease\`. If it conflicts, run \`git rebase --abort\` and surface the conflict to the user — do not attempt unilateral resolution.

2. Invoke the /review skill on this PR. Resolve the PR number with \`gh pr view --json number,url\` on the current branch.

3. Categorize each /review finding by whether it requires architectural / design-level judgment:
   - Architectural / design-level (the finding asks you to rethink the approach, do a multi-file refactor outside the PR's scope, introduce a new abstraction, or otherwise raises a question of design judgment a senior engineer should weigh in on): list for the user, do NOT auto-apply, do NOT merge.
   - Everything else (nit, minor, style, naming, redundant code, formatting, dead import, local logic fix, security fix with a localized edit, missing-test addition, missing edge-case handler, doc/comment tweak): apply via Edit/Write.

4. If you applied any fixes, commit with subject \"review: address /review nits\" and \`git push\`.

5. If any architectural findings were surfaced in step 3: stop here. Summarize each one in one line for the user and DO NOT merge. The user decides whether to address them.

6. Otherwise, auto-merge:
   a. Resolve the PR number with \`gh pr view --json number\`.
   b. Wait for green CI: \`gh pr checks <N> --watch\`. If any check fails, surface the failure to the user — do not retry the merge or use \`--admin\`. (Exception: the documented \`Run Tests\` silent-stop anomaly — see CLAUDE.md's CI section; fall back to local \`pytest\` + manual \`gh pr merge --admin\` only if the user confirms.)
   c. Merge: \`gh pr merge <N> --squash\`. Do NOT use \`--delete-branch\` — it fails in worktrees (see CLAUDE.md's worktree section).
   d. Delete the remote branch: \`git push origin --delete <branch-name>\`. Resolve the branch name from \`git branch --show-current\` before the merge step in case the worktree HEAD changes.

7. End-of-turn summary: report what was applied (one line for the commit), what (if anything) was surfaced for user judgment, and whether the PR was merged.

Separately: if this session had a non-routine moment — user corrected your approach mid-flight, a CLAUDE.md stop-rule bit you, or something went unusually well because of a specific rule — invoke the post-session-critique skill before moving on, to capture the prompt lesson. Skip if the session was routine (do not run the skill just because this hook fired). See .claude/skills/post-session-critique/SKILL.md."

# Inject context via the PostToolUse hook protocol. Using --arg keeps the
# multi-line payload readable here and lets jq handle JSON escaping.
/usr/bin/jq -n --arg ctx "$ctx" '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: $ctx
  }
}'

exit 0
