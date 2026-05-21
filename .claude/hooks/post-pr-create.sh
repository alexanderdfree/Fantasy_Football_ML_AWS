#!/bin/bash
# PostToolUse hook: after `gh pr create`, inject two system reminders.
# (1) Post-PR-create review workflow — rebase onto latest origin/main, invoke
#     /review on the PR, auto-apply nit-level findings, surface major findings
#     for user judgment, push fixes.
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

3. Categorize each /review finding:
   - Nit / minor / style (typo, naming, redundant code, formatting, dead import): apply via Edit/Write.
   - Major / blocker / behavioral (logic change, architectural concern, security flag, missing test coverage): list for the user, do not auto-apply.

4. If you applied any nit fixes, commit with subject \"review: address /review nits\" and \`git push\`.

5. End-of-turn summary: list non-nit findings (for user judgment) + one-line summary of nit fixes applied.

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
