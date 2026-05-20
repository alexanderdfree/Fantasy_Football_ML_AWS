#!/bin/bash
# PostToolUse hook: after `gh pr create`, inject a system reminder nudging
# Claude to consider invoking the post-session-critique skill. The skill
# captures *prompt* lessons (CLAUDE.md or memory edits) the way TODO.md's
# Fixed archive captures *code* lessons. Without this nudge, the skill is
# effectively manual — Claude doesn't reliably notice triggers mid-session.
# Spotify Honk Part 2: "After a session, the agent itself is in a surprisingly
# good position to tell you what was missing."
set -eu

input=$(cat)
cmd=$(printf '%s' "$input" | /usr/bin/jq -r '.tool_input.command // empty')

# Match `gh pr create` at a word boundary; skip otherwise.
if ! [[ "$cmd" =~ (^|[[:space:]&|;\(])gh[[:space:]]+pr[[:space:]]+create([[:space:]]|$|[&|;\)]) ]]; then
  exit 0
fi

# Inject context via the PostToolUse hook protocol. Claude sees this as a
# system reminder alongside the tool result. The skill itself encodes the
# "skip if routine" guard, so unconditional firing is fine.
/usr/bin/jq -n '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: "PR opened. If this session had a non-routine moment — user corrected your approach mid-flight, a CLAUDE.md stop-rule bit you, or something went unusually well because of a specific rule — invoke the post-session-critique skill before moving on, to capture the prompt lesson. Skip if the session was routine (do not run the skill just because this hook fired). See .claude/skills/post-session-critique/SKILL.md."
  }
}'

exit 0
