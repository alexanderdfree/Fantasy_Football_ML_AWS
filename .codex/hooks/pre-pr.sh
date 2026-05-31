#!/usr/bin/env bash
# PreToolUse hook: Codex wrapper around the repo's deterministic pre-PR gate.
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
export CLAUDE_PROJECT_DIR="$root"

if [ -x "$root/.claude/hooks/pre-pr.sh" ]; then
  printf '%s' "$input" | "$root/.claude/hooks/pre-pr.sh"
  exit $?
fi

echo "Codex pre-pr hook could not find .claude/hooks/pre-pr.sh; run ruff, format check, pytest -m unit, and benchmark freshness manually." >&2
exit 2
