#!/usr/bin/env bash
# BeforeTool hook (matcher: run_shell_command): Gemini/Antigravity wrapper around
# the repo's single-source deterministic pre-PR gate. Parity twin of
# .codex/hooks/pre-pr.sh — it detects a top-level `gh pr create`, then delegates
# to .claude/hooks/pre-pr.sh (ruff check + format check + pytest -m unit +
# benchmark freshness). A gate failure exits 2, which Antigravity reads as a tool
# block with the gate's stderr as the reason.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.gemini/hooks/lib.sh
. "$script_dir/lib.sh"

jq_bin="$(gemini_find_jq)" || exit 0
input="$(cat)"
cmd="$(gemini_tool_command "$input" "$jq_bin")"

if ! gemini_command_invokes_gh_pr_create "$cmd"; then
  exit 0
fi

root="$(gemini_project_root "$input" "$jq_bin")"
export CLAUDE_PROJECT_DIR="$root"
export GEMINI_PRE_PR_WRAPPER=1

if [ -x "$root/.claude/hooks/pre-pr.sh" ]; then
  printf '%s' "$input" \
    | "$jq_bin" '.tool_input.command = "gh pr create"' \
    | "$root/.claude/hooks/pre-pr.sh"
  exit $?
fi

echo "Gemini pre-pr hook could not find .claude/hooks/pre-pr.sh; run ruff, format check, pytest -m unit, and benchmark freshness manually." >&2
exit 2
