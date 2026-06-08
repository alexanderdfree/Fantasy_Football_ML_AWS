#!/usr/bin/env bash
# SessionStart hook: add Codex-specific setup reminders. Unlike Claude's remote
# SessionStart hook, Codex hooks cannot persist shell exports into later tool
# calls, so this is context-only.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.codex/hooks/lib.sh
. "$script_dir/lib.sh"

jq_bin="$(codex_find_jq)" || exit 0
input="$(cat)"
root="$(codex_project_root "$input" "$jq_bin")"

if [ -x "$root/scripts/agent-memory-sync.sh" ]; then
  (cd "$root" && bash scripts/agent-memory-sync.sh codex pull) || true
fi

context="Final-Project Codex context: read AGENTS.md before non-trivial edits. This repo has tracked Codex hooks under .codex/hooks; review/trust them with /hooks after changes. Custom prompt wrappers live in .codex/prompts, shared cross-agent workflow behavior lives in agent-workflows, and Codex only loads user-home prompts, so run scripts/bootstrap-codex-local.sh and restart Codex when those templates change."
main_worktree="$(codex_main_worktree "$root")"

if [ ! -d "$root/.venv" ]; then
  if [ -n "$main_worktree" ] && [ -d "$main_worktree/.venv" ]; then
    context="$context The current worktree has no .venv; the main worktree has one, and the pre-PR hook will probe it for ruff/pytest."
  else
    context="$context No .venv was found in this worktree or the main worktree; use SETUP.md before running the full local gates."
  fi
fi

if ! codex_is_clean_codex_worktree "$root" "$main_worktree"; then
  context="$context This session is not running from a clean Codex worktree; SessionStart cannot move an already-running session. Start future sessions with scripts/codex-fresh-worktree.sh to reuse or create a clean Codex worktree."
fi

codex_json_context "SessionStart" "$context" "$jq_bin"
