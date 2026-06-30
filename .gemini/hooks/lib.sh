#!/usr/bin/env bash
# Helpers for the Gemini/Antigravity (.gemini/) hooks.
#
# The provider-neutral core (gh-pr tokenizer, find_jq, main_worktree, abs_path,
# tool_command) lives once in scripts/agent-hooks-lib.sh (audit P4); this file
# sources it and re-exports those under the gemini_* names the hooks/tests call,
# then defines the genuinely Gemini-specific bits. Antigravity passes
# `tool_name` + `tool_input`, with write_file/replace carrying `.tool_input.file_path`
# (same field as Claude) and run_shell_command carrying `.tool_input.command`
# (same field as Codex).

_gemini_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/agent-hooks-lib.sh
. "$_gemini_lib_dir/../../scripts/agent-hooks-lib.sh"

# Re-export the shared core under the gemini_* prefix.
gemini_find_jq() { agent_hooks_find_jq "$@"; }
gemini_main_worktree() { agent_hooks_main_worktree "$@"; }
gemini_abs_path() { agent_hooks_abs_path "$@"; }
gemini_tool_command() { agent_hooks_tool_command "$@"; }
gemini_is_env_assignment() { agent_hooks_is_env_assignment "$@"; }
gemini_pr_subcommand_segment_matches() { agent_hooks_pr_subcommand_segment_matches "$@"; }
gemini_command_invokes_gh_pr_subcommand() { agent_hooks_command_invokes_gh_pr_subcommand "$@"; }
gemini_command_invokes_gh_pr_create() { agent_hooks_command_invokes_gh_pr_create "$@"; }
gemini_command_invokes_gh_pr_merge() { agent_hooks_command_invokes_gh_pr_merge "$@"; }

# --- Gemini-specific stdin parsing -------------------------------------------

# Project root: prefer the runtime-provided absolute path (Antigravity exports
# GEMINI_PROJECT_DIR and a CLAUDE_PROJECT_DIR compat alias), else the stdin `cwd`,
# else $PWD; then resolve to the git toplevel.
gemini_project_root() {
  local input="$1"
  local jq_bin="$2"
  local candidate="${GEMINI_PROJECT_DIR:-${CLAUDE_PROJECT_DIR:-}}"

  if [ -z "$candidate" ] && [ -n "$jq_bin" ]; then
    candidate=$(printf '%s' "$input" | "$jq_bin" -r '.cwd // empty' 2>/dev/null || true)
  elif [ -z "$candidate" ] && command -v python3 >/dev/null 2>&1; then
    candidate=$(printf '%s' "$input" | python3 -c 'import json, sys
try:
    print(json.load(sys.stdin).get("cwd") or "")
except Exception:
    sys.exit(0)' 2>/dev/null || true)
  fi
  if [ -z "$candidate" ]; then
    candidate="$PWD"
  fi

  git -C "$candidate" rev-parse --show-toplevel 2>/dev/null || printf '%s\n' "$candidate"
}

# write_file/replace carry the target path in .tool_input.file_path (no apply_patch
# headers — that is Codex-only).
gemini_tool_paths() {
  local input="$1"
  local jq_bin="$2"
  if [ -n "$jq_bin" ]; then
    printf '%s' "$input" | "$jq_bin" -r '.tool_input.TargetFile // .tool_input.file_path // empty' 2>/dev/null || true
  elif command -v python3 >/dev/null 2>&1; then
    printf '%s' "$input" | python3 -c 'import json, sys
try:
    ti = json.load(sys.stdin).get("tool_input") or {}
except Exception:
    sys.exit(0)
print(ti.get("TargetFile") or ti.get("file_path") or "")' 2>/dev/null || true
  fi
}
