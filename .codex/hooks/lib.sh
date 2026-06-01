#!/usr/bin/env bash

codex_find_jq() {
  local candidate
  for candidate in jq /usr/bin/jq /usr/local/bin/jq /opt/homebrew/bin/jq /home/linuxbrew/.linuxbrew/bin/jq; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

codex_project_root() {
  local input="$1"
  local jq_bin="$2"
  local candidate="${CODEX_PROJECT_DIR:-${CLAUDE_PROJECT_DIR:-}}"

  if [ -z "$candidate" ]; then
    candidate=$(printf '%s' "$input" | "$jq_bin" -r '.cwd // empty' 2>/dev/null || true)
  fi
  if [ -z "$candidate" ]; then
    candidate="$PWD"
  fi

  git -C "$candidate" rev-parse --show-toplevel 2>/dev/null || printf '%s\n' "$candidate"
}

codex_main_worktree() {
  local root="$1"
  git -C "$root" worktree list --porcelain 2>/dev/null \
    | awk 'NR == 1 && /^worktree / { print substr($0, 10); exit }' \
    | tr -d '\r'
}

codex_hook_command() {
  local input="$1"
  local jq_bin="$2"
  printf '%s' "$input" | "$jq_bin" -r '.tool_input.command // empty' 2>/dev/null || true
}

codex_patch_paths() {
  sed -n \
    -e 's/^\*\*\* Add File: //p' \
    -e 's/^\*\*\* Update File: //p' \
    -e 's/^\*\*\* Delete File: //p' \
    -e 's/^\*\*\* Move to: //p'
}

codex_tool_paths() {
  local input="$1"
  local jq_bin="$2"
  local direct
  direct=$(printf '%s' "$input" | "$jq_bin" -r '.tool_input.file_path // .tool_input.notebook_path // empty' 2>/dev/null || true)
  if [ -n "$direct" ]; then
    printf '%s\n' "$direct"
  fi
  codex_hook_command "$input" "$jq_bin" | codex_patch_paths
}

codex_abs_path() {
  local root="$1"
  local path="$2"
  case "$path" in
    /*) printf '%s\n' "$path" ;;
    ./*) printf '%s/%s\n' "$root" "${path#./}" ;;
    *) printf '%s/%s\n' "$root" "$path" ;;
  esac
}

codex_json_context() {
  local event="$1"
  local context="$2"
  local jq_bin="${3:-}"
  if [ -z "$jq_bin" ]; then
    jq_bin="$(codex_find_jq)" || return 0
  fi
  "$jq_bin" -n --arg event "$event" --arg context "$context" '{
    hookSpecificOutput: {
      hookEventName: $event,
      additionalContext: $context
    }
  }'
}
