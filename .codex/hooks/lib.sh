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

codex_worktrees_dir() {
  printf '%s/worktrees\n' "${CODEX_HOME:-$HOME/.codex}"
}

codex_is_clean_codex_worktree() {
  local root="$1"
  local main_worktree="${2:-}"
  local worktrees_dir repo_name

  [ -n "$main_worktree" ] || return 1
  [ "$root" != "$main_worktree" ] || return 1

  worktrees_dir="$(codex_worktrees_dir)"
  repo_name="$(basename "$main_worktree")"
  case "$root" in
    "$worktrees_dir"/*/"$repo_name") ;;
    *) return 1 ;;
  esac

  [ -z "$(git -C "$root" status --porcelain 2>/dev/null)" ]
}

codex_hook_command() {
  local input="$1"
  local jq_bin="$2"
  printf '%s' "$input" | "$jq_bin" -r '.tool_input.command // empty' 2>/dev/null || true
}

codex_is_env_assignment() {
  [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]
}

codex_pr_create_segment_matches() {
  if [ "$#" -eq 0 ]; then
    return 1
  fi

  local -a words=("$@")
  local idx=0

  while [ "$idx" -lt "${#words[@]}" ] && codex_is_env_assignment "${words[$idx]}"; do
    idx=$((idx + 1))
  done

  if [ $((idx + 2)) -ge "${#words[@]}" ]; then
    return 1
  fi

  case "${words[$idx]}" in
    gh | */gh) ;;
    *) return 1 ;;
  esac

  [ "${words[$((idx + 1))]}" = "pr" ] && [ "${words[$((idx + 2))]}" = "create" ]
}

codex_command_invokes_gh_pr_create() {
  local cmd="$1"
  local ch next quote="" token=""
  local escaped=0
  local in_comment=0
  local -a words=()
  local i

  for ((i = 0; i < ${#cmd}; i++)); do
    ch="${cmd:i:1}"

    if [ "$in_comment" -eq 1 ]; then
      if [ "$ch" = $'\n' ]; then
        in_comment=0
        if [ "${#words[@]}" -gt 0 ] && codex_pr_create_segment_matches "${words[@]}"; then
          return 0
        fi
        words=()
      fi
      continue
    fi

    if [ "$escaped" -eq 1 ]; then
      token+="$ch"
      escaped=0
      continue
    fi

    if [ -n "$quote" ]; then
      if [ "$ch" = "$quote" ]; then
        quote=""
      elif [ "$quote" = '"' ] && [ "$ch" = "\\" ]; then
        escaped=1
      else
        token+="$ch"
      fi
      continue
    fi

    case "$ch" in
      "\\")
        escaped=1
        ;;
      "'" | '"')
        quote="$ch"
        ;;
      "#")
        if [ -z "$token" ]; then
          in_comment=1
        else
          token+="$ch"
        fi
        ;;
      $'\n')
        if [ -n "$token" ]; then
          words+=("$token")
          token=""
        fi
        if [ "${#words[@]}" -gt 0 ] && codex_pr_create_segment_matches "${words[@]}"; then
          return 0
        fi
        words=()
        ;;
      " " | $'\t' | $'\r')
        if [ -n "$token" ]; then
          words+=("$token")
          token=""
        fi
        ;;
      ";" | "|")
        if [ -n "$token" ]; then
          words+=("$token")
          token=""
        fi
        if [ "${#words[@]}" -gt 0 ] && codex_pr_create_segment_matches "${words[@]}"; then
          return 0
        fi
        words=()
        if [ "$ch" = "|" ]; then
          next="${cmd:$((i + 1)):1}"
          if [ "$next" = "|" ]; then
            i=$((i + 1))
          fi
        fi
        ;;
      "&")
        if [ -n "$token" ]; then
          words+=("$token")
          token=""
        fi
        if [ "${#words[@]}" -gt 0 ] && codex_pr_create_segment_matches "${words[@]}"; then
          return 0
        fi
        words=()
        next="${cmd:$((i + 1)):1}"
        if [ "$next" = "&" ]; then
          i=$((i + 1))
        fi
        ;;
      *)
        token+="$ch"
        ;;
    esac
  done

  if [ -n "$token" ]; then
    words+=("$token")
  fi

  [ "${#words[@]}" -gt 0 ] && codex_pr_create_segment_matches "${words[@]}"
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
