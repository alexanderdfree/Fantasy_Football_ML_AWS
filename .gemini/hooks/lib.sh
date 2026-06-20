#!/usr/bin/env bash
# Shared helpers for the Gemini/Antigravity (.gemini/) hooks.
#
# Parity twin of .codex/hooks/lib.sh and .claude/hooks (see
# todo/cross-model-parity-audit.md). Antigravity (`agy`) is the Gemini-CLI-lineage
# local runtime; its hooks receive JSON on stdin with `tool_name` + `tool_input`
# and the `$GEMINI_PROJECT_DIR` env var. write_file/replace carry the target in
# `.tool_input.file_path` (same field as Claude) and run_shell_command carries the
# command in `.tool_input.command` (same field as Codex), so the extractors below
# are the Codex ones minus the Codex-only apply_patch header parsing.
#
# The gh-pr tokenizer (gemini_command_invokes_gh_pr_create and friends) is a
# verbatim copy of the Codex matcher and is safety-critical: it gates the pre-PR
# deterministic check. It is pinned by tests/scripts/test_gemini_hooks.py — keep
# the three providers' tokenizers in sync (P4 of the audit consolidates them).

gemini_find_jq() {
  local candidate
  for candidate in jq /usr/bin/jq /usr/local/bin/jq /opt/homebrew/bin/jq /home/linuxbrew/.linuxbrew/bin/jq; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

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

gemini_main_worktree() {
  local root="$1"
  git -C "$root" worktree list --porcelain 2>/dev/null \
    | awk 'NR == 1 && /^worktree / { print substr($0, 10); exit }' \
    | tr -d '\r'
}

# run_shell_command carries the command string in .tool_input.command.
gemini_tool_command() {
  local input="$1"
  local jq_bin="$2"
  if [ -n "$jq_bin" ]; then
    printf '%s' "$input" | "$jq_bin" -r '.tool_input.command // empty' 2>/dev/null || true
  elif command -v python3 >/dev/null 2>&1; then
    printf '%s' "$input" | python3 -c 'import json, sys
try:
    ti = json.load(sys.stdin).get("tool_input") or {}
except Exception:
    sys.exit(0)
print(ti.get("command") or "")' 2>/dev/null || true
  fi
}

# write_file/replace carry the target path in .tool_input.file_path.
gemini_tool_paths() {
  local input="$1"
  local jq_bin="$2"
  if [ -n "$jq_bin" ]; then
    printf '%s' "$input" | "$jq_bin" -r '.tool_input.file_path // empty' 2>/dev/null || true
  elif command -v python3 >/dev/null 2>&1; then
    printf '%s' "$input" | python3 -c 'import json, sys
try:
    ti = json.load(sys.stdin).get("tool_input") or {}
except Exception:
    sys.exit(0)
print(ti.get("file_path") or "")' 2>/dev/null || true
  fi
}

gemini_abs_path() {
  local root="$1"
  local path="$2"
  case "$path" in
    /*) printf '%s\n' "$path" ;;
    ./*) printf '%s/%s\n' "$root" "${path#./}" ;;
    *) printf '%s/%s\n' "$root" "$path" ;;
  esac
}

# --- gh pr subcommand tokenizer (verbatim parity with .codex/hooks/lib.sh) -----

gemini_is_env_assignment() {
  [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]
}

gemini_pr_subcommand_segment_matches() {
  if [ "$#" -lt 1 ]; then
    return 1
  fi
  local subcmd="$1"
  shift

  if [ "$#" -eq 0 ]; then
    return 1
  fi

  local -a words=("$@")
  local idx=0

  while [ "$idx" -lt "${#words[@]}" ] && gemini_is_env_assignment "${words[$idx]}"; do
    idx=$((idx + 1))
  done

  if [ "$idx" -lt "${#words[@]}" ]; then
    case "${words[$idx]}" in
      env | */env)
        idx=$((idx + 1))
        while [ "$idx" -lt "${#words[@]}" ]; do
          case "${words[$idx]}" in
            -u | --unset | -C | --chdir | -S | --split-string)
              idx=$((idx + 2))
              ;;
            -*)
              idx=$((idx + 1))
              ;;
            *=*)
              idx=$((idx + 1))
              ;;
            *)
              break
              ;;
          esac
        done
        ;;
    esac
  fi

  if [ $((idx + 2)) -ge "${#words[@]}" ]; then
    return 1
  fi

  case "${words[$idx]}" in
    gh | */gh) ;;
    *) return 1 ;;
  esac

  [ "${words[$((idx + 1))]}" = "pr" ] && [ "${words[$((idx + 2))]}" = "$subcmd" ]
}

gemini_command_invokes_gh_pr_subcommand() {
  local subcmd="$1"
  local cmd="$2"
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
        if [ "${#words[@]}" -gt 0 ] && gemini_pr_subcommand_segment_matches "$subcmd" "${words[@]}"; then
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
        if [ "${#words[@]}" -gt 0 ] && gemini_pr_subcommand_segment_matches "$subcmd" "${words[@]}"; then
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
        if [ "${#words[@]}" -gt 0 ] && gemini_pr_subcommand_segment_matches "$subcmd" "${words[@]}"; then
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
        if [ "${#words[@]}" -gt 0 ] && gemini_pr_subcommand_segment_matches "$subcmd" "${words[@]}"; then
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

  [ "${#words[@]}" -gt 0 ] && gemini_pr_subcommand_segment_matches "$subcmd" "${words[@]}"
}

gemini_command_invokes_gh_pr_create() {
  gemini_command_invokes_gh_pr_subcommand create "$1"
}
