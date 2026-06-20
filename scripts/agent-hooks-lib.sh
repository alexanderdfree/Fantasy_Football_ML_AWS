#!/usr/bin/env bash
# Shared, provider-neutral helpers for the agent hook libs.
#
# Single home for the functions that were verbatim-identical across
# .claude/hooks/lib.sh, .codex/hooks/lib.sh, and .gemini/hooks/lib.sh (audit P4,
# todo/cross-model-parity-audit.md). Each provider lib.sh sources this file and
# re-exports the canonical functions under its own prefix (claude_* / codex_* /
# gemini_*), so existing hooks and tests keep calling the prefixed names while the
# IMPLEMENTATION lives here once.
#
# Consolidated here (provably byte-identical before this change, pinned by
# tests/hooks/test_pr_tokenizer.py + tests/scripts/test_{claude,codex,gemini}_hooks.py):
#   - the gh-pr subcommand tokenizer (the safety-critical destructive-action gate),
#   - find_jq, main_worktree, abs_path, tool_command.
# Deliberately NOT consolidated (kept per-provider): refresh_parent_main /
# promote_worktree_splits (git-mutating, provider-divergent call sites + signatures,
# pinned by per-provider merge tests — a lower-value, higher-risk follow-up), and
# the genuinely provider-specific project_root / tool_paths / json_context.

# Resolve jq: prefer PATH, fall back to common absolute install locations so the
# hooks work whether or not jq lives at /usr/bin (WSL/dev boxes differ from CI).
agent_hooks_find_jq() {
  local candidate
  for candidate in jq /usr/bin/jq /usr/local/bin/jq /opt/homebrew/bin/jq /home/linuxbrew/.linuxbrew/bin/jq; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

# The main/parent checkout = the FIRST entry of `git worktree list` (git always
# lists the primary working tree first). $1 = a dir inside the repo (default CWD),
# unifying the Claude no-arg form and the Codex root-arg form.
agent_hooks_main_worktree() {
  local root="${1:-.}"
  git -C "$root" worktree list --porcelain 2>/dev/null \
    | awk 'NR == 1 && /^worktree / { print substr($0, 10); exit }' \
    | tr -d '\r'
}

# Resolve a tool path to an absolute path under the repo root ($1).
agent_hooks_abs_path() {
  local root="$1"
  local path="$2"
  case "$path" in
    /*) printf '%s\n' "$path" ;;
    ./*) printf '%s/%s\n' "$root" "${path#./}" ;;
    *) printf '%s/%s\n' "$root" "$path" ;;
  esac
}

# Extract the shell command from a run_shell_command tool call's hook JSON
# (.tool_input.command). jq when available; python3 fallback so jq-less boxes keep
# the command-gated hooks armed.
agent_hooks_tool_command() {
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

# --- gh pr subcommand tokenizer (the safety-critical destructive-action gate) ---

agent_hooks_is_env_assignment() {
  [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]
}

# Given an expected `gh pr` subcommand ($1) and a command segment's argv words
# ($2..), return 0 iff the segment invokes `gh pr <subcommand>` (after skipping
# leading `VAR=val` assignments and an `env [opts]` wrapper).
agent_hooks_pr_subcommand_segment_matches() {
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

  while [ "$idx" -lt "${#words[@]}" ] && agent_hooks_is_env_assignment "${words[$idx]}"; do
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

# Back-compat wrapper: `gh pr create` segment matcher (named in tests/comments).
agent_hooks_pr_create_segment_matches() {
  agent_hooks_pr_subcommand_segment_matches create "$@"
}

# Return 0 iff `cmd` ($2) actually invokes `gh pr <subcommand>` ($1) at the top
# level. Strips single-/double-quoted strings, backslash escapes, and `#`
# comments, and splits on `; | & && || newline` so quoted/argument/heredoc
# occurrences do not match.
agent_hooks_command_invokes_gh_pr_subcommand() {
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
        if [ "${#words[@]}" -gt 0 ] && agent_hooks_pr_subcommand_segment_matches "$subcmd" "${words[@]}"; then
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
        if [ "${#words[@]}" -gt 0 ] && agent_hooks_pr_subcommand_segment_matches "$subcmd" "${words[@]}"; then
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
        if [ "${#words[@]}" -gt 0 ] && agent_hooks_pr_subcommand_segment_matches "$subcmd" "${words[@]}"; then
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
        if [ "${#words[@]}" -gt 0 ] && agent_hooks_pr_subcommand_segment_matches "$subcmd" "${words[@]}"; then
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

  [ "${#words[@]}" -gt 0 ] && agent_hooks_pr_subcommand_segment_matches "$subcmd" "${words[@]}"
}

agent_hooks_command_invokes_gh_pr_create() {
  agent_hooks_command_invokes_gh_pr_subcommand create "$1"
}

agent_hooks_command_invokes_gh_pr_merge() {
  agent_hooks_command_invokes_gh_pr_subcommand merge "$1"
}
