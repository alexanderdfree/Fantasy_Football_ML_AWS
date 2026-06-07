#!/usr/bin/env bash
# Shared helpers for the Claude Code hooks (pre-pr.sh / post-pr-create.sh).
#
# The `gh pr create` matcher is the parity twin of
# `.codex/hooks/lib.sh::codex_command_invokes_gh_pr_create`. Both hooks gate on
# whether a Bash command ACTUALLY invokes `gh pr create`; a flat `=~` regex over
# the raw command text (the previous implementation) also matched the literal
# token sequence inside quoted strings, heredocs, comments, and other commands'
# arguments — so grepping a log for "gh pr create", echoing release notes, or a
# `# gh pr create` comment would wedge the PreToolUse gate (exit 2) or fire the
# post-PR workflow with no PR opened. This tokenizer strips quotes/escapes/
# comments and splits on `; | & && || newline` before testing each segment's
# leading words, so only a real top-level invocation matches.

# `VAR=value` env-assignment prefix (skipped before the command word).
claude_is_env_assignment() {
  [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]
}

# Given a command segment's argv words, return 0 iff it invokes `gh pr create`
# (after skipping leading `VAR=val` assignments and an `env [opts]` wrapper).
claude_pr_create_segment_matches() {
  if [ "$#" -eq 0 ]; then
    return 1
  fi

  local -a words=("$@")
  local idx=0

  while [ "$idx" -lt "${#words[@]}" ] && claude_is_env_assignment "${words[$idx]}"; do
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

  [ "${words[$((idx + 1))]}" = "pr" ] && [ "${words[$((idx + 2))]}" = "create" ]
}

# Return 0 iff `cmd` actually invokes `gh pr create` at the top level. Strips
# single-/double-quoted strings, backslash escapes, and `#` comments, and splits
# on `; | & && || newline` so quoted/argument/heredoc occurrences do not match.
claude_command_invokes_gh_pr_create() {
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
        if [ "${#words[@]}" -gt 0 ] && claude_pr_create_segment_matches "${words[@]}"; then
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
        if [ "${#words[@]}" -gt 0 ] && claude_pr_create_segment_matches "${words[@]}"; then
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
        if [ "${#words[@]}" -gt 0 ] && claude_pr_create_segment_matches "${words[@]}"; then
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
        if [ "${#words[@]}" -gt 0 ] && claude_pr_create_segment_matches "${words[@]}"; then
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

  [ "${#words[@]}" -gt 0 ] && claude_pr_create_segment_matches "${words[@]}"
}
