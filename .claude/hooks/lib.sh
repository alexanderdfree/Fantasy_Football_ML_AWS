#!/usr/bin/env bash
# Shared helpers for the Claude Code hooks (pre-pr.sh / post-pr-create.sh /
# post-pr-merge.sh).
#
# The `gh pr <subcommand>` matcher is the parity twin of
# `.codex/hooks/lib.sh::codex_command_invokes_gh_pr_*`. Both hooks gate on
# whether a Bash command ACTUALLY invokes `gh pr <subcommand>`; a flat `=~` regex
# over the raw command text (the previous implementation) also matched the literal
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

# Given an expected `gh pr` subcommand ($1) and a command segment's argv words
# ($2..), return 0 iff the segment invokes `gh pr <subcommand>` (after skipping
# leading `VAR=val` assignments and an `env [opts]` wrapper).
claude_pr_subcommand_segment_matches() {
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

  [ "${words[$((idx + 1))]}" = "pr" ] && [ "${words[$((idx + 2))]}" = "$subcmd" ]
}

# Back-compat wrapper: `gh pr create` segment matcher (named in tests/comments).
claude_pr_create_segment_matches() {
  claude_pr_subcommand_segment_matches create "$@"
}

# Return 0 iff `cmd` ($2) actually invokes `gh pr <subcommand>` ($1) at the top
# level. Strips single-/double-quoted strings, backslash escapes, and `#`
# comments, and splits on `; | & && || newline` so quoted/argument/heredoc
# occurrences do not match.
claude_command_invokes_gh_pr_subcommand() {
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
        if [ "${#words[@]}" -gt 0 ] && claude_pr_subcommand_segment_matches "$subcmd" "${words[@]}"; then
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
        if [ "${#words[@]}" -gt 0 ] && claude_pr_subcommand_segment_matches "$subcmd" "${words[@]}"; then
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
        if [ "${#words[@]}" -gt 0 ] && claude_pr_subcommand_segment_matches "$subcmd" "${words[@]}"; then
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
        if [ "${#words[@]}" -gt 0 ] && claude_pr_subcommand_segment_matches "$subcmd" "${words[@]}"; then
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

  [ "${#words[@]}" -gt 0 ] && claude_pr_subcommand_segment_matches "$subcmd" "${words[@]}"
}

# Public matchers: back-compat `create` + new `merge`.
claude_command_invokes_gh_pr_create() {
  claude_command_invokes_gh_pr_subcommand create "$1"
}

claude_command_invokes_gh_pr_merge() {
  claude_command_invokes_gh_pr_subcommand merge "$1"
}

# Resolve jq: prefer PATH, fall back to common absolute install locations so the
# hooks work whether or not jq lives at /usr/bin (WSL/dev boxes differ from CI).
# Prints the resolved binary on stdout; returns 1 if none found.
claude_find_jq() {
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
# lists the primary working tree first). Derived from the current CWD's repo.
claude_main_worktree() {
  git worktree list --porcelain 2>/dev/null \
    | awk 'NR == 1 && /^worktree / { print substr($0, 10); exit }' \
    | tr -d '\r'
}

# Best-effort fast-forward of the main/parent checkout's `main` branch to
# origin/main. GUARDED so it never clobbers another agent's work: the parent can
# hold a `codex/*` branch with uncommitted WIP (AGENTS.md "Worktree workflow").
# Skips unless the parent is on `main` with a clean tree; uses `pull --ff-only`
# so it can never create a merge commit. Echoes ONE status line; always succeeds.
claude_refresh_parent_main() {
  local main_wt branch worktree_status short
  main_wt="$(claude_main_worktree)"
  if [ -z "$main_wt" ] || [ ! -d "$main_wt" ]; then
    echo "parent refresh skipped: main checkout not found"
    return 0
  fi
  branch="$(git -C "$main_wt" branch --show-current 2>/dev/null || true)"
  if [ "$branch" != "main" ]; then
    echo "parent refresh skipped: $main_wt on '$branch' (not main)"
    return 0
  fi
  worktree_status="$(git -C "$main_wt" status --porcelain 2>/dev/null || true)"
  if [ -n "$worktree_status" ]; then
    echo "parent refresh skipped: $main_wt has uncommitted changes"
    return 0
  fi
  if git -C "$main_wt" pull --ff-only --quiet origin main >/dev/null 2>&1; then
    short="$(git -C "$main_wt" rev-parse --short HEAD 2>/dev/null || true)"
    echo "parent refresh: $main_wt fast-forwarded to origin/main ($short)"
  else
    echo "parent refresh skipped: ff-only pull failed (diverged or offline)"
  fi
  return 0
}
