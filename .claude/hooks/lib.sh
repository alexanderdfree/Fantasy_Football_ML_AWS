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

# Link the parent checkout's gitignored data/{raw,splits} into a worktree so the
# pre-PR `pytest -m unit` (reads data/{raw,splits}/*.parquet) works without a slow
# first pull. Mirrors scripts/codex-fresh-worktree.sh's linker. Idempotent +
# fail-open; no-op in the main checkout, when the parent has no prebuilt data, or
# when the worktree already has its own real data dir (a locally-built splits).
# $1 = worktree root (defaults to CWD).
claude_link_worktree_data() {
  local wt parent src dst d linked=""
  wt="$(git -C "${1:-.}" rev-parse --show-toplevel 2>/dev/null || true)"
  [ -n "$wt" ] || return 0
  parent="$(cd "$wt" 2>/dev/null && claude_main_worktree)"
  { [ -n "$parent" ] && [ -d "$parent" ] && [ "$wt" != "$parent" ]; } || return 0
  for d in raw splits; do
    src="$parent/data/$d"
    dst="$wt/data/$d"
    [ -e "$src" ] || continue        # parent has nothing to link
    [ -e "$dst" ] && continue        # already present (real dir OR resolving symlink)
    [ -L "$dst" ] && rm -f "$dst"     # dangling symlink from a prior parent state
    mkdir -p "$wt/data"
    ln -s "$src" "$dst" 2>/dev/null && linked="$linked $d" || true
  done
  [ -n "$linked" ] && echo "worktree data: linked$linked from $parent"
  return 0
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

# Promote a worktree's locally-built data/splits to the parent/main checkout on
# merge, so the parent (and every worktree symlinked to it) gets the fresh,
# code-matching splits without a ~10 min rebuild or the ~12 min wait for
# refresh-splits.yml to upload them to S3. data/splits is gitignored shared data,
# so this is independent of the parent's git branch/cleanliness (unlike the main
# fast-forward above). Acts ONLY when:
#   - we're in a worktree (toplevel != parent), and
#   - the worktree has its OWN data/splits (a real dir, not the parent symlink),
#     with all three parquets — i.e. the dev rebuilt/pulled splits locally, and
#   - the merge touched splits-affecting code, per the canonical mapping in
#     src/scripts/scope_positions.py (pure-stdlib; runs on vanilla python3).
# Copies only the parquets that differ. Emits a status line to STDOUT only when
# it actually copies; benign skips go to STDERR (kept out of the injected note).
claude_promote_worktree_splits() {
  local parent wt wt_splits parent_splits f py changed positions copied=0
  parent="$(claude_main_worktree)"
  wt="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  { [ -n "$parent" ] && [ -d "$parent" ]; } || {
    echo "splits promote: main checkout not found" >&2
    return 0
  }
  [ -n "$wt" ] || {
    echo "splits promote: not in a git worktree" >&2
    return 0
  }
  [ "$wt" != "$parent" ] || {
    echo "splits promote: running in the main checkout, nothing to promote" >&2
    return 0
  }
  wt_splits="$wt/data/splits"
  parent_splits="$parent/data/splits"
  if [ -L "$wt_splits" ] || [ ! -d "$wt_splits" ]; then
    echo "splits promote: worktree has no local data/splits (shares the parent's)" >&2
    return 0
  fi
  for f in train val test; do
    [ -f "$wt_splits/$f.parquet" ] || {
      echo "splits promote: worktree data/splits missing $f.parquet" >&2
      return 0
    }
  done
  py="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
  [ -n "$py" ] || {
    echo "splits promote: python3 not found; cannot check scope_positions" >&2
    return 0
  }
  git -C "$wt" fetch origin main --quiet 2>/dev/null || true
  changed="$(git -C "$wt" diff --name-only origin/main~1 origin/main 2>/dev/null || true)"
  [ -n "$changed" ] || {
    echo "splits promote: could not resolve the merged commit's changed files" >&2
    return 0
  }
  positions="$(printf '%s\n' "$changed" | (cd "$wt" && "$py" -m src.scripts.scope_positions) 2>/dev/null || true)"
  if [ -z "$positions" ]; then
    echo "splits promote: merge did not touch splits-affecting code" >&2
    return 0
  fi
  for f in train val test; do
    if ! cmp -s "$wt_splits/$f.parquet" "$parent_splits/$f.parquet" 2>/dev/null; then
      mkdir -p "$parent_splits"
      if cp -f "$wt_splits/$f.parquet" "$parent_splits/$f.parquet"; then
        copied=$((copied + 1))
      fi
    fi
  done
  if [ "$copied" -gt 0 ]; then
    echo "splits promote: copied $copied parquet(s) from $wt_splits to $parent_splits (merge touched: $positions)"
  else
    echo "splits promote: parent splits already in sync with the worktree" >&2
  fi
  return 0
}

# List memory files present in the Claude memory dir ($1) but NOT referenced in
# its MEMORY.md index — the "orphan" signature of an index edit lost to the
# last-writer-wins S3 sync. MEMORY.md is rewritten wholesale by every session
# that records a memory, so two overlapping (often cross-platform) sessions each
# overwrite it from the same base: the later push wins and the earlier session's
# index line vanishes. The topic file itself survives (push is additive, no
# --delete), leaving it present-but-unindexed → recall never surfaces it.
#
# Pure + READ-ONLY: prints one basename per orphan to stdout, nothing when the
# index is complete or absent. session-start.sh only WARNS on the output; it
# deliberately does not auto-edit MEMORY.md, which would add yet another
# concurrent writer to the very file that is the collision point. $1 = memory dir.
claude_list_unindexed_memories() {
  local memdir="${1:-}" index f base
  [ -n "$memdir" ] && [ -f "$memdir/MEMORY.md" ] || return 0
  index="$memdir/MEMORY.md"
  for f in "$memdir"/*.md; do
    [ -e "$f" ] || continue
    base="$(basename "$f")"
    [ "$base" = "MEMORY.md" ] && continue
    # Match the exact markdown link target "](<base>)" (fixed-string) so a slug
    # that is a substring of another (foo.md vs foobar.md) is not a false match.
    grep -qF "]($base)" "$index" || printf '%s\n' "$base"
  done
}
