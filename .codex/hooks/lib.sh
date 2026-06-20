#!/usr/bin/env bash
# Helpers for the Codex (.codex/) hooks.
#
# The provider-neutral core (gh-pr tokenizer, find_jq, main_worktree, abs_path,
# tool_command) lives once in scripts/agent-hooks-lib.sh (audit P4); this file
# sources it and re-exports those under the codex_* names the hooks/tests call,
# then defines the genuinely Codex-specific bits (apply_patch path parsing,
# CODEX_HOME worktree classification, the parent-housekeeping helpers).

_codex_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/agent-hooks-lib.sh
. "$_codex_lib_dir/../../scripts/agent-hooks-lib.sh"

# Re-export the shared core under the codex_* prefix. codex_hook_command is the
# Codex name for the shared tool_command extractor.
codex_find_jq() { agent_hooks_find_jq "$@"; }
codex_main_worktree() { agent_hooks_main_worktree "$@"; }
codex_abs_path() { agent_hooks_abs_path "$@"; }
codex_hook_command() { agent_hooks_tool_command "$@"; }
codex_is_env_assignment() { agent_hooks_is_env_assignment "$@"; }
codex_pr_subcommand_segment_matches() { agent_hooks_pr_subcommand_segment_matches "$@"; }
codex_pr_create_segment_matches() { agent_hooks_pr_create_segment_matches "$@"; }
codex_command_invokes_gh_pr_subcommand() { agent_hooks_command_invokes_gh_pr_subcommand "$@"; }
codex_command_invokes_gh_pr_create() { agent_hooks_command_invokes_gh_pr_create "$@"; }
codex_command_invokes_gh_pr_merge() { agent_hooks_command_invokes_gh_pr_merge "$@"; }

# --- Codex-specific helpers ---------------------------------------------------

codex_project_root() {
  local input="$1"
  local jq_bin="$2"
  local candidate="${CODEX_PROJECT_DIR:-${CLAUDE_PROJECT_DIR:-}}"

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

# Best-effort fast-forward of the main/parent checkout's `main` branch to
# origin/main. GUARDED so it never clobbers another agent's work: the parent can
# hold a `codex/*` branch with uncommitted WIP (AGENTS.md "Worktree workflow").
# Skips unless the parent is on `main` with a clean tree; uses `pull --ff-only`
# so it can never create a merge commit. Echoes ONE status line; always succeeds.
codex_refresh_parent_main() {
  local root="$1"
  local main_wt branch worktree_status short
  main_wt="$(codex_main_worktree "$root")"
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
# merge (parity twin of claude_promote_worktree_splits). data/splits is
# gitignored shared data, so this is independent of the parent's git branch.
# Acts ONLY when: $1 (worktree root) != parent, the worktree has its OWN
# data/splits (real dir, not the parent symlink) with all three parquets, and the
# merge touched splits-affecting code (src/scripts/scope_positions.py, pure
# stdlib). Copies only differing parquets; STDOUT line on copy, STDERR on skip.
codex_promote_worktree_splits() {
  local wt="$1"
  local parent wt_splits parent_splits f py changed positions copied=0
  [ -n "$wt" ] || {
    echo "splits promote: no worktree root" >&2
    return 0
  }
  parent="$(codex_main_worktree "$wt")"
  { [ -n "$parent" ] && [ -d "$parent" ]; } || {
    echo "splits promote: main checkout not found" >&2
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
  local direct=""
  if [ -n "$jq_bin" ]; then
    direct=$(printf '%s' "$input" | "$jq_bin" -r '.tool_input.file_path // .tool_input.notebook_path // empty' 2>/dev/null || true)
  elif command -v python3 >/dev/null 2>&1; then
    direct=$(printf '%s' "$input" | python3 -c 'import json, sys
try:
    ti = json.load(sys.stdin).get("tool_input") or {}
except Exception:
    sys.exit(0)
print(ti.get("file_path") or ti.get("notebook_path") or "")' 2>/dev/null || true)
  fi
  if [ -n "$direct" ]; then
    printf '%s\n' "$direct"
  fi
  codex_hook_command "$input" "$jq_bin" | codex_patch_paths
}

codex_json_context() {
  local event="$1"
  local context="$2"
  local jq_bin="${3:-}"
  if [ -z "$jq_bin" ]; then
    jq_bin="$(agent_hooks_find_jq)" || return 0
  fi
  "$jq_bin" -n --arg event "$event" --arg context "$context" '{
    hookSpecificOutput: {
      hookEventName: $event,
      additionalContext: $context
    }
  }'
}
