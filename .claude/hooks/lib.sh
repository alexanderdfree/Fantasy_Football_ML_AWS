#!/usr/bin/env bash
# Helpers for the Claude Code (.claude/) hooks.
#
# The provider-neutral core (gh-pr tokenizer, find_jq, main_worktree) lives once
# in scripts/agent-hooks-lib.sh (audit P4); this file sources it and re-exports
# those under the claude_* names the hooks/tests call (the tokenizer is pinned by
# tests/hooks/test_pr_tokenizer.py), then defines the genuinely Claude-specific
# bits (worktree data-link, parent housekeeping, the memory orphan-index check).

_claude_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/agent-hooks-lib.sh
. "$_claude_lib_dir/../../scripts/agent-hooks-lib.sh"

# Re-export the shared core under the claude_* prefix.
claude_find_jq() { agent_hooks_find_jq "$@"; }
claude_main_worktree() { agent_hooks_main_worktree "$@"; }
claude_is_env_assignment() { agent_hooks_is_env_assignment "$@"; }
claude_pr_subcommand_segment_matches() { agent_hooks_pr_subcommand_segment_matches "$@"; }
claude_pr_create_segment_matches() { agent_hooks_pr_create_segment_matches "$@"; }
claude_command_invokes_gh_pr_subcommand() { agent_hooks_command_invokes_gh_pr_subcommand "$@"; }
claude_command_invokes_gh_pr_create() { agent_hooks_command_invokes_gh_pr_create "$@"; }
claude_command_invokes_gh_pr_merge() { agent_hooks_command_invokes_gh_pr_merge "$@"; }

# --- Claude-specific helpers --------------------------------------------------

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
