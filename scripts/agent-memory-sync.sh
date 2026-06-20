#!/usr/bin/env bash
# Sync agent auto-memory between machines via private S3.
#
# Remote layout is intentionally split by agent:
#   s3://$FF_MEMORY_S3_BUCKET/claude-memory/<repo>/memory/
#   s3://$FF_MEMORY_S3_BUCKET/codex-memory/<repo>/memories/
#
# Claude memory is project-scoped. Codex memory is global to CODEX_HOME, so this
# script syncs the markdown memory tree only, not Codex's SQLite runtime state.
set -euo pipefail

log() { echo "[memory-sync] $*" >&2; }

usage() {
  log "usage: $(basename "$0") {claude|codex|all} {pull|push|status|path|generate} [--prune] [--dry-run]"
}

agent="${1:-}"
cmd="${2:-}"
if [ -z "$agent" ] || [ -z "$cmd" ]; then
  usage
  exit 2
fi
shift 2

prune=0
dry=0
for arg in "$@"; do
  case "$arg" in
    --prune) prune=1 ;;
    --dry-run) dry=1 ;;
    *) log "unknown option: $arg"; usage; exit 2 ;;
  esac
done

case "$agent" in
  claude | codex | all) ;;
  *) log "unknown agent: $agent"; usage; exit 2 ;;
esac
case "$cmd" in
  pull | push | status | path | generate) ;;
  *) log "unknown command: $cmd"; usage; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/.." rev-parse --show-toplevel 2>/dev/null || (cd "$script_dir/.." && pwd))"

main_worktree() {
  git -C "$repo_root" worktree list --porcelain 2>/dev/null \
    | awk 'NR == 1 && /^worktree / { print substr($0, 10); exit }' \
    | tr -d '\r'
}

repo_id_from_remote() {
  local remote remote_base id
  remote="$(git -C "$repo_root" config --get remote.origin.url 2>/dev/null || true)"
  if [ -n "$remote" ]; then
    remote_base="${remote##*/}"
    id="${remote_base%.git}"
  else
    id=""
  fi
  [ -n "$id" ] || id="$(basename "$(main_worktree || true)")"
  [ -n "$id" ] || id="$(basename "$repo_root")"
  printf '%s\n' "$id"
}

claude_memory_dir() {
  local main slug
  main="$(main_worktree || true)"
  [ -n "$main" ] || main="$repo_root"
  slug="$(printf '%s' "$main" | sed 's#[/.]#-#g')"
  printf '%s\n' "${HOME}/.claude/projects/${slug}/memory"
}

codex_memory_dir() {
  local codex_home="${CODEX_HOME:-${HOME}/.codex}"
  printf '%s\n' "${codex_home}/memories"
}

remote_for_agent() {
  local one="$1"
  local repo_id prefix leaf
  repo_id="$(repo_id_from_remote)"
  case "$one" in
    claude)
      prefix="${FF_CLAUDE_MEMORY_S3_PREFIX:-${FF_MEMORY_S3_PREFIX:-claude-memory/${repo_id}}}"
      leaf="memory"
      ;;
    codex)
      prefix="${FF_CODEX_MEMORY_S3_PREFIX:-codex-memory/${repo_id}}"
      leaf="memories"
      ;;
    *) return 2 ;;
  esac
  prefix="${prefix%/}"
  printf 's3://%s/%s/%s\n' "$FF_MEMORY_S3_BUCKET" "$prefix" "$leaf"
}

preflight_s3() {
  command -v aws >/dev/null 2>&1 || {
    log "aws CLI not found - skipping memory sync (no-op)."
    exit 0
  }
  if [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ -z "${AWS_PROFILE:-}" ] && [ ! -f "${HOME}/.aws/credentials" ]; then
    log "no AWS credentials detected - skipping memory sync (no-op)."
    exit 0
  fi
}

sync_one() {
  local one="$1"
  local mem_dir remote desc region
  local -a flags
  local -a exclude_flags=()

  case "$one" in
    claude)
      mem_dir="$(claude_memory_dir)"
      # MEMORY.md is a GENERATED projection of the topic files (scripts/memory_index.py),
      # rebuilt locally each SessionStart. Excluding it from sync keeps it machine-local — never
      # shared mutable state — which is what makes the index non-racy: orphans came from
      # concurrent sessions overwriting a SYNCED MEMORY.md. Topic files still sync (additive).
      exclude_flags=(--exclude "MEMORY.md")
      ;;
    codex)
      mem_dir="$(codex_memory_dir)"
      # .git: Codex's SQLite/runtime state. *.DS_Store: macOS cruft that would
      # otherwise be pushed to S3 (it was, until #697 follow-up cleanup).
      exclude_flags=(--exclude ".git" --exclude ".git/*" --exclude "*.DS_Store")
      ;;
    *) return 2 ;;
  esac

  remote="$(remote_for_agent "$one")"
  region="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"

  desc=""
  [ "$dry" -eq 1 ] && desc="$desc (dry-run)"
  [ "$prune" -eq 1 ] && desc="$desc (prune)"

  flags=(--region "$region" --no-progress)
  [ "$dry" -eq 1 ] && flags+=(--dryrun)
  [ "$prune" -eq 1 ] && flags+=(--delete)

  case "$cmd" in
    pull)
      mkdir -p "$mem_dir"
      log "$one pull: $remote -> $mem_dir$desc"
      aws s3 sync "$remote" "$mem_dir" "${flags[@]}" ${exclude_flags[@]+"${exclude_flags[@]}"} --exact-timestamps \
        || { log "WARN: $one pull failed; local memory left intact."; return 0; }
      ;;
    push)
      [ -d "$mem_dir" ] || {
        log "$one: no local memory dir at $mem_dir - nothing to push (no-op)."
        return 0
      }
      log "$one push: $mem_dir -> $remote$desc"
      aws s3 sync "$mem_dir" "$remote" "${flags[@]}" ${exclude_flags[@]+"${exclude_flags[@]}"} \
        || { log "WARN: $one push failed; remote left intact."; return 0; }
      ;;
    status)
      mkdir -p "$mem_dir"
      log "$one status (dry-run, nothing is written):"
      log "  local : $mem_dir"
      log "  remote: $remote"
      aws s3 sync "$mem_dir" "$remote" --region "$region" --no-progress --dryrun ${exclude_flags[@]+"${exclude_flags[@]}"} 2>&1 \
        | sed "s/^/  $one would push> /" || true
      aws s3 sync "$remote" "$mem_dir" --region "$region" --no-progress --dryrun ${exclude_flags[@]+"${exclude_flags[@]}"} --exact-timestamps 2>&1 \
        | sed "s/^/  $one would pull> /" || true
      ;;
  esac
}

# `path`: print the resolved local memory dir(s) for the agent, then exit. Needs
# no S3/credentials, so it short-circuits before preflight. This is the single
# source of truth for the dir, consumed by .claude/hooks/session-start.sh's
# orphan-index check (so that check never re-derives the slug independently).
if [ "$cmd" = "path" ]; then
  case "$agent" in
    claude) claude_memory_dir ;;
    codex) codex_memory_dir ;;
    all)
      claude_memory_dir
      codex_memory_dir
      ;;
  esac
  exit 0
fi

# `generate`: rebuild Claude's MEMORY.md (the auto-loaded index) as a deterministic projection of
# the topic files via scripts/memory_index.py, so the index is a generated cache rather than
# shared mutable state. Claude-only (Codex has no MEMORY.md index). Needs no S3, so it
# short-circuits before preflight. Atomic write (temp + mv); fail-open (never breaks the hook).
if [ "$cmd" = "generate" ]; then
  if [ "$agent" != "claude" ]; then
    log "generate: only the claude index is generated (no-op for '$agent')."
    exit 0
  fi
  gen_py="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
  gen_script="$script_dir/memory_index.py"
  if [ -z "$gen_py" ] || [ ! -f "$gen_script" ]; then
    log "generate: python3 or memory_index.py missing; left MEMORY.md as-is (no-op)."
    exit 0
  fi
  gen_dir="$(claude_memory_dir)"
  mkdir -p "$gen_dir"
  gen_tmp="$gen_dir/.MEMORY.md.tmp.$$"
  if "$gen_py" "$gen_script" generate "$gen_dir" >"$gen_tmp" 2>"$gen_tmp.warn" && [ -s "$gen_tmp" ]; then
    mv -f "$gen_tmp" "$gen_dir/MEMORY.md"
    log "generate: rebuilt $gen_dir/MEMORY.md from topic files."
    [ -s "$gen_tmp.warn" ] && sed 's/^/  /' "$gen_tmp.warn" >&2
  else
    log "generate: index build failed/empty; left MEMORY.md as-is (no-op)."
  fi
  rm -f "$gen_tmp" "$gen_tmp.warn"
  exit 0
fi

: "${FF_MEMORY_S3_BUCKET:=ff-predictor-training}"
preflight_s3

if [ "$agent" = "all" ]; then
  sync_one claude
  sync_one codex
else
  sync_one "$agent"
fi
