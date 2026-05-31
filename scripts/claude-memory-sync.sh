#!/usr/bin/env bash
# Sync Claude Code auto-memory (the memory/ subtree only) between machines via
# private S3. Memory CONTENT never enters git — only this script does.
#
# Why this exists: the project repo is public, so the per-project auto-memory
# (~/.claude/projects/<slug>/memory/ — the hand-written markdown facts the
# assistant recalls across sessions, plus the MEMORY.md index) cannot live in
# git, but it should follow the developer between the Mac and the WSL box. This
# syncs just that memory/ subtree through the existing private S3 bucket, under
# a MACHINE-INDEPENDENT key derived from the git remote — so the two boxes'
# divergent local project-dir slugs (-Users-alex-... vs -home-alex-...) converge
# on one S3 location.
#
# Usage:
#   scripts/claude-memory-sync.sh status          # dry-run both directions; changes nothing
#   scripts/claude-memory-sync.sh pull            # S3 -> local   (run at the start of a work block)
#   scripts/claude-memory-sync.sh push            # local -> S3   (run at the end)
#   scripts/claude-memory-sync.sh push --prune    # mirror: also delete S3 files removed locally
#   scripts/claude-memory-sync.sh pull --dry-run  # preview a pull without writing
#
# Env overrides:
#   FF_MEMORY_S3_BUCKET   default: ff-predictor-training
#   FF_MEMORY_S3_PREFIX   default: claude-memory/<repo-id from git remote>
#   AWS_REGION            default: AWS_DEFAULT_REGION or us-east-1
#
# Safety / design:
#   * Scoped to the memory/ subtree only, so sibling session transcripts
#     (*.jsonl) and subagent metadata one level up CANNOT be uploaded.
#   * Additive by default (no --delete): a memory created on the other box is
#     never silently dropped. --prune opts into mirror-delete (the bucket has
#     versioning enabled, which is the recovery net for a bad prune).
#   * Clean no-op (exit 0) when the aws CLI or credentials are absent, so it is
#     safe to wire into a SessionStart/Stop hook. Mirrors the best-effort,
#     env-gated contract of src/benchmarking/benchmark.py::_maybe_upload_to_s3.
set -euo pipefail

log() { echo "[memory-sync] $*" >&2; }

# --- args ---------------------------------------------------------------------
cmd="${1:-}"
prune=0
dry=0
for a in "${@:2}"; do
  case "$a" in
    --prune) prune=1 ;;
    --dry-run) dry=1 ;;
    *) log "unknown option: $a"; exit 2 ;;
  esac
done
case "$cmd" in
  pull | push | status) ;;
  *) log "usage: $(basename "$0") {pull|push|status} [--prune] [--dry-run]"; exit 2 ;;
esac

# --- local memory dir (machine-specific; keyed to the MAIN project root) ------
# `git worktree list`'s first entry is always the main worktree; `git rev-parse
# --show-toplevel` would return the CURRENT worktree, whose slug has no memory
# dir (auto-memory is keyed to the main project root, even from a worktree).
main_root="$(git worktree list --porcelain 2>/dev/null \
  | awk 'NR==1 && /^worktree /{print substr($0,10); exit}')" || true
[ -n "$main_root" ] || main_root="$PWD"
# Claude names the per-project dir by replacing both '/' and '.' in the absolute
# path with '-' (so /Users/alex/compsci372/Final-Project ->
# -Users-alex-compsci372-Final-Project).
slug="$(printf '%s' "$main_root" | sed 's#[/.]#-#g')"
MEM_DIR="${HOME}/.claude/projects/${slug}/memory"

# --- machine-INDEPENDENT remote key (identical on every box for this repo) ----
repo_id="$(basename -s .git "$(git config --get remote.origin.url 2>/dev/null)" 2>/dev/null)" || true
[ -n "$repo_id" ] || repo_id="$(basename "$main_root")"
: "${FF_MEMORY_S3_BUCKET:=ff-predictor-training}"
: "${FF_MEMORY_S3_PREFIX:=claude-memory/${repo_id}}"
region="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
REMOTE="s3://${FF_MEMORY_S3_BUCKET}/${FF_MEMORY_S3_PREFIX}/memory"

# --- preflight: clean no-op when we can't reach S3 (keeps a hook non-fatal) ---
command -v aws >/dev/null 2>&1 || { log "aws CLI not found — skipping memory sync (no-op)."; exit 0; }
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ -z "${AWS_PROFILE:-}" ] && [ ! -f "${HOME}/.aws/credentials" ]; then
  log "no AWS credentials detected — skipping memory sync (no-op)."
  exit 0
fi

# --- sync ---------------------------------------------------------------------
desc=""
[ "$dry" -eq 1 ] && desc="$desc (dry-run)"
[ "$prune" -eq 1 ] && desc="$desc (prune)"

flags=(--region "$region" --no-progress)
[ "$dry" -eq 1 ] && flags+=(--dryrun)
[ "$prune" -eq 1 ] && flags+=(--delete)

case "$cmd" in
  pull)
    mkdir -p "$MEM_DIR"
    log "pull: $REMOTE -> $MEM_DIR$desc"
    aws s3 sync "$REMOTE" "$MEM_DIR" "${flags[@]}" --exact-timestamps \
      || { log "WARN: pull failed; local memory left intact."; exit 0; }
    ;;
  push)
    [ -d "$MEM_DIR" ] || { log "no local memory dir at $MEM_DIR — nothing to push (no-op)."; exit 0; }
    log "push: $MEM_DIR -> $REMOTE$desc"
    aws s3 sync "$MEM_DIR" "$REMOTE" "${flags[@]}" \
      || { log "WARN: push failed; remote left intact."; exit 0; }
    ;;
  status)
    mkdir -p "$MEM_DIR"
    log "status (dry-run, nothing is written):"
    log "  local : $MEM_DIR"
    log "  remote: $REMOTE"
    aws s3 sync "$MEM_DIR" "$REMOTE" --region "$region" --no-progress --dryrun 2>&1 \
      | sed 's/^/  would push> /' || true
    aws s3 sync "$REMOTE" "$MEM_DIR" --region "$region" --no-progress --dryrun --exact-timestamps 2>&1 \
      | sed 's/^/  would pull> /' || true
    ;;
esac
