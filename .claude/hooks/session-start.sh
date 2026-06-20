#!/bin/bash
set -euo pipefail

repo_root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

if [ -x "$repo_root/scripts/agent-memory-sync.sh" ]; then
  (cd "$repo_root" && bash scripts/agent-memory-sync.sh claude pull) || true
  # Rebuild MEMORY.md from the just-pulled topic files. The index is a GENERATED, machine-local
  # projection (excluded from sync) — that's what makes it non-racy. Per Claude Code's load order
  # (memory loads BEFORE SessionStart hooks) the rebuild is picked up NEXT session; combined with
  # exclude-from-sync it keeps the index complete + orphan-free without being shared mutable state.
  (cd "$repo_root" && bash scripts/agent-memory-sync.sh claude generate) || true
fi

# Auto-link the parent checkout's gitignored data/{raw,splits} into this worktree
# so the pre-PR `pytest -m unit` works without a slow first pull. No-op in the main
# checkout / remote single-clone sessions. Runs BEFORE the remote early-exit below
# so local worktree sessions (the case that needs it) are covered.
if [ -f "$repo_root/.claude/hooks/lib.sh" ]; then
  # shellcheck source=.claude/hooks/lib.sh
  . "$repo_root/.claude/hooks/lib.sh"
  claude_link_worktree_data "$repo_root" || true

  # Canary: after the regenerate above, every topic file should be indexed. If this still warns,
  # the generator no-op'd (e.g. python3 missing) and the index is a synced/stale copy — see
  # claude_list_unindexed_memories in lib.sh. Warn only; never auto-edit the index (that would
  # add another concurrent writer to the collision point).
  if [ -x "$repo_root/scripts/agent-memory-sync.sh" ]; then
    _ff_memdir="$(cd "$repo_root" && bash scripts/agent-memory-sync.sh claude path 2>/dev/null || true)"
    if [ -n "${_ff_memdir:-}" ]; then
      _ff_orphans="$(claude_list_unindexed_memories "$_ff_memdir" || true)"
      if [ -n "${_ff_orphans:-}" ]; then
        echo "[memory-sync] WARN: $(printf '%s\n' "$_ff_orphans" | grep -c .) memory file(s) present but NOT in MEMORY.md (add a one-line index entry so recall surfaces them):"
        printf '%s\n' "$_ff_orphans" | sed 's/^/  - /'
      fi
    fi
    unset _ff_memdir _ff_orphans
  fi
fi

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$repo_root"

if [ ! -d .venv ]; then
  # Prefer uv (it fetches the right interpreter); fall back so a missing
  # python3.12 binary doesn't abort the whole bootstrap under `set -euo pipefail`.
  # Matches SETUP.md's `uv venv --python 3.12`.
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
  else
    python3.12 -m venv .venv 2>/dev/null || python3 -m venv .venv
  fi
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-dev.txt

{
  echo "export VIRTUAL_ENV=$repo_root/.venv"
  echo "export PATH=$repo_root/.venv/bin:\$PATH"
  echo "export PYTHONPATH=$repo_root"
} >> "$CLAUDE_ENV_FILE"
