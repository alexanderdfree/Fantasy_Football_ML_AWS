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

# ---------------------------------------------------------------------------
# Remote (Claude Code on the web) venv bootstrap. Mirrors CI (tests.yml):
# Python 3.12 + `uv pip install -r requirements-dev.txt`, which pulls in
# requirements.txt (numpy/pandas/sklearn/lightgbm/boto3/…), the CPU torch wheel
# via that file's --extra-index-url, and pytest/ruff/optuna. Idempotent: a warm
# re-run (resume/compact, or a container restored from cache) audits in ~1s.
#
# Every install goes through the venv's OWN interpreter (`--python .venv/bin/python`),
# never a bare `pip`: `uv venv` ships no pip, so inside the activated venv `pip`
# resolved to the system /usr/bin/pip (Python 3.11) and the previous bootstrap
# aborted before exporting anything — sessions started with an empty .venv and
# no pandas/pytest/ruff on PATH.
# ---------------------------------------------------------------------------
cd "$repo_root"
venv="$repo_root/.venv"
py="$venv/bin/python"

# uv: fetches a Python 3.12 when the image lacks one and installs far faster than
# pip. The web image ships it in ~/.local/bin; self-heal if a future image doesn't.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[session-start] uv not found — installing via astral.sh"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Recreate a half-built venv (dir present, no interpreter) instead of trusting
# `-d .venv`; a venv with a working interpreter is kept and topped up in place.
if [ ! -x "$py" ]; then
  rm -rf "$venv"
  # --seed adds pip so an ad-hoc `pip install …` in-session lands in the venv
  # instead of silently targeting the system interpreter.
  uv venv --python 3.12 --seed "$venv"
fi
"$py" -m pip --version >/dev/null 2>&1 || uv pip install --python "$py" pip

# Mirrors tests.yml's workflow-level UV_INDEX_STRATEGY: the PyTorch CPU index
# (requirements-dev.txt's --extra-index-url) hosts older numpy/requests/etc.;
# without it uv stops at the first index that carries a package and fails to resolve.
export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
uv pip install --python "$py" -r requirements-dev.txt

# Persist for every later Bash call in this session (the harness sources this file).
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export VIRTUAL_ENV=$venv"
    echo "export PATH=$venv/bin:$HOME/.local/bin:\$PATH"
    echo "export PYTHONPATH=$repo_root"
  } >> "$CLAUDE_ENV_FILE"
fi

# Fail loudly (non-zero exit) if the venv still lacks the core stack.
"$py" - <<'PY'
import sys

import pandas
import torch

print(
    f"[session-start] venv ready: python {sys.version.split()[0]} "
    f"pandas {pandas.__version__} torch {torch.__version__}"
)
PY
"$venv/bin/pytest" --version >/dev/null
"$venv/bin/ruff" --version >/dev/null
echo "[session-start] uv $(uv --version 2>/dev/null | awk '{print $2}') at $(command -v uv)"
