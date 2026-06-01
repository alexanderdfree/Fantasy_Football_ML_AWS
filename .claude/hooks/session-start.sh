#!/bin/bash
set -euo pipefail

repo_root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

if [ -x "$repo_root/scripts/agent-memory-sync.sh" ]; then
  (cd "$repo_root" && bash scripts/agent-memory-sync.sh claude pull) || true
fi

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$repo_root"

if [ ! -d .venv ]; then
  python3.12 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-dev.txt

{
  echo "export VIRTUAL_ENV=$repo_root/.venv"
  echo "export PATH=$repo_root/.venv/bin:\$PATH"
  echo "export PYTHONPATH=$repo_root"
} >> "$CLAUDE_ENV_FILE"
