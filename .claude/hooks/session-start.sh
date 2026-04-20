#!/bin/bash
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

if [ ! -d .venv ]; then
  python3.12 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

pip install --no-deps nfl_data_py==0.3.3

pip install -r requirements-dev.txt

{
  echo "export VIRTUAL_ENV=$CLAUDE_PROJECT_DIR/.venv"
  echo "export PATH=$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH"
  echo "export PYTHONPATH=$CLAUDE_PROJECT_DIR"
} >> "$CLAUDE_ENV_FILE"
