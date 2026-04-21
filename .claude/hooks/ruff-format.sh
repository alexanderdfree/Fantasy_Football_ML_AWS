#!/bin/bash
# PostToolUse hook: auto-format Python files after Edit/Write.
# Reads the Claude Code tool-call JSON from stdin and runs `ruff format` on the
# edited file if it is a .py file inside this project. Silent on success.
set -eu

input=$(cat)
file=$(printf '%s' "$input" | /usr/bin/jq -r '.tool_input.file_path // empty')

[ -z "$file" ] && exit 0
case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
case "$file" in
  "$CLAUDE_PROJECT_DIR"/*) ;;
  *) exit 0 ;;
esac

if [ -x "$CLAUDE_PROJECT_DIR/.venv/bin/ruff" ]; then
  ruff="$CLAUDE_PROJECT_DIR/.venv/bin/ruff"
elif command -v ruff >/dev/null 2>&1; then
  ruff="ruff"
else
  exit 0
fi

"$ruff" format "$file" >/dev/null 2>&1 || true
