#!/bin/bash
# PostToolUse hook: auto-format Python files after Edit/Write.
# Reads the Claude Code tool-call JSON from stdin and runs `ruff format` on the
# edited file if it is a .py file inside this project. Silent on success.
set -eu

# Resolve jq: prefer PATH, fall back to common absolute install locations so the
# hook works whether or not jq lives at /usr/bin (WSL/dev boxes differ from CI).
jq_bin=""
for _c in jq /usr/bin/jq /usr/local/bin/jq /opt/homebrew/bin/jq /home/linuxbrew/.linuxbrew/bin/jq; do
  if command -v "$_c" >/dev/null 2>&1; then jq_bin="$_c"; break; fi
done
[ -n "$jq_bin" ] || exit 0  # no jq → skip auto-format (non-critical)

input=$(cat)
file=$(printf '%s' "$input" | "$jq_bin" -r '.tool_input.file_path // empty')

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
