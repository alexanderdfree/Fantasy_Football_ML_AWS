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

# Resolve ruff from a venv, then PATH. Worktrees usually symlink data/ but NOT
# .venv (symlinking it breaks sys.path — AGENTS.md), so also probe the MAIN
# worktree's .venv (git lists it first). Otherwise a worktree with no local .venv
# and no ruff on PATH silently skips formatting, and pre-pr.sh's
# `ruff format --check` fails later. Mirrors the venv resolution in pre-pr.sh
# (incl. the Windows Scripts/ layout).
venv_roots=("$CLAUDE_PROJECT_DIR/.venv")
main_wt=$(git -C "$CLAUDE_PROJECT_DIR" worktree list --porcelain 2>/dev/null \
  | awk 'NR==1 && /^worktree /{print substr($0, 10); exit}' | tr -d '\r')
if [ -n "$main_wt" ] && [ -d "$main_wt/.venv" ]; then
  venv_roots+=("$main_wt/.venv")
fi
ruff=""
for vr in "${venv_roots[@]}"; do
  if [ -x "$vr/bin/ruff" ]; then ruff="$vr/bin/ruff"; break; fi
  if [ -x "$vr/Scripts/ruff.exe" ]; then ruff="$vr/Scripts/ruff.exe"; break; fi
done
if [ -z "$ruff" ] && command -v ruff >/dev/null 2>&1; then
  ruff="ruff"
fi
[ -n "$ruff" ] || exit 0

"$ruff" format "$file" >/dev/null 2>&1 || true
