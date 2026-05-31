#!/usr/bin/env bash
# PostToolUse hook: run ruff format on Python files touched by Codex apply_patch.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.codex/hooks/lib.sh
. "$script_dir/lib.sh"

jq_bin="$(codex_find_jq)" || exit 0
input="$(cat)"
root="$(codex_project_root "$input" "$jq_bin")"

ruff=""
if [ -x "$root/.venv/bin/ruff" ]; then
  ruff="$root/.venv/bin/ruff"
else
  main_worktree="$(codex_main_worktree "$root")"
  if [ -n "$main_worktree" ] && [ -x "$main_worktree/.venv/bin/ruff" ]; then
    ruff="$main_worktree/.venv/bin/ruff"
  elif command -v ruff >/dev/null 2>&1; then
    ruff="ruff"
  fi
fi
[ -n "$ruff" ] || exit 0

seen=""
while IFS= read -r path; do
  [ -n "$path" ] || continue
  case "$path" in
    *.py) ;;
    *) continue ;;
  esac
  abs="$(codex_abs_path "$root" "$path")"
  case "$abs" in
    "$root"/*) ;;
    *) continue ;;
  esac
  [ -f "$abs" ] || continue
  case " $seen " in
    *" $abs "*) continue ;;
  esac
  seen="$seen $abs"
  "$ruff" format "$abs" >/dev/null 2>&1 || true
done <<EOF
$(codex_tool_paths "$input" "$jq_bin")
EOF

exit 0
