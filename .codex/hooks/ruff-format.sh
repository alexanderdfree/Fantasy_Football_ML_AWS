#!/usr/bin/env bash
# PostToolUse hook: run ruff format on Python files touched by Codex apply_patch.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.codex/hooks/lib.sh
. "$script_dir/lib.sh"

jq_bin="$(codex_find_jq)" || exit 0
input="$(cat)"
root="$(codex_project_root "$input" "$jq_bin")"

# Resolve ruff from a venv, then PATH. Probe the local .venv AND the MAIN
# worktree's .venv (worktrees don't symlink .venv), each in the Unix bin/ and the
# Windows Scripts/ layout — mirrors .claude/hooks/ruff-format.sh.
venv_roots=("$root/.venv")
main_worktree="$(codex_main_worktree "$root")"
if [ -n "$main_worktree" ] && [ "$main_worktree" != "$root" ]; then
  venv_roots+=("$main_worktree/.venv")
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
  # --no-cache: mirrors .claude/hooks/ruff-format.sh — the mtime+size cache can
  # false-skip a same-size rewrite landing in the same mtime tick.
  "$ruff" format --no-cache "$abs" >/dev/null 2>&1 || true
done <<EOF
$(codex_tool_paths "$input" "$jq_bin")
EOF

exit 0
