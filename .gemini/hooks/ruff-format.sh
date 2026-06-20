#!/usr/bin/env bash
# AfterTool hook (matcher: write_file|replace): run ruff format on Python files
# Gemini/Antigravity just wrote. Parity twin of .claude/hooks/ruff-format.sh and
# the Codex one. Non-blocking (AfterTool) and fail-open.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.gemini/hooks/lib.sh
. "$script_dir/lib.sh"

jq_bin="$(gemini_find_jq || true)"
input="$(cat)"
root="$(gemini_project_root "$input" "$jq_bin")"

# Resolve ruff from a venv, then PATH. Probe the local .venv AND the MAIN
# worktree's .venv (worktrees don't symlink .venv), each in the Unix bin/ and the
# Windows Scripts/ layout — mirrors .claude/hooks/ruff-format.sh.
venv_roots=("$root/.venv")
main_worktree="$(gemini_main_worktree "$root")"
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
  abs="$(gemini_abs_path "$root" "$path")"
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
$(gemini_tool_paths "$input" "$jq_bin")
EOF

exit 0
