#!/usr/bin/env bash
# PreToolUse hook: block Codex file edits that target the main checkout while
# this session is running in a git worktree.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.codex/hooks/lib.sh
. "$script_dir/lib.sh"

# Python resolves filesystem aliases as well as parsing JSON without jq. Do not
# approve an edit if its destination cannot be checked.
jq_bin="$(codex_find_jq || true)"
input="$(cat)"
if [ -z "$codex_python_bin" ]; then
  echo "guard-worktree-path: a Python 3 interpreter is required to validate resolved edit paths" >&2
  exit 2
fi
cwd="$(codex_project_cwd "$input" "$jq_bin")"
root="$(codex_project_root "$input" "$jq_bin")"
main_worktree="$(codex_main_worktree "$root")"

[ -n "$main_worktree" ] || exit 0
root="$(codex_abs_path "$root" .)" || exit 2
main_worktree="$(codex_abs_path "$main_worktree" .)" || exit 2
[ "$root" != "$main_worktree" ] || exit 0

blocked=0
while IFS= read -r path; do
  [ -n "$path" ] || continue
  abs="$(codex_abs_path "$cwd" "$path")" || exit 2
  case "$abs" in
    "$root"/*) ;;
    "$main_worktree"/*)
      corrected="$root/${abs#"$main_worktree"/}"
      {
        echo "BLOCK: file edit targets the main checkout, not this Codex worktree."
        echo "  target:        $abs"
        echo "  main checkout: $main_worktree"
        echo "  worktree:      $root"
        echo "Re-prefix the path to the worktree and retry:"
        echo "  $corrected"
      } >&2
      blocked=1
      ;;
  esac
done <<EOF
$(codex_tool_paths "$input" "$jq_bin")
EOF

if [ "$blocked" -ne 0 ]; then
  exit 2
fi

exit 0
