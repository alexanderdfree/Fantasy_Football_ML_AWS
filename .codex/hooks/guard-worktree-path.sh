#!/usr/bin/env bash
# PreToolUse hook: block Codex file edits that target the main checkout while
# this session is running in a git worktree.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.codex/hooks/lib.sh
. "$script_dir/lib.sh"

jq_bin="$(codex_find_jq)" || exit 0
input="$(cat)"
root="$(codex_project_root "$input" "$jq_bin")"
main_worktree="$(codex_main_worktree "$root")"

[ -n "$main_worktree" ] || exit 0
[ "$root" != "$main_worktree" ] || exit 0

blocked=0
while IFS= read -r path; do
  [ -n "$path" ] || continue
  abs="$(codex_abs_path "$root" "$path")"
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
