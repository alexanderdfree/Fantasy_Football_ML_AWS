#!/usr/bin/env bash
# PreToolUse hook: block Codex file edits that target the main checkout while
# this session is running in a git worktree.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.codex/hooks/lib.sh
. "$script_dir/lib.sh"

# Resolve jq, but do NOT exit when it is absent: this guard is the deterministic
# backstop for the parent-checkout-write footgun (parity with
# .claude/hooks/guard-worktree-path.sh). The codex_* extractors fall back to
# python3's JSON parser when jq_bin is empty, so the guard stays ARMED. It only
# truly disarms — with a WARNING, never silently — if neither jq nor python3 exists.
jq_bin="$(codex_find_jq || true)"
input="$(cat)"
if [ -z "$jq_bin" ] && ! command -v python3 >/dev/null 2>&1; then
  echo "guard-worktree-path: neither jq nor python3 found; cannot validate paths (guard disarmed for this edit)" >&2
  exit 0
fi
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
