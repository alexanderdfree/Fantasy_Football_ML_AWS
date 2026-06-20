#!/usr/bin/env bash
# SessionStart hook: best-effort pull of Gemini/Antigravity memory from S3.
# Parity twin of .codex/hooks/session-start.sh (memory pull only). Never blocks.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.gemini/hooks/lib.sh
. "$script_dir/lib.sh"

jq_bin="$(gemini_find_jq || true)"
input="$(cat)"
root="$(gemini_project_root "$input" "$jq_bin")"

if [ -x "$root/scripts/agent-memory-sync.sh" ]; then
  (cd "$root" && bash scripts/agent-memory-sync.sh gemini pull) || true
fi

exit 0
