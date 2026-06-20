#!/usr/bin/env bash
# SessionEnd hook: best-effort push of local agent memories to their separate S3
# prefixes. Parity twin of .codex/hooks/memory-sync-stop.sh — pushes ALL trees
# (claude + codex + gemini) so cross-agent updates made on this machine are not
# stranded in one local store. Never blocks.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.gemini/hooks/lib.sh
. "$script_dir/lib.sh"

jq_bin="$(gemini_find_jq || true)"
input="$(cat)"
root="$(gemini_project_root "$input" "$jq_bin")"

if [ -x "$root/scripts/agent-memory-sync.sh" ]; then
  (cd "$root" && bash scripts/agent-memory-sync.sh all push) || true
fi

exit 0
