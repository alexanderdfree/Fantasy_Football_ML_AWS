#!/usr/bin/env bash
# Stop hook: best-effort push of local agent memories to their separate S3
# prefixes. Push both trees so cross-agent memory updates made on this machine
# are not stranded in one local store.
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.codex/hooks/lib.sh
. "$script_dir/lib.sh"

jq_bin="$(codex_find_jq)" || exit 0
input="$(cat)"
root="$(codex_project_root "$input" "$jq_bin")"

if [ -x "$root/scripts/agent-memory-sync.sh" ]; then
  (cd "$root" && bash scripts/agent-memory-sync.sh all push) || true
fi
