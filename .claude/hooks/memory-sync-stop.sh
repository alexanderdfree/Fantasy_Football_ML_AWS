#!/usr/bin/env bash
# Stop hook: best-effort push of local agent memories to their separate S3
# prefixes. Push both trees so cross-agent memory updates made on this machine
# are not stranded in one local store.
set -u

repo_root="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$repo_root" ]; then
  repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi

if [ -x "$repo_root/scripts/agent-memory-sync.sh" ]; then
  (cd "$repo_root" && bash scripts/agent-memory-sync.sh all push) || true
fi
