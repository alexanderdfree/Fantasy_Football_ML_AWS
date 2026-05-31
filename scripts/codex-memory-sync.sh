#!/usr/bin/env bash
# Convenience wrapper for Codex-only memory sync.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/agent-memory-sync.sh" codex "$@"
