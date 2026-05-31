#!/usr/bin/env bash
# Backwards-compatible wrapper for Claude-only memory sync.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/agent-memory-sync.sh" claude "$@"
