#!/usr/bin/env bash
# Install machine-local Codex prompt templates for this repo.
set -euo pipefail

WITH_MEMORY_SYNC=0
for arg in "$@"; do
  case "$arg" in
    --with-memory-sync) WITH_MEMORY_SYNC=1 ;;
    -h | --help)
      echo "usage: $(basename "$0") [--with-memory-sync]"
      exit 0
      ;;
    *)
      echo "[bootstrap] unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

repo_root="$(git rev-parse --show-toplevel)"
codex_home="${CODEX_HOME:-$HOME/.codex}"
prompt_src="$repo_root/.codex/prompts"
prompt_dst="$codex_home/prompts"

if [ ! -d "$prompt_src" ]; then
  echo "No Codex prompt templates found at $prompt_src" >&2
  exit 1
fi

mkdir -p "$prompt_dst"

for prompt in "$prompt_src"/*.md; do
  [ -f "$prompt" ] || continue
  cp "$prompt" "$prompt_dst/$(basename "$prompt")"
done

if [ "$WITH_MEMORY_SYNC" -eq 1 ]; then
  echo "[bootstrap] doing an initial Codex memory pull from S3 (best-effort)..."
  bash "$repo_root/scripts/codex-memory-sync.sh" pull || true
fi

cat <<EOF
Installed Final-Project Codex prompts into:
  $prompt_dst

Restart Codex so custom prompts reload, then invoke them as:
  /prompts:pre-pr-judge
  /prompts:pre-pr-gate
  /prompts:post-pr-followup
  /prompts:post-session-critique
  /prompts:solve-issues

Project hooks are tracked under:
  $repo_root/.codex/hooks.json

Open /hooks in Codex after changes and trust the reviewed project hooks.

To also seed Codex memories from S3 on a fresh machine, re-run:
  scripts/bootstrap-codex-local.sh --with-memory-sync
EOF
