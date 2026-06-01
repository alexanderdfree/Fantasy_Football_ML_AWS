#!/usr/bin/env bash
# Start Codex from a clean Codex-owned worktree, creating one when needed.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/codex-fresh-worktree.sh [launcher options] [--] [codex args...]

Launcher options:
  --force-new       Always create a new worktree.
  --base <ref>      Base new worktrees on <ref> (default: origin/main).
  --branch <name>   Use <name> for the new worktree branch.
  --no-fetch        Skip fetching origin/main before creating a worktree.
  --print-path      Print the selected worktree path instead of launching Codex.
  -h, --help        Show this help.

Codex args after the launcher options are passed to: codex --cd <worktree>.
Use -- before Codex args when a Codex prompt or option starts with a launcher
option name.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/.." rev-parse --show-toplevel)"
# shellcheck source=.codex/hooks/lib.sh
. "$repo_root/.codex/hooks/lib.sh"

force_new=0
fetch_main=1
print_path=0
base_ref="origin/main"
branch_name=""
codex_args=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --force-new)
      force_new=1
      shift
      ;;
    --base)
      if [ "$#" -lt 2 ]; then
        echo "missing value for --base" >&2
        exit 2
      fi
      base_ref="$2"
      shift 2
      ;;
    --branch)
      if [ "$#" -lt 2 ]; then
        echo "missing value for --branch" >&2
        exit 2
      fi
      branch_name="$2"
      shift 2
      ;;
    --no-fetch)
      fetch_main=0
      shift
      ;;
    --print-path)
      print_path=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      codex_args=("$@")
      break
      ;;
    *)
      codex_args=("$@")
      break
      ;;
  esac
done

current_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$current_root" ]; then
  echo "codex-fresh-worktree must run from inside a git checkout." >&2
  exit 2
fi

main_worktree="$(codex_main_worktree "$current_root")"
if [ -z "$main_worktree" ]; then
  echo "could not resolve the main git worktree." >&2
  exit 2
fi

repo_name="$(basename "$main_worktree")"
worktrees_dir="$(codex_worktrees_dir)"

if [ "$force_new" -eq 0 ] && codex_is_clean_codex_worktree "$current_root" "$main_worktree"; then
  target="$current_root"
else
  if [ "$fetch_main" -eq 1 ]; then
    git -C "$main_worktree" fetch origin main --quiet
  fi

  if [ -n "$branch_name" ]; then
    short_id="${branch_name##*/}"
    target="$worktrees_dir/$short_id/$repo_name"
    if [ -e "$target" ]; then
      echo "target worktree already exists: $target" >&2
      exit 2
    fi
  else
    while :; do
      short_id="$(od -An -N2 -tx1 /dev/urandom | tr -d ' \n')"
      branch_name="codex/session-$short_id"
      target="$worktrees_dir/$short_id/$repo_name"
      if [ ! -e "$target" ] && ! git -C "$main_worktree" show-ref --verify --quiet "refs/heads/$branch_name"; then
        break
      fi
    done
  fi

  mkdir -p "$(dirname "$target")"
  git -C "$main_worktree" worktree add -b "$branch_name" "$target" "$base_ref" >/dev/null

  for data_dir in raw splits; do
    source_dir="$main_worktree/data/$data_dir"
    dest_dir="$target/data/$data_dir"
    if [ -e "$source_dir" ] && [ ! -e "$dest_dir" ]; then
      mkdir -p "$target/data"
      ln -s "$source_dir" "$dest_dir" || true
    fi
  done
fi

if [ "$print_path" -eq 1 ]; then
  printf '%s\n' "$target"
  exit 0
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found on PATH; selected worktree: $target" >&2
  exit 127
fi

exec codex --cd "$target" "${codex_args[@]}"
