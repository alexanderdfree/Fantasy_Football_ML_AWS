#!/usr/bin/env bash
# shellcheck shell=bash
#
# Train all six positions in parallel on a many-core CUDA box (WSL2 / RTX 5080 /
# 9950X3D). Thin wrapper over ``python -m src.benchmarking.parallel_train``:
#
#   1. sources scripts/wsl-env.sh — BLAS caps + FF_MODEL_S3_BUCKET so the run reaches
#      the website History tab (metrics only; never deploys weights).
#   2. warns if no AWS creds are visible (the History sync would silently no-op).
#   3. execs the orchestrator, which autodetects the box, partitions the physical cores
#      across the active positions, re-pins survivors onto freed cores as positions
#      finish, then merges into one benchmark record and mirrors it to S3.
#
# Usage:
#   scripts/train-local-parallel.sh                 # all 6, concurrency autodetected
#   scripts/train-local-parallel.sh QB RB WR        # subset
#   scripts/train-local-parallel.sh -j 4            # cap concurrency
#   scripts/train-local-parallel.sh --rolling-origin # walk-forward report
#   scripts/train-local-parallel.sh --dry-run       # show the plan, launch nothing
#   scripts/train-local-parallel.sh --no-sync       # don't touch the website
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# BLAS caps + FF_MODEL_S3_BUCKET / FF_MODEL_S3_PREFIX for the History sync.
# shellcheck source=/dev/null
source "$SCRIPT_DIR/wsl-env.sh"

# The orchestrator's S3 sync is on by default; warn early if creds are absent so the
# operator isn't surprised when the run doesn't appear on the site. Skip the warning
# for --dry-run / --no-sync.
case " $* " in
  *" --dry-run "* | *" --no-sync "*) ;;
  *)
    if [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ -z "${AWS_PROFILE:-}" ] && [ ! -f "$HOME/.aws/credentials" ]; then
      echo "[train-local-parallel] WARNING: no AWS creds detected (env or ~/.aws/credentials);" \
           "History sync will no-op. Set creds, or pass --no-sync to silence." >&2
    fi
    ;;
esac

exec python -m src.benchmarking.parallel_train "$@"
