#!/usr/bin/env bash
# Initialize missing model manifests from local, smoke-tested model directories.
# Existing manifests are verified, never overwritten. Requires the project
# Python environment and AWS credentials. PYTHON may select the interpreter.
# Run from any directory: bash /path/to/repo/infra/aws/seed_s3_models.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
exec "${PYTHON:-python}" -m src.scripts.seed_s3_models "$@"
