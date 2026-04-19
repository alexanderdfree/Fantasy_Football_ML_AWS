#!/usr/bin/env bash
# Seed s3://ff-predictor-training/models/{POS}/model.tar.gz from the
# git-committed local model directories. Unblocks production serving while
# the EC2 GPU quota is still 0 — once a real training run lands, it will
# overwrite these tarballs and the running Fargate task will pick up the new
# artifacts on its next restart.
#
# Safe to run repeatedly; each upload is atomic.
#
# Prereqs:
#   - AWS CLI v2 with credentials for the target account.
#   - Local {POS}/outputs/models/ directories populated (true today — the
#     git-committed .pkl / .pt files are the fallback models).
#   - S3 bucket ff-predictor-training exists (it does — training already
#     writes data/raw/ there).
#
# Run from the repo root:  bash infra/aws/seed_s3_models.sh

set -euo pipefail

REGION="us-east-1"
BUCKET="ff-predictor-training"
PREFIX="models"
POSITIONS=(QB RB WR TE K DST)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

log() { echo "[seed-s3] $*"; }

for POS in "${POSITIONS[@]}"; do
  SRC_DIR="$REPO_ROOT/$POS/outputs/models"
  if [ ! -d "$SRC_DIR" ]; then
    log "ERROR: missing $SRC_DIR"
    exit 1
  fi
  if [ -z "$(ls -A "$SRC_DIR" 2>/dev/null)" ]; then
    log "ERROR: $SRC_DIR is empty — refusing to upload an empty tarball."
    exit 1
  fi
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for POS in "${POSITIONS[@]}"; do
  SRC_DIR="$REPO_ROOT/$POS/outputs/models"
  TARBALL="$TMP/$POS.tar.gz"
  # -C so tar entries are flat (nn_scaler.pkl, lightgbm/..., not QB/outputs/models/nn_scaler.pkl).
  # This matches the layout batch/train.py:upload_artifacts produces and
  # shared.model_sync._extract_tarball expects.
  log "Tarring $POS from $SRC_DIR..."
  tar -czf "$TARBALL" -C "$SRC_DIR" .
  SIZE_MB=$(du -m "$TARBALL" | cut -f1)

  S3_KEY="$PREFIX/$POS/model.tar.gz"
  log "Uploading s3://$BUCKET/$S3_KEY ($SIZE_MB MB)..."
  aws s3 cp "$TARBALL" "s3://$BUCKET/$S3_KEY" --region "$REGION"
done

log "Verifying all 6 keys exist in S3..."
for POS in "${POSITIONS[@]}"; do
  aws s3api head-object --bucket "$BUCKET" --key "$PREFIX/$POS/model.tar.gz" --region "$REGION" >/dev/null
done
log "All tarballs uploaded and verified."
