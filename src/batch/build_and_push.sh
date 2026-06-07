#!/usr/bin/env bash
# Build and push the training image for AWS Batch.
#
# Combines the cold-start optimizations from docs/batch_design.md:
#   - 2b: explicit COPYs in Dockerfile.train (nothing to do here — just `docker build`)
#   - 2c: pull-through cache base image (PULL_THROUGH_PREFIX build-arg)
#
# (SOCI lazy-loading was removed 2026-06-07 — the ECS agent does not pull through
# the soci snapshotter on Batch's ECS-managed EC2, so the index was never
# consumed; see docs/batch_design.md §2a.)
#
# Prereqs:
#   - AWS CLI authenticated:            aws sts get-caller-identity
#   - Docker running
#   - ECR repo exists:                  aws ecr describe-repositories --repository-names "$ECR_REPO"
#   - (optional, for 2c) ECR pull-through cache rule for Docker Hub, ecrPrefix=dockerhub
#
# Env overrides (all optional):
#   AWS_REGION             default us-east-1
#   ECR_REPO               default ff-training
#   IMAGE_TAG              default latest
#   USE_PULL_THROUGH       1 (default) to use <acct>.dkr.ecr/dockerhub/ base, 0 to hit Docker Hub directly
#   PULL_THROUGH_PREFIX    override the prefix string entirely
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO="${ECR_REPO:-ff-training}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
USE_PULL_THROUGH="${USE_PULL_THROUGH:-1}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

if [[ -z "${PULL_THROUGH_PREFIX:-}" && "$USE_PULL_THROUGH" == "1" ]]; then
  PULL_THROUGH_PREFIX="${REGISTRY}/dockerhub/"
fi
PULL_THROUGH_PREFIX="${PULL_THROUGH_PREFIX:-}"

echo "==> Account:    ${ACCOUNT_ID}"
echo "==> Region:     ${AWS_REGION}"
echo "==> Image:      ${IMAGE_URI}"
echo "==> Base prefix: ${PULL_THROUGH_PREFIX:-<Docker Hub>}"

echo "==> docker login to ECR"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

echo "==> docker build"
docker build \
  --platform linux/amd64 \
  --build-arg PULL_THROUGH_PREFIX="$PULL_THROUGH_PREFIX" \
  -f src/batch/Dockerfile.train \
  -t "$IMAGE_URI" \
  .

echo "==> docker push"
docker push "$IMAGE_URI"

echo "==> Done. ${IMAGE_URI} is live."
