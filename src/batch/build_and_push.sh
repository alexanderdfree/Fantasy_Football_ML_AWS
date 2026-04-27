#!/usr/bin/env bash
# Build, push, and SOCI-index the training image for AWS Batch.
#
# Combines the three cold-start optimizations from docs/batch_design.md:
#   - 2b: explicit COPYs in Dockerfile.train (nothing to do here — just `docker build`)
#   - 2c: pull-through cache base image (PULL_THROUGH_PREFIX build-arg)
#   - 2a: SOCI index pushed alongside the image tag in ECR
#
# Prereqs:
#   - AWS CLI authenticated:            aws sts get-caller-identity
#   - Docker running
#   - ECR repo exists:                  aws ecr describe-repositories --repository-names "$ECR_REPO"
#   - (optional, for 2c) ECR pull-through cache rule for Docker Hub, ecrPrefix=dockerhub
#   - (optional, for 2a) soci CLI:      https://github.com/awslabs/soci-snapshotter/releases
#
# Env overrides (all optional):
#   AWS_REGION             default us-east-1
#   ECR_REPO               default ff-training
#   IMAGE_TAG              default latest
#   USE_PULL_THROUGH       1 (default) to use <acct>.dkr.ecr/dockerhub/ base, 0 to hit Docker Hub directly
#   PULL_THROUGH_PREFIX    override the prefix string entirely
#   SKIP_SOCI              set to 1 to skip SOCI index creation even if the CLI is present
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO="${ECR_REPO:-ff-training}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
USE_PULL_THROUGH="${USE_PULL_THROUGH:-1}"
SKIP_SOCI="${SKIP_SOCI:-0}"

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
  -f batch/Dockerfile.train \
  -t "$IMAGE_URI" \
  .

echo "==> docker push"
docker push "$IMAGE_URI"

if [[ "$SKIP_SOCI" == "1" ]]; then
  echo "==> SKIP_SOCI=1; skipping SOCI index step"
elif ! command -v soci >/dev/null 2>&1; then
  echo "==> 'soci' CLI not found; skipping SOCI index step"
  echo "    Install from https://github.com/awslabs/soci-snapshotter/releases to enable"
  echo "    lazy loading. Batch jobs will still work, they just start slower."
else
  # soci create reads the image from the local containerd content store.
  # On a Docker-only host (e.g. Docker Desktop) this requires a recent soci
  # build that supports the Docker graph driver, OR running on a Linux host
  # with containerd. If `soci create` fails, the note above explains the
  # trade-off — the image still works, cold starts just aren't accelerated.
  echo "==> soci create ${IMAGE_URI}"
  soci create "$IMAGE_URI"

  echo "==> soci push ${IMAGE_URI}"
  soci push --user "AWS:$(aws ecr get-login-password --region "$AWS_REGION")" "$IMAGE_URI"
fi

echo "==> Done. ${IMAGE_URI} is live."
