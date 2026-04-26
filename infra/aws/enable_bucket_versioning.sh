#!/usr/bin/env bash
# Enable S3 bucket versioning on the artifacts bucket. Idempotent — re-running
# is a no-op once versioning is "Enabled".
#
# Why we need this on top of manifest-based promotion:
#   The manifest's stable/current/previous slots already give us logical
#   versioning, and history/{ts}-{sha7}/model.tar.gz already preserves prior
#   uploads. But a buggy retention prune (shared/artifact_gc.py) or a
#   misclick on the AWS console could permanently delete the artifact pointed
#   to by `stable`. With bucket versioning enabled, S3 keeps a non-current
#   version on every overwrite/delete, so we can recover from operator error.
#
# Cost: each non-current version is billed as standard storage. Artifacts are
# small (~50 MB per position), so leaving versioning on indefinitely is
# cheap. If costs grow, add a lifecycle rule to expire non-current versions
# after N days — see the comment block at the bottom of this file.
#
# Run once, with bucket-owner IAM:
#   bash infra/aws/enable_bucket_versioning.sh

set -euo pipefail

BUCKET="${S3_BUCKET:-ff-predictor-training}"
REGION="${AWS_REGION:-us-east-1}"

echo "Enabling versioning on s3://${BUCKET} (region=${REGION})..."
aws s3api put-bucket-versioning \
    --bucket "$BUCKET" \
    --versioning-configuration Status=Enabled \
    --region "$REGION"

echo "Current versioning state:"
aws s3api get-bucket-versioning --bucket "$BUCKET" --region "$REGION"

# Optional — uncomment and customize to bound non-current-version storage cost:
#
# aws s3api put-bucket-lifecycle-configuration \
#     --bucket "$BUCKET" \
#     --region "$REGION" \
#     --lifecycle-configuration '{
#         "Rules": [{
#             "ID": "expire-noncurrent-model-versions",
#             "Status": "Enabled",
#             "Filter": {"Prefix": "models/"},
#             "NoncurrentVersionExpiration": {"NoncurrentDays": 90}
#         }]
#     }'
