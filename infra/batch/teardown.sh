#!/usr/bin/env bash
# Tear down Batch + Spot infrastructure created by setup.sh.
# Idempotent — skips resources that don't exist; safe to rerun.
#
# Service-linked roles, the ECR pull-through cache rule, ECR repositories,
# and the /aws/batch/job log group are NOT removed (shared / preserve history).
# Remove those manually if you want a complete wipe.

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
COMPUTE_ENVS=("ff-gpu-spot-g5" "ff-gpu-spot")
JOB_QUEUE="ff-training-queue"
JOB_DEF="ff-training-job"
JOB_ROLE="BatchTrainingRole"
INSTANCE_ROLE="ecsInstanceRole"
INSTANCE_PROFILE="ecsInstanceRole"
TASK_EXEC_ROLE="ecsTaskExecutionRole"
SG_NAME="ff-batch-sg"
LAUNCH_TEMPLATE_NAME="ff-batch-lt"

log() { echo "[batch-teardown] $*"; }

# --- 1. Deregister all active Job Definition revisions ------------------
log "Deregistering active Job Definition revisions for $JOB_DEF..."
REVISIONS=$(aws batch describe-job-definitions \
  --job-definition-name "$JOB_DEF" \
  --status ACTIVE \
  --region "$REGION" \
  --query 'jobDefinitions[].revision' \
  --output text 2>/dev/null || echo "")
for rev in $REVISIONS; do
  log "  deregistering $JOB_DEF:$rev"
  aws batch deregister-job-definition \
    --job-definition "$JOB_DEF:$rev" \
    --region "$REGION" || true
done

# --- 2. Disable + delete Job Queue --------------------------------------
JQ_EXISTS=$(aws batch describe-job-queues \
  --job-queues "$JOB_QUEUE" \
  --region "$REGION" \
  --query 'jobQueues[0].jobQueueName' \
  --output text 2>/dev/null || echo "None")
if [ "$JQ_EXISTS" != "None" ] && [ -n "$JQ_EXISTS" ] && [ "$JQ_EXISTS" != "null" ]; then
  log "Disabling Job Queue $JOB_QUEUE..."
  aws batch update-job-queue --job-queue "$JOB_QUEUE" --state DISABLED --region "$REGION" || true
  log "Waiting for JQ to reach VALID (after disable)..."
  for i in $(seq 1 30); do
    STATUS=$(aws batch describe-job-queues \
      --job-queues "$JOB_QUEUE" \
      --region "$REGION" \
      --query 'jobQueues[0].status' --output text)
    [ "$STATUS" = "VALID" ] && break
    sleep 5
  done
  log "Deleting Job Queue $JOB_QUEUE..."
  aws batch delete-job-queue --job-queue "$JOB_QUEUE" --region "$REGION" || true
  for i in $(seq 1 30); do
    EXISTS=$(aws batch describe-job-queues \
      --job-queues "$JOB_QUEUE" \
      --region "$REGION" \
      --query 'jobQueues[0].jobQueueName' --output text 2>/dev/null || echo "None")
    [ "$EXISTS" = "None" ] || [ -z "$EXISTS" ] && break
    sleep 5
  done
fi

# --- 3. Disable + delete Compute Environments ---------------------------
for compute_env in "${COMPUTE_ENVS[@]}"; do
  CE_EXISTS=$(aws batch describe-compute-environments \
    --compute-environments "$compute_env" \
    --region "$REGION" \
    --query 'computeEnvironments[0].computeEnvironmentName' \
    --output text 2>/dev/null || echo "None")
  if [ "$CE_EXISTS" != "None" ] && [ -n "$CE_EXISTS" ] && [ "$CE_EXISTS" != "null" ]; then
    log "Disabling Compute Environment $compute_env..."
    aws batch update-compute-environment \
      --compute-environment "$compute_env" \
      --state DISABLED \
      --region "$REGION" || true
    log "Waiting for CE to reach VALID (after disable)..."
    for i in $(seq 1 30); do
      STATUS=$(aws batch describe-compute-environments \
        --compute-environments "$compute_env" \
        --region "$REGION" \
        --query 'computeEnvironments[0].status' --output text)
      [ "$STATUS" = "VALID" ] && break
      sleep 5
    done
    log "Deleting Compute Environment $compute_env..."
    aws batch delete-compute-environment \
      --compute-environment "$compute_env" \
      --region "$REGION" || true
  fi
done

# --- 3b. Launch Template ------------------------------------------------
# Safe to delete after CE deletion: AWS Batch holds a reference while the
# CE exists; deleting the template before the CE returns a dependency
# error. Delete after the CE is gone.
if aws ec2 describe-launch-templates \
     --launch-template-names "$LAUNCH_TEMPLATE_NAME" \
     --region "$REGION" >/dev/null 2>&1; then
  log "Deleting launch template $LAUNCH_TEMPLATE_NAME..."
  aws ec2 delete-launch-template \
    --launch-template-name "$LAUNCH_TEMPLATE_NAME" \
    --region "$REGION" >/dev/null || true
fi

# --- 4. Security group --------------------------------------------------
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' --output text --region "$REGION")
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text --region "$REGION" 2>/dev/null || echo "None")
if [ "$SG_ID" != "None" ] && [ -n "$SG_ID" ] && [ "$SG_ID" != "null" ]; then
  log "Deleting security group $SG_ID..."
  aws ec2 delete-security-group --group-id "$SG_ID" --region "$REGION" || \
    log "  (security group delete failed — may have ENIs still attached; retry shortly)"
fi

# --- 5. IAM ------------------------------------------------------------
if aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" >/dev/null 2>&1; then
  log "Removing $INSTANCE_ROLE from instance profile $INSTANCE_PROFILE..."
  aws iam remove-role-from-instance-profile \
    --instance-profile-name "$INSTANCE_PROFILE" \
    --role-name "$INSTANCE_ROLE" 2>/dev/null || true
  log "Deleting instance profile $INSTANCE_PROFILE..."
  aws iam delete-instance-profile --instance-profile-name "$INSTANCE_PROFILE" || true
fi

for role in "$JOB_ROLE" "$INSTANCE_ROLE" "$TASK_EXEC_ROLE"; do
  if aws iam get-role --role-name "$role" >/dev/null 2>&1; then
    log "Detaching managed policies from $role..."
    for arn in $(aws iam list-attached-role-policies --role-name "$role" \
                   --query 'AttachedPolicies[].PolicyArn' --output text); do
      aws iam detach-role-policy --role-name "$role" --policy-arn "$arn" || true
    done
    log "Removing inline policies from $role..."
    for name in $(aws iam list-role-policies --role-name "$role" \
                    --query 'PolicyNames' --output text); do
      aws iam delete-role-policy --role-name "$role" --policy-name "$name" || true
    done
    log "Deleting IAM role $role..."
    aws iam delete-role --role-name "$role" || true
  fi
done

cat <<EOF

────────────────────────────────────────────────────────────────
Teardown complete.
Resources preserved (delete manually if needed):
  - ECR repository ff-training (preserves built images)
  - ECR pull-through cache rule "dockerhub"
  - CloudWatch log group /aws/batch/job (preserves history)
  - Service-linked roles AWSServiceRoleForBatch, AWSServiceRoleForEC2Spot
────────────────────────────────────────────────────────────────
EOF
