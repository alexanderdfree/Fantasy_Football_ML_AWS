#!/usr/bin/env bash
# Bootstrap AWS Batch + Spot Compute Environment for ff-training.
# Idempotent — reruns skip anything that already exists.
#
# Prereqs:
#   - AWS CLI v2 with credentials for the target account.
#   - "All G and VT Spot Instance Requests" vCPU quota >= 24 (six g4dn.xlarge
#     in parallel). Script refuses to proceed otherwise.
#   - S3 bucket ff-predictor-training exists (training uses it).
#   - ECR repo ff-training exists (created by batch-image.yml's first run).
#
# Run from the repo root:  bash infra/batch/setup.sh

set -euo pipefail

REGION="us-east-1"
BUCKET="ff-predictor-training"
COMPUTE_ENV="ff-gpu-spot"
JOB_QUEUE="ff-training-queue"
JOB_DEF="ff-training-job"
JOB_ROLE="BatchTrainingRole"
INSTANCE_ROLE="ecsInstanceRole"
INSTANCE_PROFILE="ecsInstanceRole"
TASK_EXEC_ROLE="ecsTaskExecutionRole"
SG_NAME="ff-batch-sg"
ECR_REPO="ff-training"
LOG_GROUP="/aws/batch/job"
# "All G and VT Spot Instance Requests" — service quota code.
SPOT_QUOTA_CODE="L-3819A6DF"
MAX_VCPUS=24
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[batch-setup] $*"; }

# --- 1. Account ID, VPC, subnets ----------------------------------------
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' --output text --region "$REGION")
if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
  log "ERROR: no default VPC in $REGION."
  exit 1
fi
SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
  --query 'Subnets[].SubnetId' --output text --region "$REGION")
SUBNET_COUNT=$(echo "$SUBNET_IDS" | wc -w | tr -d ' ')
if [ "$SUBNET_COUNT" -lt 2 ]; then
  log "ERROR: need >=2 default subnets for Spot diversification; found $SUBNET_COUNT."
  exit 1
fi
log "Account $ACCOUNT_ID, VPC $VPC_ID, $SUBNET_COUNT default subnets"

# --- 2. Spot quota check ------------------------------------------------
log "Checking Spot G+VT vCPU quota (need >= $MAX_VCPUS)..."
QUOTA=$(aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code "$SPOT_QUOTA_CODE" \
  --region "$REGION" \
  --query 'Quota.Value' \
  --output text)
QUOTA_INT=${QUOTA%.*}
if [ "$QUOTA_INT" -lt "$MAX_VCPUS" ]; then
  log "ERROR: Spot G+VT quota is $QUOTA; need >= $MAX_VCPUS (6 x g4dn.xlarge x 4 vCPU)."
  log "Request an increase:"
  log "  aws service-quotas request-service-quota-increase --service-code ec2 \\"
  log "    --quota-code $SPOT_QUOTA_CODE --desired-value $MAX_VCPUS --region $REGION"
  exit 1
fi
log "Quota OK: $QUOTA vCPUs"

# --- 3. BatchTrainingRole (container/job role) --------------------------
if ! aws iam get-role --role-name "$JOB_ROLE" >/dev/null 2>&1; then
  log "Creating IAM role $JOB_ROLE..."
  aws iam create-role \
    --role-name "$JOB_ROLE" \
    --assume-role-policy-document "file://$SCRIPT_DIR/iam-trust-policy-job.json" \
    --description "ff-training Batch job role (S3, ECR pull, CW Logs)"
fi
log "Putting inline policy ff-batch-workload..."
aws iam put-role-policy \
  --role-name "$JOB_ROLE" \
  --policy-name ff-batch-workload \
  --policy-document "file://$SCRIPT_DIR/iam-job-policy.json"

# --- 4. ecsInstanceRole (Spot EC2 instance role) ------------------------
# Some accounts have this auto-created by Batch console; new accounts don't.
if ! aws iam get-role --role-name "$INSTANCE_ROLE" >/dev/null 2>&1; then
  log "Creating IAM role $INSTANCE_ROLE..."
  aws iam create-role \
    --role-name "$INSTANCE_ROLE" \
    --assume-role-policy-document "file://$SCRIPT_DIR/iam-trust-policy-instance.json" \
    --description "ECS-managed EC2 instance role for AWS Batch Spot fleet"
fi
log "Attaching AmazonEC2ContainerServiceforEC2Role..."
aws iam attach-role-policy \
  --role-name "$INSTANCE_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role \
  2>/dev/null || true

# --- 5. Instance profile wrapping ecsInstanceRole -----------------------
if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" >/dev/null 2>&1; then
  log "Creating instance profile $INSTANCE_PROFILE..."
  aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE"
  aws iam add-role-to-instance-profile \
    --instance-profile-name "$INSTANCE_PROFILE" \
    --role-name "$INSTANCE_ROLE"
fi

# --- 6. ecsTaskExecutionRole (image pull + log driver) ------------------
if ! aws iam get-role --role-name "$TASK_EXEC_ROLE" >/dev/null 2>&1; then
  log "Creating IAM role $TASK_EXEC_ROLE..."
  aws iam create-role \
    --role-name "$TASK_EXEC_ROLE" \
    --assume-role-policy-document "file://$SCRIPT_DIR/iam-trust-policy-job.json" \
    --description "ECS-managed task execution role (ECR pull, awslogs)"
fi
aws iam attach-role-policy \
  --role-name "$TASK_EXEC_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
  2>/dev/null || true

# --- 7. Service-linked roles (idempotent; AlreadyExists is fine) --------
log "Ensuring service-linked roles exist..."
aws iam create-service-linked-role --aws-service-name batch.amazonaws.com 2>/dev/null || true
aws iam create-service-linked-role --aws-service-name spot.amazonaws.com 2>/dev/null || true

# IAM propagation: CE creation races on it.
log "Sleeping 10s for IAM propagation..."
sleep 10

# --- 8. Security group --------------------------------------------------
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text --region "$REGION" 2>/dev/null || echo "None")
if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
  log "Creating security group $SG_NAME..."
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "ff-training Batch Spot (egress only)" \
    --vpc-id "$VPC_ID" \
    --region "$REGION" \
    --query GroupId --output text)
fi
log "Security group: $SG_ID (no ingress rules)"

# --- 9. CloudWatch log group --------------------------------------------
aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$REGION" 2>/dev/null || true
aws logs put-retention-policy \
  --log-group-name "$LOG_GROUP" \
  --retention-in-days 7 \
  --region "$REGION"

# --- 10. Compute Environment --------------------------------------------
CE_STATUS=$(aws batch describe-compute-environments \
  --compute-environments "$COMPUTE_ENV" \
  --region "$REGION" \
  --query 'computeEnvironments[0].status' \
  --output text 2>/dev/null || echo "None")
if [ "$CE_STATUS" = "None" ] || [ -z "$CE_STATUS" ] || [ "$CE_STATUS" = "null" ]; then
  log "Creating Compute Environment $COMPUTE_ENV..."
  # Build JSON array of subnet IDs from space-separated list.
  SUBNETS_JSON=$(printf '"%s",' $SUBNET_IDS | sed 's/,$//')
  aws batch create-compute-environment \
    --compute-environment-name "$COMPUTE_ENV" \
    --type MANAGED \
    --state ENABLED \
    --compute-resources "{
      \"type\": \"SPOT\",
      \"allocationStrategy\": \"SPOT_PRICE_CAPACITY_OPTIMIZED\",
      \"minvCpus\": 0,
      \"maxvCpus\": $MAX_VCPUS,
      \"instanceTypes\": [\"g4dn.xlarge\"],
      \"subnets\": [$SUBNETS_JSON],
      \"securityGroupIds\": [\"$SG_ID\"],
      \"instanceRole\": \"arn:aws:iam::$ACCOUNT_ID:instance-profile/$INSTANCE_PROFILE\"
    }" \
    --region "$REGION"
  log "Waiting for CE to reach VALID..."
  for i in $(seq 1 60); do
    STATUS=$(aws batch describe-compute-environments \
      --compute-environments "$COMPUTE_ENV" \
      --region "$REGION" \
      --query 'computeEnvironments[0].status' \
      --output text)
    if [ "$STATUS" = "VALID" ]; then
      log "CE is VALID"
      break
    fi
    if [ "$STATUS" = "INVALID" ]; then
      REASON=$(aws batch describe-compute-environments \
        --compute-environments "$COMPUTE_ENV" \
        --region "$REGION" \
        --query 'computeEnvironments[0].statusReason' \
        --output text)
      log "ERROR: CE is INVALID — $REASON"
      exit 1
    fi
    sleep 5
  done
else
  log "Compute Environment $COMPUTE_ENV already exists (status: $CE_STATUS)"
fi

# --- 11. Job Queue ------------------------------------------------------
JQ_STATUS=$(aws batch describe-job-queues \
  --job-queues "$JOB_QUEUE" \
  --region "$REGION" \
  --query 'jobQueues[0].status' \
  --output text 2>/dev/null || echo "None")
if [ "$JQ_STATUS" = "None" ] || [ -z "$JQ_STATUS" ] || [ "$JQ_STATUS" = "null" ]; then
  log "Creating Job Queue $JOB_QUEUE..."
  aws batch create-job-queue \
    --job-queue-name "$JOB_QUEUE" \
    --state ENABLED \
    --priority 1 \
    --compute-environment-order "order=1,computeEnvironment=$COMPUTE_ENV" \
    --region "$REGION"
  log "Waiting for JQ to reach VALID..."
  for i in $(seq 1 30); do
    STATUS=$(aws batch describe-job-queues \
      --job-queues "$JOB_QUEUE" \
      --region "$REGION" \
      --query 'jobQueues[0].status' \
      --output text)
    if [ "$STATUS" = "VALID" ]; then
      log "JQ is VALID"
      break
    fi
    sleep 5
  done
else
  log "Job Queue $JOB_QUEUE already exists (status: $JQ_STATUS)"
fi

# --- 12. Job Definition (register rev 1 against :latest) ----------------
# Solves the chicken-and-egg with batch-image.yml: its re-registration step
# reads the previous revision; without a seed revision, jq breaks. Once
# rev 1 exists, every push registers a fresh revision and launch.py picks
# the highest by family name.
JD_NAME=$(aws batch describe-job-definitions \
  --job-definition-name "$JOB_DEF" \
  --status ACTIVE \
  --max-results 1 \
  --region "$REGION" \
  --query 'jobDefinitions[0].jobDefinitionName' \
  --output text 2>/dev/null || echo "None")
if [ "$JD_NAME" = "None" ] || [ -z "$JD_NAME" ] || [ "$JD_NAME" = "null" ]; then
  log "Registering Job Definition $JOB_DEF rev 1..."
  aws batch register-job-definition \
    --job-definition-name "$JOB_DEF" \
    --type container \
    --platform-capabilities EC2 \
    --container-properties "{
      \"image\": \"$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest\",
      \"vcpus\": 4,
      \"memory\": 15000,
      \"jobRoleArn\": \"arn:aws:iam::$ACCOUNT_ID:role/$JOB_ROLE\",
      \"executionRoleArn\": \"arn:aws:iam::$ACCOUNT_ID:role/$TASK_EXEC_ROLE\",
      \"resourceRequirements\": [{\"type\": \"GPU\", \"value\": \"1\"}],
      \"logConfiguration\": {
        \"logDriver\": \"awslogs\",
        \"options\": {
          \"awslogs-group\": \"$LOG_GROUP\",
          \"awslogs-region\": \"$REGION\",
          \"awslogs-stream-prefix\": \"ff-training\"
        }
      }
    }" \
    --timeout '{"attemptDurationSeconds": 1800}' \
    --retry-strategy '{"attempts": 1}' \
    --region "$REGION"
else
  log "Job Definition $JOB_DEF already exists; CI will re-register on next push."
fi

cat <<EOF

────────────────────────────────────────────────────────────────
Batch + Spot infrastructure ready:
  Region:              $REGION
  Compute environment: $COMPUTE_ENV (maxVcpus=$MAX_VCPUS, SPOT_PRICE_CAPACITY_OPTIMIZED)
  Job queue:           $JOB_QUEUE
  Job definition:      $JOB_DEF (image: $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest)
  Job role:            arn:aws:iam::$ACCOUNT_ID:role/$JOB_ROLE
  Instance role:       arn:aws:iam::$ACCOUNT_ID:role/$INSTANCE_ROLE
  Task exec role:      arn:aws:iam::$ACCOUNT_ID:role/$TASK_EXEC_ROLE
  Security group:      $SG_ID
  Subnets:             $SUBNET_IDS

Next steps:
  1. (Cold-start opt) Create ECR pull-through cache for the PyTorch base image:
       aws ecr create-pull-through-cache-rule \\
         --ecr-repository-prefix dockerhub \\
         --upstream-registry-url registry-1.docker.io \\
         --region $REGION
  2. Verify CE and JQ are VALID:
       aws batch describe-compute-environments --compute-environments $COMPUTE_ENV \\
         --query 'computeEnvironments[0].[state,status]' --region $REGION
  3. Smoke test (single cheap position, ~2-3 min):
       AWS_REGION=$REGION python -m src.batch.launch --positions K --seed 42
  4. When ready to flip the active trainer from EC2 to Batch:
       gh variable set BATCH_ACTIVE --body "true"
────────────────────────────────────────────────────────────────
EOF
