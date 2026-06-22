#!/usr/bin/env bash
# Bootstrap AWS Batch + Spot Compute Environment for ff-training.
# Idempotent — reruns skip anything that already exists.
#
# Prereqs:
#   - AWS CLI v2 with credentials for the target account.
#   - "All G and VT Spot Instance Requests" vCPU quota >= 64 (sixteen 4-vCPU
#     GPU Spot hosts: two concurrent six-position fan-outs plus tune-fleet
#     headroom). Script refuses to proceed otherwise.
#   - S3 bucket ff-predictor-training exists (training uses it).
#   - ECR repo ff-training exists (created by batch-image.yml's first run).
#
# Run from the repo root:  bash infra/batch/setup.sh

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
BUCKET="ff-predictor-training"
COMPUTE_ENV="ff-gpu-spot"
JOB_QUEUE="ff-training-queue"
JOB_DEF="ff-training-job"
CPU_COMPUTE_ENV="ff-cpu-spot"
CPU_JOB_QUEUE="ff-cpu-training-queue"
CPU_JOB_DEF="ff-training-cpu-job"
JOB_ROLE="BatchTrainingRole"
INSTANCE_ROLE="ecsInstanceRole"
INSTANCE_PROFILE="ecsInstanceRole"
TASK_EXEC_ROLE="ecsTaskExecutionRole"
SG_NAME="ff-batch-sg"
ECR_REPO="ff-training"
LOG_GROUP="/aws/batch/job"
# "All G and VT Spot Instance Requests" — service quota code.
SPOT_QUOTA_CODE="L-3819A6DF"
# "All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests".
STANDARD_SPOT_QUOTA_CODE="L-34B43A08"
# Matches the live "All G and VT Spot Instance Requests" quota (raised 24 -> 64
# on 2026-06-11): two concurrent six-position fan-outs (monolithic or split-nn)
# plus tune-fleet headroom, 16 x 4-vCPU hosts max per CE.
MAX_VCPUS=64
CPU_MAX_VCPUS=64
# GPU Spot instance types for the fan-out. Both go in ONE compute environment;
# the SPOT_PRICE_CAPACITY_OPTIMIZED allocation strategy then diversifies across
# every (instance-type x AZ) Spot pool and launches wherever capacity exists.
#
# Why one CE and not a g6-primary / g5-fallback queue order: a job queue's
# compute-environment ORDER only advances to the next CE on a MISCONFIGURATION
# (maxvCpus too small, job resource requirement, unsupported config, service
# role) -- NOT on Spot capacity starvation. A healthy, below-max g6 CE that
# simply can't get g6 Spot parks jobs in RUNNABLE forever and never reaches a
# g5 CE behind it. The capacity-fallback mechanism is the allocation strategy
# *inside* a CE, so the pool must be diversified within one CE (this mirrors the
# ff-cpu-spot CE, which already lists c8a + m8a). Both types are 4 vCPU / 16 GiB
# / 1 GPU with 24 GB GPU memory and are single-GPU (so GPU=1 keeps one job per
# host), and both qualify for the sm_80+ CUDA-graph path, so the training
# container runs unchanged. PRIMARY_INSTANCE_TYPE is the representative type used
# for subnet/AZ resolution (g6 and g5 are offered in the same us-east-1 AZs).
PRIMARY_INSTANCE_TYPE="g6.xlarge"
SECONDARY_INSTANCE_TYPE="g5.xlarge"
ALL_GPU_INSTANCE_TYPES=("$PRIMARY_INSTANCE_TYPE" "$SECONDARY_INSTANCE_TYPE")
CPU_PRIMARY_INSTANCE_TYPE="c8a.xlarge"
CPU_FALLBACK_INSTANCE_TYPE="m8a.xlarge"
CPU_INSTANCE_TYPES_JSON="[\"$CPU_PRIMARY_INSTANCE_TYPE\",\"$CPU_FALLBACK_INSTANCE_TYPE\"]"
# Opt-in warm pre-pulled AMI (cold-start lever). UNSET by default => the GPU CE
# uses the default ECS-GPU AMI exactly as before, so merging this changes
# nothing live. Set FF_BATCH_AMI_ID=<ami> (built by infra/batch/build-warm-ami.sh)
# to boot fresh GPU Spot hosts from a custom AMI with the training image's base
# layers pre-pulled — skipping the ~122 s image pull that dominates the
# orchestration-bound cold-start. Attached to the GPU CE *only* (the c8a CPU
# fleet keeps the default AMI) via the ff-warm-ami-lt launch template.
FF_BATCH_AMI_ID="${FF_BATCH_AMI_ID:-}"
WARM_AMI_LT_NAME="${FF_WARM_AMI_LT_NAME:-ff-warm-ami-lt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[batch-setup] $*"; }

join_words() {
  local IFS=" "
  echo "$*"
}

json_array() {
  local item json=""
  for item in "$@"; do
    json="${json}\"${item}\","
  done
  printf '[%s]' "${json%,}"
}

DESIRED_INSTANCE_TYPES="$(join_words "${ALL_GPU_INSTANCE_TYPES[@]}")"

# --- 1. Account ID, VPC, subnets ----------------------------------------
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' --output text --region "$REGION")
if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
  log "ERROR: no default VPC in $REGION."
  exit 1
fi
resolve_default_subnets_for_instance_type() {
  local instance_type="$1"
  local offering_azs offering_azs_csv subnet_ids subnet_count

  # Default subnet per AZ, but ONLY in AZs that actually offer this GPU type.
  # Older AZs can lack newer GPU types; including such a subnet makes the Spot
  # fleet throw "InvalidFleetConfiguration - instance type not supported in your
  # requested Availability Zone" on every launch attempt.
  offering_azs=$(aws ec2 describe-instance-type-offerings \
    --location-type availability-zone \
    --filters "Name=instance-type,Values=$instance_type" \
    --query 'InstanceTypeOfferings[].Location' --output text --region "$REGION")
  if [ -z "$offering_azs" ]; then
    log "ERROR: $instance_type is not offered in any AZ of $REGION."
    exit 1
  fi
  offering_azs=$(printf '%s\n' $offering_azs | sort -u)
  offering_azs_csv=$(printf '%s\n' "$offering_azs" | paste -sd, -)
  subnet_ids=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
              "Name=availability-zone,Values=$offering_azs_csv" \
    --query 'Subnets[].SubnetId' --output text --region "$REGION")
  subnet_count=$(echo "$subnet_ids" | wc -w | tr -d ' ')
  if [ "$subnet_count" -lt 2 ]; then
    log "ERROR: need >=2 default subnets in $instance_type-offering AZs for Spot diversification; found $subnet_count (offering AZs: $(printf '%s\n' "$offering_azs" | paste -sd' ' -))."
    exit 1
  fi

  RESOLVED_SUBNET_IDS="$subnet_ids"
  RESOLVED_SUBNETS_JSON=$(printf '"%s",' $subnet_ids | sed 's/,$//')
  RESOLVED_SUBNETS_SORTED=$(printf '%s\n' $subnet_ids | sort | paste -sd' ' -)
  RESOLVED_OFFERING_AZS_DISPLAY=$(printf '%s\n' "$offering_azs" | paste -sd' ' -)
  RESOLVED_SUBNET_COUNT="$subnet_count"
}

log "Account $ACCOUNT_ID, VPC $VPC_ID, GPU Spot pool: ${ALL_GPU_INSTANCE_TYPES[*]} (single diversified CE)"

CPU_OFFERING_AZS=$(aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=$CPU_PRIMARY_INSTANCE_TYPE" \
  --query 'InstanceTypeOfferings[].Location' --output text --region "$REGION")
if [ -z "$CPU_OFFERING_AZS" ]; then
  log "ERROR: $CPU_PRIMARY_INSTANCE_TYPE is not offered in any AZ of $REGION."
  exit 1
fi
CPU_OFFERING_AZS_CSV=$(echo $CPU_OFFERING_AZS | tr ' ' ',')
CPU_SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
            "Name=availability-zone,Values=$CPU_OFFERING_AZS_CSV" \
  --query 'Subnets[].SubnetId' --output text --region "$REGION")
CPU_SUBNET_COUNT=$(echo "$CPU_SUBNET_IDS" | wc -w | tr -d ' ')
if [ "$CPU_SUBNET_COUNT" -lt 2 ]; then
  log "ERROR: need >=2 default subnets in $CPU_PRIMARY_INSTANCE_TYPE-offering AZs for CPU Spot diversification; found $CPU_SUBNET_COUNT (offering AZs: $CPU_OFFERING_AZS)."
  exit 1
fi
log "$CPU_SUBNET_COUNT CPU subnets in $CPU_PRIMARY_INSTANCE_TYPE-offering AZs ($CPU_OFFERING_AZS)"

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
  log "ERROR: Spot G+VT quota is $QUOTA; need >= $MAX_VCPUS (16 x 4-vCPU GPU Spot hosts: $DESIRED_INSTANCE_TYPES)."
  log "Request an increase:"
  log "  aws service-quotas request-service-quota-increase --service-code ec2 \\"
  log "    --quota-code $SPOT_QUOTA_CODE --desired-value $MAX_VCPUS --region $REGION"
  exit 1
fi
log "Quota OK: $QUOTA vCPUs"

log "Checking Standard Spot vCPU quota (need >= $CPU_MAX_VCPUS)..."
CPU_QUOTA=$(aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code "$STANDARD_SPOT_QUOTA_CODE" \
  --region "$REGION" \
  --query 'Quota.Value' \
  --output text)
CPU_QUOTA_INT=${CPU_QUOTA%.*}
if [ "$CPU_QUOTA_INT" -lt "$CPU_MAX_VCPUS" ]; then
  log "ERROR: Standard Spot quota is $CPU_QUOTA; need >= $CPU_MAX_VCPUS (16 x c8a.xlarge x 4 vCPU)."
  log "Request an increase:"
  log "  aws service-quotas request-service-quota-increase --service-code ec2 \\"
  log "    --quota-code $STANDARD_SPOT_QUOTA_CODE --desired-value $CPU_MAX_VCPUS --region $REGION"
  exit 1
fi
log "Standard Spot quota OK: $CPU_QUOTA vCPUs"

# --- 3. BatchTrainingRole (container/job role) --------------------------
if ! aws iam get-role --role-name "$JOB_ROLE" >/dev/null 2>&1; then
  log "Creating IAM role $JOB_ROLE..."
  aws iam create-role \
    --role-name "$JOB_ROLE" \
    --assume-role-policy-document "file://$SCRIPT_DIR/iam-trust-policy-job.json" \
    --description "ff-training Batch job role (S3, ECR pull, CW Logs)"
fi
log "Putting inline policy ff-batch-workload..."
POLICY_FILE="$(mktemp)"
sed -e "s|__REGION__|$REGION|g" "$SCRIPT_DIR/iam-job-policy.json" > "$POLICY_FILE"
aws iam put-role-policy \
  --role-name "$JOB_ROLE" \
  --policy-name ff-batch-workload \
  --policy-document "file://$POLICY_FILE"
rm -f "$POLICY_FILE"

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

# SSM access so operators can `aws ssm start-session` / Run-Command into a Batch
# host for diagnosis — the hosts are egress-only (no SSH ingress; ff-batch-sg) and
# the SSM agent needs this to register with Systems Manager. The agent acquires
# creds at boot, so a host launched after this is SSM-ready; already-running hosts
# must be replaced (Batch is scale-to-zero, so that resolves itself). Idempotent
# (re-attach is a no-op). Mirrors the warm-EC2 path (infra/ec2/launch-instance.sh).
log "Attaching AmazonSSMManagedInstanceCore (host diagnosis via SSM)..."
aws iam attach-role-policy \
  --role-name "$INSTANCE_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore \
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

# --- 10. Compute Environments -------------------------------------------
# NOTE: No launch template / custom UserData. SOCI lazy-loading was removed
# (2026-06-07) — it cannot work on AWS Batch: Batch runs on ECS-managed EC2
# and the amazon-ecs-agent does not pull through the soci snapshotter
# (Fargate-only; aws/containers-roadmap#1832). The CE uses the default
# ECS-optimized AMI. See todo/fixed-archive.md + docs/batch_design.md §2a.
wait_for_compute_environment_valid() {
  local ce_name="$1"
  local attempts="${2:-60}"
  for i in $(seq 1 "$attempts"); do
    STATUS=$(aws batch describe-compute-environments \
      --compute-environments "$ce_name" \
      --region "$REGION" \
      --query 'computeEnvironments[0].status' \
      --output text)
    if [ "$STATUS" = "VALID" ]; then
      return 0
    fi
    if [ "$STATUS" = "INVALID" ]; then
      REASON=$(aws batch describe-compute-environments \
        --compute-environments "$ce_name" \
        --region "$REGION" \
        --query 'computeEnvironments[0].statusReason' \
        --output text)
      log "ERROR: $ce_name is INVALID — $REASON"
      exit 1
    fi
    sleep 5
  done
  log "ERROR: $ce_name did not reach VALID after ~$((attempts * 5))s."
  exit 1
}

# Ensure the warm-AMI launch template exists with ImageId=$1 and set globals
# WARM_AMI_LT_ID + WARM_AMI_LT_VERSION (a concrete numeric version, so the CE
# pins an exact version and a changed AMI deterministically triggers a CE
# reconcile). Idempotent: reuses the latest version when its ImageId already
# matches, else adds a new version. Only called when FF_BATCH_AMI_ID is set, so
# the default (unset) path never touches launch templates.
ensure_warm_ami_launch_template() {
  local ami_id="$1"
  local lt_id current_ami
  lt_id="$(aws ec2 describe-launch-templates \
    --launch-template-names "$WARM_AMI_LT_NAME" \
    --region "$REGION" \
    --query 'LaunchTemplates[0].LaunchTemplateId' \
    --output text 2>/dev/null || echo None)"
  if [ "$lt_id" = "None" ] || [ -z "$lt_id" ]; then
    log "Creating launch template $WARM_AMI_LT_NAME (ImageId=$ami_id)..."
    WARM_AMI_LT_ID="$(aws ec2 create-launch-template \
      --launch-template-name "$WARM_AMI_LT_NAME" \
      --launch-template-data "{\"ImageId\": \"$ami_id\"}" \
      --region "$REGION" \
      --query 'LaunchTemplate.LaunchTemplateId' \
      --output text)"
    WARM_AMI_LT_VERSION="1"
    return
  fi
  current_ami="$(aws ec2 describe-launch-template-versions \
    --launch-template-id "$lt_id" \
    --versions '$Latest' \
    --region "$REGION" \
    --query 'LaunchTemplateVersions[0].LaunchTemplateData.ImageId' \
    --output text 2>/dev/null || echo None)"
  WARM_AMI_LT_ID="$lt_id"
  if [ "$current_ami" = "$ami_id" ]; then
    WARM_AMI_LT_VERSION="$(aws ec2 describe-launch-template-versions \
      --launch-template-id "$lt_id" --versions '$Latest' --region "$REGION" \
      --query 'LaunchTemplateVersions[0].VersionNumber' --output text)"
    log "Launch template $WARM_AMI_LT_NAME already at ImageId=$ami_id (v$WARM_AMI_LT_VERSION)"
    return
  fi
  log "Adding launch template version to $WARM_AMI_LT_NAME (ImageId=$ami_id)..."
  WARM_AMI_LT_VERSION="$(aws ec2 create-launch-template-version \
    --launch-template-id "$lt_id" \
    --launch-template-data "{\"ImageId\": \"$ami_id\"}" \
    --region "$REGION" \
    --query 'LaunchTemplateVersion.VersionNumber' \
    --output text)"
}

ensure_compute_environment() {
  local ce_name="$1"
  # Space-separated list of instance types (e.g. "g6.xlarge g5.xlarge"); all live
  # in this single CE so the Spot allocation strategy can diversify across them.
  local instance_types_str="$2"
  local primary_type="${instance_types_str%% *}"
  local instance_types_json
  instance_types_json="$(json_array $instance_types_str)"
  local desired_types_sorted
  desired_types_sorted="$(printf '%s\n' $instance_types_str | sort | paste -sd' ' -)"

  # Resolve subnets on the representative (primary) type; the listed GPU types are
  # offered in the same us-east-1 AZs, and the allocation strategy simply skips
  # any (type, AZ) pool that isn't offered.
  resolve_default_subnets_for_instance_type "$primary_type"
  local subnet_ids="$RESOLVED_SUBNET_IDS"
  local subnets_json="$RESOLVED_SUBNETS_JSON"
  local subnets_sorted="$RESOLVED_SUBNETS_SORTED"
  log "$ce_name: [$instance_types_str] offered in AZs ($RESOLVED_OFFERING_AZS_DISPLAY); using $RESOLVED_SUBNET_COUNT default subnets"

  # Warm-AMI launch template (opt-in; see FF_BATCH_AMI_ID). When set, attach a
  # version-pinned launch template so fresh Spot hosts boot from the pre-pulled
  # AMI. Empty => no launchTemplate key anywhere => the exact prior behavior.
  local lt_fragment="" lt_desired=""
  if [ -n "$FF_BATCH_AMI_ID" ]; then
    ensure_warm_ami_launch_template "$FF_BATCH_AMI_ID"
    lt_fragment=", \"launchTemplate\": {\"launchTemplateId\": \"$WARM_AMI_LT_ID\", \"version\": \"$WARM_AMI_LT_VERSION\"}"
    lt_desired="$WARM_AMI_LT_ID:$WARM_AMI_LT_VERSION"
    log "$ce_name: attaching warm AMI $FF_BATCH_AMI_ID via $WARM_AMI_LT_NAME v$WARM_AMI_LT_VERSION"
  fi

  CE_STATUS=$(aws batch describe-compute-environments \
    --compute-environments "$ce_name" \
    --region "$REGION" \
    --query 'computeEnvironments[0].status' \
    --output text 2>/dev/null || echo "None")
  if [ "$CE_STATUS" = "None" ] || [ -z "$CE_STATUS" ] || [ "$CE_STATUS" = "null" ]; then
    log "Creating Compute Environment $ce_name ([$instance_types_str])..."
    aws batch create-compute-environment \
      --compute-environment-name "$ce_name" \
      --type MANAGED \
      --state ENABLED \
      --compute-resources "{
        \"type\": \"SPOT\",
        \"allocationStrategy\": \"SPOT_PRICE_CAPACITY_OPTIMIZED\",
        \"minvCpus\": 0,
        \"maxvCpus\": $MAX_VCPUS,
        \"instanceTypes\": $instance_types_json,
        \"subnets\": [$subnets_json],
        \"securityGroupIds\": [\"$SG_ID\"],
        \"instanceRole\": \"arn:aws:iam::$ACCOUNT_ID:instance-profile/$INSTANCE_PROFILE\"$lt_fragment
      }" \
      --region "$REGION"
    log "Waiting for $ce_name to reach VALID..."
    wait_for_compute_environment_valid "$ce_name"
    log "$ce_name is VALID"
  else
    log "Compute Environment $ce_name already exists (status: $CE_STATUS)"
    # Live-account reconcile: bring an existing CE's instanceTypes/subnets in
    # line with the desired value. update-compute-environment with
    # --compute-resources requires the CE to be DISABLED first; cycle DISABLED →
    # UPDATE → ENABLED, polling VALID between steps. CE has minvCpus=0 so no
    # in-flight instances are disrupted — the next provisioning picks up the
    # change.
    CURRENT_INSTANCE_TYPES=$(aws batch describe-compute-environments \
      --compute-environments "$ce_name" \
      --region "$REGION" \
      --query 'computeEnvironments[0].computeResources.instanceTypes' \
      --output text 2>/dev/null || echo "None")
    CURRENT_INSTANCE_TYPES_SORTED=$(printf '%s\n' $CURRENT_INSTANCE_TYPES | sort | paste -sd' ' -)
    CURRENT_SUBNETS=$(aws batch describe-compute-environments \
      --compute-environments "$ce_name" \
      --region "$REGION" \
      --query 'computeEnvironments[0].computeResources.subnets' \
      --output text 2>/dev/null || echo "None")
    CURRENT_SUBNETS_SORTED=$(printf '%s\n' $CURRENT_SUBNETS | sort | paste -sd' ' -)
    CURRENT_MAX=$(aws batch describe-compute-environments \
      --compute-environments "$ce_name" \
      --region "$REGION" \
      --query 'computeEnvironments[0].computeResources.maxvCpus' \
      --output text 2>/dev/null || echo "None")
    # Warm-AMI launch-template diff — only when opted in via FF_BATCH_AMI_ID. An
    # unset FF_BATCH_AMI_ID leaves lt_desired empty, so this never fires and the
    # CE's launch template (whatever it is) is left untouched.
    lt_needs_update=0
    if [ -n "$lt_desired" ]; then
      CURRENT_LT_ID=$(aws batch describe-compute-environments \
        --compute-environments "$ce_name" --region "$REGION" \
        --query 'computeEnvironments[0].computeResources.launchTemplate.launchTemplateId' \
        --output text 2>/dev/null || echo "None")
      CURRENT_LT_VER=$(aws batch describe-compute-environments \
        --compute-environments "$ce_name" --region "$REGION" \
        --query 'computeEnvironments[0].computeResources.launchTemplate.version' \
        --output text 2>/dev/null || echo "None")
      [ "$CURRENT_LT_ID:$CURRENT_LT_VER" != "$lt_desired" ] && lt_needs_update=1
    fi
    if [ "$CURRENT_INSTANCE_TYPES_SORTED" != "$desired_types_sorted" ] || [ "$CURRENT_SUBNETS_SORTED" != "$subnets_sorted" ] || [ "$CURRENT_MAX" != "$MAX_VCPUS" ] || [ "$lt_needs_update" = "1" ]; then
      log "Reconciling $ce_name: instanceTypes ${CURRENT_INSTANCE_TYPES:-None} -> [$instance_types_str]; maxVcpus ${CURRENT_MAX:-None} -> $MAX_VCPUS; subnets refreshed for $primary_type offerings${lt_desired:+; launchTemplate -> $lt_desired}"
      log "  step 1/3: DISABLE"
      aws batch update-compute-environment \
        --compute-environment "$ce_name" \
        --state DISABLED \
        --region "$REGION" >/dev/null
      wait_for_compute_environment_valid "$ce_name" 30
      log "  step 2/3: UPDATE instanceTypes/subnets/maxvCpus"
      aws batch update-compute-environment \
        --compute-environment "$ce_name" \
        --compute-resources "{\"allocationStrategy\": \"SPOT_PRICE_CAPACITY_OPTIMIZED\", \"maxvCpus\": $MAX_VCPUS, \"instanceTypes\": $instance_types_json, \"subnets\": [$subnets_json]$lt_fragment}" \
        --region "$REGION" >/dev/null
      wait_for_compute_environment_valid "$ce_name" 30
      log "  step 3/3: ENABLE"
      aws batch update-compute-environment \
        --compute-environment "$ce_name" \
        --state ENABLED \
        --region "$REGION" >/dev/null
      wait_for_compute_environment_valid "$ce_name" 30
      log "$ce_name reconciled — instanceTypes=[$instance_types_str], maxVcpus=$MAX_VCPUS. Next Spot host picks it up."
    else
      log "$ce_name already matches desired instance types/subnets/maxVcpus ([$instance_types_str], $MAX_VCPUS)."
    fi
  fi

  PRIMARY_SUBNET_IDS="$subnet_ids"
}

ensure_compute_environment "$COMPUTE_ENV" "${ALL_GPU_INSTANCE_TYPES[*]}"

# --- 11. CPU Compute Environment ----------------------------------------
CPU_SUBNETS_JSON=$(printf '"%s",' $CPU_SUBNET_IDS | sed 's/,$//')
CPU_CE_STATUS=$(aws batch describe-compute-environments \
  --compute-environments "$CPU_COMPUTE_ENV" \
  --region "$REGION" \
  --query 'computeEnvironments[0].status' \
  --output text 2>/dev/null || echo "None")
if [ "$CPU_CE_STATUS" = "None" ] || [ -z "$CPU_CE_STATUS" ] || [ "$CPU_CE_STATUS" = "null" ]; then
  log "Creating CPU Compute Environment $CPU_COMPUTE_ENV..."
  aws batch create-compute-environment \
    --compute-environment-name "$CPU_COMPUTE_ENV" \
    --type MANAGED \
    --state ENABLED \
    --compute-resources "{
      \"type\": \"SPOT\",
      \"allocationStrategy\": \"SPOT_PRICE_CAPACITY_OPTIMIZED\",
      \"minvCpus\": 0,
      \"maxvCpus\": $CPU_MAX_VCPUS,
      \"instanceTypes\": $CPU_INSTANCE_TYPES_JSON,
      \"subnets\": [$CPU_SUBNETS_JSON],
      \"securityGroupIds\": [\"$SG_ID\"],
      \"instanceRole\": \"arn:aws:iam::$ACCOUNT_ID:instance-profile/$INSTANCE_PROFILE\"
    }" \
    --region "$REGION"
  log "Waiting for CPU CE to reach VALID..."
  for i in $(seq 1 60); do
    STATUS=$(aws batch describe-compute-environments \
      --compute-environments "$CPU_COMPUTE_ENV" \
      --region "$REGION" \
      --query 'computeEnvironments[0].status' \
      --output text)
    if [ "$STATUS" = "VALID" ]; then
      log "CPU CE is VALID"
      break
    fi
    if [ "$STATUS" = "INVALID" ]; then
      REASON=$(aws batch describe-compute-environments \
        --compute-environments "$CPU_COMPUTE_ENV" \
        --region "$REGION" \
        --query 'computeEnvironments[0].statusReason' \
        --output text)
      log "ERROR: CPU CE is INVALID — $REASON"
      exit 1
    fi
    sleep 5
  done
else
  log "CPU Compute Environment $CPU_COMPUTE_ENV already exists (status: $CPU_CE_STATUS)"
  CURRENT_CPU_MAX=$(aws batch describe-compute-environments \
    --compute-environments "$CPU_COMPUTE_ENV" \
    --region "$REGION" \
    --query 'computeEnvironments[0].computeResources.maxvCpus' \
    --output text 2>/dev/null || echo "0")
  CURRENT_CPU_TYPES=$(aws batch describe-compute-environments \
    --compute-environments "$CPU_COMPUTE_ENV" \
    --region "$REGION" \
    --query 'join(`,`, computeEnvironments[0].computeResources.instanceTypes)' \
    --output text 2>/dev/null || echo "")
  DESIRED_CPU_TYPES="$CPU_PRIMARY_INSTANCE_TYPE,$CPU_FALLBACK_INSTANCE_TYPE"
  if [ "$CURRENT_CPU_MAX" != "$CPU_MAX_VCPUS" ] || [ "$CURRENT_CPU_TYPES" != "$DESIRED_CPU_TYPES" ]; then
    log "Reconciling CPU CE resources: maxVcpus $CURRENT_CPU_MAX -> $CPU_MAX_VCPUS, instanceTypes [$CURRENT_CPU_TYPES] -> [$DESIRED_CPU_TYPES]..."
    log "  step 1/3: DISABLE"
    aws batch update-compute-environment \
      --compute-environment "$CPU_COMPUTE_ENV" \
      --state DISABLED \
      --region "$REGION" >/dev/null
    for i in $(seq 1 30); do
      STATUS=$(aws batch describe-compute-environments \
        --compute-environments "$CPU_COMPUTE_ENV" \
        --region "$REGION" \
        --query 'computeEnvironments[0].status' \
        --output text)
      [ "$STATUS" = "VALID" ] && break
      sleep 5
    done
    log "  step 2/3: UPDATE compute resources"
    aws batch update-compute-environment \
      --compute-environment "$CPU_COMPUTE_ENV" \
      --compute-resources "{\"maxvCpus\": $CPU_MAX_VCPUS, \"instanceTypes\": $CPU_INSTANCE_TYPES_JSON}" \
      --region "$REGION" >/dev/null
    for i in $(seq 1 30); do
      STATUS=$(aws batch describe-compute-environments \
        --compute-environments "$CPU_COMPUTE_ENV" \
        --region "$REGION" \
        --query 'computeEnvironments[0].status' \
        --output text)
      [ "$STATUS" = "VALID" ] && break
      sleep 5
    done
    log "  step 3/3: ENABLE"
    aws batch update-compute-environment \
      --compute-environment "$CPU_COMPUTE_ENV" \
      --state ENABLED \
      --region "$REGION" >/dev/null
    for i in $(seq 1 30); do
      STATUS=$(aws batch describe-compute-environments \
        --compute-environments "$CPU_COMPUTE_ENV" \
        --region "$REGION" \
        --query 'computeEnvironments[0].status' \
        --output text)
      [ "$STATUS" = "VALID" ] && break
      sleep 5
    done
    log "CPU CE reconciled."
  else
    log "CPU CE already matches desired resources."
  fi
fi

# --- 12. Job Queue ------------------------------------------------------
CE_ORDER_ARGS=(
  "order=1,computeEnvironment=$COMPUTE_ENV"
)
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
    --compute-environment-order "${CE_ORDER_ARGS[@]}" \
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
  log "Reconciling Job Queue compute environment: single CE $COMPUTE_ENV (g6+g5 diversified Spot pool)"
  aws batch update-job-queue \
    --job-queue "$JOB_QUEUE" \
    --state ENABLED \
    --compute-environment-order "${CE_ORDER_ARGS[@]}" \
    --region "$REGION" >/dev/null
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
fi

CPU_JQ_STATUS=$(aws batch describe-job-queues \
  --job-queues "$CPU_JOB_QUEUE" \
  --region "$REGION" \
  --query 'jobQueues[0].status' \
  --output text 2>/dev/null || echo "None")
if [ "$CPU_JQ_STATUS" = "None" ] || [ -z "$CPU_JQ_STATUS" ] || [ "$CPU_JQ_STATUS" = "null" ]; then
  log "Creating CPU Job Queue $CPU_JOB_QUEUE..."
  aws batch create-job-queue \
    --job-queue-name "$CPU_JOB_QUEUE" \
    --state ENABLED \
    --priority 1 \
    --compute-environment-order "order=1,computeEnvironment=$CPU_COMPUTE_ENV" \
    --region "$REGION"
  log "Waiting for CPU JQ to reach VALID..."
  for i in $(seq 1 30); do
    STATUS=$(aws batch describe-job-queues \
      --job-queues "$CPU_JOB_QUEUE" \
      --region "$REGION" \
      --query 'jobQueues[0].status' \
      --output text)
    if [ "$STATUS" = "VALID" ]; then
      log "CPU JQ is VALID"
      break
    fi
    sleep 5
  done
else
  log "CPU Job Queue $CPU_JOB_QUEUE already exists (status: $CPU_JQ_STATUS)"
fi

# --- 13. Job Definition (register rev 1 against :latest) ----------------
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
    --retry-strategy '{
      "attempts": 3,
      "evaluateOnExit": [
        {"onStatusReason": "Host EC2*", "action": "RETRY"},
        {"onReason": "CannotPullContainerError*", "action": "RETRY"},
        {"onReason": "*", "action": "EXIT"}
      ]
    }' \
    --region "$REGION"
else
  log "Job Definition $JOB_DEF already exists; CI will re-register on next push."
fi

CPU_JD_NAME=$(aws batch describe-job-definitions \
  --job-definition-name "$CPU_JOB_DEF" \
  --status ACTIVE \
  --max-results 1 \
  --region "$REGION" \
  --query 'jobDefinitions[0].jobDefinitionName' \
  --output text 2>/dev/null || echo "None")
if [ "$CPU_JD_NAME" = "None" ] || [ -z "$CPU_JD_NAME" ] || [ "$CPU_JD_NAME" = "null" ]; then
  log "Registering CPU Job Definition $CPU_JOB_DEF rev 1..."
  aws batch register-job-definition \
    --job-definition-name "$CPU_JOB_DEF" \
    --type container \
    --platform-capabilities EC2 \
    --container-properties "{
      \"image\": \"$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest\",
      \"vcpus\": 4,
      \"memory\": 7500,
      \"jobRoleArn\": \"arn:aws:iam::$ACCOUNT_ID:role/$JOB_ROLE\",
      \"executionRoleArn\": \"arn:aws:iam::$ACCOUNT_ID:role/$TASK_EXEC_ROLE\",
      \"environment\": [
        {\"name\": \"REQUIRE_GPU\", \"value\": \"0\"},
        {\"name\": \"FF_DEVICE\", \"value\": \"cpu\"},
        {\"name\": \"FF_CPU_BRANCH_CORES\", \"value\": \"4\"},
        {\"name\": \"LGBM_N_JOBS\", \"value\": \"1\"},
        {\"name\": \"LOKY_MAX_CPU_COUNT\", \"value\": \"4\"},
        {\"name\": \"OPENBLAS_NUM_THREADS\", \"value\": \"1\"},
        {\"name\": \"OMP_NUM_THREADS\", \"value\": \"1\"},
        {\"name\": \"MKL_NUM_THREADS\", \"value\": \"1\"},
        {\"name\": \"NUMEXPR_NUM_THREADS\", \"value\": \"1\"}
      ],
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
    --retry-strategy '{
      "attempts": 3,
      "evaluateOnExit": [
        {"onStatusReason": "Host EC2*", "action": "RETRY"},
        {"onReason": "CannotPullContainerError*", "action": "RETRY"},
        {"onReason": "*", "action": "EXIT"}
      ]
    }' \
    --region "$REGION"
else
  log "CPU Job Definition $CPU_JOB_DEF already exists; CI will re-register on next push."
fi

cat <<EOF

────────────────────────────────────────────────────────────────
Batch + Spot infrastructure ready:
  Region:              $REGION
  GPU CE:              $COMPUTE_ENV (types=${ALL_GPU_INSTANCE_TYPES[*]}, maxVcpus=$MAX_VCPUS, SPOT_PRICE_CAPACITY_OPTIMIZED)
  Job queue:           $JOB_QUEUE
  Job definition:      $JOB_DEF (image: $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest)
  CPU environment:     $CPU_COMPUTE_ENV (maxVcpus=$CPU_MAX_VCPUS, types=$CPU_PRIMARY_INSTANCE_TYPE,$CPU_FALLBACK_INSTANCE_TYPE)
  CPU job queue:       $CPU_JOB_QUEUE
  CPU job definition:  $CPU_JOB_DEF (4 vCPU, 7500 MiB, no GPU)
  Job role:            arn:aws:iam::$ACCOUNT_ID:role/$JOB_ROLE
  Instance role:       arn:aws:iam::$ACCOUNT_ID:role/$INSTANCE_ROLE
  Task exec role:      arn:aws:iam::$ACCOUNT_ID:role/$TASK_EXEC_ROLE
  Security group:      $SG_ID
  GPU subnets:         $PRIMARY_SUBNET_IDS

Next steps:
  1. (Cold-start opt) Create ECR pull-through cache for the PyTorch base image:
       aws ecr create-pull-through-cache-rule \\
         --ecr-repository-prefix dockerhub \\
         --upstream-registry-url registry-1.docker.io \\
         --region $REGION
  2. Verify CE and JQ are VALID:
       aws batch describe-compute-environments --compute-environments $COMPUTE_ENV \\
         --query 'computeEnvironments[].{name:computeEnvironmentName,state:state,status:status,instanceTypes:computeResources.instanceTypes}' --region $REGION
  3. Smoke test (single cheap position, ~2-3 min):
       AWS_REGION=$REGION python -m src.batch.launch --positions K --seed 42
  4. When ready to flip the active trainer from EC2 to Batch:
       gh variable set BATCH_ACTIVE --body "true"
────────────────────────────────────────────────────────────────
EOF
