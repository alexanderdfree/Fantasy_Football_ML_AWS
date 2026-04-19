#!/usr/bin/env bash
# Bootstrap the ff-training EC2 g4dn.xlarge: quota check, IAM, SG, instance.
# Idempotent — reruns skip anything that already exists.
#
# Prereqs:
#   - AWS CLI v2 with credentials for the target account.
#   - G-instance vCPU quota >= 4 (this script refuses to proceed otherwise).
#   - S3 bucket ff-predictor-training exists (it already does — training uses it).
#
# Run from the repo root:  bash infra/ec2/launch-instance.sh

set -euo pipefail

REGION="us-east-1"
BUCKET="ff-predictor-training"
INFRA_PREFIX="infra/ec2"
INSTANCE_TYPE="g4dn.xlarge"
ROLE_NAME="ff-training-ec2-role"
PROFILE_NAME="ff-training-ec2-profile"
SG_NAME="ff-training-ec2-sg"
KEY_NAME="ff-training-ec2-key"
LOG_GROUP="/ff/training"
INSTANCE_NAME="ff-training"
# "Running On-Demand G and VT instances" quota code.
GPU_QUOTA_CODE="L-DB2E81BA"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[launch-instance] $*"; }

# --- 1. Quota check (memory: us-east-1 quota was 0) ---------------------
log "Checking on-demand G-instance vCPU quota..."
QUOTA=$(aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code "$GPU_QUOTA_CODE" \
  --region "$REGION" \
  --query 'Quota.Value' \
  --output text)
QUOTA_INT=${QUOTA%.*}
if [ "$QUOTA_INT" -lt 4 ]; then
  log "ERROR: G-instance quota is $QUOTA; need >= 4 (g4dn.xlarge uses 4 vCPU)."
  log "Your quota increase may not be approved yet. Check:"
  log "  aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region $REGION"
  exit 1
fi
log "Quota OK: $QUOTA vCPUs"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' --output text --region "$REGION")
if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
  log "ERROR: no default VPC in $REGION."
  exit 1
fi
log "Account $ACCOUNT_ID, default VPC $VPC_ID"

# --- 2. Upload companion files to S3 ------------------------------------
log "Uploading companion files to s3://$BUCKET/$INFRA_PREFIX/..."
for f in cloudwatch-agent.json auto-shutdown.sh auto-shutdown.service auto-shutdown.timer; do
  aws s3 cp "$SCRIPT_DIR/$f" "s3://$BUCKET/$INFRA_PREFIX/$f" --region "$REGION"
done

# --- 3. IAM role + instance profile -------------------------------------
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  log "Creating IAM role $ROLE_NAME..."
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "file://$SCRIPT_DIR/iam-trust-policy.json" \
    --description "ff-training EC2 trainer (S3, ECR pull, CW Logs, SSM)"
fi

log "Attaching managed policies..."
for p in \
  arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore \
  arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy; do
  aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$p" || true
done

log "Putting inline policy ff-training-workload..."
aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name ff-training-workload \
  --policy-document "file://$SCRIPT_DIR/iam-instance-policy.json"

if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
  log "Creating instance profile $PROFILE_NAME..."
  aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME"
  aws iam add-role-to-instance-profile \
    --instance-profile-name "$PROFILE_NAME" \
    --role-name "$ROLE_NAME"
  # IAM propagation is slow — pause so run-instances doesn't race.
  sleep 10
fi

# --- 4. Security group --------------------------------------------------
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text --region "$REGION" 2>/dev/null || echo "None")
if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
  log "Creating security group $SG_NAME..."
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "ff-training trainer (egress only; SSM is mgmt channel)" \
    --vpc-id "$VPC_ID" \
    --region "$REGION" \
    --query GroupId --output text)
fi
log "Security group: $SG_ID (no ingress rules — SSM only)"

# --- 5. Key pair (break-glass only) -------------------------------------
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" >/dev/null 2>&1; then
  log "Creating break-glass key pair $KEY_NAME and storing in Secrets Manager..."
  KEY_MATERIAL=$(aws ec2 create-key-pair \
    --key-name "$KEY_NAME" \
    --region "$REGION" \
    --query KeyMaterial --output text)
  aws secretsmanager create-secret \
    --name "ff/training-ec2/ssh-key" \
    --description "Break-glass SSH key for ff-training EC2" \
    --secret-string "$KEY_MATERIAL" \
    --region "$REGION" || \
  aws secretsmanager put-secret-value \
    --secret-id "ff/training-ec2/ssh-key" \
    --secret-string "$KEY_MATERIAL" \
    --region "$REGION"
fi

# --- 6. Log group retention (pre-create so user-data ships logs) --------
aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$REGION" 2>/dev/null || true
aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 7 --region "$REGION"

# --- 7. Resolve latest DLAMI GPU PyTorch (Ubuntu 22.04) -----------------
# AMI name format (2026): "Deep Learning OSS Nvidia Driver AMI GPU PyTorch
# 2.7 (Ubuntu 22.04) 20260417". Filter explicitly excludes the ARM64 images,
# which have "ARM64" in the name and x86_64-only g4dn can't boot them.
log "Resolving latest Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
  --owners 898082745236 \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch* (Ubuntu 22.04)*" \
            "Name=architecture,Values=x86_64" \
            "Name=state,Values=available" \
  --query 'Images | sort_by(@,&CreationDate) | [-1].ImageId' \
  --output text --region "$REGION")
if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
  log "ERROR: could not resolve DLAMI."
  exit 1
fi
log "AMI: $AMI_ID"

# --- 8. Launch (skip if an instance with Name=ff-training already exists) ---
EXISTING_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$INSTANCE_NAME" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text --region "$REGION" 2>/dev/null || echo "None")
if [ "$EXISTING_ID" != "None" ] && [ -n "$EXISTING_ID" ]; then
  log "Instance already exists: $EXISTING_ID — not launching a second one."
  INSTANCE_ID="$EXISTING_ID"
else
  log "Launching $INSTANCE_TYPE..."
  INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile "Name=$PROFILE_NAME" \
    --user-data "file://$SCRIPT_DIR/user-data.sh" \
    --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}' \
    --disable-api-termination \
    --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Project,Value=fantasy-predictor},{Key=Role,Value=trainer},{Key=ManagedBy,Value=infra-ec2}]" \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' --output text)
  log "Launched: $INSTANCE_ID"
fi

cat <<EOF

────────────────────────────────────────────────────────────────
Launch summary:
  Instance ID:         $INSTANCE_ID
  AMI:                 $AMI_ID
  Instance type:       $INSTANCE_TYPE
  Security group:      $SG_ID
  Instance profile:    $PROFILE_NAME
  Log group:           $LOG_GROUP (7-day retention)
  Break-glass SSH key: secret ff/training-ec2/ssh-key

Next steps:
  1. Wait for state=running + SSM PingStatus=Online (~2-3 min):
       aws ssm describe-instance-information \\
         --filters "Key=InstanceIds,Values=$INSTANCE_ID" --region $REGION
  2. Verify bootstrap breadcrumb:
       aws ssm start-session --target $INSTANCE_ID
       # inside session:
       test -f /opt/ff/config/bootstrap-complete && echo OK
  3. Set the GitHub repo variable:
       gh variable set EC2_TRAINER_INSTANCE_ID --body "$INSTANCE_ID"
       gh variable set BATCH_ACTIVE --body "false"
  4. Smoke test: /usr/local/bin/ff-train K 42
  5. Enable auto-shutdown after you've observed usage for a day:
       aws ssm send-command --targets "Key=InstanceIds,Values=$INSTANCE_ID" \\
         --document-name AWS-RunShellScript \\
         --parameters 'commands=["systemctl enable --now ff-auto-shutdown.timer"]' \\
         --region $REGION
────────────────────────────────────────────────────────────────
EOF
