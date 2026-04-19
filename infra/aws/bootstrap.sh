#!/usr/bin/env bash
# Stand up the full ECS/ALB/ACM stack for alexfree.me.
# Idempotent — every step is gated on `describe || create`, so reruns are safe.
#
# Prereqs (run before this script):
#   1. Gap 1 merged to main (shared/model_sync.py exists in the image).
#   2. infra/aws/seed_s3_models.sh executed (S3 has all 6 model tarballs).
#   3. One seed ARM64 image pushed to ECR:
#        aws ecr get-login-password --region us-east-1 \
#          | docker login --username AWS --password-stdin \
#              "$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com"
#        docker buildx build --platform linux/arm64 \
#          -t "$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fantasy-predictor:bootstrap" \
#          --push .
#
# Outputs: infra/aws/.env.out with resource IDs; final line prints ALB DNS.
# Run from the repo root:  bash infra/aws/bootstrap.sh

set -euo pipefail

REGION="us-east-1"
ECR_REPO="fantasy-predictor"
CLUSTER="fantasy-cluster"
SERVICE="fantasy-service"
TASK_FAMILY="fantasy-predictor"
LOG_GROUP="/ecs/fantasy-predictor"
EXEC_ROLE="ecsTaskExecutionRole"
TASK_ROLE="fantasyTaskRole"
ALB_NAME="fantasy-alb"
TG_NAME="fantasy-tg"
ALB_SG_NAME="fantasy-alb-sg"
ECS_SG_NAME="fantasy-ecs-sg"
S3_BUCKET="ff-predictor-training"
DOMAIN="alexfree.me"
DOMAIN_WWW="www.alexfree.me"
IMAGE_TAG="${IMAGE_TAG:-bootstrap}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="$SCRIPT_DIR/.env.out"
: > "$OUT_FILE"

log() { echo "[bootstrap] $*"; }
out() { echo "$1=$2" >> "$OUT_FILE"; }

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
log "Account $ACCOUNT_ID in $REGION"
out ACCOUNT_ID "$ACCOUNT_ID"

# ---------------------------------------------------------------------------
# Step 1: ECR repo + verify seed image exists
# ---------------------------------------------------------------------------
if ! aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" >/dev/null 2>&1; then
  log "Creating ECR repo $ECR_REPO..."
  aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION" >/dev/null
fi
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO"
out ECR_URI "$ECR_URI"

if ! aws ecr describe-images --repository-name "$ECR_REPO" --image-ids imageTag="$IMAGE_TAG" --region "$REGION" >/dev/null 2>&1; then
  log "ERROR: no $ECR_URI:$IMAGE_TAG image found. Push the seed image first (see header)."
  exit 1
fi
log "ECR seed image verified: $ECR_URI:$IMAGE_TAG"

# ---------------------------------------------------------------------------
# Step 2: IAM roles
# ---------------------------------------------------------------------------
if ! aws iam get-role --role-name "$EXEC_ROLE" >/dev/null 2>&1; then
  log "Creating $EXEC_ROLE..."
  aws iam create-role \
    --role-name "$EXEC_ROLE" \
    --assume-role-policy-document "file://$SCRIPT_DIR/task-trust-policy.json" >/dev/null
  aws iam attach-role-policy \
    --role-name "$EXEC_ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
fi
EXEC_ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/$EXEC_ROLE"
out EXEC_ROLE_ARN "$EXEC_ROLE_ARN"

if ! aws iam get-role --role-name "$TASK_ROLE" >/dev/null 2>&1; then
  log "Creating $TASK_ROLE (S3 read for models)..."
  aws iam create-role \
    --role-name "$TASK_ROLE" \
    --assume-role-policy-document "file://$SCRIPT_DIR/task-trust-policy.json" >/dev/null
fi
aws iam put-role-policy \
  --role-name "$TASK_ROLE" \
  --policy-name "fantasy-s3-read" \
  --policy-document "file://$SCRIPT_DIR/task-role-policy.json"
TASK_ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/$TASK_ROLE"
out TASK_ROLE_ARN "$TASK_ROLE_ARN"

# ---------------------------------------------------------------------------
# Step 3: CloudWatch log group
# ---------------------------------------------------------------------------
if ! aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --region "$REGION" \
     --query 'logGroups[?logGroupName==`'"$LOG_GROUP"'`]' --output text | grep -q "$LOG_GROUP"; then
  log "Creating log group $LOG_GROUP..."
  aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$REGION"
  aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 14 --region "$REGION"
fi

# ---------------------------------------------------------------------------
# Step 4: ECS cluster
# ---------------------------------------------------------------------------
STATUS=$(aws ecs describe-clusters --clusters "$CLUSTER" --region "$REGION" \
  --query 'clusters[0].status' --output text 2>/dev/null || echo NONE)
if [ "$STATUS" != "ACTIVE" ]; then
  log "Creating ECS cluster $CLUSTER..."
  aws ecs create-cluster --cluster-name "$CLUSTER" --region "$REGION" >/dev/null
fi

# ---------------------------------------------------------------------------
# Step 5: VPC / subnets / security groups
# ---------------------------------------------------------------------------
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' --output text --region "$REGION")
if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
  log "ERROR: no default VPC in $REGION. Create one or edit this script to target a specific VPC."
  exit 1
fi
out VPC_ID "$VPC_ID"

# Pick 2 public subnets in different AZs.
SUBNET_IDS=$(aws ec2 describe-subnets --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=map-public-ip-on-launch,Values=true" \
  --query 'Subnets[].[SubnetId,AvailabilityZone]' --output text \
  | sort -u -k2,2 | head -2 | awk '{print $1}' | paste -sd, -)
if [ "$(echo "$SUBNET_IDS" | tr ',' '\n' | wc -l)" -lt 2 ]; then
  log "ERROR: need at least 2 public subnets in different AZs in VPC $VPC_ID."
  exit 1
fi
out SUBNET_IDS "$SUBNET_IDS"
log "Using subnets: $SUBNET_IDS"

ALB_SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=$ALB_SG_NAME" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)
if [ "$ALB_SG_ID" = "None" ] || [ -z "$ALB_SG_ID" ]; then
  log "Creating SG $ALB_SG_NAME..."
  ALB_SG_ID=$(aws ec2 create-security-group --region "$REGION" \
    --group-name "$ALB_SG_NAME" --vpc-id "$VPC_ID" \
    --description "ALB ingress 80/443" --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$ALB_SG_ID" \
    --protocol tcp --port 80  --cidr 0.0.0.0/0 >/dev/null
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$ALB_SG_ID" \
    --protocol tcp --port 443 --cidr 0.0.0.0/0 >/dev/null
fi
out ALB_SG_ID "$ALB_SG_ID"

ECS_SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=$ECS_SG_NAME" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)
if [ "$ECS_SG_ID" = "None" ] || [ -z "$ECS_SG_ID" ]; then
  log "Creating SG $ECS_SG_NAME..."
  ECS_SG_ID=$(aws ec2 create-security-group --region "$REGION" \
    --group-name "$ECS_SG_NAME" --vpc-id "$VPC_ID" \
    --description "ECS tasks, :8000 from ALB SG only" --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$ECS_SG_ID" \
    --protocol tcp --port 8000 --source-group "$ALB_SG_ID" >/dev/null
fi
out ECS_SG_ID "$ECS_SG_ID"

# ---------------------------------------------------------------------------
# Step 6: ALB + target group
# ---------------------------------------------------------------------------
ALB_ARN=$(aws elbv2 describe-load-balancers --region "$REGION" \
  --names "$ALB_NAME" --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null || echo None)
if [ "$ALB_ARN" = "None" ] || [ -z "$ALB_ARN" ]; then
  log "Creating ALB $ALB_NAME..."
  ALB_ARN=$(aws elbv2 create-load-balancer --region "$REGION" \
    --name "$ALB_NAME" --type application --scheme internet-facing \
    --security-groups "$ALB_SG_ID" \
    --subnets $(echo "$SUBNET_IDS" | tr ',' ' ') \
    --query 'LoadBalancers[0].LoadBalancerArn' --output text)
fi
ALB_DNS=$(aws elbv2 describe-load-balancers --region "$REGION" \
  --load-balancer-arns "$ALB_ARN" --query 'LoadBalancers[0].DNSName' --output text)
out ALB_ARN "$ALB_ARN"
out ALB_DNS "$ALB_DNS"
log "ALB: $ALB_DNS"

TG_ARN=$(aws elbv2 describe-target-groups --region "$REGION" \
  --names "$TG_NAME" --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null || echo None)
if [ "$TG_ARN" = "None" ] || [ -z "$TG_ARN" ]; then
  log "Creating target group $TG_NAME..."
  TG_ARN=$(aws elbv2 create-target-group --region "$REGION" \
    --name "$TG_NAME" --protocol HTTP --port 8000 \
    --vpc-id "$VPC_ID" --target-type ip \
    --health-check-path /health --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 5 --healthy-threshold-count 2 --unhealthy-threshold-count 3 \
    --query 'TargetGroups[0].TargetGroupArn' --output text)
fi
out TG_ARN "$TG_ARN"

# ---------------------------------------------------------------------------
# Step 7: ACM cert for alexfree.me (+ www) — DNS validation
# ---------------------------------------------------------------------------
CERT_ARN=$(aws acm list-certificates --region "$REGION" \
  --query "CertificateSummaryList[?DomainName==\`$DOMAIN\`].CertificateArn | [0]" --output text)
if [ "$CERT_ARN" = "None" ] || [ -z "$CERT_ARN" ]; then
  log "Requesting ACM cert for $DOMAIN + $DOMAIN_WWW..."
  CERT_ARN=$(aws acm request-certificate --region "$REGION" \
    --domain-name "$DOMAIN" \
    --subject-alternative-names "$DOMAIN_WWW" \
    --validation-method DNS \
    --query CertificateArn --output text)
  sleep 5
fi
out CERT_ARN "$CERT_ARN"

log "Fetching ACM DNS validation records..."
for i in $(seq 1 30); do
  RECORDS=$(aws acm describe-certificate --region "$REGION" --certificate-arn "$CERT_ARN" \
    --query 'Certificate.DomainValidationOptions[].ResourceRecord' --output json)
  if [ "$(echo "$RECORDS" | jq 'length')" -gt 0 ] \
     && [ "$(echo "$RECORDS" | jq '[.[] | select(.Name != null)] | length')" -gt 0 ]; then
    break
  fi
  sleep 5
done

echo ""
echo "======================================================================"
echo "ACM requires DNS validation. Add these records in Namecheap"
echo "(Domain List -> $DOMAIN -> Manage -> Advanced DNS). For each CNAME,"
echo "Namecheap auto-appends \".$DOMAIN\" — paste only the short host."
echo "----------------------------------------------------------------------"
echo "$RECORDS" | jq -r '.[] | "  Host:  \(.Name)\n  Value: \(.Value)\n  Type:  \(.Type)\n"'
echo "======================================================================"
read -r -p "Press ENTER once the records are in Namecheap... "

log "Polling ACM for ISSUED status (up to 40 min)..."
for i in $(seq 1 80); do
  STATUS=$(aws acm describe-certificate --region "$REGION" --certificate-arn "$CERT_ARN" \
    --query Certificate.Status --output text)
  log "ACM status: $STATUS (attempt $i/80)"
  if [ "$STATUS" = "ISSUED" ]; then break; fi
  if [ "$STATUS" = "FAILED" ]; then log "ERROR: ACM validation failed."; exit 1; fi
  sleep 30
done
[ "$STATUS" = "ISSUED" ] || { log "ERROR: ACM not ISSUED after polling."; exit 1; }

# ---------------------------------------------------------------------------
# Step 8: Listeners (443 HTTPS -> TG, 80 -> redirect 443)
# ---------------------------------------------------------------------------
HTTPS_ARN=$(aws elbv2 describe-listeners --region "$REGION" --load-balancer-arn "$ALB_ARN" \
  --query "Listeners[?Port==\`443\`].ListenerArn | [0]" --output text 2>/dev/null || echo None)
if [ "$HTTPS_ARN" = "None" ] || [ -z "$HTTPS_ARN" ]; then
  log "Creating HTTPS listener..."
  aws elbv2 create-listener --region "$REGION" --load-balancer-arn "$ALB_ARN" \
    --protocol HTTPS --port 443 \
    --certificates "CertificateArn=$CERT_ARN" \
    --ssl-policy ELBSecurityPolicy-TLS13-1-2-2021-06 \
    --default-actions "Type=forward,TargetGroupArn=$TG_ARN" >/dev/null
fi

HTTP_ARN=$(aws elbv2 describe-listeners --region "$REGION" --load-balancer-arn "$ALB_ARN" \
  --query "Listeners[?Port==\`80\`].ListenerArn | [0]" --output text 2>/dev/null || echo None)
if [ "$HTTP_ARN" = "None" ] || [ -z "$HTTP_ARN" ]; then
  log "Creating HTTP->HTTPS redirect listener..."
  aws elbv2 create-listener --region "$REGION" --load-balancer-arn "$ALB_ARN" \
    --protocol HTTP --port 80 \
    --default-actions 'Type=redirect,RedirectConfig={Protocol=HTTPS,Port=443,StatusCode=HTTP_301}' >/dev/null
fi

# ---------------------------------------------------------------------------
# Step 9: Task definition
# ---------------------------------------------------------------------------
# Require all 6 tarballs in S3. Model .pkl/.pt files are gitignored, so
# images built by CI (from a fresh git clone) contain NO model fallback.
# Only the locally-built seed image has models baked in, and we can't rely on
# that surviving the first deploy.yml run.
for POS in QB RB WR TE K DST; do
  if ! aws s3api head-object --bucket "$S3_BUCKET" --key "models/$POS/model.tar.gz" \
       --region "$REGION" >/dev/null 2>&1; then
    log "ERROR: s3://$S3_BUCKET/models/$POS/model.tar.gz missing."
    log "Run: bash infra/aws/seed_s3_models.sh  (then re-run this script)"
    exit 1
  fi
done
TASK_BUCKET_VALUE="$S3_BUCKET"
log "All 6 S3 tarballs present — task will sync from S3 at boot."

TASK_DEF_JSON=$(sed \
  -e "s|__ACCOUNT_ID__|$ACCOUNT_ID|g" \
  -e "s|__ECR_URI__|$ECR_URI|g" \
  -e "s|__IMAGE_TAG__|$IMAGE_TAG|g" \
  -e "s|__FF_MODEL_S3_BUCKET__|$TASK_BUCKET_VALUE|g" \
  "$SCRIPT_DIR/task-definition.json")
TASK_DEF_FILE="$(mktemp)"
echo "$TASK_DEF_JSON" > "$TASK_DEF_FILE"
log "Registering task definition $TASK_FAMILY..."
TASK_DEF_ARN=$(aws ecs register-task-definition --region "$REGION" \
  --cli-input-json "file://$TASK_DEF_FILE" \
  --query 'taskDefinition.taskDefinitionArn' --output text)
rm -f "$TASK_DEF_FILE"
out TASK_DEF_ARN "$TASK_DEF_ARN"

# ---------------------------------------------------------------------------
# Step 10: ECS service
# ---------------------------------------------------------------------------
SVC_STATUS=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
  --query 'services[0].status' --output text 2>/dev/null || echo MISSING)
if [ "$SVC_STATUS" = "ACTIVE" ]; then
  log "Service exists — updating to new task def + forcing new deployment..."
  aws ecs update-service --region "$REGION" \
    --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TASK_DEF_ARN" --force-new-deployment >/dev/null
else
  log "Creating ECS service $SERVICE..."
  aws ecs create-service --region "$REGION" \
    --cluster "$CLUSTER" --service-name "$SERVICE" \
    --task-definition "$TASK_DEF_ARN" \
    --desired-count 1 --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$(echo "$SUBNET_IDS" | tr ',' ' ' | sed 's/ /,/g')],securityGroups=[$ECS_SG_ID],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=$TG_ARN,containerName=fantasy-predictor,containerPort=8000" \
    --health-check-grace-period-seconds 120 >/dev/null
fi

log "Waiting for service to stabilize (can take 3-5 min)..."
aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "DONE. Resource IDs saved to $OUT_FILE"
echo "----------------------------------------------------------------------"
echo "ALB DNS:   $ALB_DNS"
echo ""
echo "Next: add these records in Namecheap Advanced DNS for $DOMAIN:"
echo "  ALIAS  @    -> $ALB_DNS"
echo "  CNAME  www  -> $ALB_DNS"
echo "Then remove any A record pointing to 192.64.119.87 (Namecheap parking)."
echo ""
echo "Smoke test (works now without DNS, will warn on cert):"
echo "  curl -I http://$ALB_DNS/health"
echo "======================================================================"
