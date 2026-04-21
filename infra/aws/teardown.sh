#!/usr/bin/env bash
# Cost-control teardown: stops the meter on the expensive resources (Fargate
# service, ALB). Preserves ECR images, IAM roles, ACM cert, log group, and
# cluster so re-running bootstrap.sh is fast.
#
# After this completes:
#   - alexfree.me stops serving (ALB is gone).
#   - You pay ~$0/month baseline.
#   - Re-run bootstrap.sh to bring it back (ACM validation NOT re-required).
#
# Run from the repo root:  bash infra/aws/teardown.sh

set -euo pipefail

REGION="us-east-1"
CLUSTER="fantasy-cluster"
SERVICE="fantasy-service"
ALB_NAME="fantasy-alb"
TG_NAME="fantasy-tg"
ALB_SG_NAME="fantasy-alb-sg"
ECS_SG_NAME="fantasy-ecs-sg"

log() { echo "[teardown] $*"; }

# Step 1: Scale service to 0, then delete it.
if aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
     --query 'services[0].status' --output text 2>/dev/null | grep -q ACTIVE; then
  log "Scaling $SERVICE to 0..."
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
    --desired-count 0 --region "$REGION" >/dev/null
  log "Deleting $SERVICE..."
  aws ecs delete-service --cluster "$CLUSTER" --service "$SERVICE" \
    --force --region "$REGION" >/dev/null
fi

# Step 2: Listeners + ALB.
ALB_ARN=$(aws elbv2 describe-load-balancers --region "$REGION" \
  --names "$ALB_NAME" --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null || echo None)
if [ "$ALB_ARN" != "None" ] && [ -n "$ALB_ARN" ]; then
  for LSN in $(aws elbv2 describe-listeners --load-balancer-arn "$ALB_ARN" --region "$REGION" \
                 --query 'Listeners[].ListenerArn' --output text); do
    log "Deleting listener $LSN..."
    aws elbv2 delete-listener --listener-arn "$LSN" --region "$REGION"
  done
  log "Deleting ALB $ALB_NAME..."
  aws elbv2 delete-load-balancer --load-balancer-arn "$ALB_ARN" --region "$REGION"
  # ALB must be fully deleted before we can remove its security group.
  aws elbv2 wait load-balancers-deleted --load-balancer-arns "$ALB_ARN" --region "$REGION" 2>/dev/null || true
fi

# Step 3: Target group.
TG_ARN=$(aws elbv2 describe-target-groups --region "$REGION" \
  --names "$TG_NAME" --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null || echo None)
if [ "$TG_ARN" != "None" ] && [ -n "$TG_ARN" ]; then
  log "Deleting target group $TG_NAME..."
  aws elbv2 delete-target-group --target-group-arn "$TG_ARN" --region "$REGION"
fi

# Step 4: Security groups (ecs-sg first — alb-sg is referenced by it).
# Scope by default VPC so a same-named SG in another VPC isn't touched.
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' --output text --region "$REGION" 2>/dev/null || echo None)
for SG_NAME in "$ECS_SG_NAME" "$ALB_SG_NAME"; do
  if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
    log "No default VPC — skipping SG $SG_NAME cleanup."
    continue
  fi
  SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)
  if [ "$SG_ID" != "None" ] && [ -n "$SG_ID" ]; then
    log "Deleting SG $SG_NAME ($SG_ID)..."
    aws ec2 delete-security-group --group-id "$SG_ID" --region "$REGION" 2>/dev/null \
      || log "  (SG still in use — will retry next run)"
  fi
done

log "Done. Cluster, IAM roles, ECR repo, ACM cert, and log group preserved."
log "Monthly floor after teardown: ~\$0 (ECR image storage is the only lingering cost, ~\$0.10/mo)."
