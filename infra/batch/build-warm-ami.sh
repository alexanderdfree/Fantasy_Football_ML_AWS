#!/usr/bin/env bash
# Build a warm pre-pulled GPU AMI for the AWS Batch training fleet.
#
# WHY: on a cold Spot host the largest *controllable* slice of the ~258 s
# cold-start is the ~122 s training-image pull (decompress + extract to the
# container store), re-paid on every fresh host. Production training is
# orchestration-bound, not GPU-bound — so the highest-leverage training-time win
# is killing that pull, not optimizing the ~1-min GPU step. This bakes the
# training image's layers into a custom AMI built FROM the latest ECS-GPU-
# optimized AMI, so a fresh host boots with the layers already in the container
# store and the ECS agent's pull finds them cached.
#
# REBUILD CADENCE: rebuild only when the image's *base* layers change (the
# torch / CUDA pin or requirements.txt). The app-code layer drifts on every
# src/** push, but a stale app layer costs only the small app-delta pull — the
# heavy base layers (the ~1 GB torch wheel) stay cached on the AMI. So this is a
# rare, manual rebuild, not a per-push step.
#
# SOCI is NOT this: SOCI lazy-loading was removed 2026-06-07 because the ECS
# agent ignores the snapshotter on ECS-managed EC2 (Fargate-only). A warm AMI
# needs no snapshotter and no UserData daemon — just baked layers — so it
# sidesteps both that limitation and the UserData-MIME CE-INVALID footgun.
#
# Usage:
#   infra/batch/build-warm-ami.sh <ecr-image-uri[:tag]>            # build
#   infra/batch/build-warm-ami.sh <ecr-image-uri[:tag]> --dry-run  # print plan
#
# Prints the new AMI id as the last stdout line. Activate it with:
#   FF_BATCH_AMI_ID=<ami-id> bash infra/batch/setup.sh
# (setup.sh attaches it via the ff-warm-ami-lt launch template; default-unset =
# no change. Roll back by re-running setup.sh with FF_BATCH_AMI_ID unset is NOT
# enough — see the rollback note in infra/batch/README.md.)
#
# Prereqs:
#   - AWS CLI v2 with credentials for the target account.
#   - The builder instance profile (default: ecsInstanceRole) must have BOTH
#     ECR pull permission AND AmazonSSMManagedInstanceCore (this script drives
#     the in-instance `docker pull` via SSM Run Command). Attach the SSM managed
#     policy once if missing.
#   - A subnet with outbound internet (NAT/IGW) for the ECR pull + SSM.

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
IMAGE_URI="${1:-}"
DRY_RUN=0
[ "${2:-}" = "--dry-run" ] && DRY_RUN=1

if [ -z "$IMAGE_URI" ]; then
  echo "usage: $0 <ecr-image-uri[:tag]> [--dry-run]" >&2
  exit 2
fi

# Reuse the Batch fleet's identity by default so the AMI is GPU/ECS-compatible
# and the builder can pull from the same ECR repo.
BUILDER_TYPE="${FF_WARM_AMI_BUILDER_TYPE:-g6.xlarge}"
INSTANCE_PROFILE="${FF_WARM_AMI_INSTANCE_PROFILE:-ecsInstanceRole}"
SG_NAME="${FF_WARM_AMI_SG_NAME:-ff-batch-sg}"
# Latest ECS GPU-optimized Amazon Linux 2 AMI (matches the Batch default lineage:
# NVIDIA driver + ECS agent + Docker). Keeping the SAME OS family as the default
# CE AMI is deliberate — a custom AMI only adds pre-pulled layers, nothing else.
SSM_AMI_PARAM="/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id"
AMI_NAME="ff-warm-$(date -u +%Y%m%d-%H%M%S)"
TAG_SPEC="ResourceType=instance,Tags=[{Key=Name,Value=ff-warm-ami-builder},{Key=ff-purpose,Value=warm-ami-build}]"

log() { echo "[warm-ami] $*"; }

# Resolve the source AMI (read-only; safe to run even in dry-run so the plan is
# concrete).
SOURCE_AMI="$(aws ssm get-parameters \
  --names "$SSM_AMI_PARAM" \
  --region "$REGION" \
  --query 'Parameters[0].Value' \
  --output text)"
if [ -z "$SOURCE_AMI" ] || [ "$SOURCE_AMI" = "None" ]; then
  echo "ERROR: could not resolve ECS-GPU AMI from SSM ($SSM_AMI_PARAM)" >&2
  exit 1
fi
log "source ECS-GPU AMI: $SOURCE_AMI"
log "image to bake:      $IMAGE_URI"
log "builder type:       $BUILDER_TYPE  (profile=$INSTANCE_PROFILE, sg=$SG_NAME)"

if [ "$DRY_RUN" = "1" ]; then
  cat <<EOF
[dry-run] would, in order:
  1. run-instances from $SOURCE_AMI ($BUILDER_TYPE, profile $INSTANCE_PROFILE, sg $SG_NAME)
  2. wait until the instance is SSM-online
  3. SSM RunShellScript on it:
       aws ecr get-login-password --region $REGION \\
         | docker login --username AWS --password-stdin <registry>
       docker pull $IMAGE_URI
       docker image inspect $IMAGE_URI >/dev/null   # assert layers resident
  4. stop-instances + wait instance-stopped
  5. create-image --name $AMI_NAME --no-reboot  (from the stopped builder)
  6. wait image-available; tag the AMI (Name, source-image, built-at)
  7. terminate the builder instance
  8. print the new AMI id

Activate:  FF_BATCH_AMI_ID=<ami-id> bash infra/batch/setup.sh
EOF
  exit 0
fi

# Resolve the security group id from its name.
SG_ID="$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" \
  --region "$REGION" \
  --query 'SecurityGroups[0].GroupId' \
  --output text)"
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
  echo "ERROR: security group '$SG_NAME' not found in $REGION" >&2
  exit 1
fi

INSTANCE_ID=""
cleanup() {
  if [ -n "$INSTANCE_ID" ]; then
    log "cleanup: terminating builder $INSTANCE_ID"
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

REGISTRY="${IMAGE_URI%%/*}"  # <acct>.dkr.ecr.<region>.amazonaws.com

log "launching builder instance..."
INSTANCE_ID="$(aws ec2 run-instances \
  --image-id "$SOURCE_AMI" \
  --instance-type "$BUILDER_TYPE" \
  --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --security-group-ids "$SG_ID" \
  --tag-specifications "$TAG_SPEC" \
  --region "$REGION" \
  --query 'Instances[0].InstanceId' \
  --output text)"
log "builder: $INSTANCE_ID — waiting for running + SSM-online..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

# Poll until the SSM agent registers the instance (the ECS-optimized AMI ships
# the agent; this needs AmazonSSMManagedInstanceCore on the instance profile).
for _ in $(seq 1 60); do
  ONLINE="$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --region "$REGION" \
    --query 'InstanceInformationList[0].PingStatus' \
    --output text 2>/dev/null || echo None)"
  [ "$ONLINE" = "Online" ] && break
  sleep 10
done
if [ "${ONLINE:-}" != "Online" ]; then
  echo "ERROR: builder $INSTANCE_ID never came SSM-online (check AmazonSSMManagedInstanceCore on $INSTANCE_PROFILE)" >&2
  exit 1
fi

log "pulling $IMAGE_URI on the builder via SSM..."
PULL_CMDS="set -euo pipefail
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REGISTRY
docker pull $IMAGE_URI
docker image inspect $IMAGE_URI >/dev/null"
CMD_ID="$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "warm-ami pre-pull" \
  --parameters "commands=[$(printf '%s' "$PULL_CMDS" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')]" \
  --timeout-seconds 1200 \
  --region "$REGION" \
  --query 'Command.CommandId' \
  --output text)"

# Wait for the pull command to finish.
for _ in $(seq 1 120); do
  CMD_STATUS="$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' \
    --output text 2>/dev/null || echo Pending)"
  case "$CMD_STATUS" in
    Success) break ;;
    Failed | Cancelled | TimedOut)
      echo "ERROR: pull command $CMD_ID ended $CMD_STATUS" >&2
      aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
        --region "$REGION" --query 'StandardErrorContent' --output text >&2 || true
      exit 1
      ;;
  esac
  sleep 10
done
[ "${CMD_STATUS:-}" = "Success" ] || { echo "ERROR: pull command did not complete" >&2; exit 1; }
log "image layers resident on builder"

log "stopping builder for a clean snapshot..."
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"

log "creating AMI $AMI_NAME..."
AMI_ID="$(aws ec2 create-image \
  --instance-id "$INSTANCE_ID" \
  --name "$AMI_NAME" \
  --description "ECS-GPU AMI ($SOURCE_AMI) with $IMAGE_URI pre-pulled" \
  --no-reboot \
  --region "$REGION" \
  --query 'ImageId' \
  --output text)"
log "AMI $AMI_ID creating — waiting for available..."
aws ec2 wait image-available --image-ids "$AMI_ID" --region "$REGION"
aws ec2 create-tags \
  --resources "$AMI_ID" \
  --tags "Key=Name,Value=$AMI_NAME" "Key=ff-source-ami,Value=$SOURCE_AMI" \
         "Key=ff-baked-image,Value=$IMAGE_URI" \
  --region "$REGION" >/dev/null

log "done. AMI ready:"
echo "$AMI_ID"
