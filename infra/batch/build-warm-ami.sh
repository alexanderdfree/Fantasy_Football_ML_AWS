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
# Prints the new AMI id as the last stdout line and writes a bake manifest.
# Run infra/batch/warm_ami.py canary, then activate with its passing evidence.
# See infra/batch/WARM_AMI.md for the measured gate and recorded rollback.
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
# Latest ECS GPU-optimized Amazon Linux 2023 AMI (matches the Batch lineage:
# NVIDIA driver + ECS agent + Docker). Keeping the SAME OS family as the default
# CE AMI is deliberate — a custom AMI only adds pre-pulled layers, nothing else.
SSM_AMI_PARAM="/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id"
AMI_NAME="ff-warm-$(date -u +%Y%m%d-%H%M%S)"
MANIFEST_OUT="${FF_WARM_AMI_MANIFEST:-${TMPDIR:-/tmp}/${AMI_NAME}.json}"
# AMI_NAME is unique per run (UTC timestamp) and doubles as the per-run instance
# tag + run-instances client token, so the cleanup trap can recover a leaked
# builder by tag without ever terminating a concurrent build's host.
TAG_SPEC="ResourceType=instance,Tags=[{Key=Name,Value=ff-warm-ami-builder},{Key=ff-purpose,Value=warm-ami-build},{Key=ff-warm-ami-run,Value=$AMI_NAME}]"

log() { echo "[warm-ami] $*"; }

# Resolve the source AMI (read-only; safe to run even in dry-run so the plan is
# concrete).
SOURCE_AMI="${FF_WARM_AMI_SOURCE_AMI:-$(aws ssm get-parameters \
  --names "$SSM_AMI_PARAM" \
  --region "$REGION" \
  --query 'Parameters[0].Value' \
  --output text)}"
if [ -z "$SOURCE_AMI" ] || [ "$SOURCE_AMI" = "None" ]; then
  echo "ERROR: could not resolve ECS-GPU AMI from SSM ($SSM_AMI_PARAM)" >&2
  exit 1
fi
if [[ ! "$SOURCE_AMI" =~ ^ami-[0-9a-f]+$ ]]; then
  echo "ERROR: source AMI must be a concrete AMI ID" >&2
  exit 1
fi
SOURCE_NAME="$(aws ec2 describe-images --image-ids "$SOURCE_AMI" --region "$REGION" \
  --filters Name=architecture,Values=x86_64 Name=state,Values=available \
  --query 'Images[0].Name' --output text)"
if [[ "$SOURCE_NAME" != *al2023*gpu* ]]; then
  echo "ERROR: source AMI is not the AL2023 ECS GPU family: $SOURCE_NAME" >&2
  exit 1
fi
if [[ ! "$IMAGE_URI" =~ ^[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com(\.cn)?/[a-z0-9._/-]+(:[A-Za-z0-9_.-]+|@sha256:[0-9a-f]{64})$ ]]; then
  echo "ERROR: supply a tagged or digest-qualified ECR image" >&2
  exit 1
fi
REGISTRY="${IMAGE_URI%%/*}"
IMAGE_PATH="${IMAGE_URI#*/}"
if [[ "$IMAGE_PATH" == *@* ]]; then
  ECR_REPOSITORY="${IMAGE_PATH%@*}"
  IMAGE_SELECTOR="imageDigest=${IMAGE_PATH#*@}"
else
  ECR_REPOSITORY="${IMAGE_PATH%:*}"
  IMAGE_SELECTOR="imageTag=${IMAGE_PATH##*:}"
fi
IMAGE_DIGEST="$(aws ecr describe-images --repository-name "$ECR_REPOSITORY" \
  --image-ids "$IMAGE_SELECTOR" --region "$REGION" --query 'imageDetails[0].imageDigest' --output text)"
if [[ ! "$IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "ERROR: could not resolve the exact training image digest" >&2
  exit 1
fi
IMAGE_URI="${REGISTRY}/${ECR_REPOSITORY}@${IMAGE_DIGEST}"
SOURCE_SHA="$(aws ecr describe-images --repository-name "$ECR_REPOSITORY" \
  --image-ids "imageDigest=$IMAGE_DIGEST" --region "$REGION" \
  --query 'imageDetails[0].imageTags' --output json | python3 -c '
import json,re,sys
tags=[v for v in json.load(sys.stdin) if re.fullmatch(r"[0-9a-f]{40}", v)]
if len(tags) != 1: raise SystemExit("Image needs one unambiguous source-SHA tag")
print(tags[0])')"
# The final two filesystem layers are the application and source stamp. Fail
# closed if the selected image recipe changes that layout; otherwise an old
# dependency layer could be mislabeled as fresh.
DEPENDENCY_RECIPE="$(python3 - "$SOURCE_SHA" <<'PY'
import hashlib, subprocess, sys
def read(path):
    return subprocess.check_output(['git', 'show', sys.argv[1] + ':' + path])
dockerfile = read('src/batch/Dockerfile.train')
before, separator, after = dockerfile.partition(b'COPY src/ src/')
instructions = [line.strip().split(b' ', 1)[0] for line in after.splitlines()
                if line.strip() and not line.lstrip().startswith(b'#')]
if not separator or instructions != [b'ARG', b'RUN', b'ENTRYPOINT']:
    raise SystemExit('Training image layout changed; review the dependency boundary')
print(hashlib.sha256(before + read('src/batch/requirements.txt')).hexdigest())
PY
)"
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
       docker logout <registry>; remove builder registration state
  4. stop-instances + wait instance-stopped
  5. create-image --name $AMI_NAME --no-reboot  (from the stopped builder)
  6. wait image-available; tag the AMI (Name, source-image, built-at)
  7. terminate the builder instance
  8. write $MANIFEST_OUT and print the new AMI id

Next: python infra/batch/warm_ami.py canary --input $MANIFEST_OUT --output canary.json
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
PULL_OUTPUT=""
COMMAND_PARAMS=""
cleanup() {
  [ -z "$PULL_OUTPUT" ] || rm -f "$PULL_OUTPUT"
  [ -z "$COMMAND_PARAMS" ] || rm -f "$COMMAND_PARAMS"
  if [ -n "$INSTANCE_ID" ]; then
    log "cleanup: terminating builder $INSTANCE_ID"
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null 2>&1 || true
    return
  fi
  # INSTANCE_ID never captured — run-instances may have created the host but the
  # CLI died before the assignment landed. Recover the leaked builder by the
  # per-run unique tag so a concurrent build's host is never touched.
  local leaked
  leaked="$(aws ec2 describe-instances \
    --filters "Name=tag:ff-warm-ami-run,Values=$AMI_NAME" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --region "$REGION" --query 'Reservations[].Instances[].InstanceId' \
    --output text 2>/dev/null || true)"
  if [ -n "$leaked" ]; then
    log "cleanup: terminating leaked builder(s) by tag ff-warm-ami-run=$AMI_NAME: $leaked"
    # shellcheck disable=SC2086  # intentional word-split: space-separated ids
    aws ec2 terminate-instances --instance-ids $leaked --region "$REGION" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

REGISTRY="${IMAGE_URI%%/*}"  # <acct>.dkr.ecr.<region>.amazonaws.com
BUILDER_USER_DATA='#!/bin/bash
systemctl mask --now ecs.service
'

log "launching builder instance..."
INSTANCE_ID="$(aws ec2 run-instances \
  --image-id "$SOURCE_AMI" \
  --instance-type "$BUILDER_TYPE" \
  --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --security-group-ids "$SG_ID" \
  --tag-specifications "$TAG_SPEC" \
  --client-token "$AMI_NAME" \
  --user-data "$BUILDER_USER_DATA" \
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
docker pull --quiet $IMAGE_URI
docker image inspect $IMAGE_URI >/dev/null
docker image inspect --format '{{json .RootFS.Layers}}' $IMAGE_URI
docker logout $REGISTRY
systemctl stop ecs.service || true
rm -f /var/lib/ecs/data/agent.db /var/lib/amazon/ssm/registration
systemctl unmask ecs.service
systemctl enable ecs.service
cloud-init clean --logs
printf 'uninitialized\\n' >/etc/machine-id
rm -f /var/lib/dbus/machine-id"
COMMAND_PARAMS="$(mktemp "${TMPDIR:-/tmp}/warm-ami-command.XXXXXX")"
printf '%s' "$PULL_CMDS" | python3 -c \
  'import json,sys; json.dump({"commands": [sys.stdin.read()]}, sys.stdout)' >"$COMMAND_PARAMS"
CMD_ID="$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "warm-ami pre-pull" \
  --parameters "file://$COMMAND_PARAMS" \
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
PULL_OUTPUT="$(mktemp "${TMPDIR:-/tmp}/warm-ami-pull.XXXXXX")"
aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
  --region "$REGION" --query 'StandardOutputContent' --output text >"$PULL_OUTPUT"

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
# Record the candidate even if layer validation fails, so it can be inspected
# or removed without launching another builder.
python3 - "$MANIFEST_OUT" "$SOURCE_AMI" "$IMAGE_URI" "$AMI_ID" "$REGION" "$PULL_OUTPUT" "$SOURCE_SHA" "$DEPENDENCY_RECIPE" <<'PY'
import hashlib, json, pathlib, sys
output, source, image, ami, region, log, sha, recipe = sys.argv[1:]
manifest = {'version': 1, 'source_ami': source, 'image_uri': image,
            'candidate_ami': ami, 'region': region, 'source_sha': sha,
            'dependency_recipe': recipe, 'eligible': False}
path = pathlib.Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, indent=2) + '\n')
layers = []
for line in pathlib.Path(log).read_text().splitlines():
    try:
        candidate = json.loads(line)
    except ValueError:
        continue
    if isinstance(candidate, list) and candidate and all(isinstance(v, str) and v.startswith('sha256:') for v in candidate):
        layers = candidate
if len(layers) < 3:
    raise SystemExit('Missing image-layer evidence; candidate is not eligible for promotion')
# Dockerfile.train ends with COPY src and the baked source-SHA layer.
dependency_layers = layers[:-2]
manifest.update(image_layers=layers, dependency_layers=dependency_layers,
                dependency_fingerprint=hashlib.sha256(json.dumps(dependency_layers).encode()).hexdigest(),
                eligible=True)
path.write_text(json.dumps(manifest, indent=2) + '\n')
PY
rm -f "$PULL_OUTPUT"
# `aws ec2 wait image-available` caps at 40 polls x 15s = 10 min, but a warm AMI
# bakes the multi-GB training-image layers into an EBS snapshot that routinely
# takes longer — the waiter would time out and `set -e` would kill the script
# before printing the AMI id, even though the AMI keeps building. Poll directly
# with a generous ceiling instead.
log "AMI $AMI_ID creating — waiting for available (up to ~40 min; large GPU AMI snapshots exceed the default 10-min waiter)..."
img_state=""
for _ in $(seq 1 120); do  # 120 x 20s = 40 min
  img_state="$(aws ec2 describe-images --image-ids "$AMI_ID" --region "$REGION" \
    --query 'Images[0].State' --output text 2>/dev/null || echo pending)"
  case "$img_state" in
    available) break ;;
    failed | error | invalid)
      echo "ERROR: AMI $AMI_ID entered terminal state '$img_state'" >&2
      exit 1
      ;;
  esac
  sleep 20
done
if [ "$img_state" != "available" ]; then
  echo "ERROR: AMI $AMI_ID not available after ~40 min (last state=$img_state)" >&2
  exit 1
fi
aws ec2 create-tags \
  --resources "$AMI_ID" \
  --tags "Key=Name,Value=$AMI_NAME" "Key=ff-source-ami,Value=$SOURCE_AMI" \
         "Key=ff-baked-image,Value=$IMAGE_URI" \
         "Key=ff-purpose,Value=warm-ami" \
  --region "$REGION" >/dev/null

log "manifest: $MANIFEST_OUT"
log "done. AMI ready (canary validation is required before activation):"
echo "$AMI_ID"
