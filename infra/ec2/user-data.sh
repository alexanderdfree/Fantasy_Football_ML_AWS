#!/bin/bash
# Cloud-init for the ff-training g4dn.xlarge. Runs once as root on first boot.
# Companion files (CW agent config, auto-shutdown units) are fetched from
# s3://ff-predictor-training/infra/ec2/ using the instance profile.
#
# On success, writes /opt/ff/config/bootstrap-complete. CI refuses to run
# training until this breadcrumb exists.

set -euxo pipefail

mkdir -p /var/log/ff-train
exec > >(tee -a /var/log/ff-train/user-data.log) 2>&1

echo "=== ff-training bootstrap $(date -Iseconds) ==="

REGION="us-east-1"
BUCKET="ff-predictor-training"
INFRA_PREFIX="infra/ec2"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/ff-training:latest"

# --- 1. NVMe instance store at /opt/ff/scratch --------------------------
NVME_DEV=$(lsblk -dpno NAME,MODEL | awk '/Amazon EC2 NVMe Instance Storage/ {print $1; exit}')
mkdir -p /opt/ff/scratch /opt/ff/logs /opt/ff/config
if [ -n "${NVME_DEV:-}" ] && ! mountpoint -q /opt/ff/scratch; then
  mkfs.xfs -f "$NVME_DEV"
  # nofail: a fresh instance store after stop/start is unformatted — we
  # reformat on every boot (below), so fstab nofail protects boot from
  # racing the mount.
  grep -q "/opt/ff/scratch" /etc/fstab || echo "$NVME_DEV /opt/ff/scratch xfs defaults,nofail 0 2" >> /etc/fstab
  mount /opt/ff/scratch
fi
chown -R ubuntu:ubuntu /opt/ff
chmod 755 /opt/ff /opt/ff/scratch /opt/ff/logs /opt/ff/config

# --- 2. Ensure Docker + SSM agent are enabled ---------------------------
systemctl enable --now docker amazon-ssm-agent

# --- 3. ECR credential helper (fixes 12h token expiry) ------------------
install -d -m 755 -o ubuntu -g ubuntu /home/ubuntu/.docker
cat > /home/ubuntu/.docker/config.json <<'JSON'
{"credsStore": "ecr-login"}
JSON
chown ubuntu:ubuntu /home/ubuntu/.docker/config.json
# Root also pulls images (SSM runs as root by default).
install -d -m 755 /root/.docker
cp /home/ubuntu/.docker/config.json /root/.docker/config.json

# --- 4. CloudWatch agent for /var/log/ff-train/*.log --------------------
# DLAMI variants don't all ship amazon-cloudwatch-agent preinstalled.
# Install if missing, then configure. Tolerate install failure — SSM command
# output goes straight to /ff/training via --cloud-watch-output-config, so
# CW-agent is defense-in-depth, not the only log path.
CW_CTL="/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl"
if [ ! -x "$CW_CTL" ]; then
  wget -q https://s3.${REGION}.amazonaws.com/amazoncloudwatch-agent-${REGION}/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb \
    -O /tmp/cw-agent.deb && dpkg -i /tmp/cw-agent.deb || \
    echo "CW agent install failed, skipping"
fi
if [ -x "$CW_CTL" ]; then
  CW_CONFIG_DIR="/opt/aws/amazon-cloudwatch-agent/etc"
  mkdir -p "$CW_CONFIG_DIR"
  aws s3 cp "s3://${BUCKET}/${INFRA_PREFIX}/cloudwatch-agent.json" \
    "${CW_CONFIG_DIR}/amazon-cloudwatch-agent.json" --region "$REGION"
  "$CW_CTL" -a fetch-config -m ec2 -s \
    -c "file:${CW_CONFIG_DIR}/amazon-cloudwatch-agent.json" || \
    echo "CW agent configure failed, continuing"
fi

# --- 5. /usr/local/bin/ff-train — single entry point for SSM and humans
cat > /usr/local/bin/ff-train <<EOF
#!/usr/bin/env bash
# Usage: ff-train POS [SEED]
# Runs training for one position inside the ff-training container. A flock
# prevents concurrent invocations from racing on the scratch dir or S3
# artifact. last-activity timestamp feeds the auto-shutdown timer.
set -euo pipefail
POS="\$1"
SEED="\${2:-42}"
IMAGE="${IMAGE}"

exec 200>/var/lock/ff-train.lock
flock -n 200 || { echo "ff-train: another run is in progress"; exit 2; }

mkdir -p /opt/ff/scratch/input /opt/ff/scratch/model /opt/ff/logs
docker pull "\$IMAGE"
docker run --rm --gpus all \\
  -e S3_BUCKET=${BUCKET} \\
  -e S3_DATA_PREFIX=data \\
  -e LOG_EVERY=1 \\
  -e REQUIRE_GPU=1 \\
  -e AWS_DEFAULT_REGION=${REGION} \\
  -v /opt/ff/scratch/input:/opt/ml/input/data/training \\
  -v /opt/ff/scratch/model:/opt/ml/model \\
  "\$IMAGE" --position "\$POS" --seed "\$SEED" 2>&1 \\
  | tee -a "/var/log/ff-train/\${POS}.log"

date -Iseconds > /opt/ff/logs/last-activity
EOF
chmod 755 /usr/local/bin/ff-train

# --- 6. Auto-shutdown units (installed but disabled) --------------------
aws s3 cp "s3://${BUCKET}/${INFRA_PREFIX}/auto-shutdown.sh" \
  /usr/local/bin/ff-auto-shutdown --region "$REGION"
chmod 755 /usr/local/bin/ff-auto-shutdown
aws s3 cp "s3://${BUCKET}/${INFRA_PREFIX}/auto-shutdown.service" \
  /etc/systemd/system/ff-auto-shutdown.service --region "$REGION"
aws s3 cp "s3://${BUCKET}/${INFRA_PREFIX}/auto-shutdown.timer" \
  /etc/systemd/system/ff-auto-shutdown.timer --region "$REGION"
systemctl daemon-reload
# Intentionally NOT enabling the timer. Operator enables via SSM after
# observing real usage patterns: systemctl enable --now ff-auto-shutdown.timer

# --- 7. GPU driver + warm image pull ------------------------------------
nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk -F. '{
  if (int($1) < 525) { print "Driver too old: " $0; exit 1 }
  else { print "Driver OK: " $0 }
}'

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
docker pull "$IMAGE"

# --- 8. Breadcrumb ------------------------------------------------------
date -Iseconds > /opt/ff/config/bootstrap-complete
echo "=== ff-training bootstrap complete ==="
