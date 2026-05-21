#!/bin/bash
# Cloud-init for the ff-gpu-spot Batch container instances.
# Runs once as root on each fresh Spot host BEFORE ecs.service starts.
#
# Purpose: activate SOCI v2 lazy-loading. The ECS-optimized AL2 AMI Batch
# picks does NOT run `soci-snapshotter-grpc` by default, so the SOCI
# indexes published to ECR by .github/workflows/batch-image.yml are
# silently ignored — every fresh Spot host does a full ~122s image pull.
# This script installs the snapshotter, registers it as a containerd
# proxy plugin, and starts it as a systemd unit ordered Before=ecs.service
# so the first pull is lazy.
#
# Wired in by infra/batch/setup.sh as base64-encoded launch-template
# UserData. See infra/batch/README.md §"Cold-start optimization" and
# docs/batch_design.md §2a for the full design.
#
# Expected impact: ~122s pull → ~5–10s pull, total cold-start
# ~258s → ~135s (measured baseline 2026-05-20).
#
# Failure mode: if the snapshotter doesn't come up, exit 1 fails cloud-init
# and marks the instance unhealthy in the CE — the next Spot host retries.
# Better to fail loudly than to silently fall back to full pull.

set -euxo pipefail
exec > >(tee -a /var/log/soci-userdata.log) 2>&1

echo "=== soci bootstrap $(date -Iseconds) ==="

# Pin to a release. MUST match SOCI_VERSION in .github/workflows/batch-image.yml's
# "Publish SOCI index" step — version skew between publisher and host
# snapshotter can break lazy-load (the manifest format evolves).
SOCI_VERSION="0.13.0"
SOCI_TARBALL="soci-snapshotter-${SOCI_VERSION}-linux-amd64.tar.gz"
SOCI_URL="https://github.com/awslabs/soci-snapshotter/releases/download/v${SOCI_VERSION}/${SOCI_TARBALL}"

# --- 1. Install snapshotter binaries ------------------------------------
# fuse is the runtime dep; the AL2 base image typically has it but not
# always on minimal variants.
yum install -y fuse

curl -fsSL -o "/tmp/${SOCI_TARBALL}" "$SOCI_URL"
tar -xzf "/tmp/${SOCI_TARBALL}" -C /usr/local/bin/ soci soci-snapshotter-grpc
chmod +x /usr/local/bin/soci /usr/local/bin/soci-snapshotter-grpc
rm -f "/tmp/${SOCI_TARBALL}"

# --- 2. Register snapshotter as containerd proxy plugin ------------------
# Append the [proxy_plugins.soci] stanza to /etc/containerd/config.toml.
# ECS-optimized AL2 ships a default config; if it's missing or empty,
# generate one with `containerd config default`. Idempotent: grep guards
# against re-appending on userdata re-execution.
mkdir -p /etc/containerd
if [ ! -s /etc/containerd/config.toml ]; then
  containerd config default > /etc/containerd/config.toml
fi
if ! grep -q '\[proxy_plugins.soci\]' /etc/containerd/config.toml; then
  cat >> /etc/containerd/config.toml <<'TOML'

[proxy_plugins]
  [proxy_plugins.soci]
    type = "snapshot"
    address = "/run/soci-snapshotter-grpc/soci-snapshotter-grpc.sock"
TOML
fi

# Restart containerd so the plugin is registered. ecs.service is ordered
# After=containerd.service on AL2, so restarting containerd here cascades
# to a clean state before ecs.service starts.
systemctl restart containerd

# --- 3. Snapshotter systemd unit -----------------------------------------
# After=containerd.service  — wait for containerd socket
# Before=ecs.service        — block ECS agent until snapshotter is up so
#                              the first image pull doesn't race to full-pull
cat > /etc/systemd/system/soci-snapshotter.service <<'UNIT'
[Unit]
Description=SOCI snapshotter
After=containerd.service
Wants=containerd.service
Before=ecs.service

[Service]
Type=notify
ExecStart=/usr/local/bin/soci-snapshotter-grpc
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now soci-snapshotter

# --- 4. Socket-wait belt-and-suspenders ----------------------------------
# `Before=ecs.service` should be sufficient, but cloud-init userdata
# ordering vs. ecs.service is occasionally flaky on AL2 — block userdata
# completion until the snapshotter socket exists. ecs.service can't start
# until userdata finishes (cloud-init.target ordering), so this guarantees
# the first image pull uses SOCI.
for i in $(seq 1 60); do
  if [ -S /run/soci-snapshotter-grpc/soci-snapshotter-grpc.sock ]; then
    echo "=== soci bootstrap complete ($(date -Iseconds)) ==="
    exit 0
  fi
  sleep 1
done

echo "ERROR: soci-snapshotter socket never appeared after 60s" >&2
echo "  /var/log/soci-userdata.log has the bootstrap trace" >&2
echo "  journalctl -u soci-snapshotter for daemon logs" >&2
exit 1
