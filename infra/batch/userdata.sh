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

# --- 2. Snapshotter systemd unit -----------------------------------------
# Mirrors the canonical soci-snapshotter.service from
# https://github.com/awslabs/soci-snapshotter/blob/v0.13.0/soci-snapshotter.service
#
# Ordering: soci-snapshotter is a containerd PROXY PLUGIN — containerd
# resolves proxy plugins at startup by connecting to the configured socket.
# If containerd starts before the snapshotter socket exists, the plugin is
# marked unavailable and containerd falls back to overlayfs (silent
# regression of the entire Option B win). Hence Before=containerd.service.
# ecs.service is transitively After=containerd.service on AL2, so we don't
# need an explicit Before=ecs.service.
cat > /etc/systemd/system/soci-snapshotter.service <<'UNIT'
[Unit]
Description=SOCI snapshotter
Documentation=https://github.com/awslabs/soci-snapshotter
After=network.target
Before=containerd.service

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

# Wait for the snapshotter socket BEFORE we restart containerd. If
# containerd starts without the socket present, the proxy plugin loads as
# "unavailable" for the rest of the boot — image pulls silently fall back
# to overlayfs and we lose the Option B win without any error signal.
for i in $(seq 1 60); do
  [ -S /run/soci-snapshotter-grpc/soci-snapshotter-grpc.sock ] && break
  sleep 1
done
if [ ! -S /run/soci-snapshotter-grpc/soci-snapshotter-grpc.sock ]; then
  echo "ERROR: soci-snapshotter socket never appeared after 60s" >&2
  echo "  /var/log/soci-userdata.log has the bootstrap trace" >&2
  echo "  journalctl -u soci-snapshotter for daemon logs" >&2
  exit 1
fi

# --- 3. Register snapshotter as containerd proxy plugin ------------------
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

# Restart containerd AFTER the snapshotter socket is up and the config
# references it. containerd discovers proxy plugins at startup by
# connecting to the socket — the snapshotter MUST be running first.
# ecs.service is After=containerd.service on AL2 (and cloud-init.target
# blocks ecs.service until userdata completes), so this cascades into a
# clean ECS-agent claim with the soci snapshotter ready.
systemctl restart containerd

echo "=== soci bootstrap complete ($(date -Iseconds)) ==="
exit 0
