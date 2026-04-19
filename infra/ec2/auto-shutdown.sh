#!/usr/bin/env bash
# Stop the instance when it has been idle past the threshold.
# "Idle" = /opt/ff/logs/last-activity is missing or older than IDLE_HOURS.
# The timer also does a best-effort prune of stale Docker images so the
# 100 GB root volume doesn't fill up.
#
# Uses `shutdown -h`, NOT `terminate-instances` — CI wakes it again with
# `aws ec2 start-instances`.

set -euo pipefail

IDLE_HOURS="${IDLE_HOURS:-4}"
ACTIVITY_FILE="/opt/ff/logs/last-activity"

# Stale image prune — keep only images referenced in the last 7 days.
docker image prune -af --filter "until=168h" >/dev/null 2>&1 || true

now=$(date +%s)

if [ ! -f "$ACTIVITY_FILE" ]; then
  # First boot with no training yet — count "now" as activity so we don't
  # shut down before CI has had a chance to wake up the instance.
  date -Iseconds > "$ACTIVITY_FILE"
  exit 0
fi

last=$(date -d "$(cat "$ACTIVITY_FILE")" +%s)
age=$(( now - last ))
threshold=$(( IDLE_HOURS * 3600 ))

if [ "$age" -gt "$threshold" ]; then
  logger -t ff-auto-shutdown "idle ${age}s > ${threshold}s, stopping"
  /sbin/shutdown -h +1 "ff-training: idle ${IDLE_HOURS}h, stopping"
fi
