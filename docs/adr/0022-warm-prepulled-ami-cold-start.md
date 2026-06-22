# ADR-0022: Warm Pre-pulled GPU AMI For Batch Cold-Start

**Status:** Accepted
**Date:** 2026-06-22
**Supersedes:** none
**Related:** ADR-0013 (Spot fan-out via AWS Batch), ADR-0019 (split Batch training)

## Context

Production training runs on AWS Batch GPU Spot hosts (g6/L4, g5/A10G). The
attention NN is launch-bound, but the *production wall-clock* is
orchestration-bound, not GPU-bound: the ~1-min GPU step sits inside a ~258 s
cold-start — ~120 s Spot provisioning (a fixed G-family floor) + **~122 s
training-image pull** + ~10 s container start — plus Spot queue latency. The
image pull is the single largest *controllable* chunk, re-paid on every fresh
Spot host.

SOCI lazy-loading (the prior attempt at this) was removed 2026-06-07 because the
ECS agent ignores the snapshotter on ECS-managed EC2 — SOCI-on-ECS is
Fargate-only, and Fargate has no GPU (see ADR-0013 changelog, §2a of
[batch_design.md](../batch_design.md)). Its `ff-batch-lt` launch template +
`infra/batch/userdata.sh` daemon were deleted. The CE has paid the full pull
since.

The W0 instrumentation (`src/batch/launch.py`, `[timing] phase=batch_lifecycle`)
now decomposes each job's wall-clock into `queue_provision` (submit→RUNNING,
including the pull) vs `run` (training), giving a live baseline this lever is
measured against.

## Decision

Add an **opt-in** warm pre-pulled custom AMI:

- `infra/batch/build-warm-ami.sh <ecr-image-uri[:tag]>` builds an AMI **from**
  the latest ECS-GPU-optimized AMI (same NVIDIA driver + ECS agent + Docker
  lineage) with the training image's layers pre-pulled into the container store,
  via SSM Run Command on a short-lived builder instance. `--dry-run` prints the
  plan. It prints the new AMI id.
- `FF_BATCH_AMI_ID=<ami> bash infra/batch/setup.sh` attaches the AMI to the
  **GPU CE only** via a version-pinned `ff-warm-ami-lt` launch template, reusing
  the existing `DISABLE → update-compute-environment → ENABLE` reconcile. The
  launch template carries **only** `ImageId` — no UserData — so it sidesteps both
  the SOCI ECS-agent limitation and the UserData-MIME `CE-INVALID` footgun.
- **Unset `FF_BATCH_AMI_ID` is the default and is a no-op for this feature**:
  `setup.sh` adds no `launchTemplate` key and never auto-detaches one, so a plain
  run is byte-identical to prior behavior.

A fresh host then boots with the layers cached; the ECS agent's pull finds them
and skips the ~122 s extract. **Rebuild only when the image's base layers change**
(torch/CUDA pin or `requirements.txt`) — a stale app-code layer costs only the
small app-delta pull, so this is a rare manual rebuild, not per-push.

## Consequences

- Up to ~120 s/host off the cold-start; since the six fan-out hosts cold-start in
  parallel, ~120 s off the *critical path* (~20-25 % of the production wall).
- Zero metric/accuracy risk: pure infra, no training-path or numerics change. The
  AMI only adds pre-pulled layers to the same OS/driver/agent lineage.
- A new operational object (the AMI + `ff-warm-ami-lt` launch template) with a
  manual rebuild cadence and an explicit `launchTemplate={}` rollback. The AMI is
  GPU-CE-scoped; the c8a CPU fleet is untouched.
- Composes with the already-shipped image-shrink (§2d) and ECR pull-through cache
  (§2c) — it removes the residual pull those leave.

## Rejected Alternatives

- **SOCI lazy-loading.** Removed 2026-06-07 — ECS agent ignores the snapshotter
  on ECS-managed EC2 (Fargate-only). This is the reason a warm AMI (no
  snapshotter) is the surviving lever.
- **`imageId` directly in `computeResources`.** Simpler, but on AWS's deprecation
  path for ECS-on-EC2 and gives no home for version pinning; the launch template
  is the supported, future-proof path and matches the prior-art reconcile.
- **Always-on warm pool (`minvCpus > 0`).** Keeps a host hot to avoid provision +
  pull entirely, but pays for idle GPU Spot capacity continuously; the AMI gets
  most of the win at zero idle cost.
- **Bottlerocket GPU AMI.** Changes OS family (driver baking, agent
  customization, debugging surface); higher risk than baking layers onto the
  existing AL2 ECS-GPU lineage.

## Changelog

- 2026-06-22 · Initial decision: opt-in warm pre-pulled GPU AMI
  (`build-warm-ami.sh` + `FF_BATCH_AMI_ID` launch-template wiring in
  `setup.sh`), default-unset/no-op, GPU-CE-scoped. Capability added; live
  activation is a separate gated step.
