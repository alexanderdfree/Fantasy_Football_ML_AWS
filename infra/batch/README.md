# ff-training AWS Batch + Spot

_Last verified: 2026-05-20._

Provisions an AWS Batch managed compute environment on Spot g4dn.xlarge so all
six position pipelines train in parallel (one position per Spot instance). This
is the **default push-driven trainer since 2026-05-20** — [train-batch.yml](../../.github/workflows/train-batch.yml)
runs when the `BATCH_ACTIVE` repo variable is `true`; flipping it to `false`
falls back to the warm-EC2 path ([infra/ec2/](../ec2/)).

See [`docs/batch_design.md`](../../docs/batch_design.md) for the full design
including cold-start optimizations (SOCI lazy loading + ECR pull-through
cache).

## First-time setup

Prereqs: AWS CLI v2, `gh` CLI, credentials with rights to create IAM + Batch
resources. The `ff-training` ECR repo must exist (created by the first run of
[batch-image.yml](../../.github/workflows/batch-image.yml)).

```
bash infra/batch/setup.sh
```

Idempotent — reruns skip anything that already exists. It performs a quota
preflight ("All G and VT Spot Instance Requests" >= 24 vCPU) and refuses to
proceed if the quota hasn't been approved.

The script creates:

| Resource | Name |
|---|---|
| IAM role (job/container) | `BatchTrainingRole` |
| IAM role (EC2 instance) | `ecsInstanceRole` |
| IAM role (task execution) | `ecsTaskExecutionRole` |
| Instance profile | `ecsInstanceRole` |
| Security group | `ff-batch-sg` (egress only) |
| Compute environment | `ff-gpu-spot` (SPOT, max 24 vCPU, g4dn.xlarge) |
| Job queue | `ff-training-queue` |
| Job definition | `ff-training-job` (rev 1; CI re-registers on every push) |
| CloudWatch log group | `/aws/batch/job` (7-day retention) |

## Cold-start optimization (recommended)

The cold-start tax (image pull on a fresh Spot instance) is the original reason
this project moved off Batch. Two AWS-side optimizations were planned to bring
cold-start from ~120s pull to ~60–90s total; both are wired into
[batch-image.yml](../../.github/workflows/batch-image.yml). **Measured
2026-05-20: ~258 s total cold-start (~120 s Spot+boot + ~122 s full image pull
+ ~10 s container start)** — the SOCI snapshotter is not active on the default
Batch AMI, so the SOCI indexes in ECR are ignored. See
[docs/batch_design.md §2a](../../docs/batch_design.md) for activation options
(pin a Bottlerocket GPU AMI, or pin AL2 + userdata install of
`soci-snapshotter-grpc`). The pull-through cache rule below is independently
useful at GHA build time.

**Run the pull-through cache rule creation once after `setup.sh`:**

```
aws ecr create-pull-through-cache-rule \
  --ecr-repository-prefix dockerhub \
  --upstream-registry-url registry-1.docker.io \
  --region us-east-1
```

After this, every CI build pulls the PyTorch base image from your ECR's local
endpoints instead of Docker Hub. The build step in
[batch-image.yml](../../.github/workflows/batch-image.yml) auto-detects the
cache rule and only sets `PULL_THROUGH_PREFIX` when it exists, so it's safe to
merge this CI change before running `create-pull-through-cache-rule`.

The CI's AWS credentials (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` repo
secrets) need these ECR permissions on `repository/dockerhub/*` for the first
build to hydrate the cache from Docker Hub:

```
ecr:BatchImportUpstreamImage
ecr:CreateRepository
ecr:BatchCheckLayerAvailability
ecr:GetDownloadUrlForLayer
ecr:BatchGetImage
```

A SOCI index is also published alongside the image (cold-start opt 2a). The
SOCI publish step is `continue-on-error: true` — if it fails, the image still
works, just with a slower first pull on each fresh Spot instance. **As of
2026-05-20 the index is published successfully but unused at runtime** —
the default Batch AMI doesn't run `soci-snapshotter-grpc`; see
[docs/batch_design.md §2a](../../docs/batch_design.md) for the snapshotter
activation paths.

## Verification

1. **CE and JQ are VALID:**
   ```
   aws batch describe-compute-environments --compute-environments ff-gpu-spot \
     --query 'computeEnvironments[0].[state,status]' --region us-east-1
   aws batch describe-job-queues --job-queues ff-training-queue \
     --query 'jobQueues[0].[state,status]' --region us-east-1
   ```
   Both should return `["ENABLED","VALID"]`.

2. **Smoke test** (single CPU-only position, ~2–3 min on Spot):
   ```
   AWS_REGION=us-east-1 python -m src.batch.launch --positions K --seed 42
   ```
   Expect: job submitted → RUNNABLE → STARTING → RUNNING → SUCCEEDED, then
   model.tar.gz in `s3://ff-predictor-training/models/K/`.

3. **Full parallel fanout test:**
   ```
   AWS_REGION=us-east-1 python -m src.batch.launch \
     --positions QB RB WR TE K DST --seed 42
   ```
   All six should reach RUNNING simultaneously (six g4dn.xlarge instances,
   exactly saturating the 24 vCPU Spot quota). Total wall-clock for the
   "Submit Batch jobs and wait" step measured ~10 min on 2026-05-21
   — the slowest position dominates, not the sum.

4. **End-to-end CI:** `gh workflow run train-batch.yml -f positions=K -f seed=42`
   should run the detect → launch → freshness → benchmark commit → ECS refresh
   pipeline and write a fresh `benchmark_history/{run_id}.json`.

## Switch the active trainer

Both training workflows guard on `vars.BATCH_ACTIVE`. The current default is `true`:

| `BATCH_ACTIVE` | Active path |
|---|---|
| `true` (default since 2026-05-20) | [train-batch.yml](../../.github/workflows/train-batch.yml) — parallel Spot fanout |
| `false` (or unset) | [train-ec2.yml](../../.github/workflows/train-ec2.yml) — warm g4dn.xlarge OD (rollback) |

Flip with one command:

```
gh variable set BATCH_ACTIVE --body "true"   # → Batch (active)
gh variable set BATCH_ACTIVE --body "false"  # → EC2 rollback
```

The next push to `main` picks up the new value. The EC2 instance can stay
stopped in parallel — flipping back is instant.

## Cost

Spot g4dn.xlarge in us-east-1: ~$0.16/hr × 6 positions × ~10 min ≈ **~$0.16 per
full retrain** (measured 2026-05-21; cost dropped from the original ~$0.40 est.
as per-position training came in ~2 min not ~5). Single-position retrains scale
down linearly. At zero capacity
when idle, the CE has no standing cost. CloudWatch logs and ECR storage are
free-tier territory for this volume.

Compared to the warm-EC2 path's ~$8/mo idle EBS + ~$0.53/hr active OD, Spot
fanout is cheaper *and* faster.

## Teardown

```
bash infra/batch/teardown.sh
```

Disables and deletes JQ → CE → SG → IAM roles + profile. Preserves the ECR
repository, the pull-through cache rule, the log group, and service-linked
roles (`AWSServiceRoleForBatch`, `AWSServiceRoleForEC2Spot`). Remove those
manually if you want a complete wipe.

## What lives where

| File | Purpose |
|---|---|
| `setup.sh` | Idempotent provisioning: IAM, SG, CE, JQ, JD seed revision. |
| `teardown.sh` | Reverse-order tear down; idempotent. |
| `iam-trust-policy-job.json` | `ecs-tasks.amazonaws.com` trust (job + execution roles). |
| `iam-trust-policy-instance.json` | `ec2.amazonaws.com` trust (EC2 instance role). |
| `iam-job-policy.json` | Inline policy on `BatchTrainingRole` — S3 r/w, ECR pull (incl. pull-through hydration), CW Logs. |

## Why Batch + Spot, not warm EC2

See [`docs/batch_design.md`](../../docs/batch_design.md). Short version: warm
EC2 forced sequential training because one T4 can't host six concurrent NN
jobs; Spot fanout gives each position its own T4, parallelizing the workload.
Cold-start (the original blocker) is partially mitigated — SOCI indexes are
published but the snapshotter is not active on the default AMI, so measured
2026-05-20 cold-start is ~258 s vs the ~60–90 s design target. See
[docs/batch_design.md §2a](../../docs/batch_design.md) for activation paths.
