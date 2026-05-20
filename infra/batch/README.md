# ff-training AWS Batch + Spot

_Last verified: 2026-05-19._

Provisions an AWS Batch managed compute environment on Spot g4dn.xlarge so all
six position pipelines train in parallel (one position per Spot instance). The
active trainer is [train-batch.yml](../../.github/workflows/train-batch.yml)
when the `BATCH_ACTIVE` repo variable is `true`; otherwise the warm-EC2 path
([infra/ec2/](../ec2/)) handles training.

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
this project moved off Batch. Two AWS-side optimizations bring it down from
~120s to ~60–90s; both are wired into [batch-image.yml](../../.github/workflows/batch-image.yml).
**Run the pull-through cache rule creation once after `setup.sh`:**

```
aws ecr create-pull-through-cache-rule \
  --ecr-repository-prefix dockerhub \
  --upstream-registry-url registry-1.docker.io \
  --region us-east-1
```

After this, every CI build pulls the PyTorch base image from your ECR's local
endpoints instead of Docker Hub. SOCI index publishing happens automatically in
the same CI step (gated on the `soci` CLI install succeeding in the runner).

If your CI's GitHub Actions role doesn't already have
`ecr:BatchImportUpstreamImage` on `repository/dockerhub/*`, the first build
falls back to upstream Docker Hub (no failure, just slower). Add it to the
CI role for repeat builds to hit the cache.

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
   exactly saturating the 24 vCPU Spot quota). Total wall-clock should be
   ~25–30 min — the slowest position dominates, not the sum.

4. **End-to-end CI:** `gh workflow run train-batch.yml -f positions=K -f seed=42`
   should run the detect → launch → freshness → benchmark commit → ECS refresh
   pipeline and write a fresh `benchmark_history/{run_id}.json`.

## Switch the active trainer

Both training workflows guard on `vars.BATCH_ACTIVE`:

| `BATCH_ACTIVE` | Active path |
|---|---|
| `false` (or unset) | [train-ec2.yml](../../.github/workflows/train-ec2.yml) — warm g4dn.xlarge OD |
| `true` | [train-batch.yml](../../.github/workflows/train-batch.yml) — parallel Spot fanout |

Flip with one command:

```
gh variable set BATCH_ACTIVE --body "true"   # → Batch
gh variable set BATCH_ACTIVE --body "false"  # → EC2 rollback
```

The next push to `main` picks up the new value. The EC2 instance can stay
stopped in parallel — flipping back is instant.

## Cost

Spot g4dn.xlarge in us-east-1: ~$0.16/hr × 6 positions × ~25 min ≈ **~$0.40 per
full retrain**. Single-position retrains scale down linearly. At zero capacity
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
Cold-start (the original blocker) is mitigated by SOCI + ECR pull-through.
