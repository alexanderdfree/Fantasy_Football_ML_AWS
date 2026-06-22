# ff-training AWS Batch + Spot

_Last verified: 2026-06-15._

Provisions AWS Batch managed Spot compute environments so all six position
pipelines train in parallel (one position per Spot instance). The job queue
prefers `g6.xlarge`/L4 and falls back to `g5.xlarge`/A10G when g6 Spot capacity
is unavailable. This is the **default push-driven trainer since 2026-05-20** —
[train-batch.yml](../../.github/workflows/train-batch.yml) runs when the
`BATCH_ACTIVE` repo variable is `true`; flipping it to `false` falls back to the
warm-EC2 path ([infra/ec2/](../ec2/)).

See [`docs/batch_design.md`](../../docs/batch_design.md) for the full design
including the ECR pull-through cache cold-start optimization.

## First-time setup

Prereqs: AWS CLI v2, `gh` CLI, credentials with rights to create IAM + Batch
resources. The `ff-training` ECR repo must exist (created by the first run of
[batch-image.yml](../../.github/workflows/batch-image.yml)).

```
bash infra/batch/setup.sh
```

Idempotent — reruns skip anything that already exists. It performs a quota
preflight ("All G and VT Spot Instance Requests" >= 64 vCPU; raised from 24 on
2026-06-11) and refuses to proceed if the quota hasn't been approved.

The script creates:

| Resource | Name |
|---|---|
| IAM role (job/container) | `BatchTrainingRole` |
| IAM role (EC2 instance) | `ecsInstanceRole` |
| IAM role (task execution) | `ecsTaskExecutionRole` |
| Instance profile | `ecsInstanceRole` |
| Security group | `ff-batch-sg` (egress only) |
| GPU compute environment | `ff-gpu-spot` (SPOT, max 64 vCPU, diversified `g6.xlarge` + `g5.xlarge`, `SPOT_PRICE_CAPACITY_OPTIMIZED`) |
| Job queue | `ff-training-queue` |
| Job definition | `ff-training-job` (rev 1; CI re-registers on every push) |
| CloudWatch log group | `/aws/batch/job` (7-day retention) |

## Cold-start optimization

The cold-start tax (image pull on a fresh Spot instance) is the original
reason this project moved off Batch. A fresh Spot host pays the full
container image pull (~120 s) before training starts; baseline 2026-05-20
cold-start was **~258 s** (~120 s Spot+boot + ~122 s full image pull +
~10 s container start). What makes the Spot fan-out faster than warm-EC2
anyway is **parallelism** — that pull is paid once per host with all six
positions training concurrently, not six times in sequence (see
[ADR-0013](../../docs/adr/0013-spot-fan-out-via-aws-batch.md)).

The remaining AWS-side optimization is the **ECR pull-through cache** below
(it cuts GHA build time on cold base layers). SOCI lazy-loading was
evaluated to cut the runtime pull further but **removed 2026-06-07** — it
cannot work on ECS-managed Batch; see the historical note below.

### ECR pull-through cache (one-time, after `setup.sh`)

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

### SOCI snapshotter — ⚠️ REMOVED 2026-06-07 (did not work on Batch)

> **SOCI lazy-loading was removed.** It cannot work on AWS Batch: Batch runs on
> ECS-managed EC2 and the `amazon-ecs-agent` does not pull through the soci
> snapshotter (SOCI on ECS is **Fargate-only** —
> [containers-roadmap#1832](https://github.com/aws/containers-roadmap/issues/1832);
> and Fargate has no GPU). Verified: with a fully-correct soci config a prod K
> job still pulled in ~103 s (overlayfs) — same on g4dn and g6 (the limit is the
> ECS agent, not the GPU arch). The `ff-batch-lt` launch template, its
> `userdata.sh`, and the CI "Publish SOCI index" step were all removed; the CE
> uses the default ECS-optimized AMI and pays the ~120 s pull. Full root-cause:
> [../../docs/batch_design.md](../../docs/batch_design.md) §2a +
> [../../todo/fixed-archive.md](../../todo/fixed-archive.md). **Text below is
> retained for history only.**

A SOCI index is published alongside the image by
[batch-image.yml](../../.github/workflows/batch-image.yml)'s `Publish SOCI
index` step (`continue-on-error: false`, version-pinned to `0.13.0`). The
`ff-gpu-spot` CE uses the `ff-batch-lt` launch template, whose UserData
([userdata.sh](userdata.sh)) installs `soci-snapshotter-grpc` v0.13.0 on
the AL2 host, registers it as a containerd proxy plugin, and starts it
as a systemd unit ordered `Before=ecs.service`. First image pull on a
fresh Spot host streams lazily via SOCI instead of doing a full ~122 s
pull.

**Two-gate bootstrap (see [userdata.sh](userdata.sh) steps 3 + 4)**:

1. **Socket-wait** (after `systemctl enable --now soci-snapshotter`) —
   verifies the snapshotter daemon is up before containerd is asked to
   load it as a proxy plugin.
2. **Plugin-status-wait** (after `systemctl restart containerd`) —
   polls `ctr plugin ls` until the soci row reports `STATUS=ok`. Closes
   a race where `systemctl restart` returns on containerd's `Type=notify`
   READY signal (daemon socket open) but proxy-plugin discovery is still
   in progress — without this gate, ECS agent claims a task during the
   discovery window and the first pull silently falls back to overlayfs
   (SOCI [#190](https://github.com/awslabs/soci-snapshotter/issues/190)
   documents the silent-fallback issue).

Both gates exit cloud-init with code 1 on 60s timeout → instance marked
unhealthy in the CE; Batch's `Host EC2*` retry catches the next host
rather than letting a single broken host serve overlayfs pulls
indefinitely.

**SOCI_VERSION discipline**: the version pin MUST stay aligned between
[userdata.sh](userdata.sh) (host snapshotter) and
[batch-image.yml](../../.github/workflows/batch-image.yml) (index
publisher). Manifest format evolves between minor releases.

**Live-account migration**: running `bash infra/batch/setup.sh` against
an existing CE reconciles the launch template via `DISABLE →
update-compute-environment → ENABLE`, polling VALID between each step.
The CE's `minvCpus=0` means there are no in-flight instances to disrupt
— the next training run gets the new launch template.

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
   a fresh `manifest.json` at `s3://ff-predictor-training/models/K/manifest.json` pointing at the new `models/K/history/{ts}-{sha7}/model.tar.gz` artifact (the flat `models/K/model.tar.gz` mirror was removed in #282/#288).

3. **Full parallel fanout test:**
   ```
   AWS_REGION=us-east-1 python -m src.batch.launch \
     --positions QB RB WR TE K DST --seed 42
   ```
   All six should reach RUNNING simultaneously (six 4-vCPU GPU Spot hosts;
   the 64 vCPU Spot quota leaves headroom for a second concurrent fan-out or a
   tune fleet; one diversified g6+g5 Spot CE). Total wall-clock for the
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

Spot g6.xlarge in us-east-1: ~$0.35/hr × 6 positions × ~10 min ≈ **~$0.35 per
full retrain** (estimate post-migration from g4dn; measured baseline was ~$0.16
on g4dn 2026-05-21). `g5.xlarge` shares the pool and may cost slightly more on
the runs the allocation strategy places there, but it only picks g5 when g6 is
pricier or scarcer. Single-position retrains scale down
linearly. At zero capacity when idle, the CEs have no standing cost. CloudWatch
logs and ECR storage are free-tier territory for this volume.

Compared to the warm-EC2 g4dn path's ~$8/mo idle EBS + ~$0.53/hr active OD,
Spot fanout is still cheaper *and* faster. The g4dn → g6 swap was chosen for
BF16 support (T4 lacked sm_80+) and `torch.compile` re-eligibility (rejected on
T4 in D12), not cost — absolute annual delta is ~$25.

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
| `setup.sh` | Idempotent provisioning: IAM, SG, CE (`ff-gpu-spot`, diversified `g6.xlarge` + `g5.xlarge` Spot pool), JQ, JD seed revision. Re-runs skip anything that already exists. |
| `teardown.sh` | Reverse-order tear down (JQ → CE → SG → IAM roles + profile); idempotent. |
| `iam-trust-policy-job.json` | `ecs-tasks.amazonaws.com` trust (job + execution roles). |
| `iam-trust-policy-instance.json` | `ec2.amazonaws.com` trust (EC2 instance role). |
| `iam-job-policy.json` | Inline policy on `BatchTrainingRole` — S3 r/w, ECR pull (incl. pull-through hydration), CW Logs. |

## Why Batch + Spot, not warm EC2

See [`docs/batch_design.md`](../../docs/batch_design.md). Short version: warm
EC2 forced sequential training because one T4 can't host six concurrent NN
jobs; Spot fanout gives each position its own single-GPU Spot host from a
diversified g6.xlarge/L4 + g5.xlarge/A10G pool, parallelizing the workload.
Cold-start (the original blocker) is paid once per host (~120 s full image
pull) while all six positions train concurrently, so it no longer
dominates wall-clock the way the sequential warm-EC2 loop did. SOCI
lazy-loading was evaluated to cut it further but **removed 2026-06-07** —
it cannot work on ECS-managed Batch; see
[docs/batch_design.md §2a](../../docs/batch_design.md).
