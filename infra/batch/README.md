# ff-training AWS Batch + Spot

_Last verified: 2026-05-21._

Provisions AWS Batch managed Spot compute environments so all six position
pipelines train in parallel (one position per Spot instance). The job queue
prefers `g6.xlarge`/L4 and falls back to `g5.xlarge`/A10G when g6 Spot capacity
is unavailable. This is the **default push-driven trainer since 2026-05-20** —
[train-batch.yml](../../.github/workflows/train-batch.yml) runs when the
`BATCH_ACTIVE` repo variable is `true`; flipping it to `false` falls back to the
warm-EC2 path ([infra/ec2/](../ec2/)).

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
| Launch template | `ff-batch-lt` (UserData = [infra/batch/userdata.sh](userdata.sh), installs `soci-snapshotter-grpc` v0.13.0) |
| Compute environment | `ff-gpu-spot` (SPOT, max 24 vCPU, g6.xlarge, queue order 1) |
| Fallback compute environment | `ff-gpu-spot-g5` (SPOT, max 24 vCPU, g5.xlarge, queue order 2) |
| Job queue | `ff-training-queue` |
| Job definition | `ff-training-job` (rev 1; CI re-registers on every push) |
| CloudWatch log group | `/aws/batch/job` (7-day retention) |

## Cold-start optimization

The cold-start tax (image pull on a fresh Spot instance) is the original
reason this project moved off Batch. Two AWS-side optimizations bring it
down: the `ff-batch-lt` launch template (installs `soci-snapshotter-grpc`
on the AL2 host pre-boot via [userdata.sh](userdata.sh)) cuts the runtime
image pull from ~122s to ~5–10s; the ECR pull-through cache cuts GHA
build time on cold base layers. Baseline 2026-05-20 cold-start was
**~258 s** (~120 s Spot+boot + ~122 s full image pull + ~10 s container
start); expected post-Option-B cold-start is **~135 s** (~115 s saved
per job). See [docs/batch_design.md §2a](../../docs/batch_design.md) for
the full activation story.

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
   All six should reach RUNNING simultaneously (six 4-vCPU GPU Spot hosts,
   exactly saturating the 24 vCPU Spot quota; g6 preferred, g5 fallback). Total wall-clock for the
   "Submit Batch jobs and wait" step measured ~10 min on 2026-05-21
   — the slowest position dominates, not the sum.

4. **End-to-end CI:** `gh workflow run train-batch.yml -f positions=K -f seed=42`
   should run the detect → launch → freshness → benchmark commit → ECS refresh
   pipeline and write a fresh `benchmark_history/{run_id}.json`.

5. **SOCI is actually used (post-Option-B):** while a fresh K job is RUNNING,
   capture the image-pull window:
   ```
   JOB_ID=<from step 4>
   CLUSTER=$(aws batch describe-compute-environments \
     --compute-environments ff-gpu-spot --region us-east-1 \
     --query 'computeEnvironments[0].ecsClusterArn' --output text)
   TASK_ARN=$(aws batch describe-jobs --jobs $JOB_ID --region us-east-1 \
     --query 'jobs[0].container.taskArn' --output text)
   aws ecs describe-tasks --cluster $CLUSTER --tasks $TASK_ARN \
     --region us-east-1 \
     --query 'tasks[0].[pullStartedAt,pullStoppedAt]'
   ```
   Expect: pull window **~1–2 s** (down from ~122 s baseline 2026-05-20). Anything
   over ~30 s means SOCI didn't activate on that host — run "Rollback SOCI launch
   template" below and inspect `/var/log/soci-userdata.log` on the failing host
   via `aws ssm start-session --target <ec2InstanceId>` (`ecsInstanceRole` carries
   `AmazonSSMManagedInstanceCore`, attached by `setup.sh` §4; the SSM agent
   registers at boot, so this works on any Batch host launched after setup).

6. **Host-side plugin status** (when SSM is available on the host): the
   userdata.sh step-4 gate that prevents the race documented in §"SOCI
   snapshotter" can be re-checked at any time via:
   ```
   ctr plugin ls id==soci
   ```
   Expect (column STATUS = `ok`):
   ```
   TYPE                                  ID    PLATFORMS    STATUS
   io.containerd.snapshotter.v1          soci  linux/amd64  ok
   ```
   `STATUS != ok` on a live host = SOCI silently inactive; cycle the
   instance (Tier-1 rollback or just terminate to let Batch replace it).

## Rollback SOCI launch template

If userdata breaks instance boot (CE goes `INVALID`, jobs stuck `RUNNABLE`, or
first job fails repeatedly with `CannotPullContainerError` despite the per-job
retry), detach the launch template from the CE — next Spot host uses the
default ECS-optimized GPU AMI again, training resumes at the pre-Option-B ~122 s
pull:

```
aws batch update-compute-environment --compute-environment ff-gpu-spot \
  --state DISABLED --region us-east-1
# Wait VALID:
until [ "$(aws batch describe-compute-environments --compute-environments ff-gpu-spot \
  --region us-east-1 --query 'computeEnvironments[0].status' --output text)" = "VALID" ]; do
  sleep 5
done

aws batch update-compute-environment --compute-environment ff-gpu-spot \
  --compute-resources 'launchTemplate={}' --region us-east-1
until [ "$(aws batch describe-compute-environments --compute-environments ff-gpu-spot \
  --region us-east-1 --query 'computeEnvironments[0].status' --output text)" = "VALID" ]; do
  sleep 5
done

aws batch update-compute-environment --compute-environment ff-gpu-spot \
  --state ENABLED --region us-east-1
```

This is the Tier-1 rollback (fast, no PR revert). Tier-2 is the existing full
flip: `gh variable set BATCH_ACTIVE --body "false"` — returns push-driven
training to the warm-EC2 path, unaffected by the launch template.

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
on g4dn 2026-05-21). g5.xlarge fallback may cost more when used, but only when
g6 cannot provide suitable capacity. Single-position retrains scale down
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
| `setup.sh` | Idempotent provisioning: IAM, SG, launch template, CE (with `ff-batch-lt`), JQ, JD seed revision. Reconciles launch template onto existing CE on re-run. |
| `teardown.sh` | Reverse-order tear down (incl. launch template after CE delete); idempotent. |
| `userdata.sh` | EC2 launch-template UserData. Installs `soci-snapshotter-grpc` v0.13.0 on AL2 + configures containerd proxy plugin + starts systemd unit `Before=ecs.service`. |
| `iam-trust-policy-job.json` | `ecs-tasks.amazonaws.com` trust (job + execution roles). |
| `iam-trust-policy-instance.json` | `ec2.amazonaws.com` trust (EC2 instance role). |
| `iam-job-policy.json` | Inline policy on `BatchTrainingRole` — S3 r/w, ECR pull (incl. pull-through hydration), CW Logs. |

## Why Batch + Spot, not warm EC2

See [`docs/batch_design.md`](../../docs/batch_design.md). Short version: warm
EC2 forced sequential training because one T4 can't host six concurrent NN
jobs; Spot fanout gives each position its own g6.xlarge L4 host when available,
or a g5.xlarge A10G fallback host when needed, parallelizing the workload.
Cold-start (the original blocker) is mitigated by SOCI lazy-loading on the
Spot host via `ff-batch-lt` launch template + [userdata.sh](userdata.sh)
(Option B, default since 2026-05-21). Baseline 2026-05-20 cold-start was
~258 s (snapshotter inactive); expected post-Option-B is ~135 s. See
[docs/batch_design.md §2a](../../docs/batch_design.md) for the activation
story.
