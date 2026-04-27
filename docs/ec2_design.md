# EC2 24/7 Training Design Doc

_Last verified: 2026-04-21._

## Problem

AWS Batch scale-to-zero costs 3-5 minutes of cold-start per run (compute-env provisioning, image pull, container start) for training that only takes ~2 minutes on GPU. That overhead dominates during "constant fine-tuning" where a push should produce a new model in minutes, not minutes-plus-cold-start.

Goals:

1. Eliminate cold start — keep an instance warm with the training image pre-pulled.
2. Drive training from CI on push to `main` so commit ↔ model is one-to-one.
3. Keep the existing S3-backed data/artifact flow and the 6-position fanout unchanged.
4. Cap cost with an idle auto-shutdown (instance stops after 4h quiet, CI wakes it on next push).

## Architecture

```
GITHUB ACTIONS (push to main)                AWS
─────────────────────────────        ──────────────────────────────

.github/workflows/
  batch-image.yml ─── builds image ─> ECR: ff-training:latest

  train-ec2.yml (workflow_run after image built)
    detect job
    └─ git diff HEAD^ HEAD → scope positions
         (shared|src|batch|requirements.txt → all 6;
          otherwise only the {POS}/ dirs that changed)

    train job  (skipped if detect.positions is empty)
    ├─ aws ec2 start-instances ────> EC2 g4dn.xlarge
    │                                  (stopped if idle > 4h,
    │                                   else already running)
    │
    ├─ wait SSM PingStatus=Online
    ├─ verify /opt/ff/config/bootstrap-complete
    │
    ├─ aws ssm send-command ───────> /usr/local/bin/ff-train QB
    │   single cmd wrapping             │
    │   for POS in $POSITIONS           ├─ docker pull (credsStore auth)
    │     (sequential — T4)             ├─ docker run --gpus all
    │                                   │    python -m src.batch.train
    │                                   │      --position $POS
    │                                   │      (reads s3://ff-predictor-training/data/)
    │                                   │      (writes s3://…/models/$POS/model.tar.gz)
    │                                   └─ date > /opt/ff/logs/last-activity
    │
    ├─ manual poll get-command-invocation (30-min deadline)
    ├─ stream stdout/stderr into Actions log
    ├─ aws s3api head-object (freshness check + summary table)
    └─ python -m src.batch.benchmark --download-only --backend ec2 …
         ├─ write {run_id}.json under benchmark_history/
         └─ commit + push (retry-rebase up to 3×)
```

## Directory structure (new)

```
infra/ec2/
  README.md                  # operator runbook + teardown
  launch-instance.sh         # one-shot bootstrap (quota check, IAM, SG, run-instances)
  user-data.sh               # cloud-init (NVMe mount, credsStore, ff-train helper, breadcrumb)
  iam-trust-policy.json      # ec2.amazonaws.com assume-role trust
  iam-instance-policy.json   # least-priv inline policy
  cloudwatch-agent.json      # ships /var/log/ff-train/*.log to /ff/training
  auto-shutdown.sh           # stops the instance after idle threshold
  auto-shutdown.service      # systemd unit wrapping the script
  auto-shutdown.timer        # fires every 15 min
```

## Data Flow

Same as Batch. Training container reads `s3://ff-predictor-training/data/{train,val,test}.parquet`, writes `s3://ff-predictor-training/models/{POS}/model.tar.gz`. S3 remains source of truth; instance store (`/opt/ff/scratch`) is ephemeral scratch bind-mounted into the container.

## Training Container

Reused as-is from Batch path. `batch/Dockerfile.train` produces the image, CI pushes to ECR `ff-training:latest`, the EC2 instance pulls it via nvidia-docker with `--gpus all`. Env vars passed to the container (`S3_BUCKET`, `S3_DATA_PREFIX`, `LOG_EVERY`, `REQUIRE_GPU`) are unchanged.

## Instance Spec

- **AMI**: Deep Learning AMI GPU PyTorch (Ubuntu 22.04), resolved at launch to the latest. Brings NVIDIA driver, Docker, nvidia-container-toolkit, SSM agent, `amazon-ecr-credential-helper`.
- **Type**: `g4dn.xlarge` (4 vCPU, 16 GB, 1× T4, 125 GB NVMe).
- **Root EBS**: 100 GB gp3 encrypted, DeleteOnTermination.
- **Scratch**: NVMe instance store mounted at `/opt/ff/scratch` (ephemeral; wiped on stop/start, re-mounted by cloud-init).
- **Security group**: egress all, no ingress — SSM is the only management channel.
- **IMDSv2** required, `HopLimit=2` so containers can reach IMDS for instance-profile credentials.
- **Termination protection** on.

## IAM

Role `ff-training-ec2-role`, instance profile `ff-training-ec2-profile`.

Managed policies: `AmazonSSMManagedInstanceCore`, `CloudWatchAgentServerPolicy`.

Inline `ff-training-workload`:
- S3 `GetObject/PutObject/DeleteObject/ListBucket` on `ff-predictor-training[/*]`.
- ECR `GetAuthorizationToken` (`*`) + pull verbs scoped to `repository/ff-training`.
- CloudWatch Logs on `log-group:/ff/training:*`.
- Deliberately excluded: `iam:*`, `ec2:*`, `batch:*`, ECR push.

## CI: .github/workflows/train-ec2.yml

Triggers:
- `workflow_run` after `batch-image.yml` succeeds on `main` — the image-build path filter (`batch/**`, `shared/**`, position dirs, `src/**`, `requirements.txt`) gates whether this fires at all, so non-model commits never train.
- `workflow_dispatch` with inputs `positions`, `seed` — break-glass for re-runs.

A `detect` job runs first and scopes which positions to train:
- On `workflow_run`, `git diff --name-only HEAD^ HEAD` on the merged commit. Changes under `src/shared/`, `src/batch/`, `src/data/`, `src/features/`, `src/models/`, `src/training/`, `src/evaluation/`, `src/config.py`, or `requirements.txt` are treated as global triggers and retrain all six positions; changes under a single `src/{POS}/` subtree retrain only that position. Edits scoped to non-training subtrees (e.g. `src/serving/`, `src/tuning/`, `src/analysis/`, `src/benchmarking/`, `src/scripts/`, `src/**/tests/`) do not trigger retraining. No model-relevant changes → `train` job skipped.
- On `workflow_dispatch`, the explicit `positions` input is honored verbatim. Empty input falls back to all six (legacy semantics for "retrain after data refresh").

Depends on `batch-image.yml` via `workflow_run` — training only fires after the new image has been pushed for the same commit.

Steps (train job, after `detect` scopes positions):
1. Configure AWS credentials (existing secrets).
2. Checkout repo with admin PAT (so the later benchmark-history push bypasses branch protection) and install `requirements-dev.txt` — torch is needed by `src/batch/benchmark.py`'s import chain.
3. Resolve instance ID from repo variable `EC2_TRAINER_INSTANCE_ID`.
4. `aws ec2 start-instances` + `aws ec2 wait instance-running` (no-op if already running).
5. Poll `aws ssm describe-instance-information` until `PingStatus=Online` (30 × 10s).
6. Verify `/opt/ff/config/bootstrap-complete` exists via an SSM `test -f` probe.
7. `aws ssm send-command` with a single command wrapping a sequential `for POS in …; do /usr/local/bin/ff-train $POS $SEED; done` (T4 can't fit concurrent NN runs; one command ID keeps polling simple).
8. Manual poll of `get-command-invocation` (30-min deadline — `aws ssm wait command-executed` caps at ~100 s and doesn't honor `AWS_MAX_ATTEMPTS`, so long runs would otherwise be mis-reported as failures).
9. Stream stdout/stderr into the Actions log via `get-command-invocation --query` (`if: always()`).
10. `aws s3api head-object` per position → summary table to `$GITHUB_STEP_SUMMARY`; fail if any artifact is missing or older than 20 min.
11. Write a per-run JSON file under `benchmark_history/` via `python -m src.batch.benchmark --download-only --backend ec2 --instance-type "g4dn.xlarge (On-Demand)" --positions $POSITIONS --note "EC2 auto-run (${sha::7})"`, then commit + push (retry-rebase up to 3×).

Concurrency: `group: train-ec2, cancel-in-progress: true` — rapid-iteration pushes supersede in-flight runs.

## Cost

~$50/mo active weeks (on-demand while running + EBS + logs), ~$8/mo idle (EBS only). Auto-shutdown after 4h idle is the knob — raise to reduce wake-ups, lower to save more.

Detailed line items:
- g4dn.xlarge on-demand us-east-1: $0.526/hr → $383/mo at 24/7, ~$42/mo at 4h/day active 5 days/week.
- Root EBS 100 GB gp3: ~$8/mo.
- CloudWatch Logs (7-day retention): ~$0.03/mo.
- SSM / IAM: ~$0.40/mo.

## Setup Steps

See [`infra/ec2/README.md`](../infra/ec2/README.md).

## Rollback

Batch remains a drop-in fallback:
1. Set repo variable `BATCH_ACTIVE=true` (re-enables job-definition registration in `batch-image.yml`).
2. Run `python batch/launch.py` locally — Batch resources already provisioned; no infra work needed.
3. Optional: disable `train-ec2.yml` by renaming or setting a repo variable guard.

## Reactivating Batch image-sync

`batch-image.yml` keeps building/pushing on every push to `main` regardless of which trainer is active — the EC2 trainer depends on the same image. Only the "Register new job definition revision" step is gated behind `BATCH_ACTIVE`, so re-syncing the job definition with HEAD is a one-flip operation.
