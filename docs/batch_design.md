# AWS Batch Training Design Doc

> **Status (2026-05-31): Active when `BATCH_ACTIVE=true`.** Parallel Spot fan-out via [.github/workflows/train-batch.yml](../.github/workflows/train-batch.yml) — six g6.xlarge Spot instances, one position per host (migrated from g4dn.xlarge 2026-05-31 — T4 → L4 for BF16 + torch.compile re-eligibility; see ARCHITECTURE.md Update history). **Measured 2026-05-21: ~10 min for the "Submit Batch jobs and wait" step (training itself), which is now train-batch.yml's dominant wall-clock cost — PR #330 removed the forced "Refresh ECS service" (`update-service --force-new-deployment`) step from train-batch.yml (per-position in-flight model refresh via `src/shared/model_sync.py` makes new artifacts live within one ~30s poll interval; ECS rolling redeploy via deploy.yml's ALB-tuned ~3 min path is now decoupled from the train workflow). The original 2026-05-20 design estimate was ~25–30 min — actual came in faster mainly because per-position training is closer to ~2 min than the 5 min estimate.** Warm-EC2 rollback path: ~120-min sequential loop. Cold-start mitigation: SOCI v2 indexes published by PR #216 + `soci-snapshotter-grpc` v0.13.0 installed on the AL2 Spot host via the `ff-batch-lt` launch template (Option B, this PR — see §2a for the active configuration). Image kept ~5–6 GB via aggressive `.dockerignore` + explicit `COPY`. Baseline 2026-05-20 cold-start ~258 s; expected ~135 s once the launch template is attached to the live CE (post-merge `bash infra/batch/setup.sh` reconciles). See [ADR-0013 (Spot fan-out via AWS Batch)](adr/0013-spot-fan-out-via-aws-batch.md) for the decision; [infra/batch/README.md](../infra/batch/README.md) is the operator runbook.
>
> Rollback: `gh variable set BATCH_ACTIVE --body "false"` returns push-driven training to the warm-EC2 path ([docs/ec2_design.md](ec2_design.md)) on the next push to `main`. Both paths remain provisioned.

## Problem

Managed training services add 3-5 minutes of cold-start overhead (instance provisioning,
container pull, dependency install) for training runs that only take ~2 minutes on GPU.
Goals:

1. Minimize orchestration overhead with a lean job submission path
2. Cut compute cost ~60% via Spot pricing (~$0.35/hr g6 Spot vs $0.80/hr g6 on-demand; the historical g4dn comparison was ~$0.16/hr Spot vs $0.53/hr OD pre-2026-05-31 migration)
3. Keep the same parallel-6-positions pattern and S3 artifact flow
4. Scale to zero when idle (no cost between runs)

## Architecture

```
LOCAL (your laptop)                          AWS
─────────────────                   ─────────────────────────────

src/batch/launch.py ─────────────> S3: s3://ff-training/data/
  uploads data/splits/*.parquet        train.parquet
  submits 6 Batch jobs                 val.parquet
       │                               test.parquet
       │
       ├─> Batch Job: ff-rb-xxx ───> CloudWatch Logs
       │     (g6.xlarge Spot)          stdout/stderr streamed
       │     src.batch.train --position RB
       │       ├─ boto3: download data from S3
       │       ├─ src.rb.run_pipeline.run(train_df, val_df, test_df)
       │       ├─ save benchmark_metrics.json
       │       └─ boto3: upload model.tar.gz to S3
       │
       ├─> Batch Job: ff-wr-xxx ───> CloudWatch Logs
       ├─> Batch Job: ff-qb-xxx ───> CloudWatch Logs
       ├─> Batch Job: ff-te-xxx ───> CloudWatch Logs
       ├─> Batch Job: ff-k-xxx  ───> CloudWatch Logs
       └─> Batch Job: ff-dst-xxx ──> CloudWatch Logs
                                           │
src/batch/launch.py <──────────────────────┘
  polls describe_jobs() for status   S3: s3://ff-training/models/
  resolves rb/manifest.json -> rb/    rb/manifest.json
    history/{ts}-{sha7}/model.tar.gz  rb/history/{ts}-{sha7}/model.tar.gz
  downloads + extracts to             wr/manifest.json
    src/rb/outputs/models/            wr/history/{ts}-{sha7}/model.tar.gz
    src/wr/outputs/models/            ...
```

### Data Staging

The container handles data staging (S3 → container) and artifact packaging
(container → S3) explicitly via boto3 — about 20 extra lines in `train.py`, but artifact
locations are predictable and the launcher (`launch.py`) stays simple.

## Directory Structure

```
src/batch/
  __init__.py
  train.py              ← container entry point
  launch.py             ← job submitter (boto3 Batch client)
  benchmark.py          ← Batch-side benchmark runner (downloads metrics from S3)
  Dockerfile.train      ← GPU training image
  Dockerfile.train.dockerignore
  build_and_push.sh     ← ECR login + buildx + SOCI index publish
  requirements.txt      ← container-only deps (no torch — base image provides it)

tests/batch/            ← Batch tests live under the top-level tests/ tree
  __init__.py
  test_launch.py
  test_launch_main.py
  test_train.py
  test_train_smoke.py
  ...

src/benchmarking/benchmark.py ← Local multi-position benchmark runner (separate from src/batch/benchmark.py).
```

## Data Flow

| Step | How |
|------|-----|
| Upload data | `boto3` `s3.upload_file()` in launch.py |
| Download data in container | train.py calls `s3.download_file()` at start |
| Run training | Batch runs the ENTRYPOINT |
| Upload artifacts | train.py calls `s3.upload_file()` at end |
| Download artifacts | launch.py downloads from known S3 path directly |

## Training Container

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime

WORKDIR /opt/ml/code

COPY src/batch/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

ENTRYPOINT ["python", "-m", "src.batch.train"]
```

- Base image has PyTorch + CUDA pre-installed (~4 GB)
- Project source baked into `/opt/ml/code/`
- ENTRYPOINT runs train.py; Batch passes `["--position", "RB", "--seed", "42"]` as command override
- Image size: ~5-6 GB total

### Container Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TRAINING_DATA_DIR` | `/opt/ml/input/data/training/` | Where parquet files are downloaded to |
| `MODEL_OUTPUT_DIR` | `/opt/ml/model/` | Where model artifacts are written |
| `LOG_EVERY` | `1` (batch) / `10` (default) | Epoch logging frequency; read by `shared.pipeline._resolve_nn_log_every` |
| `S3_BUCKET` | (required) | S3 bucket for data and artifacts |
| `S3_DATA_PREFIX` | `data` | S3 key prefix for training data |
| `REQUIRE_GPU` | `1` | Fail fast if CUDA unavailable. **Auto-skipped for K/DST** (CPU-only pipelines). |

### Launcher Environment Variables (`src/batch/launch.py`)

| Variable | Default | Description |
|---|---|---|
| `FF_S3_BUCKET` | `ff-predictor-training` | Override bucket name (for staging accounts) |
| `FF_JOB_QUEUE` | `ff-training-queue` | Override Batch job queue |
| `FF_JOB_DEFINITION` | `ff-training-job` | Override Batch job definition (GPU) |
| `FF_JOB_DEFINITION_CPU` | (unset) | **Optional CPU job definition for K/DST.** When set, K/DST jobs submit here instead of the GPU queue — saves ~60% of Spot spend on those positions. Falls back to the GPU definition when unset. |
| `FF_WAIT_TIMEOUT` | `10800` (3h) | Wall-clock cap for `wait_for_jobs` |

### Container Dependencies (`src/batch/requirements.txt`)

Derived from root `requirements.txt`:
- **Excluded**: `torch` (in base image), `flask`, `gunicorn`, `pytest`
- **Added**: `boto3>=1.34` (S3 operations), `nflreadpy==0.1.5` + `polars==1.41.1` (nflverse data loading via the `src/data/nfl_source.py` shim; imported transitively by K/DST data modules even though no fetch happens at training time)

## Position Pipeline Invocation

Pipeline registry in `src/batch/train.py`:

```python
POSITIONS = {
    "QB":  ("src.qb.run_pipeline",  "run", True),
    "RB":  ("src.rb.run_pipeline",  "run", True),
    "WR":  ("src.wr.run_pipeline",  "run", True),
    "TE":  ("src.te.run_pipeline",  "run", True),
    "K":   ("src.k.run_pipeline",   "run", False),
    "DST": ("src.dst.run_pipeline", "run", False),
}
```

- **Standard (QB, RB, WR, TE)**: `accepts_df=True` — train.py downloads parquets from S3, passes DataFrames
- **Special (K, DST)**: `accepts_df=False` — load their own data internally, no S3 download needed

## Job Submission

`src/batch/launch.py` submits jobs via `boto3.client('batch').submit_job()`:

```python
batch.submit_job(
    jobName=f"ff-{position.lower()}-{timestamp}",
    jobQueue=JOB_QUEUE,
    jobDefinition=JOB_DEFINITION,
    containerOverrides={
        "command": ["--position", position, "--seed", str(seed)],
        "environment": [
            {"name": "S3_BUCKET", "value": S3_BUCKET},
            {"name": "S3_DATA_PREFIX", "value": "data"},
            {"name": "LOG_EVERY", "value": "1"},
        ],
    },
)
```

All 6 positions submitted in parallel via ThreadPoolExecutor. `wait_for_jobs()` polls
`describe_jobs()` every 30 seconds and prints status transitions.

## Monitoring

| What you want to see | How |
|---|---|
| Live logs in terminal | `aws logs tail /aws/batch/job --follow --filter-pattern ff-training` |
| Job status | `aws batch describe-jobs --jobs JOB_ID` |
| All jobs at a glance | `aws batch list-jobs --job-queue ff-training-queue --job-status RUNNING` |
| Benchmark metrics | Downloaded as `benchmark_metrics.json` in model artifacts |
| GPU utilisation | Packed into the tarball as `gpu_profile_{POS}.csv` (500ms nvidia-smi samples around `run_pipeline`); summarize with `python -m src.scripts.analyze_gpu_profile --s3 --positions QB RB WR TE K DST` |
| Historical logs | CloudWatch > Log Groups > `/aws/batch/job` |

Terminal stdout and `benchmark_metrics.json` provide sufficient
visibility for this project.

## AWS Resources Required

### ECR Repository

```bash
aws ecr create-repository --repository-name ff-training --region us-east-1
```

### IAM Roles

**Job Role** (container S3 access):
```
Name: BatchTrainingRole
Trust policy: ecs-tasks.amazonaws.com
Policies:
  - S3 read/write on ff-predictor-training bucket
  - CloudWatch Logs write
```

**Execution Role** (ECS image pull + logging): Reuse existing `ecsTaskExecutionRole`.

### Compute Environment

```bash
aws batch create-compute-environment \
  --compute-environment-name ff-gpu-spot \
  --type MANAGED \
  --state ENABLED \
  --compute-resources '{
    "type": "SPOT",
    "allocationStrategy": "SPOT_PRICE_CAPACITY_OPTIMIZED",
    "minvCpus": 0,
    "maxvCpus": 24,
    "instanceTypes": ["g6.xlarge"],
    "subnets": ["SUBNET_A", "SUBNET_B"],
    "securityGroupIds": ["DEFAULT_SG"],
    "instanceRole": "ecsInstanceRole",
    "spotIamFleetRole": "arn:aws:iam::ACCOUNT_ID:role/aws-ec2-spot-fleet-tagging-role"
  }'
```

- `type=SPOT` — 70% cheaper than on-demand
- `minvCpus=0` — scales to zero when idle (no cost)
- `maxvCpus=24` — up to 6 concurrent g6.xlarge (4 vCPUs each, same as g4dn)
- `allocationStrategy=SPOT_PRICE_CAPACITY_OPTIMIZED` — AWS-recommended strategy that weighs *both* capacity (lowest current reclaim risk) and Spot price. Strict superset of `SPOT_CAPACITY_OPTIMIZED`: same reclaim-avoidance behaviour plus price awareness

### Job Queue

```bash
aws batch create-job-queue \
  --job-queue-name ff-training-queue \
  --state ENABLED \
  --priority 1 \
  --compute-environment-order order=1,computeEnvironment=ff-gpu-spot
```

### Job Definition

```bash
aws batch register-job-definition \
  --job-definition-name ff-training-job \
  --type container \
  --platform-capabilities EC2 \
  --container-properties '{
    "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ff-training:latest",
    "vcpus": 4,
    "memory": 15000,
    "jobRoleArn": "arn:aws:iam::ACCOUNT_ID:role/BatchTrainingRole",
    "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
    "resourceRequirements": [{"type": "GPU", "value": "1"}],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/aws/batch/job",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ff-training"
      }
    }
  }' \
  --timeout '{"attemptDurationSeconds": 1800}' \
  --retry-strategy '{
    "attempts": 3,
    "evaluateOnExit": [
      {"onStatusReason": "Host EC2*", "action": "RETRY"},
      {"onReason": "CannotPullContainerError*", "action": "RETRY"},
      {"onReason": "*", "action": "EXIT"}
    ]
  }'
```

- `vcpus=4, memory=15000` — matches g6.xlarge (4 vCPU, 16 GB RAM; identical sizing to g4dn pre-2026-05-31)
- `resourceRequirements: GPU=1` — ensures GPU scheduling
- `timeout: 1800` — 30-minute max **per attempt** (wall-clock cap)
- `retry-strategy` — up to 3 attempts; `evaluateOnExit` retries only on Spot reclaim (`Host EC2*`) or transient ECR pull errors. Anything else (`onReason: "*"`) is treated as a genuine app failure and fails immediately so app crashes don't loop. With the 30-min per-attempt cap, worst-case wall-clock is bounded at 90 min per position.
- `command` not set here — overridden per-job via `containerOverrides`

**Two layers of retry strategy — keep them in sync.** The job definition above is the fallback for any submitter (e.g. a manual `aws batch submit-job`), but production training goes through [src/batch/launch.py](../src/batch/launch.py), which passes its own `retryStrategy={...}` to `submit_job()` — that per-submission override wins for those jobs. The job definition's retry strategy is hardcoded in [.github/workflows/batch-image.yml](../.github/workflows/batch-image.yml)'s registration step (the workflow file is the source of truth — `infra/batch/setup.sh` is only the seed for brand-new accounts, since `setup.sh` is idempotent and the workflow re-registers on every image push). All three locations — launch.py, the workflow, and setup.sh — must stay in sync.

### Build and Push Image

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

# Build
docker build -f src/batch/Dockerfile.train -t ff-training:latest .

# Authenticate
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Tag and push
docker tag ff-training:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/ff-training:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/ff-training:latest
```

## Cost

| Resource | Estimate |
|---|---|
| Batch GPU compute per full run (6 g6 Spot hosts x measured ~10 min wait) | ~$0.35 (Spot) |
| Idle cost | $0 (scales to zero) |
| Service fee | Free (pay only EC2) |
| ECR image storage | ~$0.50/month for 5 GB |
| S3 storage (data + models) | < $0.01/month |
| CloudWatch logs | Free tier |

## Setup Steps

1. Create ECR repository (`ff-training`)
2. Create IAM roles (`BatchTrainingRole` + reuse `ecsTaskExecutionRole`)
3. Create Compute Environment (`ff-gpu-spot`)
4. Create Job Queue (`ff-training-queue`)
5. Register Job Definition (`ff-training-job`)
6. Build and push training image to ECR
7. Run: `python -m src.batch.launch`

## Rollback

The existing Flask Dockerfile and `src/serving/app.py` inference code are completely unaffected.
CUDA auto-detection in `src/shared/pipeline.py` falls back to CPU. Local pipeline scripts
(`python -m src.qb.run_pipeline`) work identically without any AWS dependencies.

## CPU-only Queue for K/DST (optional)

K and DST pipelines are Ridge/LGBM only — they never touch CUDA. Running them on
g6.xlarge Spot costs ~$0.35/hr of GPU time they won't use (higher than the g4dn
era's ~$0.16/hr; the gap is now bigger so the CPU-queue optimization is more
worthwhile). To route them to a cheaper CPU Spot pool:

1. Register a CPU compute env (e.g. `c6i.large` Spot) + job queue + CPU job
   definition (`ff-training-job-cpu`) pointing at the same ECR image.
2. Export `FF_JOB_DEFINITION_CPU=ff-training-job-cpu` before running
   `python -m src.batch.launch`. K and DST will submit there; QB/RB/WR/TE stay on the GPU
   queue.
3. When `FF_JOB_DEFINITION_CPU` is unset, K/DST fall back to the GPU definition —
   so it's safe to deploy this code before the CPU infra exists.

## CI/CD

Two workflows cover the training image and the inference service:

- `.github/workflows/batch-image.yml` — builds `src/batch/Dockerfile.train`, pushes
  to ECR (`ff-training`), and registers a new revision of the `ff-training-job`
  Batch job definition pinned to the new image SHA. Triggered by any change
  under `src/**` (excluding `**/tests/**` and `**/*.md`) or `requirements.txt`.
- `.github/workflows/deploy.yml` — builds the inference `Dockerfile`, pushes to
  ECR (`fantasy-predictor`), and updates the ECS service. Now gated on the
  full test suite.
- `.github/workflows/tests.yml` — runs pytest across **all** position test
  directories plus `src/batch/` and `src/shared/` on every push and PR.

## Cold-start optimization (image pull acceleration)

The largest chunk of per-job wall time on a cold Spot instance is pulling the
~7–8 GB `pytorch/pytorch:*-cuda12.6-cudnn9-runtime` base image from a public
registry. Three stacking optimizations target this:

### 2b. Explicit COPYs in Dockerfile.train

`src/batch/Dockerfile.train` used to end with `COPY . .`, shipping the Flask UI
(`src/serving/app.py`, `src/serving/static/`, `src/serving/templates/`), scratch scripts (`src/tuning/tune_*.py`,
`src/benchmarking/benchmark.py`, `src/analysis/analysis_*.py`), and everything else at the repo root into
the training image. The Dockerfile now copies only the dirs that
`src/batch/train.py` actually imports: `src/batch/`, `src/shared/`, `src/`, and the six
position dirs. `.dockerignore` handles the coarse exclusions (caches, outputs,
`*.db` files, `data/`).

### 2c. ECR pull-through cache for the base image

One-time AWS setup:

```bash
aws ecr create-pull-through-cache-rule \
  --ecr-repository-prefix dockerhub \
  --upstream-registry-url registry-1.docker.io \
  --region us-east-1
```

Ensure the Batch instance role (and any CI role doing the build) has
`ecr:BatchImportUpstreamImage` in addition to the standard pull permissions —
without it the pull-through rule silently falls back to the upstream fetch.

The Dockerfile accepts a `PULL_THROUGH_PREFIX` build arg so the base `FROM`
can be routed through ECR:

```bash
--build-arg PULL_THROUGH_PREFIX=<account>.dkr.ecr.us-east-1.amazonaws.com/dockerhub/
```

After the first pull seeds the cache, every subsequent Batch instance in the
region pulls the base layers from ECR's local endpoints instead of Docker Hub.

### 2a. SOCI (Seekable OCI) lazy loading

SOCI v2 lazy-loading lets a container start executing before its full
image is pulled — essential files stream first, the rest loads in the
background. It requires both a SOCI index in ECR alongside the image
**and** the `soci-snapshotter-grpc` plugin active on the ECS host's
containerd.

**Active configuration (2026-05-21, Option B)**: indexes are published
to the `ff-training` ECR repo by [batch-image.yml](../.github/workflows/batch-image.yml)'s
`Publish SOCI index` step (`continue-on-error: false`, version-pinned
to `0.13.0`). The `ff-gpu-spot` Compute Environment uses the
`ff-batch-lt` EC2 launch template, whose UserData
([infra/batch/userdata.sh](../infra/batch/userdata.sh)) installs
`soci-snapshotter-grpc` v0.13.0 on the AL2 host pre-boot, registers it
as a containerd proxy plugin, and starts it as a systemd unit ordered
`Before=containerd.service` (ecs.service is transitively `After=containerd.service` on AL2, so no explicit `Before=ecs.service` is needed). Two gates guard the bootstrap:

1. **Socket-wait** (after `systemctl enable --now soci-snapshotter`,
   before the `/etc/containerd/config.toml` edit) — verifies the
   snapshotter daemon is up and its socket exists before containerd is
   asked to load it as a proxy plugin.
2. **Plugin-status-wait** (after `systemctl restart containerd`) —
   polls `ctr plugin ls` until the soci snapshotter row reports
   `STATUS=ok`. Closes a race where `systemctl restart` returns on
   containerd's `Type=notify` READY signal (daemon socket open) but
   proxy plugin discovery is still in progress; the ECS agent
   (correctly After=cloud-final.service) can otherwise issue its first
   pull during the discovery window, fall back to overlayfs, and stay
   in fallback for the rest of the boot. SOCI [issue #190](https://github.com/awslabs/soci-snapshotter/issues/190)
   documents the silent-fallback failure mode. Live evidence motivating
   this gate: post-PR #295 production data showed a 4/5 success ratio
   (1/5 hosts measured 131 s full pull instead of 1–2 s lazy-load).

Both gates exit cloud-init with code 1 on timeout, marking the
instance unhealthy in the CE rather than serving overlayfs pulls.

Baseline 2026-05-20 cold-start was ~258 s (snapshotter inactive,
~122 s full image pull). Expected post-Option-B cold-start is ~135 s
(120 s instance provisioning + ~5 s SOCI lazy-start + 10 s container
start), saving ~115 s per Spot cold-start. Operator activates Option B
on the live account by running `bash infra/batch/setup.sh`, which
reconciles the launch template onto the existing CE via
`DISABLE → update-compute-environment → ENABLE`. Post-merge measurement
captures the actual pull window via `aws ecs describe-tasks
--query 'tasks[0].[pullStartedAt,pullStoppedAt]'` on a smoke-test job.

**Rejected (Option A)**: Bottlerocket GPU AMI. Migration changes OS
family — nvidia driver baking, ECS agent customization, debugging
surface all shift. Higher risk than userdata install on the existing
AL2 lineage; revisit only if Option B repeatedly fails.

**Rollback**: detach the launch template via
`aws batch update-compute-environment --compute-environment ff-gpu-spot
--compute-resources 'launchTemplate={}'` (with disable/enable
bracketing). Next Spot host uses the default AMI again; pull window
returns to ~122 s. See [infra/batch/README.md](../infra/batch/README.md)
"Rollback SOCI launch template" for the full sequence.

**Version pin discipline**: `SOCI_VERSION` in
[infra/batch/userdata.sh](../infra/batch/userdata.sh) (host snapshotter)
and [batch-image.yml](../.github/workflows/batch-image.yml) (index
publisher) MUST stay aligned. SOCI manifest format evolves between
minor releases; publisher/consumer skew breaks lazy-load silently.

One-time developer setup for local builds: install the `soci` CLI from
[soci-snapshotter releases](https://github.com/awslabs/soci-snapshotter/releases).

### Build & push

`src/batch/build_and_push.sh` wires all three together:

```bash
./src/batch/build_and_push.sh                        # defaults: us-east-1, ff-training:latest
IMAGE_TAG=$(git rev-parse --short HEAD) ./src/batch/build_and_push.sh
USE_PULL_THROUGH=0 ./src/batch/build_and_push.sh     # bypass pull-through (for debugging)
SKIP_SOCI=1 ./src/batch/build_and_push.sh            # skip SOCI index even if soci is installed
```

The script logs in to ECR, builds with the pull-through base, pushes the
image, and then (if `soci` is present) creates and pushes the SOCI index
next to the image tag. If `soci` isn't installed, the script warns and
exits cleanly — image still works, cold starts just aren't accelerated.

### Cold-start (baseline 2026-05-20 and Option B target)

Cold-start on the active Batch path (six g6.xlarge Spot post-2026-05-31; was
g4dn.xlarge when these baseline measurements were taken), measured
2026-05-20 via live ECS `describe-tasks` + Batch `describe-jobs`
across a workflow_run, and the expected window after Option B
activation (launch template + SOCI snapshotter, see §2a). The instance-floor
phase is GPU-family-independent; the swap to g6 doesn't change the cold-start
math materially:

| Phase | Baseline (2026-05-20, pre-Option-B) | Post-Option-B target |
|---|---|---|
| Spot fulfillment + EC2 boot + ECS agent register + Batch claim | ~120 s | ~120 s (unchanged — instance floor) |
| Image pull | **~122 s** (containerd full pull, snapshotter inactive) | **~5–10 s** (SOCI lazy-load) |
| Container start + `ENTRYPOINT` | ~10 s | ~10 s (unchanged) |
| **Total cold-start** | **~258 s** (outlier: ~365 s when 6th Spot instance takes ~6 min) | **~135 s** (~115 s saved per job) |

Post-merge measurement step (after operator runs
`bash infra/batch/setup.sh` to reconcile the launch template onto the
live CE): trigger `gh workflow run train-batch.yml -f positions=K
-f seed=42`, then `aws ecs describe-tasks --query
'tasks[0].[pullStartedAt,pullStoppedAt]'` on the resulting task. The
pull window drop is the canonical Option B success signal.

The original ~60–90 s design target assumed (a) snapshotter active and
(b) instance provisioning ~30 s. (b) was optimistic — instance
provisioning on G-family Spot (g4dn pre-migration, g6 now) is ~120 s floor regardless. (a) was
addressed by Option B; the achievable post-Option-B total is ~135 s,
not 60–90 s, but that's still the single largest realizable win on
this path.
