# AWS Batch Training Design Doc

> **Status (2026-04-21): Standby path.** The active training path is EC2 warm-host ([docs/ec2_design.md](ec2_design.md)). Reactivate Batch by setting repo variable `BATCH_ACTIVE=true`.
>
> Image builds still track HEAD ([`.github/workflows/batch-image.yml`](../.github/workflows/batch-image.yml)) so reactivation is one repo-variable flip and a push — Batch resources ([`src/batch/launch.py`](../src/batch/launch.py), [`src/batch/train.py`](../src/batch/train.py), [`src/batch/Dockerfile.train`](../src/batch/Dockerfile.train)) remain in place.

## Problem

Managed training services add 3-5 minutes of cold-start overhead (instance provisioning,
container pull, dependency install) for training runs that only take ~2 minutes on GPU.
Goals:

1. Minimize orchestration overhead with a lean job submission path
2. Cut compute cost ~70% via Spot pricing ($0.16/hr vs $0.53/hr for g4dn.xlarge)
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
       │     (g4dn.xlarge Spot)        stdout/stderr streamed
       │     src.batch.train --position RB
       │       ├─ boto3: download data from S3
       │       ├─ run_rb_pipeline(train_df, val_df, test_df)
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
  downloads model artifacts             rb/model.tar.gz
  extracts to src/RB/outputs/models/    wr/model.tar.gz
             src/WR/outputs/models/     ...
             ...
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
  requirements.txt      ← container-only deps (no torch — base image provides it)
  tests/
    __init__.py
    conftest.py
    test_launch.py
    test_train.py
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
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

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
- **Added**: `boto3>=1.34` (S3 operations), `nfl_data_py==0.3.3` (K/DST data loading)

## Position Pipeline Invocation

Pipeline registry in `src/batch/train.py`:

```python
POSITIONS = {
    "QB": ("QB.run_qb_pipeline", "run_qb_pipeline", True),
    "RB": ("RB.run_rb_pipeline", "run_rb_pipeline", True),
    "WR": ("WR.run_wr_pipeline", "run_wr_pipeline", True),
    "TE": ("TE.run_te_pipeline", "run_te_pipeline", True),
    "K":  ("K.run_k_pipeline",   "run_k_pipeline",  False),
    "DST": ("DST.run_dst_pipeline", "run_dst_pipeline", False),
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
    "allocationStrategy": "SPOT_CAPACITY_OPTIMIZED",
    "minvCpus": 0,
    "maxvCpus": 24,
    "instanceTypes": ["g4dn.xlarge"],
    "subnets": ["SUBNET_A", "SUBNET_B"],
    "securityGroupIds": ["DEFAULT_SG"],
    "instanceRole": "ecsInstanceRole",
    "spotIamFleetRole": "arn:aws:iam::ACCOUNT_ID:role/aws-ec2-spot-fleet-tagging-role"
  }'
```

- `type=SPOT` — 70% cheaper than on-demand
- `minvCpus=0` — scales to zero when idle (no cost)
- `maxvCpus=24` — up to 6 concurrent g4dn.xlarge (4 vCPUs each)
- `allocationStrategy=SPOT_CAPACITY_OPTIMIZED` — best Spot availability

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
  --retry-strategy '{"attempts": 1}'
```

- `vcpus=4, memory=15000` — matches g4dn.xlarge (4 vCPU, 16 GB RAM)
- `resourceRequirements: GPU=1` — ensures GPU scheduling
- `timeout: 1800` — 30-minute max
- `command` not set here — overridden per-job via `containerOverrides`

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
| Compute per full run (6 positions x ~2 min) | ~$0.03 (Spot) |
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
(`python -m src.QB.run_qb_pipeline`) work identically without any AWS dependencies.

## CPU-only Queue for K/DST (optional)

K and DST pipelines are Ridge/LGBM only — they never touch CUDA. Running them on
g4dn.xlarge Spot costs ~$0.16/hr of GPU time they won't use. To route them to a
cheaper CPU Spot pool:

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

SOCI v2 is enabled by default on all ECS accounts (and AWS Batch uses ECS
under the hood). Publishing a SOCI index alongside the image in ECR lets the
container start executing before the full image is pulled — essential files
stream first, the rest loads in the background.

One-time developer setup: install the `soci` CLI from
[soci-snapshotter releases](https://github.com/awslabs/soci-snapshotter/releases).
No Batch or ECS configuration changes are required.

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

### Expected impact

Pull phase of a cold Batch job startup: ~120 s → ~20–40 s. Combined with the
existing ~30–60 s of boot + data load, total Batch startup lands at ~60–90 s
on the current `g4dn.xlarge`.

