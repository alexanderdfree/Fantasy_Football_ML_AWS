# AWS Batch Training Design Doc

> **Status (2026-06-10): Active when `BATCH_ACTIVE=true`.** Default mode remains the D13 parallel Spot fan-out via [.github/workflows/train-batch.yml](../.github/workflows/train-batch.yml): six 4-vCPU GPU Spot hosts, one position per host. The GPU job queue prefers `g6.xlarge` / L4 (`ff-gpu-spot`, order 1) and falls back to `g5.xlarge` / A10G (`ff-gpu-spot-g5`, order 2) when g6 cannot provide suitable capacity. **Split mode is opt-in via `BATCH_SPLIT_ACTIVE=true` / `src.batch.launch --split`: NN work stays on the GPU queue, while Ridge+LightGBM run on a new c8a CPU Spot queue and a merge job publishes the complete artifact only after both staged branches validate (ADR-0019).** Measured 2026-05-21 monolithic path: ~10 min for the "Submit Batch jobs and wait" step. Warm-EC2 rollback path: ~120-min sequential loop. Cold-start: ~120 s image pull on a fresh Spot host; SOCI lazy-loading was removed 2026-06-07 because ECS-EC2 Batch cannot use it. Image size is now dominated by the torch CUDA wheel; `.dockerignore` + explicit `COPY` trim the app side. See [ADR-0013 (Spot fan-out via AWS Batch)](adr/0013-spot-fan-out-via-aws-batch.md) and [ADR-0019 (split Batch training)](adr/0019-split-batch-training-gpu-nn-cpu-ridge-lgbm.md); [infra/batch/README.md](../infra/batch/README.md) is the operator runbook.
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
       │     (g6 Spot, g5 fallback)    stdout/stderr streamed
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

### Split GPU/CPU Mode

`src.batch.launch --split` submits three jobs per position:

1. `nn` branch on `ff-training-queue` / `ff-training-job`: trains base NN and
   attention NN only, then uploads a staged tarball under
   `split-runs/{run_id}/{POS}/nn/`.
2. `cpu` branch on `ff-cpu-training-queue` / `ff-training-cpu-job`: trains Ridge
   and LightGBM only, with `FF_CORE_POOL_ADDR` active so Ridge CV and LightGBM
   lease all 4 c8a cores dynamically. Staged output lands under
   `split-runs/{run_id}/{POS}/cpu/`.
3. `merge` branch on the CPU queue with Batch `dependsOn` both branch jobs:
   downloads staged artifacts, checks branch/position/SHA/size/checksum, merges
   the model files into one normal artifact directory, runs the existing smoke
   test, and only then promotes `models/{POS}/manifest.json`.

The CPU compute environment is `ff-cpu-spot` with `c8a.xlarge` primary,
`m8a.xlarge` fallback, 4 vCPU / 7500 MiB per job, and `maxvCpus=64` so the
fleet can run up to 16 CPU branch/merge jobs concurrently. The old monolithic
path remains available by leaving `--split` off or setting
`BATCH_SPLIT_ACTIVE=false`.

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
  build_and_push.sh     ← ECR login + buildx + push
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

Simplified shape (the real file uses uv + a build-time import smoke + the
`PULL_THROUGH_PREFIX` arg — see [src/batch/Dockerfile.train](../src/batch/Dockerfile.train)
and §"Cold-start optimization" 2d):

```dockerfile
FROM nvidia/cuda:12.6.3-base-ubuntu24.04          # slim CUDA base (~86 MB), no torch
RUN apt-get install -y python3 python3-dev g++ libgomp1 ca-certificates
WORKDIR /opt/ml/code
COPY src/batch/requirements.txt .
RUN uv pip install --system -r requirements.txt   # incl. torch==2.12.0+cu126 (cu126 index)
COPY src/ src/
ENTRYPOINT ["python", "-m", "src.batch.train"]
```

- Slim `nvidia/cuda:*-base` base; **torch installed via pip** (`2.12.0+cu126`, the
  wheel brings its own CUDA libs) rather than baked into a conda base (2d)
- Project source baked into `/opt/ml/code/`
- ENTRYPOINT runs train.py; Batch passes `["--position", "RB", "--seed", "42"]` as command override

### Container Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TRAINING_DATA_DIR` | `/opt/ml/input/data/training/` | Where parquet files are downloaded to |
| `MODEL_OUTPUT_DIR` | `/opt/ml/model/` | Where model artifacts are written |
| `LOG_EVERY` | `1` (batch) / `10` (default) | Epoch logging frequency; read by `shared.pipeline._resolve_nn_log_every` |
| `S3_BUCKET` | (required) | S3 bucket for data and artifacts |
| `S3_DATA_PREFIX` | `data` | S3 key prefix for training data |
| `REQUIRE_GPU` | `1` | Fail fast if CUDA unavailable. **Auto-skipped for K/DST** — relaxes the *assertion* only; K/DST still train a GPU attention NN when one is present. Split CPU jobs set `REQUIRE_GPU=0` and disable NN work. |

### Launcher Environment Variables (`src/batch/launch.py`)

| Variable | Default | Description |
|---|---|---|
| `FF_S3_BUCKET` | `ff-predictor-training` | Override bucket name (for staging accounts) |
| `FF_JOB_QUEUE` | `ff-training-queue` | Override Batch job queue |
| `FF_JOB_DEFINITION` | `ff-training-job` | Override Batch job definition (GPU) |
| `FF_JOB_QUEUE_CPU` | (unset) | CPU queue for split `cpu` and `merge` branch jobs (`ff-cpu-training-queue`) |
| `FF_JOB_DEFINITION_CPU` | (unset) | CPU job definition for split `cpu` and `merge` branch jobs (`ff-training-cpu-job`) |
| `FF_JOB_DEFINITION_REVISION` | (unset) | GPU job-definition revision pin written by `batch-image.yml` |
| `FF_JOB_DEFINITION_CPU_REVISION` | (unset) | CPU job-definition revision pin written by `batch-image.yml` under `job-def-revisions/cpu/{sha}.txt` |
| `FF_SPLIT_RUN_ID` | (unset) | Optional split artifact namespace; `launch.py --split` auto-generates one when absent |
| `FF_WAIT_TIMEOUT` | `10800` (3h) | Wall-clock cap for `wait_for_jobs`. `train-batch.yml` overrides to `18000` (5h) — Spot queue time dominates the ~1 min/position training, and the 3h default expired minutes before Spot-starved jobs SUCCEEDED on 2026-06-08/06-10, silently skipping the benchmark append (see ADR-0013 changelog) |
| `FF_BATCH_JOB_IDS_FILE` | (unset) | Path where `launch.py` records submitted job ids + expected positions as JSON. Set by `train-batch.yml` (`batch_job_ids.json`) so its recovery step can re-check the exact jobs after a failed wait; unset writes nothing |

### Container Dependencies (`src/batch/requirements.txt`)

Derived from root `requirements.txt`:
- **Excluded**: `flask`, `gunicorn`, `pytest`
- **Added (CUDA)**: `torch==2.12.0+cu126` via `--extra-index-url https://download.pytorch.org/whl/cu126` — the slim `nvidia/cuda:*-base` base ships no torch (was previously provided by the conda base; see §"Cold-start optimization" 2d)
- **Added**: `boto3>=1.34` (S3 operations), `nflreadpy==0.1.5` + `polars==1.41.1` (nflverse data loading via the `src/data/nfl_source.py` shim; imported transitively by K/DST data modules even though no fetch happens at training time)

## Position Pipeline Invocation

Pipeline dispatch lives in `src/shared/registry.py` (the single source of truth, consumed by `train.py`, `app.py`, and `benchmark.py`). Positions come from the `Position` StrEnum in `src/shared/position.py` (`ALL_POSITIONS = Position.values()`); each position's runner module and `accepts_dataframes` flag live on its `POSITION_CONFIG`. `train.py` calls `get_runner(pos)` and reads `INFERENCE_REGISTRY[pos]` — there is no `POSITIONS` dict in `train.py` (the legacy `_POSITION_META` table is gone):

```python
# src/shared/registry.py
ALL_POSITIONS = Position.values()          # ["QB", "RB", "WR", "TE", "K", "DST"]
run_fn = get_runner(pos)                   # lazy-imports src.{pos}.run_pipeline.run
accepts_df = _position_config(pos).accepts_dataframes  # True for QB/RB/WR/TE, False for K/DST
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
    "maxvCpus": 64,
    "instanceTypes": ["g6.xlarge"],          # primary CE
    "subnets": ["SUBNET_A", "SUBNET_B"],
    "securityGroupIds": ["DEFAULT_SG"],
    "instanceRole": "ecsInstanceRole",
    "spotIamFleetRole": "arn:aws:iam::ACCOUNT_ID:role/aws-ec2-spot-fleet-tagging-role"
  }'
```

- `type=SPOT` — 70% cheaper than on-demand
- `minvCpus=0` — scales to zero when idle (no cost)
- `maxvCpus=64` — up to 16 concurrent 4-vCPU GPU hosts per CE (`g6.xlarge` primary, `g5.xlarge` fallback); matches the Spot G+VT quota raised 24 → 64 on 2026-06-11, so two six-position fan-outs (or a fan-out plus a tune fleet) no longer starve each other at the CE ceiling
- `allocationStrategy=SPOT_PRICE_CAPACITY_OPTIMIZED` — AWS-recommended strategy that weighs *both* capacity (lowest current reclaim risk) and Spot price. Strict superset of `SPOT_CAPACITY_OPTIMIZED`: same reclaim-avoidance behaviour plus price awareness

### Job Queue

```bash
aws batch create-job-queue \
  --job-queue-name ff-training-queue \
  --state ENABLED \
  --priority 1 \
  --compute-environment-order \
    order=1,computeEnvironment=ff-gpu-spot \
    order=2,computeEnvironment=ff-gpu-spot-g5
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

- `vcpus=4, memory=15000` — matches both g6.xlarge and g5.xlarge (4 vCPU, 16 GB RAM; identical CPU/RAM sizing to g4dn pre-2026-05-31)
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
3. Create GPU Compute Environments (`ff-gpu-spot` primary, `ff-gpu-spot-g5` fallback)
4. Create CPU Compute Environment (`ff-cpu-spot`)
5. Create Job Queues (`ff-training-queue`, `ff-cpu-training-queue`)
6. Register Job Definitions (`ff-training-job`, `ff-training-cpu-job`)
7. Build and push training image to ECR
8. Run: `python -m src.batch.launch`

## Rollback

The existing Flask Dockerfile and `src/serving/app.py` inference code are completely unaffected.
CUDA auto-detection in `src/shared/pipeline.py` falls back to CPU. Local pipeline scripts
(`python -m src.qb.run_pipeline`) work identically without any AWS dependencies.

## CPU Split Queue

The CPU queue is for split `cpu` and `merge` branches, not for routing K/DST
monolithic jobs away from GPU. K and DST train attention NNs, so a full K/DST
job still belongs on the GPU queue. `train-batch.yml` exports
`FF_JOB_DEFINITION_CPU` and `FF_JOB_QUEUE_CPU` only when
`BATCH_SPLIT_ACTIVE=true` — **the production default since 2026-06-11** (first
full split run validated Δ=0.0000 vs the monolithic baseline across all
models/positions; see ADR-0019's changelog). Unset the variable to fall back
to the monolithic GPU-backed path.

The CPU job definition uses the same image as the GPU job definition but no GPU
resource requirement:

- `vcpus=4`, `memory=7500`
- `REQUIRE_GPU=0`, `FF_DEVICE=cpu`
- `FF_CPU_BRANCH_CORES=4`
- BLAS/OMP/LightGBM fallback caps set to 1, with `LOKY_MAX_CPU_COUNT=4`

## CI/CD

Three workflows cover the training image and the inference service:

- `.github/workflows/batch-image.yml` — builds `src/batch/Dockerfile.train`, pushes
  to ECR (`ff-training`), and registers new revisions of both `ff-training-job`
  and `ff-training-cpu-job` pinned to the new image SHA. Triggered by any change
  under `src/**` (excluding `**/tests/**` and `**/*.md`) or `requirements.txt`.
- `.github/workflows/deploy.yml` — builds the inference `Dockerfile`, pushes to
  ECR (`fantasy-predictor`), and updates the ECS service. Now gated on the
  full test suite.
- `.github/workflows/tests.yml` — runs pytest across **all** position test
  directories plus `src/batch/` and `src/shared/` on every push and PR.

## Cold-start optimization (image pull acceleration)

The largest chunk of per-job wall time on a cold Spot instance is pulling the
training image (decompress + extract to overlayfs), re-paid on every fresh Spot
host. With SOCI dead on Batch (§2a) and the ~120 s Spot-provisioning half a fixed
G-family floor, the live levers are **shrinking the image** (2d, below) and a
**warm pre-pulled custom AMI** (deferred Stage 2 — not yet built; would reuse
`infra/batch/setup.sh`'s `DISABLE → update-compute-environment → ENABLE` reconcile
to host an AMI with the base layers pre-seeded into containerd). The stacking
build-time optimizations:

### 2d. Slim CUDA base + pip torch (primary image-size lever, 2026-06-07)

`src/batch/Dockerfile.train` bases on `nvidia/cuda:12.6.3-base-ubuntu24.04`
(~86 MB compressed) and installs `torch==2.12.0+cu126` via the uv layer, instead
of the conda-based `pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime` (~3.7 GB
compressed) that bundled torch+CUDA+mamba+MKL in one re-pulled layer. The `*-base`
flavor ships **no** CUDA math libs, so the torch wheel brings its own — using a
`*-cudnn-runtime` base (~2 GB) instead would double-count cudnn/cublas and erase
the win. **torch version is held identical** (2.12.0 / cu126, the build the old
base shipped) so this is packaging, not an upgrade: deterministic Ridge MAE stays
byte-identical (data-identity tell) while NN/attn rows rebaseline within
single-seed noise (different torch *distribution*). System deps the conda base
provided are re-added explicitly (`python3`+`python3-dev`, `g++` for the Inductor
probe, `libgomp1` for LightGBM/sklearn OpenMP, `ca-certificates`). A build-time
`python -c "import torch, ..."` smoke fails the CI build on an incoherent install
rather than on the first GPU job. **`batch-image.yml` builds only on
push-to-`main`/dispatch (not PRs), so the new image is first built on merge** (or
a `workflow_dispatch` branch build) — there is no PR-time or local build gate.
Targets the image-pull half of cold-start; actual pull-window drop measured
post-merge via `aws ecs describe-tasks` `pullStartedAt`/`pullStoppedAt`.

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

**Verify the rule is active** (the per-job pull-time win is entirely contingent
on it — `batch-image.yml`'s build step auto-detects the rule and only routes the
base through ECR when it exists, else falls back to Docker Hub silently):

```bash
# 1. The rule exists in the training region (non-empty list = active):
aws ecr describe-pull-through-cache-rules \
  --ecr-repository-prefixes dockerhub --region us-east-1 \
  --query 'pullThroughCacheRules[].ecrRepositoryPrefix' --output text

# 2. The most recent image build actually used it — the build log line should read
#    "Pull-through cache 'dockerhub' detected; base image routed through ECR."
#    (the alternative "...pulls from dockerhub directly" means the rule is missing).
```

If step 1 is empty, run the `create-pull-through-cache-rule` command above once;
no code change is needed — the next `batch-image.yml` build picks it up automatically.

### 2a. SOCI (Seekable OCI) lazy loading

> **⚠️ REMOVED / DOES NOT WORK ON BATCH (2026-06-07).** AWS Batch runs on
> **ECS-managed EC2**, and the `amazon-ecs-agent` does not pull images through
> the soci snapshotter — SOCI on ECS is **Fargate-only**
> ([containers-roadmap#1832](https://github.com/aws/containers-roadmap/issues/1832),
> open since 2022), and Fargate has no GPU, so GPU training can never use it.
> Verified empirically: with a fully-correct soci config (transfer-service
> snapshotter selection + ECR credential helper + `soci_v1` pull mode — validated
> to lazy-load a `ctr` pull in 7 s / 9 KiB on a fresh AL2023 host), a prod K Batch
> job still pulled in **103 s** (overlayfs) because the agent ignored the config.
> The launch template, the CI index-publish step, and the userdata daemon were all
> removed; the CE uses the default ECS AMI and pays the ~120 s pull. This holds on
> g6/L4 too (the limitation is the ECS agent, not the GPU arch). Full root-cause:
> [todo/fixed-archive.md](../todo/fixed-archive.md). **Text below is retained for
> history only.**

SOCI v2 lazy-loading lets a container start executing before its full
image is pulled — essential files stream first, the rest loads in the
background. It requires both a SOCI index in ECR alongside the image
**and** the `soci-snapshotter-grpc` plugin active on the ECS host's
containerd.

> **⚠️ REMOVED 2026-06-07 — this section is historical.** SOCI never worked on
> ECS-managed EC2 (the ECS agent ignores the snapshotter; SOCI-on-ECS is
> Fargate-only). The `ff-batch-lt` launch template, `infra/batch/userdata.sh`,
> and the SOCI-index publish step were deleted; cold-start is the ~120 s full
> image pull. See the [ADR-0013 changelog](adr/0013-spot-fan-out-via-aws-batch.md).

**Option B configuration (2026-05-21, since removed)**: indexes were published
to the `ff-training` ECR repo by [batch-image.yml](../.github/workflows/batch-image.yml)'s
`Publish SOCI index` step (`continue-on-error: false`, version-pinned
to `0.13.0`). The `ff-gpu-spot` Compute Environment uses the
`ff-batch-lt` EC2 launch template, whose UserData
(`infra/batch/userdata.sh`) installs
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
returns to ~122 s. (Historical — the launch template and this rollback
procedure were removed 2026-06-07.)

**Version pin discipline**: `SOCI_VERSION` in
`infra/batch/userdata.sh` (host snapshotter)
and [batch-image.yml](../.github/workflows/batch-image.yml) (index
publisher) MUST stay aligned. SOCI manifest format evolves between
minor releases; publisher/consumer skew breaks lazy-load silently.

### Build & push

`src/batch/build_and_push.sh` wires the pull-through build + push together:

```bash
./src/batch/build_and_push.sh                        # defaults: us-east-1, ff-training:latest
IMAGE_TAG=$(git rev-parse --short HEAD) ./src/batch/build_and_push.sh
USE_PULL_THROUGH=0 ./src/batch/build_and_push.sh     # bypass pull-through (for debugging)
```

The script logs in to ECR, builds with the pull-through base, and pushes the
image.

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
