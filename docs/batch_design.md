# AWS Batch Training Design Doc

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

batch/launch.py ─────────────────> S3: s3://ff-training/data/
  uploads data/splits/*.parquet        train.parquet
  submits 6 Batch jobs                 val.parquet
       │                               test.parquet
       │
       ├─> Batch Job: ff-rb-xxx ───> CloudWatch Logs
       │     (g4dn.xlarge Spot)        stdout/stderr streamed
       │     train.py --position RB
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
batch/launch.py <──────────────────────────┘
  polls describe_jobs() for status   S3: s3://ff-training/models/
  downloads model artifacts             rb/model.tar.gz
  extracts to RB/outputs/models/        wr/model.tar.gz
             WR/outputs/models/         ...
             ...
```

### Data Staging

The container handles data staging (S3 → container) and artifact packaging
(container → S3) explicitly via boto3 — about 20 extra lines in `train.py`, but artifact
locations are predictable and the launcher (`launch.py`) stays simple.

## Directory Structure

```
batch/
  __init__.py
  train.py              ← container entry point
  launch.py             ← job submitter (boto3 Batch client)
  benchmark.py          ← benchmark suite
  Dockerfile.train      ← GPU training image
  requirements.txt      ← container-only deps (no torch — base image provides it)
  tests/
    __init__.py
    conftest.py
    test_launch.py
    test_train.py
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

COPY batch/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "batch/train.py"]
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
| `LOG_EVERY` | `1` | Epoch logging frequency |
| `S3_BUCKET` | (required) | S3 bucket for data and artifacts |
| `S3_DATA_PREFIX` | `data` | S3 key prefix for training data |

### Container Dependencies (`batch/requirements.txt`)

Derived from root `requirements.txt`:
- **Excluded**: `torch` (in base image), `flask`, `gunicorn`, `pytest`
- **Added**: `boto3>=1.34` (S3 operations), `nfl_data_py==0.3.3` (K/DST data loading)

## Position Pipeline Invocation

Pipeline registry in `batch/train.py`:

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

`batch/launch.py` submits jobs via `boto3.client('batch').submit_job()`:

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
docker build -f batch/Dockerfile.train -t ff-training:latest .

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
7. Run: `python batch/launch.py`

## Rollback

The existing Flask Dockerfile and `app.py` inference code are completely unaffected.
CUDA auto-detection in `shared/pipeline.py` falls back to CPU. Local pipeline scripts
(`python -m QB.run_qb_pipeline`) work identically without any AWS dependencies.
