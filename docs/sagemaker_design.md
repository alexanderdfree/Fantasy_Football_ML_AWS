# SageMaker Training Design Doc

## Problem

All model training runs on local CPU (`torch.device("cpu")`) in `shared/pipeline.py`.
Full pipeline across 6 positions takes ~30 min sequentially. Goals:

1. Run training on GPU to speed up neural network epochs
2. Run all 6 positions in parallel (separate SageMaker jobs)
3. See training progress in real time without SSH-ing into anything

## Architecture

```
LOCAL (your laptop)                          AWS
─────────────────                   ─────────────────────────────
                                    
sagemaker/launch.py ──────────────> S3: s3://ff-training/data/
  uploads data/splits/*.parquet        train.parquet
  uploads source code tarball          val.parquet
  submits 6 Training Jobs              test.parquet
       │                               source.tar.gz
       │
       ├─> SageMaker Job: train-rb ──> CloudWatch Logs + Metrics
       ├─> SageMaker Job: train-wr ──> CloudWatch Logs + Metrics
       ├─> SageMaker Job: train-qb ──> CloudWatch Logs + Metrics
       ├─> SageMaker Job: train-te ──> CloudWatch Logs + Metrics
       ├─> SageMaker Job: train-k  ──> CloudWatch Logs + Metrics
       └─> SageMaker Job: train-dst ─> CloudWatch Logs + Metrics
                                           │
sagemaker/launch.py <──────────────────────┘
  waits for completion              S3: s3://ff-training/models/
  downloads model artifacts            rb/model.tar.gz
  extracts to RB/outputs/models/       wr/model.tar.gz
             WR/outputs/models/        ...
             ...
```

## Monitoring Training Progress

The current trainers (`MultiHeadTrainer`, `MultiHeadHistoryTrainer` in
`shared/training.py:280-291`) already print structured logs every 10 epochs:

```
Epoch  10 | Train: 0.4523 | Val: 0.5012 | MAE total: 4.231 | rushing_floor: 1.823 | receiving_floor: 1.456 | td_points: 0.952
```

SageMaker captures all stdout/stderr and streams it to **CloudWatch Logs** automatically.
We exploit this in three ways:

### 1. Live tail from the CLI (easiest)

SageMaker's Python SDK streams logs to your terminal while you wait:

```python
# In launch.py — this prints logs as they happen
estimator.fit(inputs, wait=True, logs="All")
```

You'll see the same epoch-by-epoch output you see locally, streamed in real time.
Since we launch 6 jobs, we run each in a thread and prefix lines with position:

```
[RB] Epoch  10 | Train: 0.4523 | Val: 0.5012 | MAE total: 4.231
[WR] Epoch  10 | Train: 0.3891 | Val: 0.4502 | MAE total: 3.892
[QB] Epoch  20 | Train: 0.5102 | Val: 0.5634 | MAE total: 5.103
```

### 2. CloudWatch Metrics (live charts in AWS Console)

SageMaker can parse custom metrics from stdout via regex. We define metric
definitions when creating the Estimator:

```python
metric_definitions = [
    {"Name": "train:loss",     "Regex": r"Train: ([0-9.]+)"},
    {"Name": "val:loss",       "Regex": r"Val: ([0-9.]+)"},
    {"Name": "val:mae_total",  "Regex": r"MAE total: ([0-9.]+)"},
]
```

These appear as **live-updating line charts** in the SageMaker Console under
Training Jobs > [job name] > Monitor. You can watch loss curves update every
10 epochs without any code changes to the trainer.

### 3. Optional: epoch-level logging for denser charts

The current trainers only print every 10 epochs. For denser CloudWatch charts,
we add an optional `log_every` parameter (default 10, set to 1 for SageMaker).
This is a small change to `MultiHeadTrainer.train()` and
`MultiHeadHistoryTrainer.train()` — change the `if (epoch + 1) % 10 == 0`
condition to use the configurable interval.

This is passed via the config dict:

```python
cfg["nn_log_every"] = int(os.environ.get("SM_LOG_EVERY", 10))
```

## AWS Resources Required

### S3 Bucket

```
s3://ff-predictor-training/    (or any name you choose)
  data/
    train.parquet
    val.parquet
    test.parquet
  models/
    rb/model.tar.gz
    wr/model.tar.gz
    ...
```

### IAM Role

A SageMaker execution role with:
- `AmazonSageMakerFullAccess` (managed policy)
- S3 read/write on your training bucket

### Instance Type

`ml.g4dn.xlarge` — 1x NVIDIA T4 GPU (16 GB), 4 vCPUs, 16 GB RAM.
At ~$0.53/hr, a full 6-position parallel run costs ~$0.05-0.10.

## Code Changes

### 1. `shared/pipeline.py` — CUDA auto-detection (3 lines)

Lines 253, 326, 798: replace `torch.device("cpu")` with:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Safe — falls back to CPU locally. `app.py:295` already does this for inference.

### 2. `shared/training.py` — configurable log interval

In both `MultiHeadTrainer.train()` and `MultiHeadHistoryTrainer.train()`,
accept `log_every` in the constructor and replace the hardcoded `% 10`:

```python
class MultiHeadTrainer:
    def __init__(self, ..., log_every=10):
        ...
        self.log_every = log_every

    def train(self, ...):
        ...
        if (epoch + 1) % self.log_every == 0:
            ...
```

### 3. `sagemaker/train.py` — SageMaker entry point (new file)

```python
"""SageMaker training entry point.

SageMaker invokes this as: python train.py --position RB --seed 42

Environment variables set by SageMaker:
  SM_CHANNEL_TRAINING  = /opt/ml/input/data/training/
  SM_MODEL_DIR         = /opt/ml/model/
  SM_NUM_GPUS          = 1
"""
import argparse, os, sys

# Add project root to path (SageMaker extracts source.tar.gz to /opt/ml/code/)
sys.path.insert(0, "/opt/ml/code")

import pandas as pd
from shared.pipeline import run_pipeline

# Position registry — maps name to (config, pipeline runner)
POSITIONS = {
    "RB": ("RB.rb_config", "RB.run_rb_pipeline"),
    "WR": ("WR.wr_config", "WR.run_wr_pipeline"),
    "QB": ("QB.qb_config", "QB.run_qb_pipeline"),
    "TE": ("TE.te_config", "TE.run_te_pipeline"),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = os.environ.get("SM_CHANNEL_TRAINING", "data/splits")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    train_df = pd.read_parquet(f"{data_dir}/train.parquet")
    val_df   = pd.read_parquet(f"{data_dir}/val.parquet")
    test_df  = pd.read_parquet(f"{data_dir}/test.parquet")

    # Import position config + runner
    cfg_mod = __import__(POSITIONS[args.position][0], fromlist=["CFG"])
    cfg = cfg_mod.CFG
    cfg["nn_log_every"] = 1           # dense logging for CloudWatch
    cfg["output_dir"] = model_dir     # SageMaker uploads this to S3

    runner = __import__(POSITIONS[args.position][1], fromlist=["run"])
    runner.run(cfg, train_df, val_df, test_df)

if __name__ == "__main__":
    main()
```

### 4. `sagemaker/launch.py` — local job launcher (new file)

```python
"""Launch parallel SageMaker training jobs for all positions.

Usage:
    python sagemaker/launch.py                     # all positions
    python sagemaker/launch.py --positions RB WR   # subset
    python sagemaker/launch.py --wait false         # fire and forget
"""
import argparse, sagemaker, boto3
from sagemaker.pytorch import PyTorch
from concurrent.futures import ThreadPoolExecutor, as_completed

S3_BUCKET     = "ff-predictor-training"   # configure once
ROLE          = "arn:aws:iam::role/SageMakerTrainingRole"
INSTANCE_TYPE = "ml.g4dn.xlarge"
ALL_POSITIONS = ["RB", "WR", "QB", "TE"]

METRIC_DEFINITIONS = [
    {"Name": "train:loss",    "Regex": r"Train: ([0-9.]+)"},
    {"Name": "val:loss",      "Regex": r"Val: ([0-9.]+)"},
    {"Name": "val:mae_total", "Regex": r"MAE total: ([0-9.]+)"},
]

def launch_one(position, wait=True):
    estimator = PyTorch(
        entry_point="sagemaker/train.py",
        source_dir=".",                    # tars up entire project
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        framework_version="2.1",
        py_version="py310",
        hyperparameters={"position": position, "seed": 42},
        metric_definitions=METRIC_DEFINITIONS,
        output_path=f"s3://{S3_BUCKET}/models/{position}",
        base_job_name=f"ff-{position.lower()}",
        max_run=1800,                      # 30 min timeout
    )
    estimator.fit(
        {"training": f"s3://{S3_BUCKET}/data/"},
        wait=wait,
        logs="All" if wait else "None",
    )
    return position, estimator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", nargs="+", default=ALL_POSITIONS)
    parser.add_argument("--wait", default="true")
    args = parser.parse_args()
    wait = args.wait.lower() == "true"

    # Upload data splits to S3
    session = sagemaker.Session()
    session.upload_data("data/splits", bucket=S3_BUCKET, key_prefix="data")

    # Launch all positions in parallel threads
    with ThreadPoolExecutor(max_workers=len(args.positions)) as pool:
        futures = {
            pool.submit(launch_one, pos, wait): pos
            for pos in args.positions
        }
        for future in as_completed(futures):
            pos, estimator = future.result()
            print(f"[{pos}] Complete. Artifacts: {estimator.model_data}")

    # Download model artifacts back to local dirs
    if wait:
        s3 = boto3.client("s3")
        for pos in args.positions:
            # SageMaker saves to output_path/job-name/output/model.tar.gz
            # download and extract to {pos}/outputs/models/
            ...  # extraction logic

if __name__ == "__main__":
    main()
```

### 5. `sagemaker/requirements.txt` — container dependencies (new file)

```
torch>=2.1.0
numpy>=2.0
pandas>=2.0
scikit-learn>=1.4
mord>=0.7
joblib>=1.3
scipy>=1.12
matplotlib>=3.8
```

## Monitoring Cheat Sheet

| What you want to see | How |
|---|---|
| Live epoch logs in terminal | `python sagemaker/launch.py` (streams automatically) |
| Loss curves in browser | AWS Console > SageMaker > Training Jobs > select job > Monitor tab |
| Historical logs | AWS Console > CloudWatch > Log Groups > `/aws/sagemaker/TrainingJobs` |
| Job status (running/failed/done) | `aws sagemaker list-training-jobs --name-contains ff-rb` |
| GPU utilization | SageMaker Console > Training Jobs > Monitor > System metrics |
| All 6 jobs at a glance | SageMaker Console > Training Jobs (filter by `ff-`) |

## Cost

| Resource | Cost |
|---|---|
| `ml.g4dn.xlarge` (T4 GPU) | ~$0.53/hr |
| 6 parallel jobs x ~2 min each | ~$0.05-0.10 per full run |
| S3 storage (48 MB data + models) | < $0.01/month |
| CloudWatch logs | Free tier covers this volume |

## Sequence of Steps to Set Up

1. Create S3 bucket (`ff-predictor-training`)
2. Create IAM role with `AmazonSageMakerFullAccess` + S3 access
3. Install SageMaker SDK locally: `pip install sagemaker boto3`
4. Configure AWS CLI: `aws configure` (set access key, region)
5. Update `ROLE` and `S3_BUCKET` in `sagemaker/launch.py`
6. Apply the 3 `device` line changes in `shared/pipeline.py`
7. Add `log_every` to trainers in `shared/training.py`
8. Run: `python sagemaker/launch.py`

## Rollback

All changes are backwards-compatible. The CUDA auto-detection falls back to CPU,
`log_every` defaults to 10, and local pipeline scripts work identically without
any AWS dependencies installed.
