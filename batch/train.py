"""AWS Batch training entry point.

Batch runs this as: python batch/train.py --position RB --seed 42

Environment variables set via job definition / container overrides:
  TRAINING_DATA_DIR  = /opt/ml/input/data/training/
  MODEL_OUTPUT_DIR   = /opt/ml/model/
  LOG_EVERY          = 1
  S3_BUCKET          = ff-predictor-training
  S3_DATA_PREFIX     = data
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tarfile
import tempfile

# Ensure project root is on path (baked into /opt/ml/code/ in the container)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import boto3
import numpy as np
import pandas as pd
import torch


def _assert_gpu():
    """Log GPU status and fail fast if REQUIRE_GPU=1 and CUDA is unavailable.

    This catches the silent-CPU-on-GPU-billed-instance failure mode where
    the Batch job definition forgets `resourceRequirements: [{type: GPU, ...}]`.
    """
    available = torch.cuda.is_available()
    print(f"[gpu] torch.cuda.is_available() = {available}")
    print(f"[gpu] torch.version.cuda        = {torch.version.cuda}")
    print(f"[gpu] torch.__version__         = {torch.__version__}")
    if available:
        print(f"[gpu] device count              = {torch.cuda.device_count()}")
        print(f"[gpu] device 0 name             = {torch.cuda.get_device_name(0)}")
    require_gpu = os.environ.get("REQUIRE_GPU", "1") == "1"
    if require_gpu and not available:
        raise RuntimeError(
            "REQUIRE_GPU=1 but torch.cuda.is_available() is False. "
            "Check the Batch job definition's resourceRequirements for GPU=1 "
            "and the compute environment's ECS GPU-optimized AMI."
        )


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Position registry: (module_path, runner_function_name, accepts_dataframes)
POSITIONS = {
    "QB": ("QB.run_qb_pipeline", "run_qb_pipeline", True),
    "RB": ("RB.run_rb_pipeline", "run_rb_pipeline", True),
    "WR": ("WR.run_wr_pipeline", "run_wr_pipeline", True),
    "TE": ("TE.run_te_pipeline", "run_te_pipeline", True),
    "K":  ("K.run_k_pipeline",   "run_k_pipeline",  False),
    "DST": ("DST.run_dst_pipeline", "run_dst_pipeline", False),
}


def download_data(s3_bucket, s3_prefix, local_dir):
    """Download training parquet files from S3 to the container."""
    s3 = boto3.client("s3")
    os.makedirs(local_dir, exist_ok=True)
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        s3_key = f"{s3_prefix}/{name}"
        local_path = os.path.join(local_dir, name)
        print(f"Downloading s3://{s3_bucket}/{s3_key} -> {local_path}")
        s3.download_file(s3_bucket, s3_key, local_path)
    print("Data download complete.")


def upload_artifacts(s3_bucket, position, model_dir):
    """Tar model artifacts and upload to S3."""
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            for item in os.listdir(model_dir):
                full_path = os.path.join(model_dir, item)
                tar.add(full_path, arcname=item)

        s3_key = f"models/{position}/model.tar.gz"
        print(f"Uploading artifacts to s3://{s3_bucket}/{s3_key}")
        s3.upload_file(tmp_path, s3_bucket, s3_key)
        print("Artifact upload complete.")
    finally:
        os.unlink(tmp_path)


def _extract_metrics(position, result):
    """Extract JSON-serializable benchmark metrics from pipeline result."""
    metrics = {"position": position}

    for model_key in ["ridge", "nn", "attn_nn", "lgbm"]:
        m_key = f"{model_key}_metrics"
        r_key = f"{model_key}_ranking"
        if m_key not in result:
            continue
        m = result[m_key]
        metrics[m_key] = {
            "total": {k: round(v, 4) for k, v in m["total"].items()},
        }
        for t in m:
            if t != "total":
                metrics[m_key][t] = {k: round(v, 4) for k, v in m[t].items()}
        if r_key in result:
            ranking = result[r_key]
            metrics[r_key] = {
                "season_avg_hit_rate": round(ranking["season_avg_hit_rate"], 4),
            }
            if "season_avg_spearman" in ranking:
                metrics[r_key]["season_avg_spearman"] = round(ranking["season_avg_spearman"], 4)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", required=True, choices=list(POSITIONS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pos = args.position

    # Print build fingerprint so stale container images are immediately obvious.
    _fingerprint_file = os.path.join(os.path.dirname(__file__), "train.py")
    with open(_fingerprint_file, "rb") as _f:
        _hash = hashlib.sha256(_f.read()).hexdigest()[:12]
    print(f"[batch/train.py] build fingerprint: {_hash}")

    _assert_gpu()
    _seed_everything(args.seed)

    s3_bucket = os.environ.get("S3_BUCKET", "ff-predictor-training")
    s3_prefix = os.environ.get("S3_DATA_PREFIX", "data")
    data_dir = os.environ.get("TRAINING_DATA_DIR", "/opt/ml/input/data/training")
    model_dir = os.environ.get("MODEL_OUTPUT_DIR", "/opt/ml/model")
    log_every = int(os.environ.get("LOG_EVERY", "1"))

    os.makedirs(model_dir, exist_ok=True)

    # Patch run_pipeline to inject nn_log_every BEFORE importing runners.
    import shared.pipeline as pipeline_mod
    _orig_run_pipeline = pipeline_mod.run_pipeline

    def _patched_run_pipeline(position, cfg, *a, **kw):
        cfg["nn_log_every"] = log_every
        return _orig_run_pipeline(position, cfg, *a, **kw)

    pipeline_mod.run_pipeline = _patched_run_pipeline

    mod_path, func_name, accepts_df = POSITIONS[pos]
    mod = __import__(mod_path, fromlist=[func_name])
    run_fn = getattr(mod, func_name)

    if accepts_df:
        # Download data from S3 into the container
        download_data(s3_bucket, s3_prefix, data_dir)
        train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
        val_df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
        test_df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
        print(f"Loaded data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        result = run_fn(train_df, val_df, test_df, seed=args.seed)
    else:
        # K/DST: self-contained data loading
        result = run_fn(seed=args.seed)

    # Copy model artifacts to output dir FIRST so a later metrics write cannot
    # be clobbered by a same-named file under src_model_dir.
    src_model_dir = os.path.join(pos, "outputs", "models")
    if os.path.isdir(src_model_dir):
        print(f"Copying model artifacts from {src_model_dir} to {model_dir}")
        shutil.copytree(src_model_dir, model_dir, dirs_exist_ok=True)
    else:
        print(f"WARNING: No model directory found at {src_model_dir}")

    # Save benchmark metrics as JSON (after artifacts so it can't be overwritten)
    if result is not None:
        metrics = _extract_metrics(pos, result)
        metrics_path = os.path.join(model_dir, "benchmark_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved benchmark metrics to {metrics_path}")

    # Upload artifacts to S3
    upload_artifacts(s3_bucket, pos, model_dir)


if __name__ == "__main__":
    main()
