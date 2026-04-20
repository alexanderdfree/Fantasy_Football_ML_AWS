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
import shutil
import sys
import tarfile
import tempfile

# Ensure project root is on path (baked into /opt/ml/code/ in the container)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import boto3
import pandas as pd
import torch

from shared.registry import (
    ALL_POSITIONS,
    accepts_dataframes,
    get_runner,
    is_cpu_only,
)
from shared.utils import seed_everything


def _assert_gpu(position: str):
    """Log GPU status and fail fast if REQUIRE_GPU=1 and CUDA is unavailable.

    This catches the silent-CPU-on-GPU-billed-instance failure mode where
    the Batch job definition forgets `resourceRequirements: [{type: GPU, ...}]`.

    For CPU-only positions (K, DST) we don't enforce REQUIRE_GPU even if the
    env var is set — those pipelines never touch CUDA.
    """
    available = torch.cuda.is_available()
    print(f"[gpu] torch.cuda.is_available() = {available}")
    print(f"[gpu] torch.version.cuda        = {torch.version.cuda}")
    print(f"[gpu] torch.__version__         = {torch.__version__}")
    if available:
        print(f"[gpu] device count              = {torch.cuda.device_count()}")
        print(f"[gpu] device 0 name             = {torch.cuda.get_device_name(0)}")
    if is_cpu_only(position):
        print(f"[gpu] {position} is CPU-only; skipping REQUIRE_GPU assertion")
        return
    require_gpu = os.environ.get("REQUIRE_GPU", "1") == "1"
    if require_gpu and not available:
        raise RuntimeError(
            "REQUIRE_GPU=1 but torch.cuda.is_available() is False. "
            "Check the Batch job definition's resourceRequirements for GPU=1 "
            "and the compute environment's ECS GPU-optimized AMI."
        )


def download_data(s3_bucket, s3_prefix, local_dir):
    """Download training parquet files from S3 to the container."""
    from concurrent.futures import ThreadPoolExecutor

    s3 = boto3.client("s3")
    os.makedirs(local_dir, exist_ok=True)
    names = ("train.parquet", "val.parquet", "test.parquet")

    def _download_one(name):
        s3_key = f"{s3_prefix}/{name}"
        local_path = os.path.join(local_dir, name)
        print(f"Downloading s3://{s3_bucket}/{s3_key} -> {local_path}")
        s3.download_file(s3_bucket, s3_key, local_path)

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        for _ in pool.map(_download_one, names):
            pass
    print("Data download complete.")


def sync_raw_data(s3_bucket):
    """Sync s3://{bucket}/data/raw/*.parquet into the container's data/raw/.

    Needed by shared/weather_features._load_schedules() (all positions during
    feature engineering) and by K/DST's self-contained loaders (k_data,
    dst_data). CACHE_DIR="data/raw" in src/config.py resolves relative to
    the container WORKDIR=/opt/ml/code. .dockerignore excludes data/ so these
    parquets aren't baked into the image.
    """
    s3 = boto3.client("s3")
    os.makedirs("data/raw", exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix="data/raw/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            local_path = key
            print(f"Downloading s3://{s3_bucket}/{key} -> {local_path}")
            s3.download_file(s3_bucket, key, local_path)


def upload_artifacts(s3_bucket, position, model_dir):
    """Tar model artifacts and upload to S3.

    Fails fast if model_dir is empty or missing benchmark_metrics.json — an
    empty tarball shipped to S3 would silently mask a broken pipeline run
    until someone tried to download the model weeks later.
    """
    if not os.path.isdir(model_dir):
        raise RuntimeError(
            f"Model directory {model_dir} does not exist — pipeline did not produce artifacts."
        )
    items = os.listdir(model_dir)
    if not items:
        raise RuntimeError(
            f"Model directory {model_dir} is empty — refusing to upload an "
            "empty tarball. Pipeline likely returned None or failed silently."
        )
    if "benchmark_metrics.json" not in items:
        raise RuntimeError(
            f"benchmark_metrics.json not found in {model_dir}. Contents: {sorted(items)}"
        )

    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            for item in items:
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
            "total": {
                k: (round(v, 4) if isinstance(v, (int, float)) else v)
                for k, v in m["total"].items()
            },
        }
        for t in m:
            if t != "total":
                metrics[m_key][t] = {
                    k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in m[t].items()
                }
        if r_key in result:
            ranking = result[r_key]
            metrics[r_key] = {
                "season_avg_hit_rate": round(ranking["season_avg_hit_rate"], 4),
            }
            if "season_avg_spearman" in ranking:
                metrics[r_key]["season_avg_spearman"] = round(ranking["season_avg_spearman"], 4)

    return metrics


def _dry_run_artifacts(position: str, model_dir: str, seed: int) -> None:
    """Write minimal stub artifacts for --dry-run mode.

    Exercises the post-training side of main() (artifact layout, metric
    serialization, non-None result guard) without invoking the heavy per-
    position pipeline. This lets the CLI be smoke-tested end-to-end in
    under a second with no S3 / data / GPU dependencies.
    """
    os.makedirs(model_dir, exist_ok=True)
    # Stub model file so model_dir is non-empty (upload_artifacts invariant).
    stub_path = os.path.join(model_dir, f"{position.lower()}_model.stub")
    with open(stub_path, "w") as f:
        f.write(f"dry-run stub for {position} (seed={seed})\n")
    metrics = {
        "position": position,
        "dry_run": True,
        "seed": seed,
        "ridge_metrics": {"total": {"mae": 0.0, "r2": 0.0}},
    }
    with open(os.path.join(model_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[dry-run] Wrote stub artifacts to {model_dir}")


def _replace_model_dir_contents(src: str, dst: str) -> None:
    """Replace dst's contents with src's.

    On EC2, dst is /opt/ml/model — a bind-mount from /opt/ff/scratch/model
    that persists across ff-train invocations. We cannot rmtree the mount
    point (rmdir on a mount fails; rmtree with ignore_errors leaves an
    empty dir that then trips copytree's "dst must not exist" check). So
    clear the mount's contents in place, then copytree with
    dirs_exist_ok=True. Without the clear step, sequential ff-train calls
    would accumulate every prior position's artifacts into dst — including
    PCAs fit for the wrong feature count that then crash inference.
    """
    for name in os.listdir(dst):
        child = os.path.join(dst, name)
        if os.path.isdir(child) and not os.path.islink(child):
            shutil.rmtree(child)
        else:
            os.remove(child)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", required=True, choices=ALL_POSITIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip S3 download/upload and the real pipeline. Writes stub "
        "artifacts so main() can be smoke-tested end-to-end without "
        "AWS credentials or training data.",
    )
    args = parser.parse_args()

    pos = args.position

    # Print build fingerprint so stale container images are immediately obvious.
    _fingerprint_file = os.path.join(os.path.dirname(__file__), "train.py")
    with open(_fingerprint_file, "rb") as _f:
        _hash = hashlib.sha256(_f.read()).hexdigest()[:12]
    print(f"[batch/train.py] build fingerprint: {_hash}")

    # Skip the GPU assertion in dry-run — local/CI smoke tests rarely have CUDA.
    if args.dry_run:
        print(f"[dry-run] skipping _assert_gpu for {pos}")
    else:
        _assert_gpu(pos)
    seed_everything(args.seed)

    s3_bucket = os.environ.get("S3_BUCKET", "ff-predictor-training")
    s3_prefix = os.environ.get("S3_DATA_PREFIX", "data")
    data_dir = os.environ.get("TRAINING_DATA_DIR", "/opt/ml/input/data/training")
    model_dir = os.environ.get("MODEL_OUTPUT_DIR", "/opt/ml/model")
    # LOG_EVERY is consumed directly by shared.pipeline._resolve_nn_log_every()
    # so we don't need to inject it into cfg from here. Historically we
    # monkey-patched run_pipeline, but that only worked if callers used
    # `import shared.pipeline as pipeline_mod; pipeline_mod.run_pipeline(...)`.
    # All position runners use `from shared.pipeline import run_pipeline`, so
    # the patch was dead code. Env-var resolution sidesteps the issue.

    os.makedirs(model_dir, exist_ok=True)

    if args.dry_run:
        # Stub out S3 and the pipeline — we still exercise arg parsing,
        # seed setup, model-dir setup, metrics serialization, and the
        # skip-S3 code path.
        _dry_run_artifacts(pos, model_dir, args.seed)
        print(f"[dry-run] Completed for {pos}; skipping S3 upload.")
        return

    run_fn = get_runner(pos)

    # data/raw/*.parquet is needed for weather features (all positions) and
    # for K/DST's self-contained data loaders. Sync before branching.
    sync_raw_data(s3_bucket)

    if accepts_dataframes(pos):
        # Download train/val/test splits from S3 into the container
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
        _replace_model_dir_contents(src_model_dir, model_dir)
    else:
        print(f"WARNING: No model directory found at {src_model_dir}")

    # Save benchmark metrics as JSON (after artifacts so it can't be overwritten).
    # upload_artifacts() requires benchmark_metrics.json, so this must come
    # before the upload.
    if result is None:
        raise RuntimeError(
            f"Pipeline for {pos} returned None — cannot extract metrics. "
            "Refusing to upload incomplete artifacts."
        )
    metrics = _extract_metrics(pos, result)
    metrics_path = os.path.join(model_dir, "benchmark_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved benchmark metrics to {metrics_path}")

    # Upload artifacts to S3 (raises if model_dir is empty or metrics missing)
    upload_artifacts(s3_bucket, pos, model_dir)


if __name__ == "__main__":
    main()
