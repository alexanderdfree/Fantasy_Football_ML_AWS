"""Launch parallel SageMaker training jobs for all positions.

Usage:
    python sagemaker/launch.py                     # all positions
    python sagemaker/launch.py --positions RB WR   # subset
    python sagemaker/launch.py --wait false         # fire and forget
"""
import argparse
import os
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from concurrent.futures import ThreadPoolExecutor, as_completed

S3_BUCKET = "ff-predictor-training"  # configure once
ROLE = "arn:aws:iam::123456789012:role/SageMakerTrainingRole"  # replace 123456789012 with your account ID
INSTANCE_TYPE = "ml.g4dn.xlarge"
ALL_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]

METRIC_DEFINITIONS = [
    {"Name": "train:loss", "Regex": r"Train: ([0-9.]+)"},
    {"Name": "val:loss", "Regex": r"Val: ([0-9.]+)"},
    {"Name": "val:mae_total", "Regex": r"MAE total: ([0-9.]+)"},
]


def launch_one(position, wait=True):
    """Launch a single SageMaker training job for one position."""
    estimator = PyTorch(
        entry_point="sagemaker/train.py",
        source_dir=".",
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        framework_version="2.1",
        py_version="py310",
        hyperparameters={"position": position, "seed": 42},
        metric_definitions=METRIC_DEFINITIONS,
        output_path=f"s3://{S3_BUCKET}/models/{position}",
        base_job_name=f"ff-{position.lower()}",
        max_run=1800,  # 30 min timeout
    )
    estimator.fit(
        {"training": f"s3://{S3_BUCKET}/data/"},
        wait=wait,
        logs="All" if wait else "None",
    )
    return position, estimator


def download_artifacts(positions):
    """Download model artifacts from S3 back to local position dirs."""
    s3 = boto3.client("s3")
    sm = boto3.client("sagemaker")

    for pos in positions:
        # Find the latest completed job for this position
        response = sm.list_training_jobs(
            NameContains=f"ff-{pos.lower()}",
            StatusEquals="Completed",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        if not response["TrainingJobSummaries"]:
            print(f"[{pos}] No completed job found, skipping download")
            continue

        job_name = response["TrainingJobSummaries"][0]["TrainingJobName"]
        job_desc = sm.describe_training_job(TrainingJobName=job_name)
        model_s3_uri = job_desc["ModelArtifacts"]["S3ModelArtifacts"]

        # Parse S3 URI: s3://bucket/key
        s3_parts = model_s3_uri.replace("s3://", "").split("/", 1)
        bucket, key = s3_parts[0], s3_parts[1]

        local_model_dir = os.path.join(pos, "outputs", "models")
        os.makedirs(local_model_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
            print(f"[{pos}] Downloading {model_s3_uri} ...")
            s3.download_file(bucket, key, tmp.name)
            with tarfile.open(tmp.name, "r:gz") as tar:
                tar.extractall(local_model_dir, filter="data")
            print(f"[{pos}] Extracted to {local_model_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker training jobs")
    parser.add_argument(
        "--positions", nargs="+", default=ALL_POSITIONS,
        choices=ALL_POSITIONS, help="Positions to train",
    )
    parser.add_argument(
        "--wait", default="true",
        help="Wait for jobs to complete (true/false)",
    )
    args = parser.parse_args()
    wait = args.wait.lower() == "true"

    # Upload data splits to S3
    print("Uploading data splits to S3...")
    session = sagemaker.Session()
    session.upload_data("data/splits", bucket=S3_BUCKET, key_prefix="data")
    print("Upload complete.")

    # Launch all positions in parallel threads
    print(f"\nLaunching {len(args.positions)} training jobs: {args.positions}")
    with ThreadPoolExecutor(max_workers=len(args.positions)) as pool:
        futures = {
            pool.submit(launch_one, pos, wait): pos
            for pos in args.positions
        }
        for future in as_completed(futures):
            pos = futures[future]
            try:
                pos, estimator = future.result()
                print(f"\n[{pos}] Complete. Artifacts: {estimator.model_data}")
            except Exception as e:
                print(f"\n[{pos}] FAILED: {e}")

    # Download model artifacts back to local dirs
    if wait:
        print("\nDownloading model artifacts...")
        download_artifacts(args.positions)
        print("\nAll done.")


if __name__ == "__main__":
    main()
