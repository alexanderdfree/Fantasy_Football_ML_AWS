"""Launch parallel AWS Batch training jobs for all positions.

Usage:
    python batch/launch.py                     # all positions
    python batch/launch.py --positions RB WR   # subset
    python batch/launch.py --wait false         # fire and forget
"""
import argparse
import os
import tarfile
import tempfile
import time

import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

S3_BUCKET = "ff-predictor-training"
JOB_QUEUE = "ff-training-queue"
JOB_DEFINITION = "ff-training-job"
ALL_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]

POLL_INTERVAL_SECONDS = 30
TERMINAL_STATES = {"SUCCEEDED", "FAILED"}


def upload_data(s3_bucket):
    """Upload local data splits to S3."""
    s3 = boto3.client("s3")
    data_dir = "data/splits"
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        local_path = os.path.join(data_dir, name)
        s3_key = f"data/{name}"
        print(f"Uploading {local_path} -> s3://{s3_bucket}/{s3_key}")
        s3.upload_file(local_path, s3_bucket, s3_key)
    print("Data upload complete.\n")


def submit_job(position, seed=42):
    """Submit a single Batch job for one position. Returns job ID."""
    batch = boto3.client("batch")
    timestamp = int(time.time())
    response = batch.submit_job(
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
    job_id = response["jobId"]
    print(f"[{position}] Submitted job {job_id}")
    return position, job_id


def wait_for_jobs(job_ids):
    """Poll Batch until all jobs reach a terminal state.

    Args:
        job_ids: dict mapping position -> job_id

    Returns:
        dict mapping position -> final status ("SUCCEEDED" or "FAILED")
    """
    batch = boto3.client("batch")
    remaining = dict(job_ids)  # position -> job_id
    results = {}  # position -> status
    last_status = {}  # job_id -> last printed status

    while remaining:
        ids = list(remaining.values())
        response = batch.describe_jobs(jobs=ids)

        for job in response["jobs"]:
            job_id = job["jobId"]
            status = job["status"]
            # Find position for this job_id
            pos = next(p for p, jid in remaining.items() if jid == job_id)

            if last_status.get(job_id) != status:
                print(f"[{pos}] {status}")
                last_status[job_id] = status

            if status in TERMINAL_STATES:
                results[pos] = status
                del remaining[pos]

        if remaining:
            time.sleep(POLL_INTERVAL_SECONDS)

    return results


def download_artifacts(positions):
    """Download model artifacts from S3 back to local position dirs."""
    s3 = boto3.client("s3")

    for pos in positions:
        s3_key = f"models/{pos}/model.tar.gz"
        local_model_dir = os.path.join(pos, "outputs", "models")
        os.makedirs(local_model_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
            print(f"[{pos}] Downloading s3://{S3_BUCKET}/{s3_key} ...")
            try:
                s3.download_file(S3_BUCKET, s3_key, tmp.name)
            except s3.exceptions.NoSuchKey:
                print(f"[{pos}] No artifacts found at s3://{S3_BUCKET}/{s3_key}, skipping")
                continue
            with tarfile.open(tmp.name, "r:gz") as tar:
                tar.extractall(local_model_dir, filter="data")
            print(f"[{pos}] Extracted to {local_model_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Launch AWS Batch training jobs")
    parser.add_argument(
        "--positions", nargs="+", default=ALL_POSITIONS,
        choices=ALL_POSITIONS, help="Positions to train",
    )
    parser.add_argument(
        "--wait", default="true",
        help="Wait for jobs to complete (true/false)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    wait = args.wait.lower() == "true"

    # Upload data splits to S3
    print("Uploading data splits to S3...")
    upload_data(S3_BUCKET)

    # Submit all positions in parallel
    print(f"Submitting {len(args.positions)} Batch jobs: {args.positions}")
    job_ids = {}  # position -> job_id
    with ThreadPoolExecutor(max_workers=len(args.positions)) as pool:
        futures = {
            pool.submit(submit_job, pos, args.seed): pos
            for pos in args.positions
        }
        for future in as_completed(futures):
            pos = futures[future]
            try:
                pos, job_id = future.result()
                job_ids[pos] = job_id
            except Exception as e:
                print(f"[{pos}] FAILED to submit: {e}")

    if not wait:
        print("\nJobs submitted. Use 'aws batch describe-jobs' to check status.")
        return

    # Wait for all jobs to complete
    print(f"\nWaiting for {len(job_ids)} jobs to complete (polling every {POLL_INTERVAL_SECONDS}s)...")
    results = wait_for_jobs(job_ids)

    succeeded = [pos for pos, status in results.items() if status == "SUCCEEDED"]
    failed = [pos for pos, status in results.items() if status == "FAILED"]

    if failed:
        print(f"\nFailed positions: {failed}")
    if succeeded:
        print(f"\nSucceeded: {succeeded}")
        print("Downloading model artifacts...")
        download_artifacts(succeeded)

    print("\nAll done.")


if __name__ == "__main__":
    main()
