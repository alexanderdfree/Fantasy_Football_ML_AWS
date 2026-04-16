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
import urllib.parse

import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

S3_BUCKET = "ff-predictor-training"
JOB_QUEUE = "ff-training-queue"
JOB_DEFINITION = "ff-training-job"
ALL_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]

AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
BATCH_LOG_GROUP = "/aws/batch/job"

POLL_INTERVAL_SECONDS = 30
# Hard cap so a stuck RUNNABLE job (Spot capacity, bad IAM, etc.) can't pin
# this script forever. ~3h is generous for a 6-position GPU sweep.
WAIT_TIMEOUT_SECONDS = 3 * 60 * 60
TERMINAL_STATES = {"SUCCEEDED", "FAILED"}

# Retry Spot reclaims automatically; don't retry deterministic app failures.
RETRY_STRATEGY = {
    "attempts": 3,
    "evaluateOnExit": [
        # Spot interruption: host terminated by EC2 -> retry
        {"onStatusReason": "Host EC2*", "action": "RETRY"},
        # Anything else: exit immediately so we see the real error
        {"onReason": "*", "action": "EXIT"},
    ],
}


def _cloudwatch_url(log_stream_name: str) -> str:
    """Build a console URL that opens the CloudWatch stream for a failed job."""
    # AWS console uses double-URL-encoded slashes: $252F = %2F = /
    group = urllib.parse.quote(BATCH_LOG_GROUP, safe="").replace("%", "$")
    stream = urllib.parse.quote(log_stream_name, safe="").replace("%", "$")
    return (
        f"https://{AWS_REGION}.console.aws.amazon.com/cloudwatch/home"
        f"?region={AWS_REGION}#logsV2:log-groups/log-group/{group}/log-events/{stream}"
    )


def upload_data(s3_bucket):
    """Upload local data splits to S3."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    data_dir = "data/splits"
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        local_path = os.path.join(data_dir, name)
        s3_key = f"data/{name}"
        print(f"Uploading {local_path} -> s3://{s3_bucket}/{s3_key}")
        s3.upload_file(local_path, s3_bucket, s3_key)
    print("Data upload complete.\n")


def submit_job(position, seed=42):
    """Submit a single Batch job for one position. Returns job ID."""
    batch = boto3.client("batch", region_name=AWS_REGION)
    timestamp = int(time.time())
    response = batch.submit_job(
        jobName=f"ff-{position.lower()}-{timestamp}",
        jobQueue=JOB_QUEUE,
        jobDefinition=JOB_DEFINITION,
        retryStrategy=RETRY_STRATEGY,
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


def wait_for_jobs(job_ids, timeout_seconds=WAIT_TIMEOUT_SECONDS):
    """Poll Batch until all jobs reach a terminal state (or timeout).

    Args:
        job_ids: dict mapping position -> job_id
        timeout_seconds: wall-clock cap; remaining jobs are reported as TIMED_OUT.

    Returns:
        dict mapping position -> final status
        ("SUCCEEDED", "FAILED", or "TIMED_OUT" if we gave up waiting).

    On FAILED, prints a CloudWatch console URL and an `aws logs` command so
    the log stream is one click or one paste away.
    """
    batch = boto3.client("batch", region_name=AWS_REGION)
    remaining = dict(job_ids)  # position -> job_id
    results = {}  # position -> status
    last_status = {}  # job_id -> last printed status
    deadline = time.monotonic() + timeout_seconds

    while remaining:
        if time.monotonic() > deadline:
            print(
                f"\nTimeout after {timeout_seconds}s; "
                f"giving up on {list(remaining.keys())}"
            )
            for pos in remaining:
                results[pos] = "TIMED_OUT"
            break

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
                if status == "FAILED":
                    reason = job.get("statusReason") or ""
                    container = job.get("container") or {}
                    stream = container.get("logStreamName")
                    print(f"[{pos}] FAILED reason: {reason}")
                    if stream:
                        print(f"[{pos}] log stream: {stream}")
                        print(f"[{pos}] console:    {_cloudwatch_url(stream)}")
                        print(
                            f"[{pos}] cli:        aws logs get-log-events "
                            f"--log-group-name {BATCH_LOG_GROUP} "
                            f"--log-stream-name '{stream}' --region {AWS_REGION}"
                        )
                    else:
                        print(f"[{pos}] (no log stream — job never started a container)")
                del remaining[pos]

        if remaining:
            time.sleep(POLL_INTERVAL_SECONDS)

    return results


def download_artifacts(positions):
    """Download model artifacts from S3 back to local position dirs."""
    s3 = boto3.client("s3", region_name=AWS_REGION)

    for pos in positions:
        s3_key = f"models/{pos}/model.tar.gz"
        local_model_dir = os.path.join(pos, "outputs", "models")
        os.makedirs(local_model_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
            print(f"[{pos}] Downloading s3://{S3_BUCKET}/{s3_key} ...")
            try:
                s3.download_file(S3_BUCKET, s3_key, tmp.name)
            except ClientError as e:
                # s3.download_file surfaces a 404 as ClientError, NOT NoSuchKey.
                code = e.response.get("Error", {}).get("Code")
                if code in ("404", "NoSuchKey"):
                    print(f"[{pos}] No artifacts at s3://{S3_BUCKET}/{s3_key}, skipping")
                    continue
                raise
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
