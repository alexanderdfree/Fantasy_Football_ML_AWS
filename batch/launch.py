"""Launch parallel AWS Batch training jobs for all positions.

Usage:
    python batch/launch.py                         # all positions
    python batch/launch.py --positions RB WR       # subset
    python batch/launch.py --wait false            # fire and forget
    python batch/launch.py --dry-run               # print plan, touch nothing
    python batch/launch.py --wait-timeout 1800     # override 3h default
    python batch/launch.py --force-upload          # skip ETag dedup

Config (environment variables, all optional):
    FF_S3_BUCKET        (default: ff-predictor-training)
    FF_JOB_QUEUE        (default: ff-training-queue)
    FF_JOB_DEFINITION   (default: ff-training-job)          GPU queue/def
    FF_JOB_DEFINITION_CPU  (optional)                       CPU queue/def
        If set, K and DST are submitted with this job definition instead —
        a cheap CPU Spot pool for the non-NN positions.
    FF_WAIT_TIMEOUT     (default: 10800, i.e. 3h)
"""
import argparse
import hashlib
import os
import tarfile
import tempfile
import time
import urllib.parse
import uuid

import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration (env-var overridable) ---------------------------------
S3_BUCKET = os.environ.get("FF_S3_BUCKET", "ff-predictor-training")
JOB_QUEUE = os.environ.get("FF_JOB_QUEUE", "ff-training-queue")
JOB_DEFINITION = os.environ.get("FF_JOB_DEFINITION", "ff-training-job")
# Optional CPU-only job definition. When set, K and DST route here instead of
# the default GPU definition so we don't waste g4dn Spot-hours on Ridge/LGBM.
# Set to an empty string to leave unset — we treat empty as "not configured".
JOB_DEFINITION_CPU = os.environ.get("FF_JOB_DEFINITION_CPU", "") or None

ALL_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
# Positions that don't use a neural net — safe to send to a CPU queue.
CPU_ONLY_POSITIONS = {"K", "DST"}

AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
BATCH_LOG_GROUP = "/aws/batch/job"

POLL_INTERVAL_SECONDS = 30
# Hard cap so a stuck RUNNABLE job (Spot capacity, bad IAM, etc.) can't pin
# this script forever. ~3h is generous for a 6-position GPU sweep.
WAIT_TIMEOUT_SECONDS = int(os.environ.get("FF_WAIT_TIMEOUT", 3 * 60 * 60))
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


def _file_md5(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Stream-hash a file; returns hex digest (matches S3 ETag for single-part)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _s3_object_etag(s3_client, bucket: str, key: str):
    """Return the S3 object's ETag (minus quotes) or None if the object doesn't exist."""
    try:
        resp = s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    return resp.get("ETag", "").strip('"')


def upload_data(s3_bucket, s3_client=None, force: bool = False):
    """Upload local data splits to S3, skipping files whose ETag matches the local MD5.

    Set force=True to re-upload regardless.
    """
    s3 = s3_client or boto3.client("s3", region_name=AWS_REGION)
    data_dir = "data/splits"
    uploaded = 0
    skipped = 0
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        local_path = os.path.join(data_dir, name)
        s3_key = f"data/{name}"
        if not force:
            remote_etag = _s3_object_etag(s3, s3_bucket, s3_key)
            if remote_etag is not None:
                local_md5 = _file_md5(local_path)
                if remote_etag == local_md5:
                    print(f"  [skip] s3://{s3_bucket}/{s3_key} already up to date ({local_md5[:8]}...)")
                    skipped += 1
                    continue
        print(f"  [upload] {local_path} -> s3://{s3_bucket}/{s3_key}")
        s3.upload_file(local_path, s3_bucket, s3_key)
        uploaded += 1
    print(f"Data upload complete: {uploaded} uploaded, {skipped} skipped.\n")


def _job_definition_for(position: str) -> str:
    """Pick the right job definition for a position.

    CPU-only positions use JOB_DEFINITION_CPU if it's configured; otherwise
    they fall back to the default (GPU) definition so this is safe to deploy
    before the CPU infra exists.
    """
    if position in CPU_ONLY_POSITIONS and JOB_DEFINITION_CPU:
        return JOB_DEFINITION_CPU
    return JOB_DEFINITION


def submit_job(position, seed=42, batch_client=None):
    """Submit a single Batch job for one position. Returns (position, job_id)."""
    batch = batch_client or boto3.client("batch", region_name=AWS_REGION)
    # int-seconds timestamp collides if two launches happen in the same second;
    # a short uuid suffix makes the name unique without sacrificing readability.
    timestamp = int(time.time())
    suffix = uuid.uuid4().hex[:6]
    job_definition = _job_definition_for(position)
    response = batch.submit_job(
        jobName=f"ff-{position.lower()}-{timestamp}-{suffix}",
        jobQueue=JOB_QUEUE,
        jobDefinition=job_definition,
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
    print(f"[{position}] Submitted job {job_id} (definition: {job_definition})")
    return position, job_id


def wait_for_jobs(job_ids, timeout_seconds=None, batch_client=None):
    """Poll Batch until all jobs reach a terminal state (or timeout).

    Args:
        job_ids: dict mapping position -> job_id
        timeout_seconds: wall-clock cap; remaining jobs reported as TIMED_OUT.
            Defaults to the module-level WAIT_TIMEOUT_SECONDS.
        batch_client: optional shared boto3 Batch client.

    Returns:
        dict mapping position -> (status, stopped_at_ms)
        status is "SUCCEEDED", "FAILED", or "TIMED_OUT".
        stopped_at_ms is the Batch `stoppedAt` epoch-ms, or None.

    On FAILED, prints a CloudWatch console URL and an `aws logs` command so
    the log stream is one click or one paste away.
    """
    if timeout_seconds is None:
        timeout_seconds = WAIT_TIMEOUT_SECONDS
    batch = batch_client or boto3.client("batch", region_name=AWS_REGION)
    remaining = dict(job_ids)  # position -> job_id
    results = {}  # position -> (status, stopped_at_ms)
    last_status = {}  # job_id -> last printed status
    deadline = time.monotonic() + timeout_seconds

    while remaining:
        if time.monotonic() > deadline:
            print(
                f"\nTimeout after {timeout_seconds}s; "
                f"giving up on {list(remaining.keys())}"
            )
            for pos in remaining:
                results[pos] = ("TIMED_OUT", None)
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
                stopped_at = job.get("stoppedAt")  # ms since epoch or None
                results[pos] = (status, stopped_at)
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


def download_artifacts(positions, stopped_at_by_pos=None, s3_client=None):
    """Download model artifacts from S3 back to local position dirs.

    If stopped_at_by_pos is provided (position -> ms-epoch stoppedAt), warn
    loudly when the S3 object's LastModified is older than the job finish —
    that means we're pulling a stale artifact from a prior run.
    """
    s3 = s3_client or boto3.client("s3", region_name=AWS_REGION)
    stopped_at_by_pos = stopped_at_by_pos or {}

    for pos in positions:
        s3_key = f"models/{pos}/model.tar.gz"
        local_model_dir = os.path.join(pos, "outputs", "models")
        os.makedirs(local_model_dir, exist_ok=True)

        # Stale-artifact guard: compare remote LastModified to job stoppedAt.
        try:
            head = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
            remote_modified = head.get("LastModified")  # datetime, tz-aware
            stopped_at_ms = stopped_at_by_pos.get(pos)
            if remote_modified is not None and stopped_at_ms:
                stopped_at_s = stopped_at_ms / 1000.0
                # +5s fudge for clock skew. LastModified is in UTC.
                if remote_modified.timestamp() + 5 < stopped_at_s:
                    print(
                        f"[{pos}] WARNING: s3://{S3_BUCKET}/{s3_key} LastModified "
                        f"({remote_modified.isoformat()}) is older than job "
                        f"stoppedAt ({stopped_at_s}). Artifact may be stale."
                    )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                print(f"[{pos}] No artifacts at s3://{S3_BUCKET}/{s3_key}, skipping")
                continue
            raise

        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
            print(f"[{pos}] Downloading s3://{S3_BUCKET}/{s3_key} ...")
            s3.download_file(S3_BUCKET, s3_key, tmp.name)
            with tarfile.open(tmp.name, "r:gz") as tar:
                tar.extractall(local_model_dir, filter="data")
            print(f"[{pos}] Extracted to {local_model_dir}/")


def _print_plan(positions, seed):
    """--dry-run: print what would be submitted, touch nothing."""
    print("DRY RUN — no AWS calls will be made.")
    print(f"  region:       {AWS_REGION}")
    print(f"  bucket:       {S3_BUCKET}")
    print(f"  queue:        {JOB_QUEUE}")
    print(f"  definition:   {JOB_DEFINITION}")
    if JOB_DEFINITION_CPU:
        print(f"  cpu def:      {JOB_DEFINITION_CPU} (K, DST route here)")
    print(f"  wait timeout: {WAIT_TIMEOUT_SECONDS}s")
    print(f"  seed:         {seed}")
    print("  jobs:")
    for pos in positions:
        print(f"    - {pos:<4} -> definition {_job_definition_for(pos)}")


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
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned submissions and exit without touching AWS",
    )
    parser.add_argument(
        "--wait-timeout", type=int, default=None,
        help=f"Override wait timeout in seconds (default: {WAIT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--force-upload", action="store_true",
        help="Upload data splits even if S3 ETag matches the local file",
    )
    args = parser.parse_args()
    wait = args.wait.lower() == "true"
    wait_timeout = args.wait_timeout if args.wait_timeout is not None else WAIT_TIMEOUT_SECONDS

    if args.dry_run:
        _print_plan(args.positions, args.seed)
        return

    # Shared boto3 clients — boto3 clients are thread-safe, no need per-thread.
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    batch_client = boto3.client("batch", region_name=AWS_REGION)

    # Upload data splits to S3 (skips unchanged files by default)
    print("Uploading data splits to S3...")
    upload_data(S3_BUCKET, s3_client=s3_client, force=args.force_upload)

    # Submit all positions in parallel
    print(f"Submitting {len(args.positions)} Batch jobs: {args.positions}")
    job_ids = {}  # position -> job_id
    with ThreadPoolExecutor(max_workers=len(args.positions)) as pool:
        futures = {
            pool.submit(submit_job, pos, args.seed, batch_client): pos
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
    print(f"\nWaiting for {len(job_ids)} jobs to complete (polling every {POLL_INTERVAL_SECONDS}s, timeout {wait_timeout}s)...")
    results = wait_for_jobs(job_ids, timeout_seconds=wait_timeout, batch_client=batch_client)

    succeeded = [pos for pos, (status, _) in results.items() if status == "SUCCEEDED"]
    failed = [pos for pos, (status, _) in results.items() if status == "FAILED"]
    stopped_at_by_pos = {pos: stopped_at for pos, (_, stopped_at) in results.items()}

    if failed:
        print(f"\nFailed positions: {failed}")
    if succeeded:
        print(f"\nSucceeded: {succeeded}")
        print("Downloading model artifacts...")
        download_artifacts(succeeded, stopped_at_by_pos=stopped_at_by_pos, s3_client=s3_client)

    print("\nAll done.")


if __name__ == "__main__":
    main()
