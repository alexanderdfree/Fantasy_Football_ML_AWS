"""Launch parallel AWS Batch LR-scheduler-type ablation jobs.

Mirrors ``src/tuning/launch_tune.py`` but submits
``--ablation scheduler-type`` jobs instead of ``--mode tune`` jobs. Each
position runs in its own Spot g6.xlarge/g5.xlarge container so the six positions
ablate in parallel (six 4-vCPU hosts; Spot G+VT quota is 64 vCPU since
2026-06-11). Each container runs the 3-way
(onecycle / cosine / plateau) A/B across ``--seeds`` and uploads its result JSON
to ``s3://$S3_BUCKET/ablate_scheduler/{pos}/result.json``; merge them afterwards
with ``src.tuning.aggregate_scheduler``.

Reuses ``wait_for_jobs`` / ``RETRY_STRATEGY`` / queue + def names from
``src/batch/launch.py``. ``FF_CUDA_GRAPH=0`` is forced so the scheduler-type
deltas are a bit-comparable eager A/B (CUDA graphs autodetect ON for sm_80+ but
are NOT numerically inert — see AGENTS.md / ADR-0017).

Usage:
    python -m src.tuning.launch_ablate_scheduler                          # all 6, seeds 42,43,44
    python -m src.tuning.launch_ablate_scheduler --positions K DST        # subset
    python -m src.tuning.launch_ablate_scheduler --seeds 42               # single seed (smoke)
    python -m src.tuning.launch_ablate_scheduler --wait false             # fire and forget
    python -m src.tuning.launch_ablate_scheduler --dry-run                # print plan, touch nothing
"""

import argparse
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.batch.launch import (  # noqa: E402
    AWS_REGION,
    JOB_DEFINITION,
    JOB_QUEUE,
    POLL_INTERVAL_SECONDS,
    RETRY_STRATEGY,
    S3_BUCKET,
    WAIT_TIMEOUT_SECONDS,
    wait_for_jobs,
)

SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
DEFAULT_SEEDS = "42,43,44"
# 3 seeds x 3 schedulers = 9 full pipeline runs/position (no LightGBM). On
# Batch GPU pool each run is ~1-3 min, so ~10-30 min/position. 2h attempt cap leaves
# headroom and overrides the 30-min job-def default.
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 7200
RESULT_PREFIX = "ablate_scheduler"


def submit_ablate_job(
    position: str,
    seeds: str,
    attempt_timeout: int = DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
    batch_client=None,
) -> tuple[str, str]:
    """Submit one Batch scheduler-type ablation job. Returns (position, job_id)."""
    batch = batch_client or boto3.client("batch", region_name=AWS_REGION)
    timestamp = int(time.time())
    suffix = uuid.uuid4().hex[:6]

    command = [
        "--position",
        position,
        "--ablation",
        "scheduler-type",
        "--seeds",
        seeds,
    ]
    response = batch.submit_job(
        jobName=f"ff-ablate-sched-{position.lower()}-{timestamp}-{suffix}",
        jobQueue=JOB_QUEUE,
        jobDefinition=JOB_DEFINITION,
        retryStrategy=RETRY_STRATEGY,
        timeout={"attemptDurationSeconds": int(attempt_timeout)},
        containerOverrides={
            "command": command,
            "environment": [
                {"name": "S3_BUCKET", "value": S3_BUCKET},
                {"name": "S3_DATA_PREFIX", "value": "data"},
                {"name": "FF_DEVICE", "value": "cuda"},
                # Force eager: CUDA graphs (autodetect-ON for sm_80+) are NOT
                # numerically inert, so a bit-comparable scheduler A/B needs them off.
                {"name": "FF_CUDA_GRAPH", "value": "0"},
                {"name": "FF_AMP_DTYPE", "value": "auto"},
                {"name": "FF_COMPILE", "value": "0"},
                {"name": "LOG_EVERY", "value": "20"},
            ],
        },
    )
    job_id = response["jobId"]
    print(f"[{position}] Submitted scheduler-type ablation job {job_id}")
    return position, job_id


def _print_plan(positions: list[str], seeds: str, attempt_timeout: int) -> None:
    print("DRY RUN — no AWS calls will be made.")
    print(f"  region:       {AWS_REGION}")
    print(f"  bucket:       {S3_BUCKET}")
    print(f"  queue:        {JOB_QUEUE}")
    print(f"  definition:   {JOB_DEFINITION}")
    print(f"  wait timeout: {WAIT_TIMEOUT_SECONDS}s")
    print(f"  attempt cap:  {attempt_timeout}s")
    print(f"  seeds:        {seeds}")
    print("  FF_CUDA_GRAPH: 0 (eager, bit-comparable A/B)")
    print("  jobs:")
    for pos in positions:
        print(f"    - {pos:<4} -> --ablation scheduler-type --seeds {seeds}")
    print(f"  results -> s3://{S3_BUCKET}/{RESULT_PREFIX}/{{pos}}/result.json")


def main():
    parser = argparse.ArgumentParser(description="Launch AWS Batch scheduler-type ablation jobs")
    parser.add_argument(
        "--positions",
        nargs="+",
        default=list(SUPPORTED_POSITIONS),
        choices=list(SUPPORTED_POSITIONS),
        help=f"Positions to ablate (default: all {len(SUPPORTED_POSITIONS)})",
    )
    parser.add_argument(
        "--seeds",
        default=DEFAULT_SEEDS,
        help=f"Comma-separated seeds passed to each container (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--attempt-timeout",
        type=int,
        default=DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
        help=f"AWS Batch attemptDurationSeconds (default: {DEFAULT_ATTEMPT_TIMEOUT_SECONDS})",
    )
    parser.add_argument("--wait", default="true", help="Wait for jobs to complete (true/false)")
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=None,
        help=f"Override wait timeout in seconds (default: {WAIT_TIMEOUT_SECONDS})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned submissions and exit")
    args = parser.parse_args()

    positions = [p.upper() for p in args.positions]
    wait = args.wait.lower() == "true"
    wait_timeout = args.wait_timeout if args.wait_timeout is not None else WAIT_TIMEOUT_SECONDS
    if args.attempt_timeout <= 0:
        raise SystemExit("--attempt-timeout must be > 0")
    # Validate seeds early so a typo fails locally, not in six Spot containers.
    if not [int(s) for s in args.seeds.split(",") if s.strip()]:
        raise SystemExit(f"--seeds parsed to empty list: {args.seeds!r}")

    if args.dry_run:
        _print_plan(positions, args.seeds, args.attempt_timeout)
        return

    batch_client = boto3.client("batch", region_name=AWS_REGION)
    print(f"Submitting {len(positions)} scheduler-type ablation jobs: {positions}")
    job_ids: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(positions)) as pool:
        futures = {
            pool.submit(
                submit_ablate_job,
                position=pos,
                seeds=args.seeds,
                attempt_timeout=args.attempt_timeout,
                batch_client=batch_client,
            ): pos
            for pos in positions
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
        print(f"Results land at s3://{S3_BUCKET}/{RESULT_PREFIX}/{{pos}}/result.json per position.")
        return

    print(
        f"\nWaiting for {len(job_ids)} jobs to complete (polling every "
        f"{POLL_INTERVAL_SECONDS}s, timeout {wait_timeout}s)..."
    )
    results = wait_for_jobs(job_ids, timeout_seconds=wait_timeout, batch_client=batch_client)
    succeeded = [p for p, (status, _) in results.items() if status == "SUCCEEDED"]
    failed = [p for p, (status, _) in results.items() if status == "FAILED"]
    timed_out = [p for p, (status, _) in results.items() if status == "TIMED_OUT"]
    if failed:
        print(f"\nFailed positions: {failed}")
    if timed_out:
        print(f"\nTimed-out positions: {timed_out}")
    if succeeded:
        print(f"\nSucceeded positions: {succeeded}")
        print(f"  Per-position results: s3://{S3_BUCKET}/{RESULT_PREFIX}/{{pos}}/result.json")
        print("  Run `python -m src.tuning.aggregate_scheduler` to merge per-position JSONs.")

    # Non-zero exit if nothing succeeded so the GH matrix job fails loudly.
    if not succeeded:
        raise SystemExit("No scheduler-type ablation jobs succeeded.")
    print("\nAll done.")


if __name__ == "__main__":
    main()
