"""Launch parallel AWS Batch tuning jobs for the attention NN.

Mirrors ``src/batch/launch.py`` but submits ``--mode=tune`` jobs instead of
training jobs. Each position runs in its own Spot g4dn.xlarge container so
the six positions tune in parallel (matches the 24-vCPU Spot G+VT quota
documented in the AWS-quota auto-memory).

Reuses ``submit_job`` / ``wait_for_jobs`` / ``RETRY_STRATEGY`` from
``launch.py`` by importing them — except ``submit_job`` itself, because
that one is bound to the training command shape. ``submit_tune_job`` below
is the tuning analog.

Usage:
    python -m src.batch.launch_tune                                  # all 6 positions
    python -m src.batch.launch_tune --positions QB RB                # subset
    python -m src.batch.launch_tune --n-trials 30                    # default
    python -m src.batch.launch_tune --positions QB --n-trials 5      # smoke test
    python -m src.batch.launch_tune --wait false                     # fire and forget
    python -m src.batch.launch_tune --dry-run                        # print plan, touch nothing

Config (env vars, all optional — same names as ``launch.py``):
    FF_S3_BUCKET        (default: ff-predictor-training)
    FF_JOB_QUEUE        (default: ff-training-queue)
    FF_JOB_DEFINITION   (default: ff-training-job)        GPU queue/def
    FF_WAIT_TIMEOUT     (default: 10800, i.e. 3h — bumped via --wait-timeout
                         since NN tuning can run ~30 trials × ~2 min/trial)

All six positions are now supported — K/DST were added once their ``run()``
signatures accepted a ``config=`` kwarg. The 24-vCPU Spot quota tolerates
six concurrent g4dn.xlarge jobs exactly; concurrent local launches will
queue at ``RUNNABLE`` instead of pushing over-quota.
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

# All six positions now have ``run(config=...)``; argparse choices still pin
# the input to known names so a typo fails locally instead of submitting a
# Spot job that will fail on ``get_config()``.
SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")

# Default Optuna trial budget per position. On g4dn.xlarge each trial is
# 1–2 min, so 30 trials ≈ 30–60 min per position — well under the 3h wait
# timeout. **Intentionally diverges** from ``src/tuning/tune_nn.py``'s
# ``_DEFAULT_N_TRIALS = 15``: that one is the local-laptop default (each
# trial 5–10 min on CPU, so 15 trials caps at ~2.5 hr); this one is the
# Batch-GPU default. Both module-default paths are safe because the
# retune-nn-batch.yml workflow ALWAYS forwards ``--n-trials``, so the
# divergence only matters for someone invoking ``launch_tune`` locally
# without that flag — which is the right ceiling for the Batch context.
DEFAULT_N_TRIALS = 30


def submit_tune_job(
    position: str,
    n_trials: int = DEFAULT_N_TRIALS,
    timeout: int | None = None,
    seed: int = 42,
    batch_client=None,
) -> tuple[str, str]:
    """Submit one Batch tuning job. Returns (position, job_id).

    The container's entrypoint is ``python src/batch/train.py``, which
    dispatches on ``--mode=tune`` into ``src/tuning/tune_nn``. We also
    propagate ``S3_BUCKET`` and ``S3_DATA_PREFIX`` so tune_nn's
    ``_ensure_data_from_s3`` can populate ``data/splits/`` and ``data/raw/``
    on first invocation (Spot resilience: study DB resumes via S3 if the
    job is retried after a Host EC2 interruption).
    """
    batch = batch_client or boto3.client("batch", region_name=AWS_REGION)
    timestamp = int(time.time())
    suffix = uuid.uuid4().hex[:6]

    command = [
        "--position",
        position,
        "--mode",
        "tune",
        "--seed",
        str(seed),
        "--n-trials",
        str(n_trials),
    ]
    if timeout is not None:
        command += ["--timeout", str(timeout)]

    response = batch.submit_job(
        jobName=f"ff-tune-{position.lower()}-{timestamp}-{suffix}",
        jobQueue=JOB_QUEUE,
        # Always GPU job def for tune. Attention training is GPU-bound for
        # every position, including K's nested attention and DST's two-
        # branch attention — launch.py's ``cpu_only`` CPU-job-def carve-out
        # for K/DST applies only to the training path (which falls back to
        # Ridge/LGBM for those positions when an NN isn't trained); tuning
        # always trains the attention NN.
        jobDefinition=JOB_DEFINITION,
        retryStrategy=RETRY_STRATEGY,
        containerOverrides={
            "command": command,
            "environment": [
                {"name": "S3_BUCKET", "value": S3_BUCKET},
                {"name": "S3_DATA_PREFIX", "value": "data"},
                # Quieter than training (default LOG_EVERY=1). Tuning runs N
                # trials × ~200 epochs each — 1 line per epoch would flood
                # CloudWatch with ~6000 lines/trial × 30 trials = 180k lines.
                {"name": "LOG_EVERY", "value": "20"},
            ],
        },
    )
    job_id = response["jobId"]
    print(f"[{position}] Submitted tune job {job_id}")
    return position, job_id


def _print_plan(positions: list[str], n_trials: int, timeout: int | None, seed: int) -> None:
    print("DRY RUN — no AWS calls will be made.")
    print(f"  region:       {AWS_REGION}")
    print(f"  bucket:       {S3_BUCKET}")
    print(f"  queue:        {JOB_QUEUE}")
    print(f"  definition:   {JOB_DEFINITION}")
    print(f"  wait timeout: {WAIT_TIMEOUT_SECONDS}s")
    print(f"  n_trials:     {n_trials}")
    print(f"  timeout:      {timeout if timeout is not None else 'no cap'}")
    print(f"  seed:         {seed}")
    print("  jobs:")
    for pos in positions:
        print(f"    - {pos:<4} -> --mode=tune --n-trials={n_trials}")


def main():
    parser = argparse.ArgumentParser(description="Launch AWS Batch NN tuning jobs (Optuna)")
    parser.add_argument(
        "--positions",
        nargs="+",
        default=list(SUPPORTED_POSITIONS),
        choices=list(SUPPORTED_POSITIONS),
        help=f"Positions to tune (default: all {len(SUPPORTED_POSITIONS)} supported)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Optuna trials per position (default: {DEFAULT_N_TRIALS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-position wall-clock cap in seconds (default: no cap; Batch wait timeout still applies)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wait",
        default="true",
        help="Wait for jobs to complete (true/false)",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=None,
        help=f"Override wait timeout in seconds (default: {WAIT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned submissions and exit without touching AWS",
    )
    args = parser.parse_args()

    positions = [p.upper() for p in args.positions]
    wait = args.wait.lower() == "true"
    wait_timeout = args.wait_timeout if args.wait_timeout is not None else WAIT_TIMEOUT_SECONDS

    if args.dry_run:
        _print_plan(positions, args.n_trials, args.timeout, args.seed)
        return

    batch_client = boto3.client("batch", region_name=AWS_REGION)

    print(f"Submitting {len(positions)} tune jobs: {positions}")
    job_ids: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(positions)) as pool:
        futures = {
            pool.submit(
                submit_tune_job, pos, args.n_trials, args.timeout, args.seed, batch_client
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
        print(f"Results land at s3://{S3_BUCKET}/tune_nn/{{pos}}/results.json per position.")
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
        print(f"  Per-position results: s3://{S3_BUCKET}/tune_nn/{{pos}}/results.json")
        print(f"  Per-position study DBs (resumable): s3://{S3_BUCKET}/tune_nn/{{pos}}/study.db")
        print("  Run `python -m src.tuning.aggregate_results` to merge per-position JSONs.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
