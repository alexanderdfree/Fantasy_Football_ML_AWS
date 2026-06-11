"""Launch parallel AWS Batch tuning jobs for the attention NN.

Mirrors ``src/batch/launch.py`` but submits ``--mode=tune`` jobs instead of
training jobs. Each position runs in its own Spot g6.xlarge/g5.xlarge
container so the six positions tune in parallel (matches the 24-vCPU Spot
G+VT quota documented in the AWS-quota auto-memory).

Reuses ``submit_job`` / ``wait_for_jobs`` / ``RETRY_STRATEGY`` from
``launch.py`` by importing them — except ``submit_job`` itself, because
that one is bound to the training command shape. ``submit_tune_job`` below
is the tuning analog.

Usage:
    python -m src.tuning.launch_tune                                  # all 6 positions
    python -m src.tuning.launch_tune --positions QB RB                # subset
    python -m src.tuning.launch_tune --n-trials 30                    # default
    python -m src.tuning.launch_tune --positions QB --n-trials 5      # smoke test
    python -m src.tuning.launch_tune --wait false                     # fire and forget
    python -m src.tuning.launch_tune --dry-run                        # print plan, touch nothing

Config (env vars, all optional — same names as ``launch.py``):
    FF_S3_BUCKET        (default: ff-predictor-training)
    FF_JOB_QUEUE        (default: ff-training-queue)
    FF_JOB_DEFINITION   (default: ff-training-job)        GPU queue/def
    FF_WAIT_TIMEOUT     (default: 10800, i.e. 3h — bumped via --wait-timeout
                         since NN tuning can run ~30 trials × ~2 min/trial)

All six positions are now supported — K/DST were added once their ``run()``
signatures accepted a ``config=`` kwarg. The 24-vCPU Spot quota tolerates
six concurrent 4-vCPU GPU Spot jobs exactly; concurrent local launches will
queue at ``RUNNABLE`` instead of pushing over-quota. The default
``--parallel-backend auto`` is resolved inside the Batch container by
``detect_platform()``: Batch g6/L4 or g5/A10G Linux uses NVIDIA MPS, while Mac
and 5080 hosts keep the existing thread backend.
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
    JOB_DEFINITION_REVISION,
    JOB_QUEUE,
    POLL_INTERVAL_SECONDS,
    RETRY_STRATEGY,
    S3_BUCKET,
    WAIT_TIMEOUT_SECONDS,
    wait_for_jobs,
)
from src.tuning.tune_nn_storage import resolve_search_space_version, s3_prefix  # noqa: E402

# All six positions now have ``run(config=...)``; argparse choices still pin
# the input to known names so a typo fails locally instead of submitting a
# Spot job that will fail on ``get_config()``.
SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")

# Default Optuna trial budget per position. On the Batch GPU pool each trial is
# 1–2 min, so 30 trials ≈ 30–60 min per position — well under the 3h wait
# timeout. **Intentionally diverges** from ``src/tuning/tune_nn.py``'s
# ``_DEFAULT_N_TRIALS = 15``: that one is the local-laptop default (each
# trial 5–10 min on CPU, so 15 trials caps at ~2.5 hr); this one is the
# Batch-GPU default. Both module-default paths are safe because the
# retune-nn-batch.yml workflow ALWAYS forwards ``--n-trials``, so the
# divergence only matters for someone invoking ``launch_tune`` locally
# without that flag — which is the right ceiling for the Batch context.
DEFAULT_N_TRIALS = 30
# "auto" is resolved inside the container by tune_nn._resolve_n_jobs: thread
# backend -> CPU count; mps backend -> CPU count RAM-clamped (~2 GiB/worker).
# On the g6.xlarge job shape (4 vCPU / 15000 MiB) it resolves to 4 — the
# validated ceiling (8 OOMs mid-run, 32 OOMs at startup, 2026-06-10).
DEFAULT_N_JOBS = "auto"
DEFAULT_PARALLEL_BACKEND = "auto"
DEFAULT_CUDA_GRAPH = True
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 7200


def _batch_storage_backend(parallel_backend: str) -> str:
    """The remote Batch GPU pool resolves auto to MPS there."""
    return "mps" if parallel_backend == "auto" else parallel_backend


def _tune_job_definition() -> str:
    """Resolve the job definition for a tune submission, pinning the revision.

    Mirrors ``src/batch/launch.py``'s ``_job_definition_for`` revision-append
    logic: when ``FF_JOB_DEFINITION_REVISION`` is set (``batch-image.yml``
    stashes it and ``retune-nn-batch.yml`` exports it), append ``:N`` so a
    submission pins to that exact revision instead of AWS Batch resolving the
    bare name to the latest active revision at submit time — which lets two
    concurrent image builds race onto the same (latest) revision. Tune always
    uses the GPU job def (no CPU-only carve-out — see ``submit_tune_job``), so
    there's no CPU-def guard to mirror.
    """
    if JOB_DEFINITION_REVISION:
        return f"{JOB_DEFINITION}:{JOB_DEFINITION_REVISION}"
    return JOB_DEFINITION


def submit_tune_job(
    position: str,
    n_trials: int = DEFAULT_N_TRIALS,
    timeout: int | None = None,
    seed: int = 42,
    n_jobs: int | str = DEFAULT_N_JOBS,
    parallel_backend: str = DEFAULT_PARALLEL_BACKEND,
    cuda_graph: bool = DEFAULT_CUDA_GRAPH,
    attempt_timeout: int = DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
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
        "--parallel-backend",
        parallel_backend,
        "--n-jobs",
        str(n_jobs),
    ]
    if timeout is not None:
        command += ["--timeout", str(timeout)]

    storage_version = resolve_search_space_version(
        _batch_storage_backend(parallel_backend), cuda_graph=cuda_graph
    )
    response = batch.submit_job(
        jobName=f"ff-tune-{position.lower()}-{timestamp}-{suffix}",
        jobQueue=JOB_QUEUE,
        # Always GPU job def for tune. Attention training is GPU-bound for
        # every position, including K's nested attention and DST's two-
        # branch attention — launch.py's ``cpu_only`` CPU-job-def carve-out
        # for K/DST applies only to the training path (which falls back to
        # Ridge/LGBM for those positions when an NN isn't trained); tuning
        # always trains the attention NN. Pin the revision (mirrors
        # launch.py) so concurrent image builds don't race onto the latest.
        jobDefinition=_tune_job_definition(),
        retryStrategy=RETRY_STRATEGY,
        timeout={"attemptDurationSeconds": int(attempt_timeout)},
        containerOverrides={
            "command": command,
            "environment": [
                {"name": "S3_BUCKET", "value": S3_BUCKET},
                {"name": "S3_DATA_PREFIX", "value": "data"},
                {"name": "FF_DEVICE", "value": "cuda"},
                # tune_nn re-resolves its namespace in-container from
                # cuda_graph_enabled(); on the sm_80+ tune CEs (g6/L4, g5/A10G)
                # "1" leaves autodetect ON and "0" is the force-off override,
                # so the container lands on the same storage_version predicted
                # above. Keep the env value and the cuda_graph bool coupled.
                {"name": "FF_CUDA_GRAPH", "value": "1" if cuda_graph else "0"},
                {"name": "FF_AMP_DTYPE", "value": "auto"},
                {"name": "FF_COMPILE", "value": "0"},
                {"name": "TUNE_NN_STORAGE_VERSION", "value": storage_version},
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


def _print_plan(
    positions: list[str],
    n_trials: int,
    timeout: int | None,
    seed: int,
    n_jobs: int,
    parallel_backend: str,
    cuda_graph: bool,
    attempt_timeout: int,
) -> None:
    storage_version = resolve_search_space_version(
        _batch_storage_backend(parallel_backend), cuda_graph=cuda_graph
    )
    print("DRY RUN — no AWS calls will be made.")
    print(f"  region:       {AWS_REGION}")
    print(f"  bucket:       {S3_BUCKET}")
    print(f"  queue:        {JOB_QUEUE}")
    # Render the resolved (revision-pinned) job def actually submitted, not
    # the bare name — mirrors launch.py's _print_plan.
    print(f"  definition:   {_tune_job_definition()}")
    print(f"  wait timeout: {WAIT_TIMEOUT_SECONDS}s")
    print(f"  attempt cap:  {attempt_timeout}s")
    print(f"  n_trials:     {n_trials}")
    print(f"  n_jobs:       {n_jobs}")
    print(f"  backend:      {parallel_backend}")
    print(f"  cuda graph:   {cuda_graph}")
    print(f"  storage:      {storage_version}")
    print(f"  timeout:      {timeout if timeout is not None else 'no cap'}")
    print(f"  seed:         {seed}")
    print("  jobs:")
    for pos in positions:
        print(
            f"    - {pos:<4} -> --mode=tune --n-trials={n_trials} "
            f"--parallel-backend={parallel_backend} --n-jobs={n_jobs}"
        )


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
        "--n-jobs",
        type=str,
        default=DEFAULT_N_JOBS,
        help=(
            f"Concurrent trials per Batch GPU, or 'auto' (default: {DEFAULT_N_JOBS}). "
            "auto is resolved inside the container: CPU count, RAM-clamped for "
            "the mps backend — 4 on the current g6.xlarge job shape."
        ),
    )
    parser.add_argument(
        "--parallel-backend",
        choices=["thread", "mps", "auto"],
        default=DEFAULT_PARALLEL_BACKEND,
        help=(
            "Trial concurrency backend inside each position job. auto resolves inside "
            "the Batch container: g6/L4 or g5/A10G Linux -> mps, Mac/5080/non-Batch-GPU -> thread."
        ),
    )
    parser.add_argument(
        "--cuda-graph",
        default="true" if DEFAULT_CUDA_GRAPH else "false",
        help="Enable FF_CUDA_GRAPH inside tune jobs (true/false; default true)",
    )
    parser.add_argument(
        "--attempt-timeout",
        type=int,
        default=DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
        help=(
            "AWS Batch attemptDurationSeconds for tune jobs "
            f"(default: {DEFAULT_ATTEMPT_TIMEOUT_SECONDS})"
        ),
    )
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
    cuda_graph = args.cuda_graph.strip().lower() in {"1", "true", "yes", "on"}
    n_jobs_text = str(args.n_jobs).strip().lower()
    if n_jobs_text != "auto":
        try:
            n_jobs_int = int(n_jobs_text)
        except ValueError:
            raise SystemExit("--n-jobs must be a positive integer or 'auto'") from None
        if n_jobs_int < 1:
            raise SystemExit("--n-jobs must be >= 1")
    if args.attempt_timeout <= 0:
        raise SystemExit("--attempt-timeout must be > 0")

    if args.dry_run:
        _print_plan(
            positions,
            args.n_trials,
            args.timeout,
            args.seed,
            args.n_jobs,
            args.parallel_backend,
            cuda_graph,
            args.attempt_timeout,
        )
        return

    batch_client = boto3.client("batch", region_name=AWS_REGION)

    print(f"Submitting {len(positions)} tune jobs: {positions}")
    job_ids: dict[str, str] = {}
    submit_failures: list[str] = []  # positions whose submit raised — absent from results
    with ThreadPoolExecutor(max_workers=len(positions)) as pool:
        futures = {
            pool.submit(
                submit_tune_job,
                position=pos,
                n_trials=args.n_trials,
                timeout=args.timeout,
                seed=args.seed,
                n_jobs=args.n_jobs,
                parallel_backend=args.parallel_backend,
                cuda_graph=cuda_graph,
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
                submit_failures.append(pos)

    if not wait:
        print("\nJobs submitted. Use 'aws batch describe-jobs' to check status.")
        if submit_failures:
            print(f"ERROR: {len(submit_failures)} positions failed to submit: {submit_failures}")
            sys.exit(1)
        storage_version = resolve_search_space_version(
            _batch_storage_backend(args.parallel_backend), cuda_graph=cuda_graph
        )
        print(
            f"Results land at s3://{S3_BUCKET}/{s3_prefix(storage_version)}/"
            "{pos}/results.json per position."
        )
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
        storage_version = resolve_search_space_version(
            _batch_storage_backend(args.parallel_backend), cuda_graph=cuda_graph
        )
        print(
            f"  Per-position results: s3://{S3_BUCKET}/{s3_prefix(storage_version)}/"
            "{pos}/results.json"
        )
        print(
            f"  Per-position study DBs (resumable): s3://{S3_BUCKET}/"
            f"{s3_prefix(storage_version)}/{{pos}}/study.db"
        )
        print(
            "  Run `python -m src.tuning.aggregate_results "
            f"--search-space-version {storage_version}` to merge per-position JSONs."
        )

    print("\nAll done.")

    # Exit non-zero so the retune-nn-batch step surfaces failures instead of
    # false-greening — failed/timed-out positions, plus any that never
    # submitted (absent from results), mirror src/batch/launch.py.
    if failed or timed_out or submit_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
