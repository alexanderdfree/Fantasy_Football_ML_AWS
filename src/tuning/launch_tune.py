"""Launch parallel AWS Batch tuning jobs for the attention NN.

Mirrors ``src/batch/launch.py`` but submits ``--mode=tune`` jobs instead of
training jobs. Each position runs in its own Spot g6.xlarge/g5.xlarge
container so the six positions tune in parallel (the Spot G+VT quota is
64 vCPU since 2026-06-11 — a six-host tune fleet uses 24, leaving room for a
concurrent train fan-out; see the AWS-quota auto-memory).

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
signatures accepted a ``config=`` kwarg. The 64-vCPU Spot quota fits a
six-job tune fleet alongside a train fan-out; launches beyond the quota
still queue at ``RUNNABLE`` instead of pushing over-quota. The default
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
from src.tuning.ab_ensemble_seeds import (  # noqa: E402
    DEFAULT_STACKED_SEEDS,
    ENSEMBLE_POSITIONS,
)
from src.tuning.tune_nn_storage import (  # noqa: E402
    SCOPE_ROOTS,
    SEARCH_SPACE_VERSION,
    resolve_search_space_version,
    s3_prefix,
)

# tune_nn --scope values (mirror tune_nn._DEFAULT_SCOPE / _HISTORY_SCOPE). The
# history scope tunes the attention game-history branch (seq len + per-game
# token bundles, static backbone frozen) and rides FF_TUNE_SCOPE through the
# fixed ENTRYPOINT, exactly like FF_TUNE_STACKED_SEEDS.
SCOPE_FULL = "full"
SCOPE_HISTORY = "history"
HISTORY_POSITIONS = ("QB", "RB", "WR", "TE")

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
# Full-step capture (gather+fwd+loss in one graph, FF_CUDA_GRAPH_FULL) is
# default-ON for TUNE jobs only: tuning compares trials within one regime, so
# the wider capture is pure throughput; production training keeps the
# model-only capture until a per-position A/B clears the wider scope.
DEFAULT_CUDA_GRAPH_FULL = True
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


def _stacked_suffix(stacked_seeds: int, stacked_epochs: int) -> str:
    """Mirror of tune_nn.main's stacked namespace suffix (keep in sync)."""
    return f"_ens{stacked_seeds}x{stacked_epochs}" if stacked_seeds >= 2 else ""


def submit_tune_job(
    position: str,
    n_trials: int = DEFAULT_N_TRIALS,
    timeout: int | None = None,
    seed: int = 42,
    n_jobs: int | str = DEFAULT_N_JOBS,
    parallel_backend: str = DEFAULT_PARALLEL_BACKEND,
    cuda_graph: bool = DEFAULT_CUDA_GRAPH,
    cuda_graph_full: bool = DEFAULT_CUDA_GRAPH_FULL,
    stacked_seeds: int = 0,
    stacked_epochs: int = 30,
    attempt_timeout: int = DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
    scope: str = SCOPE_FULL,
    batch_client=None,
) -> tuple[str, str]:
    """Submit one Batch tuning job. Returns (position, job_id).

    The container's entrypoint is ``python src/batch/train.py``, which
    dispatches on ``--mode=tune`` into ``src/tuning/tune_nn``. We also
    propagate ``S3_BUCKET`` and ``S3_DATA_PREFIX`` so tune_nn's
    ``_ensure_data_from_s3`` can populate ``data/splits/`` and ``data/raw/``
    on first invocation (Spot resilience: study DB resumes via S3 if the
    job is retried after a Host EC2 interruption).

    Note ``stacked_seeds`` defaults to 0 (eager) HERE — the CLI default of
    ``DEFAULT_STACKED_SEEDS`` is applied by ``main()``, which under that
    default lets K/DST flow through to this function's per-position eager
    fallback instead of pre-rejecting them (GH #1439).

    ``stacked_seeds >= 2`` rides the FF_TUNE_STACKED_SEEDS env (train.py
    forwards a fixed argv, so flags can't reach tune_nn from here): each
    trial trains a vmap-stacked N-seed ensemble in the ensemble regime —
    graphs forced OFF in-container by ``apply_ensemble_env``, so the
    predicted namespace is the graph-less base + ``_ens{N}x{E}``.

    ``n_jobs`` rides the FF_TUNE_N_JOBS env for the same reason, never the
    command: train.py's ``--n-jobs`` is ``type=int``, so the "auto" sentinel
    dies at its argparse (exit 2 — Batch job 23b157e3, 2026-06-11). tune_nn
    reads the env as its ``--n-jobs`` default; an explicit ``--n-jobs``
    in a hand-built submission's command still wins over the env.
    """
    batch = batch_client or boto3.client("batch", region_name=AWS_REGION)
    timestamp = int(time.time())
    suffix = uuid.uuid4().hex[:6]
    if stacked_seeds == 1:
        raise SystemExit("--stacked-seeds needs N >= 2 (N=1 is just the eager objective)")
    if scope == SCOPE_HISTORY and position.upper() not in HISTORY_POSITIONS:
        raise SystemExit(
            f"--scope history supports {list(HISTORY_POSITIONS)} (flat-history "
            f"game-history branch); got {position}"
        )
    # K/DST aren't flat-history, so they can't vmap-stack — they run eager even
    # under the default-on stacking, and the container falls back the same way.
    # Resolve per-position HERE so the predicted namespace matches what runs.
    stacked_seeds = stacked_seeds if position.upper() in ENSEMBLE_POSITIONS else 0
    stacked = stacked_seeds >= 2

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
        # NO --n-jobs here: it goes through FF_TUNE_N_JOBS below (see docstring).
    ]
    if timeout is not None:
        command += ["--timeout", str(timeout)]

    storage_version = resolve_search_space_version(
        _batch_storage_backend(parallel_backend),
        # In-container, apply_ensemble_env forces the graphs off before the
        # namespace resolves — predict the same graph-less base here.
        cuda_graph=cuda_graph and not stacked,
        full_graph=cuda_graph_full and not stacked,
        # --scope history lands in the history_v2 root (separate study DB).
        root=SCOPE_ROOTS.get(scope, SEARCH_SPACE_VERSION),
    ) + _stacked_suffix(stacked_seeds, stacked_epochs)
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
                # Stacked mode forces both off (apply_ensemble_env would
                # override anyway; keep the env coherent with the prediction).
                {"name": "FF_CUDA_GRAPH", "value": "1" if (cuda_graph and not stacked) else "0"},
                {
                    "name": "FF_CUDA_GRAPH_FULL",
                    "value": "1" if (cuda_graph_full and not stacked) else "0",
                },
                {"name": "FF_AMP_DTYPE", "value": "auto"},
                {"name": "FF_COMPILE", "value": "0"},
                # Trial concurrency: env, not argv — the fixed ENTRYPOINT
                # (src/batch/train.py) parses --n-jobs as type=int, so the
                # "auto" sentinel can't ride the command (exit 2). tune_nn's
                # --n-jobs default reads this (same channel as
                # FF_TUNE_STACKED_SEEDS / FF_TUNE_AB_SPEC).
                {"name": "FF_TUNE_N_JOBS", "value": str(n_jobs)},
                {"name": "TUNE_NN_STORAGE_VERSION", "value": storage_version},
                # Quieter than training (default LOG_EVERY=1). Tuning runs N
                # trials × ~200 epochs each — 1 line per epoch would flood
                # CloudWatch with ~6000 lines/trial × 30 trials = 180k lines.
                {"name": "LOG_EVERY", "value": "20"},
                *(
                    [
                        {"name": "FF_TUNE_STACKED_SEEDS", "value": str(stacked_seeds)},
                        {"name": "FF_TUNE_STACKED_EPOCHS", "value": str(stacked_epochs)},
                    ]
                    if stacked
                    else []
                ),
                # --scope history rides FF_TUNE_SCOPE through the fixed ENTRYPOINT
                # (train.py forwards a fixed argv). Only emitted off-default so a
                # full-scope submission's env stays byte-identical to before.
                *([{"name": "FF_TUNE_SCOPE", "value": scope}] if scope != SCOPE_FULL else []),
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
    n_jobs: int | str,
    parallel_backend: str,
    cuda_graph: bool,
    cuda_graph_full: bool,
    attempt_timeout: int,
    stacked_seeds: int = 0,
    stacked_epochs: int = 30,
    scope: str = SCOPE_FULL,
) -> None:
    stacked = stacked_seeds >= 2
    storage_version = resolve_search_space_version(
        _batch_storage_backend(parallel_backend),
        cuda_graph=cuda_graph and not stacked,
        full_graph=cuda_graph_full and not stacked,
        root=SCOPE_ROOTS.get(scope, SEARCH_SPACE_VERSION),
    ) + _stacked_suffix(stacked_seeds, stacked_epochs)
    print("DRY RUN — no AWS calls will be made.")
    print(f"  region:       {AWS_REGION}")
    print(f"  bucket:       {S3_BUCKET}")
    print(f"  queue:        {JOB_QUEUE}")
    # Render the resolved (revision-pinned) job def actually submitted, not
    # the bare name — mirrors launch.py's _print_plan.
    print(f"  definition:   {_tune_job_definition()}")
    print(f"  wait timeout: {WAIT_TIMEOUT_SECONDS}s")
    print(f"  attempt cap:  {attempt_timeout}s")
    print(f"  scope:        {scope}")
    print(f"  n_trials:     {n_trials}")
    print(f"  n_jobs:       {n_jobs}")
    print(f"  backend:      {parallel_backend}")
    print(f"  cuda graph:   {cuda_graph}")
    print(f"  graph full:   {cuda_graph_full}")
    print(
        f"  stacked:      {f'{stacked_seeds} seeds x {stacked_epochs} epochs' if stacked else 'off'}"
    )
    print(f"  storage:      {storage_version}")
    print(f"  timeout:      {timeout if timeout is not None else 'no cap'}")
    print(f"  seed:         {seed}")
    print("  jobs:")
    for pos in positions:
        # n_jobs rides the job environment, not the command — train.py's
        # --n-jobs is type=int and would reject the "auto" sentinel. K/DST
        # can't vmap-stack, so submit_tune_job resolves them to eager even
        # under a default-on stacked run — surface that per-position here.
        pos_stacked = stacked_seeds if (stacked and pos.upper() in ENSEMBLE_POSITIONS) else 0
        stacked_note = f" stacked={pos_stacked}x{stacked_epochs}" if pos_stacked >= 2 else " eager"
        print(
            f"    - {pos:<4} -> --mode=tune --n-trials={n_trials} "
            f"--parallel-backend={parallel_backend} FF_TUNE_N_JOBS={n_jobs}{stacked_note}"
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
        "--cuda-graph-full",
        default="true" if DEFAULT_CUDA_GRAPH_FULL else "false",
        help=(
            "Enable FF_CUDA_GRAPH_FULL (full-step capture: gather+forward+loss "
            "in one graph) inside tune jobs (true/false; default true). "
            "Requires --cuda-graph true; studies land in the *_graphfull "
            "namespaces."
        ),
    )
    parser.add_argument(
        "--stacked-seeds",
        type=int,
        default=None,
        help=(
            "N >= 2: each trial trains a vmap-stacked N-seed ensemble "
            "(seed-averaged objective; rides FF_TUNE_STACKED_SEEDS through the "
            "fixed ENTRYPOINT; QB/RB/WR/TE only — K/DST run eager; studies land "
            f"in _ens{{N}}x{{E}} namespaces with graphs forced off). DEFAULT is "
            f"{DEFAULT_STACKED_SEEDS} (the measured per-seed optimum on the Batch "
            "GPU fleet), under which any selected K/DST resolve to eager instead "
            "of being rejected; an EXPLICIT --stacked-seeds>=2 with K/DST "
            "selected is rejected. Pass 0 for eager trials."
        ),
    )
    parser.add_argument(
        "--stacked-epochs",
        type=int,
        default=30,
        help="Fixed epochs per stacked trial (default 30).",
    )
    parser.add_argument(
        "--scope",
        choices=[SCOPE_FULL, SCOPE_HISTORY],
        default=SCOPE_FULL,
        help=(
            "tune_nn search-space scope. 'full' (default) tunes attention sizing "
            "+ static backbone + scheduler. 'history' (v2 isolation) tunes ONLY "
            "the attention game-history branch (attn_max_seq_len + per-game token "
            "bundles), freezing the entire production recipe; rides FF_TUNE_SCOPE "
            "into the container, lands in the history_v2 namespace, QB/RB/WR/TE only."
        ),
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
    cuda_graph_full = args.cuda_graph_full.strip().lower() in {"1", "true", "yes", "on"}
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
    # Distinguish an EXPLICIT --stacked-seeds from the default: argparse leaves
    # None when the flag is omitted, which we resolve to DEFAULT_STACKED_SEEDS.
    # Under the default, a selected K/DST is NOT pre-rejected here — it flows
    # through to submit_tune_job's per-position eager fallback (matching the
    # documented all-six invocation, GH #1439). An explicit --stacked-seeds>=2
    # with K/DST selected is still rejected.
    stacked_explicit = args.stacked_seeds is not None
    stacked_seeds = max(0, int(args.stacked_seeds if stacked_explicit else DEFAULT_STACKED_SEEDS))
    if stacked_seeds == 1:
        raise SystemExit("--stacked-seeds needs N >= 2 (N=1 is just the eager objective)")
    if stacked_explicit and stacked_seeds:
        bad = [p for p in positions if p not in ENSEMBLE_POSITIONS]
        if bad:
            raise SystemExit(f"--stacked-seeds supports QB/RB/WR/TE (flat-history); got {bad}")
    if args.scope == SCOPE_HISTORY:
        bad = [p for p in positions if p not in HISTORY_POSITIONS]
        if bad:
            raise SystemExit(
                f"--scope history supports {list(HISTORY_POSITIONS)} (flat-history "
                f"game-history branch); got {bad}"
            )

    if args.dry_run:
        _print_plan(
            positions,
            args.n_trials,
            args.timeout,
            args.seed,
            args.n_jobs,
            args.parallel_backend,
            cuda_graph,
            cuda_graph_full,
            args.attempt_timeout,
            stacked_seeds=stacked_seeds,
            stacked_epochs=args.stacked_epochs,
            scope=args.scope,
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
                cuda_graph_full=cuda_graph_full,
                stacked_seeds=stacked_seeds,
                stacked_epochs=args.stacked_epochs,
                attempt_timeout=args.attempt_timeout,
                scope=args.scope,
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
            _batch_storage_backend(args.parallel_backend),
            cuda_graph=cuda_graph and not stacked_seeds,
            full_graph=cuda_graph_full and not stacked_seeds,
            root=SCOPE_ROOTS.get(args.scope, SEARCH_SPACE_VERSION),
        ) + _stacked_suffix(stacked_seeds, args.stacked_epochs)
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
            _batch_storage_backend(args.parallel_backend),
            cuda_graph=cuda_graph and not stacked_seeds,
            full_graph=cuda_graph_full and not stacked_seeds,
            root=SCOPE_ROOTS.get(args.scope, SEARCH_SPACE_VERSION),
        ) + _stacked_suffix(stacked_seeds, args.stacked_epochs)
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
