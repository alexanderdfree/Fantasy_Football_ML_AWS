"""Launch an eager ``ablation_runner`` ablation (e.g. ``ablate_attn_arch``) on AWS Batch GPU Spot.

Sibling of [launch_ab.py](launch_ab.py) for the eager config-injection ablations
([ablation_runner.py](ablation_runner.py)) rather than the frame-injection
[ab_harness.py](ab_harness.py) specs. Runs the ablation on the production metric
path — g6.xlarge/L4 (g5/A10G fallback) Spot hosts, FP16 AMP — instead of a local
box, for the arms that only make sense eagerly. The motivating case is
``ablate_attn_arch``'s ``selfattn`` arm: its ``nn.MultiheadAttention`` SDPA
``attn_bias`` errors under ``torch.func.vmap``, so it was dropped from the
stacked ``ab_attn_arch`` spec — this launcher is its GPU home (the "confirm once
eagerly, then delete the dead code" Tier-4 prior in
``todo/run_attn_arch_harness_priority.md``).

Shape (mirrors launch_ab / ADR-0020): each per-position job rides the
``--mode=tune`` env dispatch (``FF_TUNE_ABLATE_MOD`` routes ``tune_nn.main`` into
[ablate_batch.py](ablate_batch.py) — no ``src/batch/`` edit, no retrain trigger,
and the env channel is the only way to run *branch* code on Batch) and runs its
variant × seed cells sequentially in-container with per-cell S3 checkpointing, so
a Spot reclaim resumes instead of restarting. This launcher then collects the
per-cell JSONs and feeds them through the ablation module's own
``print_summary()`` — same decision table and Ridge data-identity sentinel as a
local run.

``--cuda-graph false`` is the **default** (unlike launch_ab's ``auto``): CUDA
graphs autodetect ON for sm_80+ but are NOT numerically inert (ADR-0017), and
``selfattn``'s fused-SDPA path under graph capture is unproven — an eager FP16
A/B is the clean, bit-comparable confirmation the eager ``ablate_attn_arch``
recipe describes. Pass ``--cuda-graph auto`` to ride the production graphed path.

Images: like launch_ab, this clones the production GPU job definition
(``ff-training-job``) into the shared **``ff-ab-job``** definition with the image
re-pointed at ``ff-training:{--image-sha}``. For an **unmerged branch**, first
dispatch batch-image.yml on that branch (``gh workflow run batch-image.yml --ref
<branch>``; non-main builds push only the SHA tag), then run this launcher with
``--image-sha <branch head SHA>``.

Usage:
    python -m src.tuning.launch_ablate --mod src.tuning.ablate_attn_arch --dry-run
    python -m src.tuning.launch_ablate --mod src.tuning.ablate_attn_arch \
        --positions RB --only selfattn --seeds 42 7 123 5 99 17 31 8
    python -m src.tuning.launch_ablate --mod ... --image-sha <sha>          # branch image
    python -m src.tuning.launch_ablate --mod ... --collect-only --run-id <id>

Results land under ``s3://$FF_S3_BUCKET/ablation_runs/{run_id}/`` (``run.json``
manifest, ``cells/*.json``, ``summary.json``). Nothing is written to ``models/``
or ``benchmark_history/``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.batch.launch import (  # noqa: E402
    AWS_REGION,
    JOB_QUEUE,
    POLL_INTERVAL_SECONDS,
    RETRY_STRATEGY,
    S3_BUCKET,
    WAIT_TIMEOUT_SECONDS,
    wait_for_jobs,
)
from src.tuning.ablate_batch import (  # noqa: E402
    DEFAULT_S3_PREFIX,
    ENV_MOD,
    ENV_RUN_ID,
    ENV_S3_PREFIX,
    ENV_SEEDS,
    ENV_VARIANTS,
    cell_key,
    cell_result_key,
    load_ablation_module,
)
from src.tuning.ablation_runner import AblationResult  # noqa: E402

# Reuse launch_ab's job-definition cloning + manifest plumbing verbatim — the
# image-swap / ff-ab-job-clone story is identical for both Batch ablation paths.
from src.tuning.launch_ab import (  # noqa: E402
    DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
    DEFAULT_MAX_CELLS,
    _git_head_sha,
    _run_prefix,
    check_image_exists,
    load_run_manifest,
    resolve_job_definition,
    write_run_manifest,
)

# Project floor for small NN deltas (single-seed overall-MAE is noise); the eager
# runner has no Optuna-style default, so default to the 8-seed set.
DEFAULT_SEEDS = [42, 7, 123, 5, 99, 17, 31, 8]


def _selected_variants(module, only: list[str] | None) -> list[str]:
    """Baseline + the requested variant subset (all variants when --only is
    absent). Validated against the module's VARIANTS so a typo fails locally."""
    baseline = module.BASELINE
    if not only:
        return [baseline, *(k for k in module.VARIANTS if k != baseline)]
    unknown = [v for v in only if v not in module.VARIANTS]
    if unknown:
        raise SystemExit(
            f"--only has unknown variant(s) {unknown}; choose from {sorted(module.VARIANTS)}"
        )
    return [baseline, *(v for v in only if v != baseline)]


def submit_ablate_job(
    position: str,
    *,
    mod_dotted: str,
    run_id: str,
    s3_prefix: str,
    job_definition: str,
    image_sha: str,
    seeds: list[int],
    only: list[str] | None,
    cuda_graph: str,
    attempt_timeout: int = DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
    batch_client=None,
) -> tuple[str, str]:
    """Submit one per-position eager-ablation job. Returns (position, job_id)."""
    import boto3

    batch = batch_client or boto3.client("batch", region_name=AWS_REGION)
    environment = [
        {"name": "S3_BUCKET", "value": S3_BUCKET},
        {"name": "S3_DATA_PREFIX", "value": "data"},
        {"name": "FF_DEVICE", "value": "cuda"},
        {"name": ENV_MOD, "value": mod_dotted},
        {"name": ENV_RUN_ID, "value": run_id},
        {"name": ENV_S3_PREFIX, "value": s3_prefix},
        {"name": ENV_SEEDS, "value": ",".join(str(s) for s in seeds)},
        # Stamped into each cell's provenance so the report records which image
        # actually ran (mirrors launch_ab / launch.py's FF_TRAIN_GIT_SHA).
        {"name": "FF_TRAIN_GIT_SHA", "value": image_sha},
        # Cells log per-epoch NN lines; 20x decimation keeps CloudWatch sane.
        {"name": "LOG_EVERY", "value": "20"},
        # AMP stays at the production default (FP16 on the L4); only the graph
        # knob below diverges, so the eager A/B keeps the production dtype.
        {"name": "FF_AMP_DTYPE", "value": "auto"},
        {"name": "FF_COMPILE", "value": "0"},
    ]
    if only:
        environment.append({"name": ENV_VARIANTS, "value": ",".join(only)})
    if cuda_graph != "auto":
        # auto = forward nothing (container autodetects ON for sm_80+, the
        # production graphed path). "false" forwards the force-off override for
        # an eager, bit-comparable A/B (the default for this launcher).
        environment.append({"name": "FF_CUDA_GRAPH", "value": "1" if cuda_graph == "true" else "0"})

    response = batch.submit_job(
        jobName=f"ff-ablate-{position.lower()}-{int(time.time())}-{uuid.uuid4().hex[:6]}",
        jobQueue=JOB_QUEUE,
        jobDefinition=job_definition,
        retryStrategy=RETRY_STRATEGY,
        timeout={"attemptDurationSeconds": int(attempt_timeout)},
        containerOverrides={
            # train.py's --mode=tune path; tune_nn.main dispatches on the env.
            "command": ["--position", position, "--mode", "tune"],
            "environment": environment,
        },
    )
    job_id = response["jobId"]
    print(f"[{position}] Submitted eager-ablation job {job_id} (definition: {job_definition})")
    return position, job_id


def collect_results(
    *,
    positions: list[str],
    variants: list[str],
    seeds: list[int],
    bucket,
    s3_prefix,
    run_id,
    s3_client,
) -> dict[str, list[AblationResult]]:
    """Download every expected cell JSON into a per-position AblationResult list;
    a missing one becomes an error result so the report surfaces the gap instead
    of silently shrinking the grid."""
    by_pos: dict[str, list[AblationResult]] = {}
    for pos in positions:
        rows: list[AblationResult] = []
        for variant in variants:
            for seed in seeds:
                key = cell_result_key(s3_prefix, run_id, cell_key(pos, variant, seed))
                try:
                    obj = s3_client.get_object(Bucket=bucket, Key=key)
                    d = json.loads(obj["Body"].read())
                    rows.append(
                        AblationResult(
                            position=d["position"],
                            seed=d["seed"],
                            variant=d["variant"],
                            metrics=d.get("metrics") or {},
                            timings=d.get("timings") or {},
                            metadata=d.get("metadata") or {},
                            error=d.get("error"),
                        )
                    )
                except Exception as e:  # noqa: BLE001 — record the gap, keep collecting
                    rows.append(
                        AblationResult(
                            position=pos,
                            seed=seed,
                            variant=variant,
                            metrics={},
                            timings={},
                            metadata={},
                            error=f"no result at s3://{bucket}/{key} ({type(e).__name__})",
                        )
                    )
        by_pos[pos] = rows
    return by_pos


def _report(module, by_pos: dict[str, list[AblationResult]], variants: list[str]) -> int:
    """Print each position's decision table via the module's own print_summary
    (3-arg attn_arch shape, 2-arg backbone_norm shape — detected). Returns the
    number of cells that errored across all positions."""
    from src.shared.registry import get_config

    print_summary = module.print_summary
    n_params = len(inspect.signature(print_summary).parameters)
    n_failed = 0
    for pos, rows in by_pos.items():
        n_failed += sum(1 for r in rows if r.error is not None)
        targets = get_config(pos)["targets"]
        print(f"\n########## {pos} ({module.ABLATION_NAME}) ##########")
        if n_params >= 3:
            print_summary(rows, targets, variants)
        else:
            print_summary(rows, targets)
    return n_failed


def _default_run_id(mod_dotted: str, image_sha: str) -> str:
    short = mod_dotted.rsplit(".", 1)[-1]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{short}-{ts}-{image_sha[:7]}"


def _print_plan(
    *, module, mod_dotted, positions, variants, seeds, run_id, image_sha, args, cells
) -> None:
    print("DRY RUN — no AWS calls will be made.")
    print(f"  region:        {AWS_REGION}")
    print(f"  bucket:        {S3_BUCKET}")
    print(f"  queue:         {JOB_QUEUE}")
    print(f"  job def:       ff-ab-job (cloned, image ff-training:{image_sha})")
    print(f"  module:        {mod_dotted}  baseline={module.BASELINE}")
    print(f"  variants:      {variants}")
    print(f"  positions:     {positions}")
    print(f"  seeds:         {seeds}")
    print(f"  cells:         {cells} ({len(variants)} variants x {len(seeds)} seeds per position)")
    print(f"  run id:        {run_id}")
    print(f"  results:       s3://{S3_BUCKET}/{_run_prefix(args.s3_prefix, run_id)}/")
    print(f"  cuda graph:    {args.cuda_graph} (false = eager bit-comparable A/B)")
    print(f"  attempt cap:   {args.attempt_timeout}s")
    print("  jobs:")
    for pos in positions:
        print(f"    - {pos:<4} -> {len(variants) * len(seeds)} sequential cells, --mode tune")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Launch an eager ablation_runner ablation on AWS Batch")
    p.add_argument(
        "--mod",
        required=True,
        help="Dotted ablation module (e.g. src.tuning.ablate_attn_arch). Must expose "
        "VARIANTS, BASELINE, _build_jobs and print_summary.",
    )
    p.add_argument("--positions", nargs="+", default=["RB"], help="Positions (default: RB)")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Seeds (default: {DEFAULT_SEEDS})",
    )
    p.add_argument("--only", nargs="+", help="Run only these variants (baseline always kept)")
    p.add_argument(
        "--image-sha",
        default=None,
        help="ff-training image tag to run (default: local git HEAD). For an unmerged "
        "branch, dispatch batch-image.yml on that branch first.",
    )
    p.add_argument("--run-id", default=None, help="Run namespace (default: {mod}-{utc}-{sha7})")
    p.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX, help=f"(default: {DEFAULT_S3_PREFIX})")
    p.add_argument(
        "--cuda-graph",
        choices=["auto", "true", "false"],
        default="false",
        help="false (default) forces eager for a bit-comparable A/B; auto leaves the "
        "container's sm_80+ autodetect ON (production graphed path).",
    )
    p.add_argument(
        "--attempt-timeout",
        type=int,
        default=DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
        help=f"AWS Batch attemptDurationSeconds per position job (default {DEFAULT_ATTEMPT_TIMEOUT_SECONDS})",
    )
    p.add_argument(
        "--max-cells",
        type=int,
        default=DEFAULT_MAX_CELLS,
        help=f"Refuse grids larger than this (cost guard, default {DEFAULT_MAX_CELLS})",
    )
    p.add_argument("--wait", default="true", help="Wait for jobs + aggregate (true/false)")
    p.add_argument(
        "--wait-timeout", type=int, default=None, help=f"(default: {WAIT_TIMEOUT_SECONDS})"
    )
    p.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip the ECR image-existence preflight (e.g. no ecr:DescribeImages)",
    )
    p.add_argument(
        "--collect-only",
        action="store_true",
        help="Skip submission; collect + report an existing --run-id from S3",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the plan, touch nothing")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    wait = args.wait.lower() == "true"
    wait_timeout = args.wait_timeout if args.wait_timeout is not None else WAIT_TIMEOUT_SECONDS

    import boto3

    s3 = None
    if args.collect_only:
        if not args.run_id:
            raise SystemExit("--collect-only requires --run-id")
        if not args.dry_run:
            s3 = boto3.client("s3", region_name=AWS_REGION)
            manifest = load_run_manifest(
                s3, bucket=S3_BUCKET, s3_prefix=args.s3_prefix, run_id=args.run_id
            )
            if manifest:
                # The manifest is authoritative for the grid (this invocation may
                # have used different overrides than the submitting one).
                args.mod = manifest.get("mod") or args.mod
                args.positions = manifest.get("positions") or args.positions
                args.seeds = manifest.get("seeds") or args.seeds
                args.only = manifest.get("only") or args.only

    module = load_ablation_module(args.mod)
    if not hasattr(module, "print_summary"):
        raise SystemExit(f"--mod {args.mod} has no print_summary; cannot aggregate on the launcher")
    positions = [p.upper() for p in args.positions]
    variants = _selected_variants(module, args.only)
    seeds = list(args.seeds)
    n_cells = len(positions) * len(variants) * len(seeds)
    if n_cells > args.max_cells:
        raise SystemExit(
            f"grid is {n_cells} cells, over the --max-cells {args.max_cells} cost guard "
            "(typo'd --seeds/--positions? raise --max-cells consciously if intended)"
        )

    image_sha = args.image_sha or _git_head_sha()
    if not image_sha and not args.collect_only:
        raise SystemExit("--image-sha is required when git HEAD cannot be resolved")
    run_id = args.run_id or _default_run_id(args.mod, image_sha or "nosha")

    if args.dry_run:
        _print_plan(
            module=module,
            mod_dotted=args.mod,
            positions=positions,
            variants=variants,
            seeds=seeds,
            run_id=run_id,
            image_sha=image_sha,
            args=args,
            cells=n_cells,
        )
        return

    if args.collect_only:
        by_pos = collect_results(
            positions=positions,
            variants=variants,
            seeds=seeds,
            bucket=S3_BUCKET,
            s3_prefix=args.s3_prefix,
            run_id=run_id,
            s3_client=s3,
        )
        n_failed = _report(module, by_pos, variants)
        sys.exit(1 if n_failed else 0)

    batch = boto3.client("batch", region_name=AWS_REGION)
    s3 = boto3.client("s3", region_name=AWS_REGION)

    if not args.skip_image_check:
        exists = check_image_exists(image_sha, boto3.client("ecr", region_name=AWS_REGION))
        if exists is False:
            raise SystemExit(
                f"image ff-training:{image_sha} not in ECR. For an unmerged branch, build it "
                "first: gh workflow run batch-image.yml --ref <branch> (then re-run). "
                "--skip-image-check bypasses this preflight."
            )

    job_definition = resolve_job_definition(image_sha, batch)

    print(f"Submitting {len(positions)} eager-ablation jobs (run_id={run_id}): {positions}")
    job_ids: dict[str, str] = {}
    submit_failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(positions)) as pool:
        futures = {
            pool.submit(
                submit_ablate_job,
                position=pos,
                mod_dotted=args.mod,
                run_id=run_id,
                s3_prefix=args.s3_prefix,
                job_definition=job_definition,
                image_sha=image_sha,
                seeds=seeds,
                only=args.only,
                cuda_graph=args.cuda_graph,
                attempt_timeout=args.attempt_timeout,
                batch_client=batch,
            ): pos
            for pos in positions
        }
        for future in as_completed(futures):
            pos = futures[future]
            try:
                pos, job_id = future.result()
                job_ids[pos] = job_id
            except Exception as e:  # noqa: BLE001 — surface per-position submit failures
                print(f"[{pos}] FAILED to submit: {e}")
                submit_failures.append(pos)

    write_run_manifest(
        s3,
        bucket=S3_BUCKET,
        s3_prefix=args.s3_prefix,
        run_id=run_id,
        manifest={
            "schema_version": 1,
            "mod": args.mod,
            "positions": positions,
            "seeds": seeds,
            "only": args.only,
            "variants": variants,
            "baseline": module.BASELINE,
            "image_sha": image_sha,
            "job_definition": job_definition,
            "cuda_graph": args.cuda_graph,
            "jobs": job_ids,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )

    if not wait:
        print("\nJobs submitted. Collect later with:")
        print(
            f"  python -m src.tuning.launch_ablate --mod {args.mod} --collect-only --run-id {run_id}"
        )
        if submit_failures:
            print(f"ERROR: {len(submit_failures)} positions failed to submit: {submit_failures}")
            sys.exit(1)
        return

    print(
        f"\nWaiting for {len(job_ids)} jobs (polling every {POLL_INTERVAL_SECONDS}s, "
        f"timeout {wait_timeout}s)..."
    )
    statuses = wait_for_jobs(job_ids, timeout_seconds=wait_timeout, batch_client=batch)
    failed = [p for p, (status, _) in statuses.items() if status == "FAILED"]
    timed_out = [p for p, (status, _) in statuses.items() if status == "TIMED_OUT"]
    if failed:
        print(f"\nFailed positions: {failed}")
    if timed_out:
        print(f"\nTimed-out positions: {timed_out}")

    by_pos = collect_results(
        positions=positions,
        variants=variants,
        seeds=seeds,
        bucket=S3_BUCKET,
        s3_prefix=args.s3_prefix,
        run_id=run_id,
        s3_client=s3,
    )
    n_failed_cells = _report(module, by_pos, variants)

    print("\nAll done.")
    if failed or timed_out or submit_failures or n_failed_cells:
        sys.exit(1)


if __name__ == "__main__":
    main()
