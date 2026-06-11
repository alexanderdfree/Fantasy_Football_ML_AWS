"""Launch a shared-harness A/B (src.tuning.ab_harness spec) on AWS Batch GPU Spot.

Runs model A/Bs on the production metric path — g6.xlarge/L4 (g5/A10G
fallback) Spot hosts, FP16 AMP, CUDA graphs autodetect-ON — instead of a
local box. Mirrors [launch_tune.py](launch_tune.py): one Batch job per
position, ``wait_for_jobs`` from ``src.batch.launch``, revisionless quota
math (a six-position A/B uses 24 of the 64 Spot G+VT vCPUs, so it coexists
with a concurrent train fan-out).

Shape (owner-approved 2026-06-11; see docs/adr/0020): each per-position job
rides the ``--mode=tune`` env dispatch (``FF_TUNE_AB_SPEC`` routes
``tune_nn.main`` into [ab_batch.py](ab_batch.py), the #1142 pattern — no
``src/batch/`` edit, no retrain trigger) and runs its variant × seed cells
sequentially in-container with per-cell S3 checkpointing, so a Spot reclaim
resumes instead of restarting. This launcher then collects the per-cell
JSONs and feeds them through the harness's own ``aggregate()`` /
``print_report()`` — same tables, same Ridge-invariance sentinel as a local
run.

Images: ``containerOverrides`` cannot swap a job definition's image, so this
launcher clones the production GPU job definition (``ff-training-job``) into
a separate **``ff-ab-job``** definition with the image re-pointed at
``ff-training:{--image-sha}`` — production job-definition names never point
at branch images. For an **unmerged branch**, first dispatch batch-image.yml
on that branch (``gh workflow run batch-image.yml --ref <branch>``; non-main
builds push only the SHA tag and skip job-definition registration), then run
this launcher with ``--image-sha <branch head SHA>``. A main-SHA image needs
no extra step.

Usage:
    python -m src.tuning.launch_ab --spec src.tuning.ab_example --dry-run
    python -m src.tuning.launch_ab --spec src.tuning.ab_example                # spec grid
    python -m src.tuning.launch_ab --spec ... --positions RB WR --seeds 42 123
    python -m src.tuning.launch_ab --spec ... --image-sha <sha>                # branch image
    python -m src.tuning.launch_ab --spec ... --wait false                     # fire and forget
    python -m src.tuning.launch_ab --spec ... --collect-only --run-id <id>     # re-aggregate

Results land under ``s3://$FF_S3_BUCKET/ab_runs/{run_id}/`` (``run.json``
manifest, ``cells/*.json``, ``summary.json``). Nothing is written to
``models/`` or ``benchmark_history/``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

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
from src.tuning.ab_batch import (  # noqa: E402
    DEFAULT_S3_PREFIX,
    ENV_ONLY,
    ENV_RUN_ID,
    ENV_S3_PREFIX,
    ENV_SEEDS,
    ENV_SPEC,
    cell_result_key,
)
from src.tuning.ab_harness import aggregate, build_cells, print_report, resolve_spec  # noqa: E402

# Separate job-definition NAME so branch images never become a revision of the
# production ff-training-job (bare-name submissions resolve to the latest
# active revision; see src/batch/launch.py's revision-pinning rationale).
AB_JOB_DEFINITION = os.environ.get("FF_AB_JOB_DEFINITION", "") or "ff-ab-job"
# 3h attempt cap: a 10-cell position slice at ~3-8 min/cell is ~30-80 min;
# the headroom covers a slow Spot start without letting a wedged job pin a
# host all day. Submit-time override — the cloned definition's baked 1800s
# training timeout would kill multi-cell jobs.
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 10800
# Cost-ceiling sanity guard, the launcher-side analog of retune-nn-batch.yml's
# 100-trial ceiling: 120 cells x ~5 min on g6 Spot is ~$3; past that is
# usually a typo'd --seeds/--positions. Raise consciously via --max-cells.
DEFAULT_MAX_CELLS = 120


def _git_head_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _swap_image_tag(image_uri: str, sha: str) -> str:
    """``...amazonaws.com/ff-training:old`` -> ``...amazonaws.com/ff-training:{sha}``."""
    base, _, _ = image_uri.rpartition(":")
    if not base:
        raise ValueError(f"job-definition image has no tag to swap: {image_uri!r}")
    return f"{base}:{sha}"


def _describe_latest_active(batch, name: str) -> dict | None:
    resp = batch.describe_job_definitions(jobDefinitionName=name, status="ACTIVE", maxResults=1)
    defs = resp.get("jobDefinitions") or []
    return defs[0] if defs else None


def resolve_job_definition(image_sha: str, batch_client) -> str:
    """Return ``ff-ab-job:{revision}`` whose image is ``ff-training:{image_sha}``.

    Clones the production GPU job definition (resource requirements, env caps,
    role ARNs) with only the image tag swapped, registering a new ``ff-ab-job``
    revision when the latest active one doesn't already point at that image.
    Mirrors batch-image.yml's jq re-registration, minus the baked training
    ``timeout`` — A/B jobs set their attempt timeout at submit time.
    """
    template = _describe_latest_active(batch_client, JOB_DEFINITION)
    if template is None:
        raise RuntimeError(f"no ACTIVE {JOB_DEFINITION} job definition to clone")
    desired_image = _swap_image_tag(template["containerProperties"]["image"], image_sha)

    existing = _describe_latest_active(batch_client, AB_JOB_DEFINITION)
    if existing is not None and existing["containerProperties"].get("image") == desired_image:
        resolved = f"{AB_JOB_DEFINITION}:{existing['revision']}"
        print(f"[ab] reusing job definition {resolved} (image already {desired_image})")
        return resolved

    container = copy.deepcopy(template["containerProperties"])
    container["image"] = desired_image
    register_kwargs = {
        "jobDefinitionName": AB_JOB_DEFINITION,
        "type": template["type"],
        "containerProperties": container,
        "retryStrategy": RETRY_STRATEGY,
    }
    for key in ("platformCapabilities", "propagateTags", "tags"):
        value = template.get(key)
        if value:
            register_kwargs[key] = value
    revision = batch_client.register_job_definition(**register_kwargs)["revision"]
    resolved = f"{AB_JOB_DEFINITION}:{revision}"
    print(f"[ab] registered {resolved} -> {desired_image}")
    return resolved


def check_image_exists(image_sha: str, ecr_client) -> bool | None:
    """True/False when ECR answers; None when we can't check (e.g. no
    ecr:DescribeImages permission) — the caller degrades to a warning."""
    try:
        ecr_client.describe_images(repositoryName="ff-training", imageIds=[{"imageTag": image_sha}])
        return True
    except ecr_client.exceptions.ImageNotFoundException:
        return False
    except Exception as e:  # noqa: BLE001 — permission/transport issues: can't check
        print(f"[ab] WARNING: could not verify image tag {image_sha} in ECR: {e!r}")
        return None


def submit_ab_job(
    position: str,
    *,
    spec_dotted: str,
    run_id: str,
    s3_prefix: str,
    job_definition: str,
    image_sha: str,
    seeds: list[int] | None,
    only: list[str] | None,
    cuda_graph: str = "auto",
    feature_cache: bool = False,
    attempt_timeout: int = DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
    batch_client=None,
) -> tuple[str, str]:
    """Submit one per-position A/B job. Returns (position, job_id)."""
    import boto3

    batch = batch_client or boto3.client("batch", region_name=AWS_REGION)
    environment = [
        {"name": "S3_BUCKET", "value": S3_BUCKET},
        {"name": "S3_DATA_PREFIX", "value": "data"},
        {"name": "FF_DEVICE", "value": "cuda"},
        {"name": ENV_SPEC, "value": spec_dotted},
        {"name": ENV_RUN_ID, "value": run_id},
        {"name": ENV_S3_PREFIX, "value": s3_prefix},
        # Stamped into each cell's provenance so the report records which
        # image actually ran (mirrors launch.py's FF_TRAIN_GIT_SHA).
        {"name": "FF_TRAIN_GIT_SHA", "value": image_sha},
        # Cells log per-epoch NN lines; 20x decimation keeps CloudWatch sane
        # across a multi-cell job (same rationale as launch_tune).
        {"name": "LOG_EVERY", "value": "20"},
    ]
    if seeds:
        environment.append({"name": ENV_SEEDS, "value": ",".join(str(s) for s in seeds)})
    if only:
        environment.append({"name": ENV_ONLY, "value": ",".join(only)})
    if cuda_graph != "auto":
        # auto = forward nothing: cuda_graph_enabled() autodetects ON for the
        # sm_80+ fleet, i.e. the production graphed metric path. "false"
        # forwards the force-off override for a bit-comparable eager A/B.
        environment.append({"name": "FF_CUDA_GRAPH", "value": "1" if cuda_graph == "true" else "0"})
    if feature_cache:
        environment.append({"name": "FF_FEATURE_CACHE_DISABLE", "value": "0"})

    response = batch.submit_job(
        jobName=f"ff-ab-{position.lower()}-{int(time.time())}-{uuid.uuid4().hex[:6]}",
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
    print(f"[{position}] Submitted A/B job {job_id} (definition: {job_definition})")
    return position, job_id


def _run_prefix(s3_prefix: str, run_id: str) -> str:
    return f"{s3_prefix.strip('/')}/{run_id}"


def write_run_manifest(s3, *, bucket: str, s3_prefix: str, run_id: str, manifest: dict) -> str:
    key = f"{_run_prefix(s3_prefix, run_id)}/run.json"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(manifest, indent=2).encode(),
        ContentType="application/json",
    )
    return key


def load_run_manifest(s3, *, bucket: str, s3_prefix: str, run_id: str) -> dict | None:
    key = f"{_run_prefix(s3_prefix, run_id)}/run.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except Exception:  # noqa: BLE001 — missing manifest degrades to CLI args
        return None
    return json.loads(obj["Body"].read())


def collect_results(spec, *, bucket: str, s3_prefix: str, run_id: str, s3_client) -> list[dict]:
    """Download every expected cell JSON; a missing one becomes a not-ok row so
    ``aggregate`` surfaces it instead of silently shrinking the grid."""
    results: list[dict] = []
    for cell in build_cells(spec):
        key = cell_result_key(s3_prefix, run_id, cell.key)
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            results.append(json.loads(obj["Body"].read()))
        except Exception as e:  # noqa: BLE001 — record the gap, keep collecting
            results.append(
                {
                    "position": cell.position,
                    "variant": cell.variant,
                    "seed": cell.seed,
                    "label": cell.variant,
                    "ok": False,
                    "metrics": {},
                    "ridge_mae": None,
                    "error": f"no result at s3://{bucket}/{key} ({type(e).__name__})",
                }
            )
    return results


def _print_plan(spec, *, args, run_id: str, image_sha: str, cells: int) -> None:
    print("DRY RUN — no AWS calls will be made.")
    print(f"  region:        {AWS_REGION}")
    print(f"  bucket:        {S3_BUCKET}")
    print(f"  queue:         {JOB_QUEUE}")
    print(
        f"  job def:       {AB_JOB_DEFINITION} (cloned from {JOB_DEFINITION}, image ff-training:{image_sha})"
    )
    print(f"  spec:          {spec.dotted}  baseline={spec.baseline}")
    print(f"  variants:      {list(spec.variants)}")
    print(f"  positions:     {spec.positions}")
    print(f"  seeds:         {spec.seeds}")
    print(
        f"  cells:         {cells} ({len(spec.variants)} variants x {len(spec.seeds)} seeds per position)"
    )
    print(f"  run id:        {run_id}")
    print(f"  results:       s3://{S3_BUCKET}/{_run_prefix(args.s3_prefix, run_id)}/")
    print(f"  cuda graph:    {args.cuda_graph} (auto = production graphed path on sm_80+)")
    print(f"  attempt cap:   {args.attempt_timeout}s")
    print("  jobs:")
    for pos in spec.positions:
        n = len(spec.variants) * len(spec.seeds)
        print(f"    - {pos:<4} -> {n} sequential cells, command --position {pos} --mode tune")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Launch a shared-harness A/B on AWS Batch GPU Spot")
    p.add_argument(
        "--spec", required=True, help="Dotted A/B spec module (e.g. src.tuning.ab_example)"
    )
    p.add_argument("--positions", nargs="+", help="Override the spec's POSITIONS")
    p.add_argument("--seeds", type=int, nargs="+", help="Override the spec's SEEDS")
    p.add_argument("--only", nargs="+", help="Run only these variant names (baseline always kept)")
    p.add_argument(
        "--image-sha",
        default=None,
        help="ff-training image tag to run (default: local git HEAD). For an "
        "unmerged branch, dispatch batch-image.yml on that branch first.",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Run namespace under the S3 prefix (default: {spec}-{utc}-{sha7})",
    )
    p.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX, help=f"(default: {DEFAULT_S3_PREFIX})")
    p.add_argument(
        "--cuda-graph",
        choices=["auto", "true", "false"],
        default="auto",
        help="auto (default) leaves the container's sm_80+ autodetect ON — the "
        "production graphed metric path; false forces the eager path for a "
        "bit-comparable A/B.",
    )
    p.add_argument(
        "--feature-cache",
        action="store_true",
        help="Re-enable the in-container feature cache (off by default for A/B "
        "correctness — see ab_harness).",
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
        "--wait-timeout",
        type=int,
        default=None,
        help=f"Override wait timeout in seconds (default: {WAIT_TIMEOUT_SECONDS})",
    )
    p.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip the ECR image-existence preflight (e.g. no ecr:DescribeImages)",
    )
    p.add_argument(
        "--collect-only",
        action="store_true",
        help="Skip submission; collect + aggregate an existing --run-id from S3",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the plan, touch nothing")
    return p


def _default_run_id(spec_dotted: str, image_sha: str) -> str:
    short = spec_dotted.rsplit(".", 1)[-1]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{short}-{ts}-{image_sha[:7]}"


def _aggregate_and_report(spec, results, *, s3, run_id: str, s3_prefix: str) -> int:
    agg = aggregate(spec, results)
    print_report(spec, agg, jobs=len(spec.positions))
    summary_key = f"{_run_prefix(s3_prefix, run_id)}/summary.json"
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=summary_key,
            Body=json.dumps(agg, indent=2, default=str).encode(),
            ContentType="application/json",
        )
        print(f"[ab] summary -> s3://{S3_BUCKET}/{summary_key}")
    except Exception as e:  # noqa: BLE001 — the printed report is the deliverable
        print(f"[ab] WARNING: summary upload failed: {e!r}")
    return len(agg["failed"])


def main() -> None:
    args = _build_parser().parse_args()
    wait = args.wait.lower() == "true"
    wait_timeout = args.wait_timeout if args.wait_timeout is not None else WAIT_TIMEOUT_SECONDS

    import boto3

    s3 = None
    manifest = None
    if args.collect_only:
        if not args.run_id:
            raise SystemExit("--collect-only requires --run-id")
        if not args.dry_run:
            s3 = boto3.client("s3", region_name=AWS_REGION)
            # The manifest is authoritative for the grid (the submitting
            # invocation may have used different overrides than this one).
            manifest = load_run_manifest(
                s3, bucket=S3_BUCKET, s3_prefix=args.s3_prefix, run_id=args.run_id
            )
        if manifest:
            args.spec = manifest["spec"]
            args.positions = manifest.get("positions") or args.positions
            args.seeds = manifest.get("seeds") or args.seeds
            args.only = manifest.get("only") or args.only

    image_sha = args.image_sha or _git_head_sha()
    # Collect-only never submits, so it doesn't need a resolvable image (it
    # always has --run-id, so the image-derived default run id is unused too).
    if not image_sha and not args.collect_only:
        raise SystemExit("--image-sha is required when git HEAD cannot be resolved")

    spec = resolve_spec(args.spec, positions=args.positions, seeds=args.seeds, only=args.only)
    cells = build_cells(spec)
    if len(cells) > args.max_cells:
        raise SystemExit(
            f"grid is {len(cells)} cells, over the --max-cells {args.max_cells} cost guard "
            "(typo'd --seeds/--positions? raise --max-cells consciously if intended)"
        )
    run_id = args.run_id or _default_run_id(spec.dotted, image_sha)

    if args.dry_run:
        _print_plan(spec, args=args, run_id=run_id, image_sha=image_sha, cells=len(cells))
        return

    if args.collect_only:
        results = collect_results(
            spec, bucket=S3_BUCKET, s3_prefix=args.s3_prefix, run_id=run_id, s3_client=s3
        )
        n_failed = _aggregate_and_report(
            spec, results, s3=s3, run_id=run_id, s3_prefix=args.s3_prefix
        )
        sys.exit(1 if n_failed else 0)

    batch = boto3.client("batch", region_name=AWS_REGION)
    s3 = boto3.client("s3", region_name=AWS_REGION)

    if not args.skip_image_check:
        exists = check_image_exists(image_sha, boto3.client("ecr", region_name=AWS_REGION))
        if exists is False:
            raise SystemExit(
                f"image ff-training:{image_sha} not in ECR. For an unmerged branch, build it "
                f"first: gh workflow run batch-image.yml --ref <branch> (then re-run; non-main "
                "builds push only the SHA tag). --skip-image-check bypasses this preflight."
            )

    job_definition = resolve_job_definition(image_sha, batch)

    print(f"Submitting {len(spec.positions)} A/B jobs (run_id={run_id}): {spec.positions}")
    job_ids: dict[str, str] = {}
    submit_failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(spec.positions)) as pool:
        futures = {
            pool.submit(
                submit_ab_job,
                position=pos,
                spec_dotted=spec.dotted,
                run_id=run_id,
                s3_prefix=args.s3_prefix,
                job_definition=job_definition,
                image_sha=image_sha,
                seeds=args.seeds,
                only=args.only,
                cuda_graph=args.cuda_graph,
                feature_cache=args.feature_cache,
                attempt_timeout=args.attempt_timeout,
                batch_client=batch,
            ): pos
            for pos in spec.positions
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
            "spec": spec.dotted,
            "positions": spec.positions,
            "seeds": args.seeds,
            "only": args.only,
            "variants": list(spec.variants),
            "baseline": spec.baseline,
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
            f"  python -m src.tuning.launch_ab --spec {spec.dotted} --collect-only --run-id {run_id}"
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

    # Collect whatever made it to S3 — per-cell checkpointing means a FAILED
    # position can still contribute completed cells, and missing cells show
    # up as not-ok rows in the report rather than vanishing.
    results = collect_results(
        spec, bucket=S3_BUCKET, s3_prefix=args.s3_prefix, run_id=run_id, s3_client=s3
    )
    n_failed_cells = _aggregate_and_report(
        spec, results, s3=s3, run_id=run_id, s3_prefix=args.s3_prefix
    )

    print("\nAll done.")
    if failed or timed_out or submit_failures or n_failed_cells:
        sys.exit(1)


if __name__ == "__main__":
    main()
