"""AWS Batch container entry for the eager ``ablation_runner`` family.

Sibling of [ab_batch.py](ab_batch.py), but for the eager config-injection
ablations built on [ablation_runner.py](ablation_runner.py) (``ablate_attn_arch``
et al.) rather than the frame-injection [ab_harness.py](ab_harness.py) specs.
The two abstractions differ — the A/B harness vmaps a stacked seed ensemble and
surfaces only ``pred_attn_nn_total``; the eager runner runs one full
``get_runner(pos)(seed, cfg)`` pipeline per cell and keeps the per-target
attention-MAE table and the Ridge data-identity sentinel. Some ablations only
make sense eager (``selfattn``'s ``nn.MultiheadAttention`` SDPA ``attn_bias``
errors under ``torch.func.vmap``, so it was dropped from the stacked
``ab_attn_arch`` spec — this path is its GPU home).

Reached via ``FF_TUNE_ABLATE_MOD=<dotted ablation module>`` in the Batch job
environment: the training image's ENTRYPOINT is pinned to ``src.batch.train``,
whose ``--mode=tune`` forwards a fixed argv into ``src.tuning.tune_nn.main``,
which dispatches here before any Optuna work — the same env-flag route as
``FF_TUNE_AB_SPEC`` / [ab_batch.py](ab_batch.py) (``containerOverrides.
environment`` passes through the entrypoint untouched, and editing
``src/batch/train.py`` would fire a 6-position retrain and can't carry branch
code anyway).

One Batch job == one position's slice of the ablation grid. The job runs that
position's variant × seed cells **sequentially** through
:func:`src.tuning.ablation_runner._run_job` (chdir + tmp-dir isolated, identical
to local execution — served ``{pos}/outputs`` are never written, no model
artifact is uploaded, ``benchmark_history/`` is untouched) and uploads each
cell's small result JSON to::

    s3://$S3_BUCKET/$FF_ABLATE_S3_PREFIX/$FF_ABLATE_RUN_ID/cells/{POS}-{variant}-{seed}.json

immediately on completion. A Spot reclaim therefore loses at most the in-flight
cell: Batch's RETRY_STRATEGY reruns the job and the resume scan skips every cell
whose JSON already landed (the same checkpoint philosophy as ab_batch / tune_nn's
S3 study DB). Position-level fan-out is the parallel axis — one g6/L4 (g5/A10G
fallback) Spot host per position.

Batch-runnable ablation module contract (``src.tuning.ablate_attn_arch``
satisfies it):

================  =============================================================
``ABLATION_NAME``  short name (str), used only for log lines
``BASELINE``       the baseline variant key (str); always kept for paired deltas
``VARIANTS``       ``dict`` whose keys are the variant names (values opaque here)
``_build_jobs``    ``_build_jobs(*, position, seeds, variants) -> list[AblationJob]``
                   with each job's ``run_fn`` self-contained (it reconstructs
                   the cfg + runner, so no Python callable crosses the S3 wire)
================  =============================================================

Env contract (set by [launch_ablate.py](launch_ablate.py)):

=====================  ======================================================
``FF_TUNE_ABLATE_MOD``  dotted ablation module (required), e.g. ``src.tuning.ablate_attn_arch``
``FF_ABLATE_RUN_ID``    run namespace under the S3 prefix (required)
``FF_ABLATE_S3_PREFIX`` S3 prefix for ablation runs (default ``ablation_runs``)
``FF_ABLATE_SEEDS``     comma-separated seeds (required; the eager runner has no
                        per-position default the way an Optuna study does)
``FF_ABLATE_VARIANTS``  comma-separated variant names (default: all of
                        ``VARIANTS``; the baseline is always kept, like ``--only``)
``S3_BUCKET``           results + data bucket (required; drives ``_ensure_data_from_s3``)
=====================  ======================================================

Aggregation happens on the launcher via the ablation module's own
``print_summary()`` over the collected per-cell JSONs.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from types import ModuleType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.tuning.ablation_runner import (  # noqa: E402
    AblationJob,
    _run_job,
    parse_seed_list,
    result_to_dict,
    select_variants,
)

ENV_MOD = "FF_TUNE_ABLATE_MOD"
ENV_RUN_ID = "FF_ABLATE_RUN_ID"
ENV_S3_PREFIX = "FF_ABLATE_S3_PREFIX"
ENV_SEEDS = "FF_ABLATE_SEEDS"
ENV_VARIANTS = "FF_ABLATE_VARIANTS"
DEFAULT_S3_PREFIX = "ablation_runs"


def cell_result_key(s3_prefix: str, run_id: str, cell_key: str) -> str:
    """S3 key for one cell's result JSON. Single source of truth — the
    launcher's collector builds the same keys from the same grid."""
    return f"{s3_prefix.strip('/')}/{run_id}/cells/{cell_key}.json"


def cell_key(position: str, variant: str, seed: int) -> str:
    """Stable per-cell identity shared by the runner and the launcher's
    collector. Mirrors ``ab_harness`` cell keys (``{pos}-{variant}-{seed}``)."""
    return f"{position}-{variant}-{seed}"


def _split_env_csv(name: str) -> list[str] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"[ablate-batch] {name} is required in the job environment")
    return value


def load_ablation_module(dotted: str) -> ModuleType:
    """Import a Batch-runnable ablation module and verify the contract up front
    so a typo fails before the (expensive) data pull, not mid-grid."""
    module = importlib.import_module(dotted)
    missing = [
        attr for attr in ("VARIANTS", "BASELINE", "_build_jobs") if not hasattr(module, attr)
    ]
    if missing:
        raise SystemExit(
            f"[ablate-batch] {dotted} is not Batch-runnable: missing {missing}. "
            "It must expose VARIANTS, BASELINE and _build_jobs(*, position, seeds, variants)."
        )
    return module


def resolve_grid(module: ModuleType, *, position: str, seeds, variants) -> list[AblationJob]:
    """Build this position's ``AblationJob``s, always including the baseline so
    the per-seed paired deltas have a reference (mirrors each ablation ``main``)."""
    selected = select_variants(
        ",".join(variants) if variants else None,
        module.VARIANTS,
        [module.BASELINE, *(k for k in module.VARIANTS if k != module.BASELINE)],
    )
    if module.BASELINE not in selected:
        selected = [module.BASELINE, *selected]
    return module._build_jobs(position=position, seeds=seeds, variants=selected)


def _list_done_cells(s3, bucket: str, s3_prefix: str, run_id: str) -> set[str]:
    """Cell keys whose result JSON already exists under the run prefix —
    completed by a prior Spot attempt; the retry skips them."""
    prefix = f"{s3_prefix.strip('/')}/{run_id}/cells/"
    done: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(prefix) :]
            if name.endswith(".json"):
                done.add(name[: -len(".json")])
    return done


def _provenance() -> dict:
    """GPU + capture facts stamped into every cell JSON so the aggregated
    report records which metric path actually ran (mirrors ab_batch)."""
    from src.shared.platform_detect import detect_platform
    from src.shared.utils import cuda_graph_enabled

    info = detect_platform()
    return {
        "git_sha": os.environ.get("FF_TRAIN_GIT_SHA", "").strip(),
        "gpu_name": info.gpu_name,
        "sm": info.sm,
        "cuda_graph_active": cuda_graph_enabled(),
    }


def run_batch_entry(position: str) -> None:
    """Run one position's ablation cells and checkpoint each result JSON to S3.

    Cell failures are recorded (the ``AblationResult.error`` shape) and uploaded
    like successes so the launcher's ``print_summary`` can surface them; the job
    still exits non-zero at the end so the Batch console shows red. RETRY_STRATEGY
    exits (no retry) on task failures, so a deterministic cell error doesn't burn
    Spot retries.
    """
    import boto3

    dotted = _require_env(ENV_MOD)
    run_id = _require_env(ENV_RUN_ID)
    bucket = _require_env("S3_BUCKET")
    s3_prefix = os.environ.get(ENV_S3_PREFIX, "").strip() or DEFAULT_S3_PREFIX
    seeds = parse_seed_list(_require_env(ENV_SEEDS))
    variants = _split_env_csv(ENV_VARIANTS)

    module = load_ablation_module(dotted)

    from src.tuning.tune_nn import _ensure_data_from_s3

    _ensure_data_from_s3()
    # The runner's output-isolation symlinks {cwd}/.cache through to the job's
    # tmp cwd (warm-once feature cache across this position's cells). Ensure the
    # target exists so the symlink resolves in a fresh container.
    os.makedirs(".cache", exist_ok=True)

    jobs = resolve_grid(module, position=position, seeds=seeds, variants=variants)
    s3 = boto3.client("s3")
    done = _list_done_cells(s3, bucket, s3_prefix, run_id)
    provenance = _provenance()
    data_dir = os.path.abspath("data")

    print(
        f"[ablate-batch] {position}: mod={dotted} run_id={run_id} "
        f"variants={sorted({j.variant for j in jobs})} seeds={seeds} -> {len(jobs)} cells "
        f"({len(done & {cell_key(j.position, j.variant, j.seed) for j in jobs})} already complete)",
        flush=True,
    )

    failures = 0
    for i, job in enumerate(jobs, 1):
        key_short = cell_key(job.position, job.variant, job.seed)
        if key_short in done:
            print(f"[ablate-batch] cell {i}/{len(jobs)} {key_short}: resume-skip", flush=True)
            continue
        print(f"[ablate-batch] cell {i}/{len(jobs)} {key_short}: running", flush=True)
        t0 = time.time()
        # _run_job captures per-job failures into an AblationResult(error=...),
        # so it never raises; a deterministic cell error is recorded, not fatal.
        result = _run_job(job, None, data_dir)
        payload = result_to_dict(result)
        payload["elapsed_sec"] = round(time.time() - t0, 1)
        payload["provenance"] = provenance
        if result.error:
            failures += 1
        s3_key = cell_result_key(s3_prefix, run_id, key_short)
        s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(payload).encode(),
            ContentType="application/json",
        )
        tag = "FAILED" if result.error else "ok"
        print(
            f"[ablate-batch] cell {key_short} {tag} in {payload['elapsed_sec']}s "
            f"-> s3://{bucket}/{s3_key}",
            flush=True,
        )

    print(f"[ablate-batch] {position}: done, {failures} cell failure(s)", flush=True)
    if failures:
        raise SystemExit(f"[ablate-batch] {position}: {failures} cell(s) failed; results are in S3")
