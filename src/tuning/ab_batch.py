"""AWS Batch container entry for the shared A/B harness ([ab_harness.py](ab_harness.py)).

Reached via ``FF_TUNE_AB_SPEC=<dotted spec module>`` in the Batch job
environment: the training image's ENTRYPOINT is pinned to ``src.batch.train``,
whose ``--mode=tune`` forwards a fixed argv into ``src.tuning.tune_nn.main``,
which dispatches here before any Optuna work — the same env-flag route as
``FF_TUNE_ENSEMBLE_AB`` / [ab_ensemble_seeds.py](ab_ensemble_seeds.py)
(``containerOverrides.environment`` passes through the entrypoint untouched,
and editing ``src/batch/train.py`` would fire a 6-position retrain).

One Batch job == one position's slice of the A/B grid. The job runs that
position's variant × seed cells **sequentially** through the harness's
:func:`~src.tuning.ab_harness.run_cell` (chdir + tmp-dir isolated, identical
to local execution — served ``{pos}/outputs`` are never written, no model
artifact is uploaded, ``benchmark_history/`` is untouched) and uploads each
cell's small result JSON to::

    s3://$S3_BUCKET/$FF_AB_S3_PREFIX/$FF_AB_RUN_ID/cells/{POS}-{variant}-{seed}.json

immediately on completion. A Spot reclaim therefore loses at most the
in-flight cell: Batch's RETRY_STRATEGY reruns the job and the resume scan
skips every cell whose JSON already landed (the same checkpoint philosophy as
tune_nn's S3 study DB). Position-level fan-out is the parallel axis — one
g6/L4 (g5/A10G fallback) Spot host per position, mirroring ADR-0015; cells
stay sequential in-container because each cell is a *full* pipeline run
(Ridge + LightGBM + both NNs) and the 4-vCPU job shape has no headroom for
concurrent cells the way the 16-core local boxes do.

Env contract (set by [launch_ab.py](launch_ab.py)):

==================  =========================================================
``FF_TUNE_AB_SPEC``  dotted spec module (required), e.g. ``src.tuning.ab_example``
``FF_AB_RUN_ID``     run namespace under the S3 prefix (required)
``FF_AB_S3_PREFIX``  S3 prefix for A/B runs (default ``ab_runs``)
``FF_AB_SEEDS``      comma-separated seed override (default: spec ``SEEDS``)
``FF_AB_ONLY``       comma-separated variant names (default: all variants;
                     the baseline is always kept, mirroring ``--only``)
``S3_BUCKET``        results + data bucket (required; same convention as the
                     train/tune paths — also drives ``_ensure_data_from_s3``)
==================  =========================================================

Aggregation happens on the launcher via the harness's own ``aggregate()`` /
``print_report()`` over the collected per-cell JSONs.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.tuning.ab_harness import (  # noqa: E402
    _cell_failed,
    build_cells,
    resolve_spec,
    run_cell,
)

ENV_SPEC = "FF_TUNE_AB_SPEC"
ENV_RUN_ID = "FF_AB_RUN_ID"
ENV_S3_PREFIX = "FF_AB_S3_PREFIX"
ENV_SEEDS = "FF_AB_SEEDS"
ENV_ONLY = "FF_AB_ONLY"
ENV_STACKED = "FF_AB_STACKED"
ENV_STACKED_EPOCHS = "FF_AB_STACKED_EPOCHS"
DEFAULT_S3_PREFIX = "ab_runs"


def cell_result_key(s3_prefix: str, run_id: str, cell_key: str) -> str:
    """S3 key for one cell's result JSON. Single source of truth — the
    launcher's collector builds the same keys from the same grid."""
    return f"{s3_prefix.strip('/')}/{run_id}/cells/{cell_key}.json"


def _split_env_csv(name: str) -> list[str] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"[ab-batch] {name} is required in the job environment")
    return value


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
    report records which metric path actually ran (mirrors train.py's
    ``_hardware_metadata``)."""
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
    """Run one position's A/B cells and checkpoint each result JSON to S3.

    Cell failures are recorded (the harness's ``_cell_failed`` shape) and
    uploaded like successes so the launcher's aggregation can surface them;
    the job still exits non-zero at the end so the Batch console shows red.
    RETRY_STRATEGY exits (no retry) on task failures, so a deterministic
    cell error doesn't burn Spot retries.
    """
    import boto3

    spec_dotted = _require_env(ENV_SPEC)
    run_id = _require_env(ENV_RUN_ID)
    bucket = _require_env("S3_BUCKET")
    s3_prefix = os.environ.get(ENV_S3_PREFIX, "").strip() or DEFAULT_S3_PREFIX
    seeds = _split_env_csv(ENV_SEEDS)
    only = _split_env_csv(ENV_ONLY)

    # Same default as the harness's local worker path: the feature cache keys
    # on data, not code, so a sibling variant's features could be silently
    # reused -> false Δ=0. launch_ab --feature-cache forwards "0" to re-enable.
    os.environ.setdefault("FF_FEATURE_CACHE_DISABLE", "1")

    from src.tuning.tune_nn import _ensure_data_from_s3

    _ensure_data_from_s3()

    spec = resolve_spec(spec_dotted, positions=[position], seeds=seeds, only=only)
    s3 = boto3.client("s3")
    done = _list_done_cells(s3, bucket, s3_prefix, run_id)
    provenance = _provenance()
    data_dir = os.path.abspath("data")

    stacked = os.environ.get(ENV_STACKED, "").strip() == "1"
    stacked_epochs = int(os.environ.get(ENV_STACKED_EPOCHS, "30") or 30)
    failures = 0
    if stacked:
        # Stacked-seeds mode (owner-sanctioned opt-in; see ab_harness): this
        # position's (variant) groups train all seeds as ONE vmap ensemble;
        # per-seed result JSONs land under the SAME cell keys, so the
        # launcher's collector is unchanged. Resume granularity is the group
        # (rerun unless every seed's JSON already landed — the stacked arms
        # are one training run, so partial resume can't splice). A K/DST
        # position submitted with --stacked falls through to eager cells.
        from src.tuning.ab_harness import (
            _group_failed,
            build_stacked_units,
            run_group_stacked,
        )

        groups, cells = build_stacked_units(spec)
        print(
            f"[ab-batch] {position}: spec={spec_dotted} run_id={run_id} STACKED "
            f"(epochs={stacked_epochs}) variants={list(spec.variants)} seeds={spec.seeds} "
            f"-> {len(groups)} groups + {len(cells)} eager cells",
            flush=True,
        )
        for gi, group in enumerate(groups, 1):
            group_cell_keys = {f"{group.position}-{group.variant}-{s}" for s in group.seeds}
            if group_cell_keys <= done:
                print(f"[ab-batch] group {gi}/{len(groups)} {group.key}: resume-skip", flush=True)
                continue
            variant = spec.variants[group.variant]
            print(f"[ab-batch] group {gi}/{len(groups)} {group.key}: running", flush=True)
            t0 = time.time()
            try:
                results = run_group_stacked(
                    group,
                    variant,
                    spec.metric_fn,
                    data_dir=data_dir,
                    stacked_epochs=stacked_epochs,
                )
            except Exception as exc:  # noqa: BLE001 — one group must not sink the job
                print(f"[ab-batch] group {group.key} FAILED: {exc}", file=sys.stderr, flush=True)
                results = _group_failed(group, variant, exc)
            elapsed = round(time.time() - t0, 1)
            for r in results:
                r["elapsed_sec"] = elapsed
                r["provenance"] = provenance
                if not r["ok"]:
                    failures += 1
                key = cell_result_key(
                    s3_prefix, run_id, f"{r['position']}-{r['variant']}-{r['seed']}"
                )
                s3.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=json.dumps(r).encode(),
                    ContentType="application/json",
                )
            tag = "ok" if all(r["ok"] for r in results) else "FAILED"
            print(f"[ab-batch] group {group.key} {tag} in {elapsed}s", flush=True)
    else:
        cells = build_cells(spec)
        print(
            f"[ab-batch] {position}: spec={spec_dotted} run_id={run_id} "
            f"variants={list(spec.variants)} seeds={spec.seeds} -> {len(cells)} cells "
            f"({len(done & {c.key for c in cells})} already complete)",
            flush=True,
        )
    for i, cell in enumerate(cells, 1):
        if cell.key in done:
            print(f"[ab-batch] cell {i}/{len(cells)} {cell.key}: resume-skip", flush=True)
            continue
        variant = spec.variants[cell.variant]
        print(f"[ab-batch] cell {i}/{len(cells)} {cell.key}: running", flush=True)
        t0 = time.time()
        try:
            result = run_cell(cell, variant, spec.metric_fn, data_dir=data_dir)
        except Exception as exc:  # noqa: BLE001 — one cell's failure must not sink the job
            print(f"[ab-batch] cell {cell.key} FAILED: {exc}", file=sys.stderr, flush=True)
            result = _cell_failed(cell, variant, exc)
        result["elapsed_sec"] = round(time.time() - t0, 1)
        result["provenance"] = provenance
        if not result["ok"]:
            failures += 1
        key = cell_result_key(s3_prefix, run_id, cell.key)
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(result).encode(),
            ContentType="application/json",
        )
        tag = "ok" if result["ok"] else "FAILED"
        print(
            f"[ab-batch] cell {cell.key} {tag} in {result['elapsed_sec']}s -> s3://{bucket}/{key}",
            flush=True,
        )

    print(f"[ab-batch] {position}: done, {failures} cell failure(s)", flush=True)
    if failures:
        raise SystemExit(f"[ab-batch] {position}: {failures} cell(s) failed; results are in S3")
