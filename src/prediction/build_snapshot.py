"""Build the serving prediction cache off-container and upload it to S3.

ADR-0018 (and the 2026-06-15 incident): the heavy ``load_raw_data`` /
``build_features`` / inference work that produces the prediction cache must NOT
run inside the 2-worker serving container — two workers recomputing concurrently
OOM the task. The serving fingerprint also drifts every time ``refresh-splits``
rewrites ``data/splits/*`` in S3, which forces a fresh container to recompute.

This script builds the cache (``predictions.parquet`` / ``metrics.json`` /
``fingerprint.json`` / ``snapshot.json``) where memory is ample (CI or an
operator box), from the SAME S3 model + data artifacts the serving container
syncs at boot, validates it, then uploads it to S3. Serving then only *hydrates*
it — the content-hash fingerprint matches by construction because both read the
same S3 ``data/splits`` + ``data/raw`` + models.

Run it after ``refresh-splits.yml`` uploads fresh splits (the drift trigger), or
after a retrain. Requires ``FF_MODEL_S3_BUCKET`` (and optional
``FF_MODEL_S3_PREFIX``); refuses to run without an S3 target.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict

_ALL_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


def expected_model_keys(s3, bucket, prefix, *, plan_id=None, dataset_id=None, legacy_run_id=None):
    """Bind a training-triggered cache build to that run's exact successful outputs."""
    from src.artifacts.receipts import load_receipt, load_run_receipt
    from src.orchestration.build_plan import load_plan

    if plan_id and legacy_run_id:
        raise RuntimeError("Serving build cannot select both plan and legacy-run identities")
    if plan_id:
        plan = load_plan(s3, bucket, plan_id)
        if dataset_id != plan["dataset_id"] or prefix != plan.get("model_prefix"):
            raise RuntimeError("Serving build source differs from its selected plan")
        return {pos: load_receipt(s3, bucket, plan_id, pos)["key"] for pos in plan["positions"]}
    if legacy_run_id:
        positions = os.environ.get("FF_BUILD_POSITIONS", "").upper().split()
        if not positions or any(pos not in _ALL_POSITIONS for pos in positions):
            raise RuntimeError("Legacy serving build requires its explicit FF_BUILD_POSITIONS")
        source_sha = os.environ.get("FF_TRAIN_GIT_SHA", "")
        image_id = os.environ.get("FF_TRAIN_IMAGE_ID")
        receipts = {
            pos: load_run_receipt(s3, bucket, prefix, source_sha, pos, legacy_run_id)
            for pos in positions
        }
        if image_id and any(receipt.get("image_id") != image_id for receipt in receipts.values()):
            raise RuntimeError("Serving build image differs from its selected training run")
        if any(receipt.get("dataset_id") != dataset_id for receipt in receipts.values()):
            raise RuntimeError("Serving build data release differs from its selected training run")
        return {pos: receipt["key"] for pos, receipt in receipts.items()}
    return {}


def stage_cache(directory, build, destination, *, generation):
    """Export verified canonical files without moving the remote serving pointer."""
    from pathlib import Path

    from src.artifacts import serving_snapshot

    _, content = serving_snapshot.read_generation(directory, generation)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in content.items():
        (target / name).write_bytes(payload)
    (target / "build.json").write_text(json.dumps(asdict(build), sort_keys=True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-directory", help="Export a candidate; do not publish it")
    args = parser.parse_args(argv)
    bucket = os.environ.get("FF_MODEL_S3_BUCKET", "").strip()
    if not bucket:
        print("FF_MODEL_S3_BUCKET unset — refusing to build a cache with no S3 source/target")
        return 1
    from pathlib import Path

    import boto3
    import numpy as np

    from src.artifacts import model_sync, serving_snapshot
    from src.artifacts import snapshot_state as state
    from src.data.providers.snapshot import assert_snapshot_sources_complete
    from src.orchestration.datasets import materialize_dataset
    from src.prediction import historical as core

    s3 = boto3.client("s3")
    prefix = os.environ.get("FF_MODEL_S3_PREFIX", "models").strip("/")
    plan_id = os.environ.get("FF_BUILD_PLAN_ID")
    dataset_id = os.environ.get("FF_DATASET_ID") or os.environ.get("FF_DATA_RELEASE")
    if os.environ.get("FF_DATA_RELEASE") and dataset_id != os.environ["FF_DATA_RELEASE"]:
        raise RuntimeError("Serving dataset and data release identities disagree")
    expected = expected_model_keys(
        s3,
        bucket,
        prefix,
        plan_id=plan_id,
        dataset_id=dataset_id,
        legacy_run_id=os.environ.get("FF_LEGACY_RUN_ID"),
    )
    token = serving_snapshot.begin_build(s3, bucket, prefix, expected_model_keys=expected)
    if dataset_id:
        from src.orchestration.build_plan import load_plan, plan_data_format

        repo = model_sync._repo_root()
        data_format = (
            plan_data_format(load_plan(s3, bucket, plan_id)) if plan_id else "data-release-v1"
        )
        materialize_dataset(
            s3,
            bucket,
            dataset_id,
            splits_dir=repo / "data/splits",
            raw_dir=repo / "data/raw",
            data_format=data_format,
        )
    else:
        model_sync.sync_data_from_s3()  # explicitly supported legacy operator/EC2 path
    synced = model_sync.sync_models_from_s3()
    actual = {entry["pos"]: entry["key"] for entry in synced["positions"]}
    desired = {pos: key for pos, _, key in token.model_generations}
    if actual != desired:
        raise RuntimeError("Serving model download did not match the captured approved generations")
    # The builder owns its state and effects. It never constructs a Flask app or
    # changes global S3 environment variables to suppress implicit publication.
    shutil.rmtree(core._PREDICTIONS_CACHE_DIR, ignore_errors=True)
    owner = state.ServingState(publish_remote=False)
    with state.use_state(owner):
        core._ensure_metrics()
        assert_snapshot_sources_complete()
        results = owner.cache.get("results")
        if results is None or results.empty:
            raise RuntimeError("Empty serving prediction snapshot")
        for pos in _ALL_POSITIONS:
            rows = results[results["position"].eq(pos)]
            columns = [f"{family}_pred_ppr" for family in ("ridge", "nn", "attn_nn", "lgbm")]
            if (
                rows.empty
                or not set(columns).issubset(rows)
                or not np.isfinite(rows[columns].to_numpy(dtype=float)).all()
            ):
                raise RuntimeError(
                    f"Incomplete {pos} model predictions; retaining the published snapshot"
                )
        # The reference is part of the sealed training release. Re-fetching it
        # here would compare model outputs against a different cohort identity.
        from src.shared.evaluation_cohorts import load_reference

        reference = load_reference(cache_dir=model_sync._repo_root() / "data/raw")
        from src.prediction.comparison_snapshot import build_comparison_snapshot

        owner.cache["comparison_snapshot"] = build_comparison_snapshot(results, reference=reference)
        # The first persistence occurred during prediction construction, before
        # comparison assembly. Seal the matching comparison into metrics.json
        # before publishing this generation's immutable files.
        completed = core._persist_cache_to_disk(required=True)
        if args.stage_directory:
            stage_cache(
                core._PREDICTIONS_CACHE_DIR,
                token,
                args.stage_directory,
                generation=completed.name,
            )
            print("[build_serving_cache] staged canonical files; serving pointer unchanged")
            return 0
        published = serving_snapshot.publish(
            s3,
            bucket,
            Path(core._PREDICTIONS_CACHE_DIR),
            token,
            prefix,
            generation=completed.name,
        )
    print(f"[build_serving_cache] published generation {published['generation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
