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
Deployments pass ``--reuse-valid`` to reuse a verified complete bundle when
possible; without that option the script always rebuilds.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

_ALL_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


def main(argv: list[str] | None = None) -> int:
    bucket = os.environ.get("FF_MODEL_S3_BUCKET", "").strip()
    if not bucket:
        print("FF_MODEL_S3_BUCKET unset — refusing to build a cache with no S3 source/target")
        return 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-valid",
        action="store_true",
        help="Reuse a complete S3 generation matching current inputs; otherwise rebuild it",
    )
    args = parser.parse_args(argv)

    # Import after the env check so the failure path stays import-light.
    import src.serving.app as app_pkg
    from src.serving import core
    from src.shared.model_sync import (
        sync_data_from_s3,
        sync_models_from_s3,
        sync_predictions_cache_from_s3,
        upload_predictions_cache_to_s3,
    )

    # 1. Pull the exact artifacts the serving container syncs at boot, so the
    #    cache we build carries the same content-hash fingerprint serving will
    #    compute (and therefore hydrates instead of recomputing).
    sync_data_from_s3()
    sync_models_from_s3()

    # Deployments must seed the new bundle protocol before replacing serving
    # tasks. A verified, complete generation can satisfy that gate without
    # inference; the default refresh-splits path deliberately rebuilds.
    if args.reuse_valid:
        synced = sync_predictions_cache_from_s3()
        with app_pkg._cache_lock:
            app_pkg._cache.clear()
            if synced and synced.get("files", 0) > 0 and core._try_hydrate_from_disk():
                results = app_pkg._cache.get("results")
                complete_positions = (
                    results is not None
                    and "position" in results.columns
                    and set(_ALL_POSITIONS).issubset(set(results["position"].unique()))
                )
                if (
                    complete_positions
                    and not app_pkg._cache.get("position_load_errors")
                    and not app_pkg._cache.get("positions_failed")
                    and not core._positions_pending()
                    and not core._any_position_sentinel_advanced()
                ):
                    print("[build_serving_cache] reusing verified complete S3 generation")
                    return 0
        print("[build_serving_cache] no reusable complete generation — rebuilding")

    # 2. Build the cache WITHOUT auto-uploading. _persist_cache_to_disk uploads as
    #    a side effect, so blank the bucket across the build and restore it after,
    #    letting us validate before anything is published. Clear any stale local
    #    cache first so _ensure_metrics recomputes rather than hydrating it.
    shutil.rmtree(core._PREDICTIONS_CACHE_DIR, ignore_errors=True)
    app_pkg._cache.clear()
    os.environ["FF_MODEL_S3_BUCKET"] = ""
    try:
        core._ensure_metrics()
    finally:
        os.environ["FF_MODEL_S3_BUCKET"] = bucket

    # 3. Validate before publishing — a partial/empty build must never overwrite a
    #    good S3 cache.
    results = app_pkg._cache.get("results")
    if results is None or len(results) == 0:
        print("ERROR: empty results frame — refusing to publish")
        return 1
    positions = set(results["position"].unique())
    missing_pos = [p for p in _ALL_POSITIONS if p not in positions]
    nflcom = int(results["nflcom_pred"].notna().sum()) if "nflcom_pred" in results.columns else 0
    print(
        f"[build_serving_cache] rows={len(results)} "
        f"positions={sorted(positions)} nflcom_non_null={nflcom}"
    )
    if missing_pos:
        print(f"ERROR: missing positions {missing_pos} — refusing to publish")
        return 1
    if (
        app_pkg._cache.get("position_load_errors")
        or app_pkg._cache.get("positions_failed")
        or core._positions_pending()
    ):
        print("ERROR: degraded or pending positions — refusing to publish")
        return 1
    # Experts are an auxiliary surface (NFL.com can be transiently unreachable), so
    # an all-null expert column is a warning, not a publish-blocker — the model
    # predictions are the primary product and are validated above.
    if nflcom == 0:
        print("WARN: nflcom_pred is all-null (expert join produced no rows this build)")

    # The reference slate is built and verified in the training data release.
    # Cache construction consumes that same immutable cohort as Batch training.

    # 4. Publish the validated generation as one bundle.
    if upload_predictions_cache_to_s3() is None:
        print("ERROR: upload returned no result — cache not published")
        return 1
    print("[build_serving_cache] published validated cache to S3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
