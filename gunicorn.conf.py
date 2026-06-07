"""Gunicorn config — S3 sync in ``on_starting``, background pre-warm in ``post_fork``.

Module-level pre-warm under ``--preload`` is forbidden — the import runs
before ``bind()`` so a slow warm caused the ALB to see TCP-refused and
mark the task unhealthy (PRs #148/#149, see TODO.md "Gunicorn --preload
pre-warm broke ALB health checks").

``on_starting`` runs in master BEFORE the ``--preload`` app import, so
the S3 syncs land here. Wall-clock is identical to running them at app
module-import time (both block before bind), but keeping the work out of
the import path means ``src.serving.app`` has no import-time side
effects — future readers can't accidentally re-introduce a slow blocking
sync by adding "just one more" line at module top.

``post_fork`` fires after the master has already bound :8000, and we hand
off to a daemon thread so the worker returns to the arbiter immediately
and starts accepting requests while warming continues in the background.
A first user request that arrives mid-warm serializes correctly on
``_cache_lock`` inside ``_ensure_metrics`` — no duplicate compute.

Per-worker: under ``--preload`` the workers fork *after* module import,
but ``_cache`` is mutated on the first warm so copy-on-write is broken
and each worker must populate its own dict. Disk + S3 cache hydration
short-circuits this once the cache exists for the current model
fingerprint, so the duplicate-compute cost only applies to the very
first container after a model retrain.
"""

import os
import threading

# Default to a 30s in-flight refresh poll. Set FF_MODEL_REFRESH_INTERVAL_S=0
# to disable (matches the prior boot-then-static behavior). The poller is a
# no-op when FF_MODEL_S3_BUCKET is unset (dev / CI), so this default is
# harmless outside ECS.
_DEFAULT_REFRESH_INTERVAL_S = 30


def on_starting(server):
    from src.shared.model_sync import (
        start_refresh_poller,
        sync_benchmark_history_from_s3,
        sync_data_from_s3,
        sync_models_from_s3,
        sync_predictions_cache_from_s3,
    )

    sync_data_from_s3()
    sync_models_from_s3()
    sync_benchmark_history_from_s3()
    sync_predictions_cache_from_s3()

    # Start the in-flight model refresh poller AFTER the boot sync so the
    # poller's bootstrap-first-call (refresh_position returns did_refresh=False
    # on last_etag=None) doesn't race the initial download. The thread is a
    # daemon — gunicorn shutdown reaps it without join.
    try:
        interval_s = int(os.environ.get("FF_MODEL_REFRESH_INTERVAL_S", _DEFAULT_REFRESH_INTERVAL_S))
    except ValueError:
        interval_s = _DEFAULT_REFRESH_INTERVAL_S
    if interval_s > 0:
        start_refresh_poller(interval_s)
        print(f"[model_sync] in-flight refresh poller started (interval={interval_s}s)")
    else:
        print("[model_sync] in-flight refresh poller disabled (FF_MODEL_REFRESH_INTERVAL_S=0)")


def post_fork(server, worker):
    def _warm():
        try:
            from src.serving import core as serving_core

            serving_core._ensure_metrics()
        except Exception as e:  # noqa: BLE001 — log + swallow; first user request will retry
            worker.log.warning("pre-warm thread failed: %r", e)

    threading.Thread(target=_warm, daemon=True, name="prewarm").start()
