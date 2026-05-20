"""Gunicorn config — background pre-warm in ``post_fork``.

Module-level pre-warm under ``--preload`` is forbidden — the import runs
before ``bind()`` so a slow warm caused the ALB to see TCP-refused and
mark the task unhealthy (PRs #148/#149, see TODO.md "Gunicorn --preload
pre-warm broke ALB health checks").

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

import threading


def post_fork(server, worker):
    def _warm():
        try:
            from src.serving import app as serving_app

            serving_app._ensure_metrics()
        except Exception as e:  # noqa: BLE001 — log + swallow; first user request will retry
            worker.log.warning("pre-warm thread failed: %r", e)

    threading.Thread(target=_warm, daemon=True, name="prewarm").start()
