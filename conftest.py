"""Root conftest for the entire test suite.

Responsibilities:
1. Cap BLAS/OpenMP/torch intra-op thread counts. xdist runs `-n auto` (one
   worker per core); without a cap each worker spawns a full threadpool and
   they oversubscribe the runner CPUs, slowing every shard. Setting these at
   module top means torch / numpy / scipy pick them up on first import.
2. Put the project root on ``sys.path`` exactly once so every test module can
   ``import src.shared.*`` / ``import src.qb.data`` etc. without each
   per-directory ``conftest.py`` having to re-wire the path.
3. Register the project-wide pytest markers as a belt-and-suspenders backup
   to ``pyproject.toml`` so ``--strict-markers`` never trips on a fresh
   checkout where ``pyproject.toml`` might be missing.
4. Redirect the raw-data cache (``src.config.CACHE_DIR``) to a per-session
   temp dir, so loader runs inside tests never write the real ``data/raw``.
"""

from __future__ import annotations

import os

# Must run before numpy / scipy / torch are imported anywhere in the test
# session. setdefault preserves any explicit override the developer set in
# their shell.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Redirect the raw nflverse cache to a per-session temp dir BEFORE any src
# import bakes ``src.config.CACHE_DIR`` into from-import copies and default
# args. Unit tests reach the cache-writing loaders (team_stats / nflcom /
# sleeper projections) through deep call chains and would otherwise
# fetch-and-write into the REAL repo ``data/raw`` — polluting a dev box's
# genuine cache (the write itself is already atomic + concurrent-read-safe
# via ``atomic_write_parquet``, the #1056/#1057 fix; this isolates WHERE it
# lands). xdist workers inherit the master's env, so all workers in one run
# share the dir — same concurrency shape as before, minus the pollution.
# Existing real cache files are symlinked in so reads (e.g. the schedules
# parquet CI fetches) still resolve; only writes are isolated. The dirs are
# small and land under the OS temp root, which reaps them. An explicit
# FF_CACHE_DIR in the shell is preserved (the same env-override shape as
# FF_DEVICE / LGBM_N_JOBS).
if "FF_CACHE_DIR" not in os.environ:
    import tempfile

    _tmp_raw = Path(tempfile.mkdtemp(prefix="ff-test-raw-"))
    _real_raw = PROJECT_ROOT / "data" / "raw"
    if _real_raw.is_dir():
        for _entry in _real_raw.iterdir():
            (_tmp_raw / _entry.name).symlink_to(_entry)
    os.environ["FF_CACHE_DIR"] = str(_tmp_raw)


_MARKERS = (
    ("unit", "fast unit tests (<1s each)"),
    ("integration", "multi-component tests (<10s)"),
    ("e2e", "full-pipeline tests (<60s)"),
    ("regression", "model quality thresholds"),
    ("slow", "excluded from default run"),
)


def pytest_configure(config):
    """Register markers as a backup to pyproject.toml + cap torch threads.

    Pytest tolerates duplicate registration, so position-level conftests
    that re-register the same markers will not conflict.

    The torch intra-op cap belts-and-suspenders the OMP_NUM_THREADS env var
    set at module top — torch reads OMP_NUM_THREADS on first import, but if
    something has already imported torch with a different value (e.g. an
    upstream plugin), this still pins it to 1 per worker.
    """
    for name, description in _MARKERS:
        config.addinivalue_line("markers", f"{name}: {description}")

    # pytest-xdist registers this whenever it's loaded (always here — addopts
    # carry ``-n auto``), but register it too so ``--strict-markers`` tolerates
    # the dynamically-added group mark below even when xdist is disabled
    # (``-p no:xdist``), where it is simply inert.
    config.addinivalue_line(
        "markers",
        "xdist_group(name): assign tests to an xdist worker group (pytest-xdist).",
    )

    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


# Memory-heavy tests (the e2e / CV / regression suites) each build a feature
# matrix and train Ridge+LightGBM+NN. Under xdist's per-file scheduling a
# position shard's THREE heavy files (test_pipeline_e2e / test_run_cv_pipeline /
# test_regression) land on DIFFERENT workers and their per-worker memory peaks
# OVERLAP at the run tail, cumulatively crossing the ~15 GiB hosted-runner
# ceiling -> a worker is OOM-SIGKILLed ("[gwN] node down", no faulthandler dump).
# Tagging every such item into one ``xdist_group`` pins them onto a SINGLE worker
# under ``--dist=loadgroup`` (the position shards, set in tests.yml) so at most
# one pipeline is resident at a time (measured single-worker peak ~1.33 GiB)
# while the light unit tests keep parallelizing on the other workers. The mark
# is INERT under the other dist modes (loadfile / worksteal / load), so local
# ``pytest -m unit``, the ``shared`` shard, and the ``serving`` shard are
# unchanged.
_TRAIN_GROUP_MARKERS = ("e2e", "regression", "integration")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Serialize the memory-heavy training tests onto one xdist worker.

    See the ``_TRAIN_GROUP_MARKERS`` comment. Only takes effect under
    ``--dist=loadgroup``; a no-op under every other dist mode.

    ``tryfirst=True`` is load-bearing: xdist's worker-side
    ``pytest_collection_modifyitems`` (remote.py, a plain hookimpl) reads the
    ``xdist_group`` mark and bakes it into each item's nodeid as an ``@group``
    suffix — which is what LoadGroupScheduling keys on. We must add the mark
    BEFORE that runs, or the grouping is silently inert and the heavy tests
    scatter across workers (the exact OOM this fix targets — caught only on CI,
    since a high-RAM box absorbs the scatter and still passes).
    """
    for item in items:
        if any(item.get_closest_marker(marker) for marker in _TRAIN_GROUP_MARKERS):
            item.add_marker(pytest.mark.xdist_group("train"))
