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

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
