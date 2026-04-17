"""Root conftest for the entire test suite.

Responsibilities:
1. Put the project root on ``sys.path`` exactly once so every test module can
   ``import shared.*`` / ``import QB.qb_data`` etc. without each position-level
   ``conftest.py`` having to re-wire the path. (The six position conftests
   still contain their own path-wiring for now; they will be retired in
   their own PRs.)
2. Register the project-wide pytest markers as a belt-and-suspenders backup
   to ``pyproject.toml`` so ``--strict-markers`` never trips on a fresh
   checkout where ``pyproject.toml`` might be missing.
"""
from __future__ import annotations

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
    """Register markers as a backup to pyproject.toml.

    Pytest tolerates duplicate registration, so position-level conftests
    that re-register the same markers will not conflict.
    """
    for name, description in _MARKERS:
        config.addinivalue_line("markers", f"{name}: {description}")
