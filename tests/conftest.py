"""Pytest configuration for top-level ``tests/``.

Registers the project-wide marker vocabulary so tests under ``tests/`` don't
emit ``PytestUnknownMarkWarning``. Per-position test folders already register
the same markers — this file covers the pipeline-E2E + reproducibility tests
that live outside any position directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when pytest is invoked from ``tests/``.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def pytest_configure(config):
    """Register markers mirroring the per-position conftests."""
    markers = [
        "unit: fast isolated test (<=1s), no external I/O, no training loops",
        "integration: multi-component test (<10s each)",
        "e2e: full-pipeline smoke test (<20s each)",
        "regression: numerical-performance assertion (may need fixture data)",
        "slow: excluded from default run",
    ]
    for m in markers:
        config.addinivalue_line("markers", m)
