"""Shared pytest config for tests/.

Ensures the project root is on ``sys.path`` so ``from src.data.loader
import ...`` works, and registers custom markers used by the loader-contract
suite so pytest doesn't emit "unknown mark" warnings.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "unit: lightweight test that only inspects checked-in fixture data",
    )
    config.addinivalue_line(
        "markers",
        "integration: test that exercises real code paths (mocked; no network)",
    )
