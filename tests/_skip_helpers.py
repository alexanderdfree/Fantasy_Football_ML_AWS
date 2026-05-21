"""Shared skip-helpers for tests that depend on engineered splits.

The E2E and reproducibility tests need ``data/splits/{train,val,test}.parquet``,
which are produced by the local data-pull workflow documented in SETUP.md.
CI does not (yet) build these — see the test-skipping audit in
.claude/plans/. Until the CI data step lands we want two behaviors:

* **Local dev**: skip silently if splits are absent (mirrors the original
  fixture behavior; preserves dev ergonomics on fresh checkouts).
* **CI**: fail loudly so the green checkmark cannot mask a missing data
  step. The ``ALLOW_SKIP_E2E=1`` env var opts back into the skip — set in
  ``.github/workflows/tests.yml`` as a TODO marker that should be removed
  the moment a CI data step is wired up.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def require_splits(splits_root: Path) -> None:
    """Skip locally / fail in CI when ``data/splits/train.parquet`` is missing.

    Call this from fixtures or tests that need engineered splits. ``CI`` is
    the standard env var set by GitHub Actions and most other CI runners.
    """
    if (splits_root / "train.parquet").exists():
        return
    msg = f"engineered splits absent at {splits_root} (run data pull — see SETUP.md)"
    if os.environ.get("CI") and not os.environ.get("ALLOW_SKIP_E2E"):
        pytest.fail(
            f"{msg}. CI must produce splits before running E2E tests; "
            "set ALLOW_SKIP_E2E=1 to opt back into the silent-skip behavior."
        )
    pytest.skip(msg)
