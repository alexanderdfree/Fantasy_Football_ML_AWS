"""Map changed file paths to the NFL position models that need retraining.

Consumed by ``.github/workflows/train-batch.yml`` and ``train-ec2.yml`` ``detect``
jobs to decide which positions the Batch fan-out / EC2 sequential trainer
should retrain. Lives in ``src/scripts/`` next to other operator CLIs, but is
deliberately pure-stdlib so the detect job can call it from vanilla
``python3`` on ubuntu-latest without installing project deps.

The bash that previously lived inline in both workflows is replaced by a
single pipe into ``python3 -m src.scripts.scope_positions``; this module is
where the rules now live and where regressions are caught by
``tests/scripts/test_scope_positions.py``.

CLI:
    git diff --name-only HEAD^ HEAD | python3 -m src.scripts.scope_positions

Contract: emit a space-separated list of positions to stdout. An empty
output (script exits 0, no stdout) means the train job should be skipped.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable

# Hardcoded so this module is importable with zero project deps (the detect
# job runs without `pip install -e .` or requirements*.txt). The drift guard
# in ``tests/scripts/test_scope_positions.py`` asserts equality with
# ``src.shared.registry.ALL_POSITIONS`` so adding a position there without
# updating here trips a unit test in CI.
ALL_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DST")

# Paths whose changes invalidate every position's model artifact. Matches the
# bash regex that used to live in both workflows. Anchored at start-of-path
# so a same-named substring deeper in the tree doesn't spuriously match.
_GLOBAL_REGEX = re.compile(
    r"^src/(shared|batch|data|features|models|training|evaluation)/"
    r"|^src/config\.py$"
    r"|^requirements\.txt$"
)


def compute_positions(changed_files: Iterable[str]) -> list[str]:
    """Return the positions to retrain given a set of changed file paths.

    - ``tests/`` paths are stripped first — test-only changes don't change
      model artifacts, so they never trigger retraining.
    - Any "global" path (``src/shared/``, ``src/batch/``, ``src/data/``,
      ``src/features/``, ``src/models/``, ``src/training/``,
      ``src/evaluation/``, ``src/config.py``, ``requirements.txt``) returns
      all six positions. Shared code affects everyone.
    - Otherwise, return the positions whose per-position dir ``src/{pos}/``
      was touched. Empty list = no model-relevant change.
    """
    files = [f for f in changed_files if not f.startswith("tests/")]
    if any(_GLOBAL_REGEX.match(f) for f in files):
        return list(ALL_POSITIONS)
    return [pos for pos in ALL_POSITIONS if any(f.startswith(f"src/{pos.lower()}/") for f in files)]


def main() -> int:
    files = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    positions = compute_positions(files)
    if positions:
        sys.stdout.write(" ".join(positions) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
