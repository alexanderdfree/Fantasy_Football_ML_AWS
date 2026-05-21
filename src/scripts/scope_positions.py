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

import argparse
import json
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
#
# Note the negative lookahead inside the ``src/batch/`` clause: files whose
# name contains ``tune`` or ``ablate`` (e.g. a misplaced ``launch_tune.py``
# or ``tune_lgbm.py`` or ``ablate_*.py``) are EXCLUDED, because that family
# of files is tuning/ablation infrastructure that fans out via the
# ``retune-nn-batch.yml`` workflow — touching them must not retrain
# production models. New tuner/ablation files should live in
# ``src/tuning/``, which isn't in this regex at all; the lookahead is
# belt-and-suspenders against a future ``src/batch/`` placement.
_GLOBAL_REGEX = re.compile(
    r"^src/(shared|data|features)/"
    r"|^src/batch/(?!.*(?:tune|ablate))"
    r"|^src/config\.py$"
    r"|^requirements\.txt$"
)


def compute_positions(changed_files: Iterable[str]) -> list[str]:
    """Return the positions to retrain given a set of changed file paths.

    - ``tests/`` paths are stripped first — test-only changes don't change
      model artifacts, so they never trigger retraining.
    - Any "global" path (``src/shared/``, ``src/batch/``, ``src/data/``,
      ``src/features/``, ``src/config.py``, ``requirements.txt``) returns
      all six positions.
      Shared code affects everyone.
    - Otherwise, return the positions whose per-position dir ``src/{pos}/``
      was touched. Empty list = no model-relevant change.
    """
    files = [f for f in changed_files if not f.startswith("tests/")]
    if any(_GLOBAL_REGEX.match(f) for f in files):
        return list(ALL_POSITIONS)
    return [pos for pos in ALL_POSITIONS if any(f.startswith(f"src/{pos.lower()}/") for f in files)]


ALL_TEST_SHARDS: tuple[str, ...] = (*ALL_POSITIONS, "shared")

_TEST_DOCS_REGEX = re.compile(r"\.md$|^docs/|^\.github/ISSUE_TEMPLATE/|^\.gitignore$|^LICENSE")
_TEST_GLOBAL_REGEX = re.compile(
    r"^src/(shared|data|features)/"
    r"|^src/(__init__|config)\.py$"
    r"|^conftest\.py$"
    r"|^tests/(conftest\.py|_pipeline_e2e_utils\.py|__init__\.py|fixtures/)"
    r"|^pyproject\.toml$"
    r"|^requirements.*\.txt$"
    r"|^\.github/workflows/tests\.yml$"
)
_TEST_SHARED_REGEX = re.compile(
    r"^src/(serving|batch|scripts|benchmarking|tuning|analysis)/"
    r"|^tests/[^/]+\.py$"
    r"|^tests/(batch|scripts|integration|shared)/"
)
_TEST_PER_POSITION_REGEX = {
    pos: re.compile(rf"^(src/{pos.lower()}/|tests/{pos.lower()}/)") for pos in ALL_POSITIONS
}


def compute_test_shards(changed_files: Iterable[str]) -> list[str]:
    """Return test matrix shards (positions + 'shared') given changed paths.

    Rules (in order):
      1. Strip docs/license-only paths. If nothing remains → [].
      2. Any global trigger (shared code, infra, deps, test plumbing) → all 7.
      3. Per-position: src/{pos}/ or tests/{pos}/ → that position.
      4. Cross-cutting dirs (src/serving, src/batch, src/scripts, src/benchmarking,
         src/tuning, src/analysis, top-level tests/*.py, tests/{batch,scripts,
         integration,shared}/) → 'shared'.
      5. Fallback: if no rule matched, run all 7 (conservative).
    """
    files = [f for f in changed_files if f]
    non_docs = [f for f in files if not _TEST_DOCS_REGEX.search(f)]
    if not non_docs:
        return []
    if any(_TEST_GLOBAL_REGEX.search(f) for f in non_docs):
        return list(ALL_TEST_SHARDS)
    shards: list[str] = [
        pos
        for pos, per_pos in _TEST_PER_POSITION_REGEX.items()
        if any(per_pos.search(f) for f in non_docs)
    ]
    if any(_TEST_SHARED_REGEX.search(f) for f in non_docs):
        shards.append("shared")
    if not shards:
        return list(ALL_TEST_SHARDS)
    return shards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    args = parser.parse_args()
    files = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    if args.mode == "test":
        sys.stdout.write(json.dumps(compute_test_shards(files)) + "\n")
    else:
        positions = compute_positions(files)
        if positions:
            sys.stdout.write(" ".join(positions) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
