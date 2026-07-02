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
# Note the negative lookahead inside the ``src/batch/`` clause. Two families
# of files under ``src/batch/`` are EXCLUDED from the global trigger:
#
#   1. ``tune`` / ``ablate`` (e.g. a misplaced ``launch_tune.py``,
#      ``tune_lgbm.py``, ``ablate_*.py``) — tuning/ablation infrastructure that
#      fans out via ``retune-nn-batch.yml``; touching it must not retrain
#      production models. New tuner/ablation files should live in
#      ``src/tuning/`` (not in this regex at all); the lookahead is
#      belt-and-suspenders against a future ``src/batch/`` placement.
#   2. Exactly ``launch.py``, ``benchmark.py``, and ``build_and_push.sh`` —
#      job-submission orchestration, read-only post-hoc benchmark aggregation,
#      and the local image build/push helper. None changes a model artifact, so
#      editing them shouldn't burn a 6-position GPU retrain. Matched by exact
#      basename (``(?:launch|benchmark)\.py$`` / ``build_and_push\.sh$``),
#      NOT substring, so a future ``benchmark_runner.py`` / ``relaunch.py``
#      still triggers conservatively. ACCEPTED RISK: ``launch.py`` also owns
#      the seed default, job-definition dispatch, and data-split upload — a
#      change there that *does* affect what trains will now silently skip the
#      retrain, so the operator must trigger ``workflow_dispatch`` manually.
#      (Test sharding is unaffected — see ``_TEST_SHARED_REGEX`` — so these
#      files' tests still run in CI.)
# ``^src/(__init__|config)\.py$`` covers both top-level globals: src/config.py
# (SEASONS / POSITIONS / scoring dicts / TOP_K_RANKING) and src/__init__.py
# (currently empty, but a re-export or package-level constant added there would
# affect every position). Mirrors the test-mode ``_TEST_GLOBAL_REGEX`` so the
# train-detect and test-detect paths agree on what counts as a global config
# change — previously only src/config.py matched, so an edit to src/__init__.py
# silently scoped to no positions and skipped the retrain.
_GLOBAL_REGEX = re.compile(
    r"^src/(shared|data|features)/"
    r"|^src/batch/(?!.*(?:tune|ablate)|(?:launch|benchmark)\.py$|build_and_push\.sh$)"
    r"|^src/(__init__|config)\.py$"
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


# --- Pre-PR benchmark-gate (B2) scoping -------------------------------------
#
# The B2 gate in .claude/hooks/pre-pr.sh needs a finer split than the train
# trigger above: shared-code changes require benchmark evidence on at least
# one position (they run the same code path for every position), while two
# _GLOBAL_REGEX families are deliberately EXEMPT from the *local* benchmark
# gate even though they retrain in CI:
#
#   - ``src/batch/**`` — Batch/ECS orchestration; not exercised by the local
#     benchmark path at all, so demanding a local run proves nothing.
#   - ``requirements.txt`` — a dep pin bump changes the *environment*, not the
#     model code; the local venv may not even have the new pin installed yet.
#
# Exempt paths are still REPORTED (the hook prints them "exempt, not gated")
# so the skip is visible, never silent. The per-position prefix rule is shared
# with compute_positions, so any file under src/{pos}/ — including
# __init__.py, diagnostic CLIs, and future additions — scopes that position.
# The soundness invariant "every path that can scope a position into the gate
# is inside that position's fingerprint manifest" is pinned by
# tests/scripts/test_bench_fingerprint.py against
# src.scripts.bench_fingerprint.GLOBAL_PATHS.
_BENCH_SHARED_REGEX = re.compile(r"^src/(shared|data|features)/|^src/(__init__|config)\.py$")
# Exempt-visibility: report EVERY src/batch/** + requirements.txt path (not
# just the _GLOBAL_REGEX-matching subset) — the lookahead-excluded batch files
# (launch.py / benchmark.py / *tune* / *ablate* / build_and_push.sh) are just
# as un-gated locally, and a silent drop would contradict the "exempt paths
# are reported, never silent" contract.
_BENCH_EXEMPT_REGEX = re.compile(r"^src/batch/|^requirements\.txt$")


def compute_benchmark_scope(changed_files: Iterable[str]) -> dict:
    """Classify changed paths for the pre-PR benchmark gate.

    Returns ``{"positions": [...], "shared": bool, "exempt": [...]}``:
    ``positions`` = per-position dirs touched (benchmark evidence required for
    each); ``shared`` = any shared/pipeline path touched (evidence required on
    at least one position); ``exempt`` = paths the train trigger treats as
    global but the local benchmark gate deliberately does not gate.
    """
    files = [f for f in changed_files if f and not f.startswith("tests/")]
    exempt = [f for f in files if _BENCH_EXEMPT_REGEX.match(f)]
    return {
        "positions": [
            pos for pos in ALL_POSITIONS if any(f.startswith(f"src/{pos.lower()}/") for f in files)
        ],
        "shared": any(_BENCH_SHARED_REGEX.match(f) for f in files),
        "exempt": exempt,
    }


ALL_TEST_SHARDS: tuple[str, ...] = (*ALL_POSITIONS, "serving", "shared")

_TEST_DOCS_REGEX = re.compile(
    r"\.md$|^docs/|^benchmark_history/|^\.github/ISSUE_TEMPLATE/|^\.gitignore$|^LICENSE"
)
_TEST_GLOBAL_REGEX = re.compile(
    r"^src/(shared|data|features)/"
    r"|^src/(__init__|config)\.py$"
    r"|^conftest\.py$"
    r"|^tests/(conftest\.py|_pipeline_e2e_utils\.py|__init__\.py|fixtures/)"
    r"|^pyproject\.toml$"
    r"|^requirements.*\.txt$"
    r"|^\.github/workflows/tests\.yml$"
)
# Serving suite (Flask app + serving libs): memory-heavy model-loading tests
# (tests/test_app*.py) that were tipping the `shared` shard over the runner's RAM
# under -n auto. Split into its own matrix shard (#1056). Anchored so a deeper
# look-alike path (tests/sub/test_app_x.py) can't spuriously match.
_TEST_SERVING_REGEX = re.compile(r"^src/serving/" r"|^tests/test_app[^/]*\.py$")
_TEST_SHARED_REGEX = re.compile(
    r"^src/(batch|scripts|benchmarking|tuning|analysis)/"
    r"|^tests/(?!test_app[^/]*\.py$)[^/]+\.py$"
    r"|^tests/(analysis|batch|hooks|scripts|integration|shared|tuning)/"
)
_TEST_PER_POSITION_REGEX = {
    pos: re.compile(rf"^(src/{pos.lower()}/|tests/{pos.lower()}/)") for pos in ALL_POSITIONS
}


def compute_test_shards(changed_files: Iterable[str]) -> list[str]:
    """Return test matrix shards (positions + 'serving' + 'shared') given changed paths.

    Rules (in order):
      1. Strip docs/license-only paths. If nothing remains → [].
      2. Any global trigger (shared code, infra, deps, test plumbing) → all 8.
      3. Per-position: src/{pos}/ or tests/{pos}/ → that position.
      4. Serving: src/serving/ or tests/test_app*.py → 'serving'.
      5. Cross-cutting dirs (src/batch, src/scripts, src/benchmarking, src/tuning,
         src/analysis, other top-level tests/*.py, tests/{analysis,batch,scripts,
         integration,shared,tuning}/) → 'shared'.
      6. Fallback: if no rule matched, run all 8 (conservative).
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
    if any(_TEST_SERVING_REGEX.search(f) for f in non_docs):
        shards.append("serving")
    if any(_TEST_SHARED_REGEX.search(f) for f in non_docs):
        shards.append("shared")
    if not shards:
        return list(ALL_TEST_SHARDS)
    return shards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "benchmark"], default="train")
    args = parser.parse_args()
    files = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    if args.mode == "test":
        sys.stdout.write(json.dumps(compute_test_shards(files)) + "\n")
    elif args.mode == "benchmark":
        sys.stdout.write(json.dumps(compute_benchmark_scope(files)) + "\n")
    else:
        positions = compute_positions(files)
        if positions:
            sys.stdout.write(" ".join(positions) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
