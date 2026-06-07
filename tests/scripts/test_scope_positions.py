"""Tests for src/scripts/scope_positions.py — train-detect path → positions map.

Pins the contract called out of ``.github/workflows/train-batch.yml`` and
``train-ec2.yml``: tests/ stripping, global-trigger fan-out, per-position
scoping, path anchoring. A regression here would silently over- or
under-retrain in CI and burn GPU-hours or ship stale models, so the cases
below are intentionally thorough.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts import scope_positions  # noqa: E402

pytestmark = pytest.mark.unit


ALL_SIX = ["QB", "RB", "WR", "TE", "K", "DST"]


# --------------------------------------------------------------------------
# Empty / filter-only inputs — should return [] (skip training)
# --------------------------------------------------------------------------


class TestEmptyOrFiltered:
    def test_empty_input(self):
        assert scope_positions.compute_positions([]) == []

    def test_tests_only_single(self):
        assert scope_positions.compute_positions(["tests/qb/test_features.py"]) == []

    def test_tests_only_shared(self):
        assert scope_positions.compute_positions(["tests/shared/test_pipeline.py"]) == []

    def test_tests_only_multiple(self):
        assert (
            scope_positions.compute_positions(
                ["tests/qb/test_a.py", "tests/rb/test_b.py", "tests/shared/test_c.py"]
            )
            == []
        )

    def test_top_level_docs(self):
        assert scope_positions.compute_positions(["README.md"]) == []

    def test_docs_dir(self):
        assert scope_positions.compute_positions(["docs/batch_design.md"]) == []

    def test_random_top_level(self):
        # Files that match neither global nor per-position patterns don't
        # trigger retraining. Train detect ignores them; tests.yml detect
        # falls back to "all shards" for safety, but that's its own logic.
        assert scope_positions.compute_positions(["pyproject.toml"]) == []
        assert scope_positions.compute_positions([".gitignore"]) == []


# --------------------------------------------------------------------------
# Global triggers — each pattern should fan out to all six
# --------------------------------------------------------------------------


class TestGlobalTriggers:
    @pytest.mark.parametrize(
        "path",
        [
            "src/shared/pipeline.py",
            "src/shared/team_box_score.py",  # the e62807e trigger
            "src/batch/train.py",
            "src/data/loader.py",
            "src/features/foo.py",
            "src/config.py",
            # src/__init__.py is a top-level global like src/config.py — a
            # package-level constant or re-export added there affects every
            # position. Previously omitted from the train-mode regex (it
            # only matched src/config.py), so an edit silently skipped the
            # retrain; the train-detect path now mirrors the test-detect
            # regex's ^src/(__init__|config)\.py$.
            "src/__init__.py",
            "requirements.txt",
        ],
    )
    def test_global_path_fans_out_to_all_six(self, path):
        assert scope_positions.compute_positions([path]) == ALL_SIX

    def test_global_wins_over_per_position(self):
        # A diff that touches both a global path AND a single position dir
        # must retrain all six (shared change might invalidate every model).
        assert (
            scope_positions.compute_positions(["src/shared/pipeline.py", "src/qb/features.py"])
            == ALL_SIX
        )

    def test_global_wins_with_tests_present(self):
        assert (
            scope_positions.compute_positions(
                ["tests/qb/test_x.py", "src/shared/models.py", "src/qb/foo.py"]
            )
            == ALL_SIX
        )


# --------------------------------------------------------------------------
# Per-position scoping — each pos's dir scopes to that pos only
# --------------------------------------------------------------------------


class TestPerPositionScoping:
    @pytest.mark.parametrize("pos", ALL_SIX)
    def test_single_position_dir(self, pos):
        assert scope_positions.compute_positions([f"src/{pos.lower()}/features.py"]) == [pos]

    def test_two_positions(self):
        assert scope_positions.compute_positions(["src/qb/features.py", "src/rb/config.py"]) == [
            "QB",
            "RB",
        ]

    def test_all_six_per_position_dirs(self):
        # Touching every per-position dir without any global hit produces
        # the same list as a global hit (a sanity convergence check).
        assert (
            scope_positions.compute_positions([f"src/{p.lower()}/x.py" for p in ALL_SIX]) == ALL_SIX
        )

    def test_position_order_preserved(self):
        # Output ordering matches ALL_POSITIONS order, not the input order.
        # The workflow consumes this as a space-separated list and the order
        # affects log readability; deterministic output also helps diffing.
        result = scope_positions.compute_positions(
            ["src/dst/features.py", "src/qb/features.py", "src/k/features.py"]
        )
        assert result == ["QB", "K", "DST"]

    def test_position_dir_and_test_dir_strips_tests(self):
        # tests/qb/ is stripped; src/qb/ stays → only QB retrains.
        assert scope_positions.compute_positions(
            ["src/qb/features.py", "tests/qb/test_features.py"]
        ) == ["QB"]


# --------------------------------------------------------------------------
# Non-training src/ subdirs — should NOT trigger retraining
# --------------------------------------------------------------------------


class TestNonTrainingSrcDirs:
    @pytest.mark.parametrize(
        "path",
        [
            "src/scripts/promote.py",
            "src/scripts/scope_positions.py",
            "src/serving/app.py",
            "src/benchmarking/benchmark.py",
            "src/tuning/tune_lgbm.py",
            "src/tuning/launch_tune.py",  # moved from src/batch/ — see PR #280 retrospective
            "src/analysis/error_analysis.py",
        ],
    )
    def test_non_training_src_dir_returns_empty(self, path):
        # These dirs ship code that doesn't affect model training output
        # (serving, scripts, post-hoc analysis). The train detect job
        # correctly excludes them — verify the script does too.
        assert scope_positions.compute_positions([path]) == []


# --------------------------------------------------------------------------
# src/batch/ tune/ablate carve-out — hardening against the PR #280 regression
# --------------------------------------------------------------------------


class TestSrcBatchTuneAblateExcluded:
    """Files inside ``src/batch/`` whose name contains ``tune`` or ``ablate``
    are tuning/ablation infrastructure. They must NOT trigger a retrain.

    PR #280 (``chore(tuning): address second-round review nits``) touched
    ``src/batch/launch_tune.py`` and triggered a full 6-position retrain on
    AWS Batch — a wasteful false positive that motivated this carve-out and
    the file move to ``src/tuning/launch_tune.py``. The lookahead inside
    the regex (`(?!.*(?:tune|ablate))`) is belt-and-suspenders against a
    future file accidentally placed back under ``src/batch/``.
    """

    @pytest.mark.parametrize(
        "path",
        [
            "src/batch/launch_tune.py",  # original PR #280 file (now moved)
            "src/batch/tune_lgbm.py",  # hypothetical future move
            "src/batch/retune_helper.py",  # any tune-named helper
            "src/batch/ablate_thing.py",  # ablation studies
            "src/batch/sub/tune_thing.py",  # nested
        ],
    )
    def test_src_batch_tune_named_files_excluded(self, path):
        assert scope_positions.compute_positions([path]) == []

    @pytest.mark.parametrize(
        "path",
        [
            "src/batch/train.py",
            "src/batch/__init__.py",
            # "turnaround" contains "tur..." but NOT "tune" as a substring —
            # confirms the regex doesn't false-match on similar-looking names.
            "src/batch/turnaround.py",
            # Precision: the launch.py/benchmark.py carve-out is an EXACT
            # basename match, so a differently-named file still fans out.
            "src/batch/benchmark_runner.py",
        ],
    )
    def test_src_batch_real_training_still_triggers(self, path):
        assert scope_positions.compute_positions([path]) == ALL_SIX

    @pytest.mark.parametrize(
        "path",
        [
            "src/batch/launch.py",
            "src/batch/benchmark.py",
            # the local image build/push helper — packaging only, never trains (#362 F13)
            "src/batch/build_and_push.sh",
        ],
    )
    def test_src_batch_orchestration_files_excluded(self, path):
        # launch.py (job submission) and benchmark.py (read-only post-hoc
        # benchmark aggregation) never change a model artifact, so editing them
        # must NOT trigger a 6-position GPU retrain. Exact-basename match —
        # benchmark_runner.py above still fans out.
        assert scope_positions.compute_positions([path]) == []

    def test_mixed_tune_and_train_triggers(self):
        # A diff that touches both a tuning file AND a real training file
        # in src/batch/ must still retrain all six — the training change is
        # what matters; the tuning file is just along for the ride.
        assert (
            scope_positions.compute_positions(["src/batch/launch_tune.py", "src/batch/train.py"])
            == ALL_SIX
        )

    def test_tune_file_alone_no_retrain(self):
        # Sanity: the literal PR #280 scope (just src/batch/launch_tune.py
        # before the file move) would produce an empty list under the new
        # regex. Documents the regression.
        assert scope_positions.compute_positions(["src/batch/launch_tune.py"]) == []


# --------------------------------------------------------------------------
# Path-anchor sanity — defensive checks against accidental substring matches
# --------------------------------------------------------------------------


class TestPathAnchoring:
    def test_vendored_src_qb_does_not_match(self):
        # If a vendor tree mirrored our layout, we still wouldn't want to
        # treat its changes as ours.
        assert scope_positions.compute_positions(["vendor/src/qb/features.py"]) == []

    def test_position_dir_prefix_collision(self):
        # `src/qbsub/` is not `src/qb/`; the trailing slash on the prefix
        # check prevents an off-by-one. Same idea for `src/datasets/`.
        assert scope_positions.compute_positions(["src/qbsub/foo.py"]) == []
        assert scope_positions.compute_positions(["src/datasets/foo.py"]) == []

    def test_global_anchor_not_substring(self):
        # `infra/src/shared/x.py` shouldn't match the global pattern.
        assert scope_positions.compute_positions(["infra/src/shared/x.py"]) == []

    def test_config_py_match_is_exact(self):
        # `src/qb/config.py` should NOT match the `src/config.py` global,
        # only the per-position rule for QB.
        assert scope_positions.compute_positions(["src/qb/config.py"]) == ["QB"]

    def test_init_py_match_is_exact(self):
        # `src/__init__.py` is a top-level global → all six. A per-position
        # `src/qb/__init__.py` is NOT the top-level global; it matches only
        # the per-position prefix rule for QB (the `^src/(__init__|config)`
        # alternation is anchored with a trailing `$`, so `src/qb/__init__.py`
        # can't satisfy it).
        assert scope_positions.compute_positions(["src/__init__.py"]) == ALL_SIX
        assert scope_positions.compute_positions(["src/qb/__init__.py"]) == ["QB"]

    def test_requirements_dev_does_not_trigger_global(self):
        # Only top-level `requirements.txt` is the global; dev/test variants
        # affect the test image but not what trains. (`requirements-dev.txt`
        # is consumed by tests.yml and the train *job*, not the *detect*.)
        assert scope_positions.compute_positions(["requirements-dev.txt"]) == []


# --------------------------------------------------------------------------
# CLI smoke tests — exercise the actual __main__ entry point
# --------------------------------------------------------------------------


class TestCLI:
    def _run(self, stdin: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "src.scripts.scope_positions"],
            input=stdin,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            check=False,
        )

    def test_cli_multi_position(self):
        result = self._run("src/qb/features.py\nsrc/rb/config.py\n")
        assert result.returncode == 0
        assert result.stdout == "QB RB\n"
        assert result.stderr == ""

    def test_cli_global_trigger(self):
        result = self._run("src/shared/pipeline.py\n")
        assert result.returncode == 0
        assert result.stdout == "QB RB WR TE K DST\n"

    def test_cli_empty_input_empty_output(self):
        result = self._run("")
        assert result.returncode == 0
        assert result.stdout == ""

    def test_cli_tests_only_empty_output(self):
        result = self._run("tests/qb/test_x.py\ntests/shared/test_y.py\n")
        assert result.returncode == 0
        assert result.stdout == ""

    def test_cli_blank_lines_ignored(self):
        # `git diff --name-only` shouldn't emit blanks, but the bash invocation
        # `printf '%s\n' $files` collapses whitespace in ways that have caused
        # off-by-one issues elsewhere; defensive.
        result = self._run("\nsrc/qb/x.py\n\nsrc/dst/y.py\n\n")
        assert result.returncode == 0
        assert result.stdout == "QB DST\n"


# --------------------------------------------------------------------------
# Drift guard — keep the hardcoded list in sync with src.shared.registry
# --------------------------------------------------------------------------


class TestRegistryDriftGuard:
    def test_all_positions_matches_registry(self):
        # scope_positions.py hardcodes ALL_POSITIONS so it stays dep-free
        # (no `from src.shared.registry import ...`, which would pull torch
        # via aggregate_targets). This test runs in the `shared` shard which
        # already has torch, so we can do the equality check here.
        from src.shared.registry import ALL_POSITIONS as REG_ALL

        assert tuple(REG_ALL) == scope_positions.ALL_POSITIONS, (
            "src.shared.registry.ALL_POSITIONS and "
            "src.scripts.scope_positions.ALL_POSITIONS have drifted. "
            "Update scope_positions.ALL_POSITIONS (and the detect jobs' "
            'fallback `ALL="..."` lists) to match.'
        )

    def test_all_positions_matches_position_enum(self):
        """``Position`` enum is the canonical source of truth; this guard
        pins ``scope_positions.ALL_POSITIONS`` to its value set so adding,
        removing, or renaming a position has to touch the enum first."""
        from src.shared.position import Position

        assert tuple(p.value for p in Position) == scope_positions.ALL_POSITIONS, (
            "src.shared.position.Position members and "
            "src.scripts.scope_positions.ALL_POSITIONS have drifted. "
            "Update the Position enum first; scope_positions stays "
            "dep-free for the CI detect job."
        )


# --------------------------------------------------------------------------
# compute_test_shards — tests.yml `detect` path → matrix shards
# --------------------------------------------------------------------------


ALL_SEVEN = list(scope_positions.ALL_TEST_SHARDS)


class TestComputeTestShards:
    @pytest.mark.parametrize(
        "files",
        [
            ["README.md"],
            ["docs/batch_design.md"],
            ["benchmark_history/2026-05-21T05-00-00Z.json"],
            [".gitignore"],
            ["LICENSE"],
            [".github/ISSUE_TEMPLATE/bug.md"],
            ["README.md", "docs/batch_design.md", ".gitignore", "LICENSE"],
        ],
    )
    def test_docs_only_returns_empty(self, files):
        assert scope_positions.compute_test_shards(files) == []

    def test_benchmark_history_markdown_is_docs_only_not_all_shards(self):
        assert scope_positions.compute_test_shards(["benchmark_history/README.md"]) == []

    @pytest.mark.parametrize(
        "path",
        [
            "src/shared/pipeline.py",
            "src/data/loader.py",
            "src/features/foo.py",
            "src/__init__.py",
            "src/config.py",
            "conftest.py",
            "tests/conftest.py",
            "tests/_pipeline_e2e_utils.py",
            "tests/__init__.py",
            "tests/fixtures/snap.parquet",
            "pyproject.toml",
            "requirements.txt",
            "requirements-dev.txt",
            ".github/workflows/tests.yml",
        ],
    )
    def test_global_path_fans_out_to_all_seven(self, path):
        assert scope_positions.compute_test_shards([path]) == ALL_SEVEN

    @pytest.mark.parametrize(
        "pos",
        ["QB", "RB", "WR", "TE", "K", "DST"],
    )
    def test_single_position_from_src(self, pos):
        assert scope_positions.compute_test_shards([f"src/{pos.lower()}/features.py"]) == [pos]

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("tests/qb/test_features.py", ["QB"]),
            ("tests/rb/test_features.py", ["RB"]),
            ("tests/wr/test_features.py", ["WR"]),
            ("tests/te/test_features.py", ["TE"]),
            ("tests/k/test_features.py", ["K"]),
            ("tests/dst/test_run_pipeline.py", ["DST"]),
        ],
    )
    def test_single_position_from_tests(self, path, expected):
        assert scope_positions.compute_test_shards([path]) == expected

    def test_multiple_positions_ordered(self):
        assert scope_positions.compute_test_shards(
            ["src/qb/features.py", "tests/wr/test_x.py"]
        ) == ["QB", "WR"]

    def test_position_order_matches_all_positions(self):
        result = scope_positions.compute_test_shards(
            ["src/dst/features.py", "src/qb/features.py", "src/k/features.py"]
        )
        assert result == ["QB", "K", "DST"]

    @pytest.mark.parametrize(
        "path",
        [
            "src/serving/app.py",
            "src/batch/launch.py",
            "src/scripts/promote.py",
            "src/benchmarking/benchmark.py",
            "src/tuning/tune_lgbm.py",
            "src/analysis/error_analysis.py",
            "tests/batch/test_foo.py",
            "tests/scripts/test_foo.py",
            "tests/analysis/test_foo.py",
            "tests/tuning/test_foo.py",
            "tests/integration/test_foo.py",
            "tests/shared/test_foo.py",
            "tests/test_top_level.py",
        ],
    )
    def test_shared_shard(self, path):
        assert scope_positions.compute_test_shards([path]) == ["shared"]

    def test_position_plus_shared(self):
        assert scope_positions.compute_test_shards(
            ["src/qb/features.py", "src/serving/app.py"]
        ) == ["QB", "shared"]

    def test_mixed_docs_and_code(self):
        assert scope_positions.compute_test_shards(["README.md", "src/qb/features.py"]) == ["QB"]

    @pytest.mark.parametrize(
        "files",
        [
            [".github/workflows/deploy.yml"],
            ["Dockerfile"],
            ["unrelated/path/file.py"],
        ],
    )
    def test_unmatched_falls_back_to_all_seven(self, files):
        assert scope_positions.compute_test_shards(files) == ALL_SEVEN

    def test_empty_input(self):
        assert scope_positions.compute_test_shards([]) == []
