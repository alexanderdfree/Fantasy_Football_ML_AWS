"""Regression test for the ``.github/workflows/tests.yml`` ``detect`` job's
``[docs-only]`` commit-range scan.

The contract pinned here is the GIT RANGE LOGIC, not the path → shard mapping
(that lives in ``tests/scripts/test_scope_positions.py``). It guards the
false-green bug confirmed via PR #367 on 2026-05-29:

    On a ``pull_request`` event GitHub checks out the synthetic merge commit
    ``refs/pull/N/merge`` ("Merge <pr-head> into <main-tip>"), so HEAD already
    contains the live main tip. The detect job scanned
    ``github.event.pull_request.base.sha..HEAD`` for a ``[docs-only]`` subject.
    ``base.sha`` lags the live main tip, so the range swept in every commit
    merged to main after the PR's base. An unrelated ``[docs-only]`` main
    commit then tripped ``shards=[]`` and skipped the entire test fan-out
    while ``tests-pass`` reported green.

The fix scopes the scan to the PR's own commits via the merge parents
(``git log HEAD^2 --not HEAD^1``) and the diff to ``HEAD^1...HEAD^2``. These
tests build a synthetic repo reproducing the #367 shape and assert the
parent-based range excludes the unrelated main commit (and that the old
``base..HEAD`` range would NOT have, documenting the bug).

The shell snippets below are kept byte-faithful to the workflow's ``scope``
step — if you change the range/scan logic in ``tests.yml``, change it here too.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    # Hermetic identity + no signing + a fixed default branch so the test runs
    # on a bare CI runner with no global git config.
    base = [
        "git",
        "-c",
        "init.defaultBranch=main",
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "-C",
        str(repo),
    ]
    return subprocess.run([*base, *args], capture_output=True, text=True, check=check)


def _rev(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", ref).stdout.strip()


def _verify(repo: Path, ref: str) -> bool:
    """Mirror the workflow's ``git rev-parse --verify <ref>`` branch predicate."""
    return _git(repo, "rev-parse", "--verify", ref, check=False).returncode == 0


def _commit(repo: Path, filename: str, content: str, message: str) -> str:
    path = repo / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    _git(repo, "add", filename)
    _git(repo, "commit", "-q", "-m", message)
    return _rev(repo, "HEAD")


# Byte-faithful copy of tests.yml::detect's `[docs-only]` subject-line scan.
# Range args (the workflow's "${commit_args[@]}") are passed positionally and
# consumed via "$@". Prints "true" / "false".
_SCAN = r"""
set -eu
docs_only=false
for sha in $(git log --format=%H "$@"); do
  if git log -1 --format=%B "$sha" | awk 'NR==1 || /^\* /' | grep -qF '[docs-only]'; then
    docs_only=true
    break
  fi
done
printf '%s\n' "$docs_only"
"""


def _scan_docs_only(repo: Path, *range_args: str) -> bool:
    out = subprocess.run(
        ["bash", "-c", _SCAN, "bash", *range_args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip() == "true"


def _changed_files(repo: Path, *diff_args: str) -> list[str]:
    out = _git(repo, "diff", "--name-only", *diff_args)
    return sorted(line for line in out.stdout.splitlines() if line)


@pytest.fixture
def pr_merge_repo(tmp_path: Path) -> Path:
    """Reproduce PR #367's shape:

    - ``base`` commit on main (the PR's base, standing in for ``39aa8a6``).
    - The PR branch forks at ``base`` and makes a real code change
      (``src/serving/app.py``).
    - Meanwhile main advances with an UNRELATED ``[docs-only]`` commit
      (standing in for PR #368's ``0a17b4e``).
    - GitHub's merge preview = ``main`` (with the docs-only commit) merged with
      the PR head, first parent = main tip, second parent = PR head — exactly
      how ``refs/pull/N/merge`` is built.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")

    _commit(repo, "README.md", "v0\n", "chore: base")

    _git(repo, "checkout", "-q", "-b", "pr")
    _commit(repo, "src/serving/app.py", "x = 1\n", "feat(serving): change app")

    _git(repo, "checkout", "-q", "main")
    _commit(
        repo,
        "notes.md",
        "docs\n",
        "[docs-only] chore(routines): version-control prompt",
    )

    # --no-ff guarantees a merge commit (not a fast-forward) with main as the
    # first parent and pr as the second — matching the merge-preview topology.
    _git(repo, "merge", "-q", "--no-ff", "-m", "Merge pr into main", "pr")
    return repo


class TestPRMergeRangeScoping:
    def test_merge_commit_has_second_parent(self, pr_merge_repo: Path):
        # The workflow's `if git rev-parse --verify HEAD^2` branch is taken on a
        # pull_request merge commit; this is what selects parent-based scoping.
        assert _verify(pr_merge_repo, "HEAD^2") is True
        assert _verify(pr_merge_repo, "HEAD^1") is True

    def test_parent_scoped_range_excludes_unrelated_main_docs_only(self, pr_merge_repo: Path):
        # THE FIX: HEAD^2 --not HEAD^1 = the PR's own commits only. The
        # unrelated main [docs-only] commit lives on HEAD^1, so it is excluded
        # and the scan correctly reports docs_only=false → the fan-out runs.
        assert _scan_docs_only(pr_merge_repo, "HEAD^2", "--not", "HEAD^1") is False

    def test_legacy_base_range_would_have_false_skipped(self, pr_merge_repo: Path):
        # THE BUG: the old `base..HEAD` range (base = the PR's fork point, which
        # the lagging github.event.pull_request.base.sha pinned to) sweeps in
        # main's [docs-only] commit and trips the false skip. This asserts the
        # exact failure the fix prevents, so a regression to base-style ranging
        # is caught here rather than in a silent CI green.
        base = _git(pr_merge_repo, "merge-base", "HEAD^1", "HEAD^2").stdout.strip()
        assert _scan_docs_only(pr_merge_repo, f"{base}..HEAD") is True

    def test_diff_is_scoped_to_pr_files_only(self, pr_merge_repo: Path):
        # The scope_positions diff moved to HEAD^1...HEAD^2 (three-dot = changes
        # since the fork point). It must surface only the PR's file, not main's
        # unrelated docs change — otherwise an unrelated main edit could
        # over-trigger shards.
        assert _changed_files(pr_merge_repo, "HEAD^1...HEAD^2") == ["src/serving/app.py"]


class TestPROwnDocsOnlyStillDetected:
    def test_parent_range_detects_pr_own_docs_only(self, tmp_path: Path):
        """Positive control: when the PR's OWN commit carries the [docs-only]
        tag, the parent-scoped range still detects it (the opt-in keeps working
        — the fix narrows the range, it does not disable the feature)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        _commit(repo, "README.md", "v0\n", "chore: base")

        _git(repo, "checkout", "-q", "-b", "pr")
        _commit(
            repo, "src/qb/config.py", "# tweak\n", "[docs-only] docs: clarify QB config comment"
        )

        _git(repo, "checkout", "-q", "main")
        _commit(repo, "other.py", "y = 2\n", "feat: unrelated main change")

        _git(repo, "merge", "-q", "--no-ff", "-m", "Merge pr into main", "pr")
        assert _scan_docs_only(repo, "HEAD^2", "--not", "HEAD^1") is True

    def test_squash_constituent_subject_detected(self, tmp_path: Path):
        """A squash-merge body lists constituent subjects as `* `-prefixed
        lines; the awk filter keeps those, so a `* [docs-only] ...` bullet in
        the PR's single commit body is detected."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        _commit(repo, "README.md", "v0\n", "chore: base")
        _git(repo, "checkout", "-q", "-b", "pr")
        # Squash-style commit: prose subject, constituent subjects as bullets.
        body = "docs: tidy comments\n\n* [docs-only] fix typo in docstring\n* chore: rewrap"
        _commit(repo, "src/qb/config.py", "# x\n", body)
        _git(repo, "checkout", "-q", "main")
        _commit(repo, "other.py", "y = 2\n", "feat: unrelated main change")
        _git(repo, "merge", "-q", "--no-ff", "-m", "Merge pr into main", "pr")
        assert _scan_docs_only(repo, "HEAD^2", "--not", "HEAD^1") is True


class TestLinearHistoryFallback:
    """Push to main / a PR whose merge couldn't be computed → HEAD is linear
    (no second parent), so the workflow falls back to HEAD~1..HEAD."""

    def _linear_repo(self, tmp_path: Path, last_message: str) -> Path:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        _commit(repo, "README.md", "v0\n", "chore: base")
        _commit(repo, "src/qb/config.py", "# x\n", last_message)
        return repo

    def test_linear_commit_has_no_second_parent(self, tmp_path: Path):
        repo = self._linear_repo(tmp_path, "feat: a change")
        assert _verify(repo, "HEAD^2") is False
        assert _verify(repo, "HEAD~1") is True

    def test_fallback_detects_docs_only_in_last_commit(self, tmp_path: Path):
        repo = self._linear_repo(tmp_path, "[docs-only] docs: comment only")
        assert _scan_docs_only(repo, "HEAD~1..HEAD") is True

    def test_fallback_clean_last_commit_runs_tests(self, tmp_path: Path):
        repo = self._linear_repo(tmp_path, "feat: real change")
        assert _scan_docs_only(repo, "HEAD~1..HEAD") is False
