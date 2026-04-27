"""Repo-wide invariants pinning the post-rename layout (#150 → #154).

These tests block the next "rename leaves stale references" PR by failing
loudly the moment a pre-rename path or symbol pattern reappears anywhere
that isn't an explicit historical archive. Cheap to run, no fixtures, no
network — they tail-grep the tracked tree.

Bug classes guarded against:

* ``src/QB/`` / ``src.QB.`` (uppercase position dirs and dotted refs)
* ``qb_config.py`` / ``qb_features.py`` / ``qb_data.py`` / ``qb_targets.py``
  (the {pos}_ filename prefix dropped in #154)
* ``QB/qb_…`` / ``QB.qb_…`` / ``QB/run.py`` / ``QB/tests/`` (legacy combos)
* ``run_qb_pipeline`` etc. (pre-rename function names; replaced by ``run``
  inside ``src/{pos}/run_pipeline.py``)
* Broken script ``Usage:`` examples like ``python -m QB.diagnose_qb_outliers``
  that nobody ever copy-pastes successfully

Excluded files (intentional historical records — do *not* lint):

* ``TODO.md`` "Fixed archive" entries
* ``instructions/SELF_ASSESSMENT_EVIDENCE.md`` historical bug rows
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files whose contents intentionally retain pre-rename references for
# historical accuracy. Add to this list with care and a short reason.
ARCHIVE_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "TODO.md",  # "Fixed archive" — incident records frozen in time
        "instructions/SELF_ASSESSMENT_EVIDENCE.md",  # rubric evidence rows
        # This test file itself contains the patterns it forbids — that's
        # the whole point of the test, but pytest collects this file too.
        "tests/test_rename_path_invariants.py",
    }
)

# Extensions to lint. Binary types and lockfiles are skipped.
LINT_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".md", ".yml", ".yaml", ".sh", ".toml", ".json", ".txt", ".cfg", ".ini"}
)


def _tracked_files() -> list[Path]:
    """List of git-tracked files under repo root, filtered to ``LINT_EXTENSIONS``.

    Uses ``git ls-files`` so untracked artifacts (benchmark dumps, model
    weights, build caches) never get linted.
    """
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    files = []
    for line in out.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        if Path(rel).suffix not in LINT_EXTENSIONS:
            continue
        if rel in ARCHIVE_EXCLUSIONS:
            continue
        files.append(REPO_ROOT / rel)
    return files


# Patterns that should never appear in non-archive tracked content. Each
# entry is (regex, human description). The regex is matched against every
# line; matches yield a reportable line.
_FORBIDDEN: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bsrc/(QB|RB|WR|TE|DST|K)/"),
        "Uppercase position directory path: write `src/{pos}/` (lowercase).",
    ),
    (
        re.compile(r"\bsrc\.(QB|RB|WR|TE|DST|K)\."),
        "Uppercase position dotted module ref: write `src.{pos}.` (lowercase).",
    ),
    (
        re.compile(r"\b(qb|rb|wr|te|dst|k)_(config|features|data|targets)\.py\b"),
        "Pre-rename `{pos}_X.py` filename: now lives at `src/{pos}/X.py`.",
    ),
    (
        re.compile(r"\b(QB|RB|WR|TE|DST|K)/(qb|rb|wr|te|dst|k)_"),
        "Legacy `POS/pos_X` path: now `src/{pos}/X`.",
    ),
    (
        re.compile(r"\b(QB|RB|WR|TE|DST|K)\.(qb|rb|wr|te|dst|k)_"),
        "Legacy `POS.pos_X` dotted ref: now `src.{pos}.X`.",
    ),
    (
        re.compile(r"\brun_(qb|rb|wr|te|dst|k)_pipeline\b"),
        "Pre-rename function name: each position's runner is now `run` "
        "(or `run_cv`) in `src/{pos}/run_pipeline.py`.",
    ),
    (
        re.compile(r"\b(QB|RB|WR|TE|DST|K)/run\.py\b"),
        "Legacy runner file: now `src/{pos}/run_pipeline.py`.",
    ),
    (
        re.compile(r"\b(QB|RB|WR|TE|DST|K)/tests/"),
        "Per-position `POS/tests/` tree: tests now live at `tests/{pos}/`.",
    ),
)


def test_no_stale_pre_rename_path_references():
    """Repo-wide grep guard.

    Every ``LINT_EXTENSIONS`` file tracked by git (minus the small
    ``ARCHIVE_EXCLUSIONS`` list of historical records) is scanned line-by-
    line. Any match against ``_FORBIDDEN`` is reported with file, line
    number, the pattern's intent, and the offending text.

    If you legitimately need to write a pre-rename path (e.g. a new
    historical bug entry), add the file to ``ARCHIVE_EXCLUSIONS`` with a
    short reason, NOT a per-line ``# noqa`` — the goal is keeping
    archives explicit.
    """
    findings: list[str] = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern, description in _FORBIDDEN:
                m = pattern.search(line)
                if m:
                    snippet = line.strip()
                    if len(snippet) > 120:
                        snippet = snippet[:117] + "..."
                    findings.append(f"{rel}:{line_no}: {m.group()!r} ({description}) → {snippet!r}")

    assert not findings, (
        f"Stale pre-rename path/symbol references ({len(findings)} hit"
        f"{'s' if len(findings) != 1 else ''}):\n  " + "\n  ".join(findings)
    )


# ---------------------------------------------------------------------------
# `Usage:` docstring validity
# ---------------------------------------------------------------------------


_USAGE_HEADER_RE = re.compile(r"^\s*Usage:?:?\s*$", re.MULTILINE)
_PY_M_INVOCATION_RE = re.compile(r"\bpython(?:[0-9]*)\s+-m\s+([\w\.]+)")
_PY_PATH_INVOCATION_RE = re.compile(r"\bpython(?:[0-9]*)\s+([A-Za-z_][\w/.-]*\.py)\b")


def _parse_module_docstring(path: Path) -> str:
    """Return the leading triple-quoted module docstring, or empty string.

    Uses a regex over the first ~300 lines instead of ``ast.parse`` so this
    test stays cheap to run on every commit and doesn't need the file to
    parse cleanly. Module docstrings are conventionally at the top of the
    file with double-triple-quote or single-triple-quote delimiters.
    """
    try:
        head = path.read_text(encoding="utf-8")[:8192]
    except (UnicodeDecodeError, OSError):
        return ""
    match = re.search(r'^(?:""".*?"""|\'\'\'.*?\'\'\')', head, re.DOTALL)
    if not match:
        return ""
    return match.group(0)


def _scripts_with_usage() -> list[Path]:
    """Source files whose module docstring contains a ``Usage:`` section.

    Restricted to ``src/`` to avoid sweeping documentation files (which use
    ``Usage:`` headers in pose, but those examples aren't always live and
    are linted by the path-references test above).
    """
    candidates: list[Path] = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        if any(part in {"__pycache__"} for part in path.parts):
            continue
        doc = _parse_module_docstring(path)
        if doc and _USAGE_HEADER_RE.search(doc):
            candidates.append(path)
    return candidates


def test_script_usage_docstrings_reference_real_modules():
    """Every ``python -m X.Y.Z`` invocation in a ``src/**.py`` script's
    ``Usage:`` docstring must name an importable module.

    Catches the failure mode from the rename refactor: ``python -m
    QB.diagnose_qb_outliers`` lingered in the docstring after the move to
    ``src/qb/diagnose_outliers.py``. Anyone copy-pasting from the
    docstring would hit ``No module named 'QB'`` and not know why.

    We only check ``importlib.util.find_spec`` (no actual import side
    effects), so the test is fast and order-independent.
    """
    bad: list[str] = []
    for path in _scripts_with_usage():
        rel = path.relative_to(REPO_ROOT).as_posix()
        doc = _parse_module_docstring(path)
        for module in _PY_M_INVOCATION_RE.findall(doc):
            try:
                spec = importlib.util.find_spec(module)
            except (ImportError, ValueError):
                spec = None
            if spec is None:
                bad.append(
                    f"{rel}: docstring `python -m {module}` does not resolve "
                    f"to an importable module."
                )
    assert not bad, "Broken `python -m` examples in script docstrings:\n  " + "\n  ".join(bad)


def test_script_usage_docstrings_reference_real_files():
    """Every ``python path/to/file.py`` invocation in a ``src/**.py``
    script's ``Usage:`` docstring must name a real path under the repo.

    Same intent as the ``-m`` test: catches things like ``python
    scripts/ablate_rb_gate.py`` after the file moved to
    ``src/tuning/ablate_rb_gate.py``.
    """
    bad: list[str] = []
    for path in _scripts_with_usage():
        rel = path.relative_to(REPO_ROOT).as_posix()
        doc = _parse_module_docstring(path)
        for raw_path in _PY_PATH_INVOCATION_RE.findall(doc):
            # Allow positional placeholder syntax like ``script.py [--flag]``;
            # only check paths that look like real ones (contain ``/``).
            if "/" not in raw_path:
                continue
            target = REPO_ROOT / raw_path
            if not target.exists():
                bad.append(
                    f"{rel}: docstring `python {raw_path}` references a "
                    f"path that does not exist at repo root."
                )
    assert not bad, (
        "Broken `python path/to/file.py` examples in script docstrings:\n  " + "\n  ".join(bad)
    )


# ---------------------------------------------------------------------------
# Filesystem-layout sanity (cheap, doesn't require git ls-files)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pos", ["qb", "rb", "wr", "te", "k", "dst"])
def test_position_directory_is_lowercase(pos):
    """``src/{pos}/`` must exist; ``src/{POS}/`` must not (case-sensitive
    string check via os.listdir, not Path.exists which case-folds on APFS).

    Pinning this with a directory-listing assertion catches a half-applied
    rename — e.g. someone moves ``src/qb/`` back to ``src/QB/`` because a
    fresh checkout on Linux confused them — before the deploy goes out.
    """
    import os

    src_entries = set(os.listdir(REPO_ROOT / "src"))
    assert pos in src_entries, f"src/{pos}/ missing — rename half-applied?"
    assert pos.upper() not in src_entries, (
        f"src/{pos.upper()}/ exists alongside src/{pos}/. The {{POS}}/→{{pos}}/ "
        "rename in #154 lowercased every position dir; an uppercase sibling "
        "means a partial revert or APFS-blind merge."
    )
