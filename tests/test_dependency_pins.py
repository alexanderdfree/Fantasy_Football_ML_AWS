"""Automated parity check across the three Python-pins sources.

The repo carries three sources of truth for Python package versions:

  1. ``requirements.txt``        — runtime deps for the Flask serving path
                                    (and the local-dev install via ``uv pip
                                    install -r``).
  2. ``src/batch/requirements.txt`` — the Batch training container deps;
                                    pinned identically to (1) for every
                                    package that appears in both. Torch is
                                    deliberately absent — it comes from the
                                    base image.
  3. ``src/batch/Dockerfile.train`` ``FROM`` line — torch + cuDNN base; the
                                    only pin for ``torch`` in the repo.

This module:

  * Asserts every package that appears in BOTH (1) and (2) is pinned to the
    same exact version string. Drift between the two breaks the implicit
    contract that ``requirements.txt`` is the source of truth and
    ``src/batch/requirements.txt`` is its training-time subset.

  * Asserts the torch line in the Dockerfile parses cleanly (the audit's
    minimum bar is that the pin exists and is grep-able — semver-comparing
    the cuDNN/cuda components is out of scope).

L-B9 in code_review_findings.md notes "three sources of Python pins (root,
Batch, and base-image torch). No automated parity check." This test is
that check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_TXT = REPO_ROOT / "requirements.txt"
BATCH_REQUIREMENTS_TXT = REPO_ROOT / "src" / "batch" / "requirements.txt"
DOCKERFILE_TRAIN = REPO_ROOT / "src" / "batch" / "Dockerfile.train"


# ``foo==1.2.3``, ``foo>=1.0,<2.0``, ``foo``, etc. The package name is the
# longest prefix of letters / digits / hyphen / underscore / period at line
# start.
_PIN_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9._-]*)\s*(.*)$")


def _parse_requirements(path: Path) -> dict[str, str]:
    """Return ``{package_name_lower: version_spec_str}`` for a requirements file.

    Skips blank lines and ``#``-prefixed comments. Inline ``;`` markers (env
    markers) and ``--`` flags are not used in this repo so they aren't
    handled here — a future addition will need to extend this.
    """
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()  # drop comments + trim
        if not line:
            continue
        m = _PIN_RE.match(line)
        if not m:
            pytest.fail(f"Unparseable line in {path.name}: {raw!r}")
        name, spec = m.group(1).lower(), m.group(2).strip()
        out[name] = spec
    return out


def test_shared_packages_have_identical_pins():
    """Every package in both requirements files must have the same version spec.

    Adds to the failure message the full pin diff so a maintainer adding a
    new dep sees the parity violation immediately rather than reading both
    files side-by-side.
    """
    top = _parse_requirements(REQUIREMENTS_TXT)
    batch = _parse_requirements(BATCH_REQUIREMENTS_TXT)
    shared = sorted(set(top) & set(batch))
    assert shared, "Expected at least one package shared between the two files."

    diffs = [(pkg, top[pkg], batch[pkg]) for pkg in shared if top[pkg] != batch[pkg]]
    assert not diffs, (
        "Pin drift between requirements.txt and src/batch/requirements.txt:\n"
        + "\n".join(f"  {pkg}: root={top!r} vs batch={batch!r}" for pkg, top, batch in diffs)
    )


def test_dockerfile_torch_pin_is_grepable():
    """The training Dockerfile must pin its torch base image explicitly.

    We don't compare the version against requirements.txt (top-level deps
    don't include torch — it comes from the base image), only that the
    pin exists and isn't a floating ``latest`` tag. Catches an accidental
    ``pytorch:latest`` regression that would land an untested cuDNN/CUDA
    combination on the next Batch run.
    """
    contents = DOCKERFILE_TRAIN.read_text()
    # Match ``FROM ... pytorch/pytorch:<tag>`` allowing the optional
    # ``${PULL_THROUGH_PREFIX}`` indirection in front.
    from_re = re.compile(r"^FROM\s+(?:--platform=\S+\s+)?\S*pytorch/pytorch:(\S+)", re.MULTILINE)
    match = from_re.search(contents)
    assert match, (
        f"Expected a `FROM ... pytorch/pytorch:<tag>` line in {DOCKERFILE_TRAIN}; "
        "did the base image source change?"
    )
    tag = match.group(1)
    assert tag != "latest", (
        f"{DOCKERFILE_TRAIN}: torch base image is pinned to 'latest' — "
        "use an explicit version tag (e.g. 2.11.0-cuda12.6-cudnn9-runtime)."
    )
    # Sanity: the tag should at least look like ``<major>.<minor>``.
    assert re.match(r"^\d+\.\d+", tag), (
        f"{DOCKERFILE_TRAIN}: torch tag {tag!r} doesn't start with a major.minor version."
    )


def test_top_level_requirements_includes_runtime_essentials():
    """Belt-and-suspenders: requirements.txt must carry the packages the
    serving + training paths depend on. A future trim that drops one of
    these would crash the Flask container at import time.
    """
    top = _parse_requirements(REQUIREMENTS_TXT)
    for essential in ("numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "flask"):
        assert essential in top, f"{essential!r} missing from requirements.txt"


def test_batch_requirements_omits_torch_and_flask():
    """The Batch image inherits torch from the base image and never serves
    Flask. Pinning either in src/batch/requirements.txt would silently
    override or duplicate the base — and an accidental flask line would
    drag in the serving-only deps the trainer doesn't need.
    """
    batch = _parse_requirements(BATCH_REQUIREMENTS_TXT)
    assert "torch" not in batch, (
        "src/batch/requirements.txt should not pin torch — comes from the "
        "pytorch/pytorch base image in Dockerfile.train."
    )
    assert "flask" not in batch, "src/batch/requirements.txt should not include flask."
    assert "gunicorn" not in batch, "src/batch/requirements.txt should not include gunicorn."
