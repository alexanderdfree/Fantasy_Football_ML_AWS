"""Automated parity check across the Python-pins sources.

The repo carries these sources of truth for Python package versions:

  1. ``requirements.txt``        — runtime deps for the Flask serving path
                                    (and the local-dev install via ``uv pip
                                    install -r``).
  2. ``src/batch/requirements.txt`` — the Batch training container deps;
                                    pinned identically to (1) for every
                                    package that appears in both. As of the
                                    2026-06-07 cold-start image-slim, this file
                                    also pins ``torch`` (the slim
                                    ``nvidia/cuda:*-base`` base no longer ships
                                    it) via an ``--extra-index-url`` to the
                                    pytorch CUDA wheel index.
  3. ``src/batch/Dockerfile.train`` ``FROM`` line — the slim CUDA base image,
                                    pinned to an explicit tag.

This module:

  * Asserts every package that appears in BOTH (1) and (2) is pinned to the
    same exact version string. Drift between the two breaks the implicit
    contract that ``requirements.txt`` is the source of truth and
    ``src/batch/requirements.txt`` is its training-time subset.

  * Asserts the Dockerfile base image is pinned (not ``latest``) and that the
    Batch requirements pin ``torch`` explicitly behind the CUDA extra index.

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

    Skips blank lines, ``#``-prefixed comments, and pip flag lines (those
    starting with ``-``, e.g. ``--extra-index-url`` / ``-r``). Inline ``;``
    markers (env markers) are not used in this repo so they aren't handled here.
    """
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()  # drop comments + trim
        if not line or line.startswith("-"):  # blank or pip flag (--extra-index-url, -r)
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


def test_dockerfile_base_image_is_pinned():
    """The training Dockerfile must pin its base image to an explicit tag.

    The base was slimmed from the conda ``pytorch/pytorch:*-runtime`` image to
    ``nvidia/cuda:*-base`` (torch now installed via pip — see
    src/batch/requirements.txt and docs/batch_design.md §"Cold-start" 2d). We
    don't semver-compare the CUDA components, only that the pin exists and isn't
    a floating ``latest`` tag (which would land an untested CUDA combo on the
    next Batch run).
    """
    contents = DOCKERFILE_TRAIN.read_text()
    # Match ``FROM <image>:<tag>`` allowing the optional ``--platform`` and
    # ``${PULL_THROUGH_PREFIX}`` indirection in front of the image reference.
    from_re = re.compile(
        r"^FROM\s+(?:--platform=\S+\s+)?\S*?([A-Za-z0-9][A-Za-z0-9._/-]*):(\S+)",
        re.MULTILINE,
    )
    match = from_re.search(contents)
    assert match, f"Expected a pinned `FROM <image>:<tag>` line in {DOCKERFILE_TRAIN}."
    image, tag = match.group(1), match.group(2)
    assert tag != "latest", (
        f"{DOCKERFILE_TRAIN}: base image {image!r} is pinned to 'latest' — use an "
        "explicit version tag (e.g. nvidia/cuda:12.6.3-base-ubuntu24.04)."
    )
    assert re.search(r"\d+\.\d+", tag), (
        f"{DOCKERFILE_TRAIN}: base tag {tag!r} doesn't contain a major.minor version."
    )


def test_batch_requirements_pins_torch_and_omits_flask():
    """The Batch image installs torch via pip now that the base is the slim
    ``nvidia/cuda:*-base`` (it no longer ships torch). The pin must be explicit
    and resolvable from the pytorch CUDA wheel index; flask/gunicorn stay out
    (the trainer never serves Flask and would otherwise drag serving-only deps).
    """
    contents = BATCH_REQUIREMENTS_TXT.read_text()
    batch = _parse_requirements(BATCH_REQUIREMENTS_TXT)

    assert "torch" in batch, (
        "src/batch/requirements.txt must pin torch — the slim nvidia/cuda base "
        "ships no torch (see docs/batch_design.md §'Cold-start' 2d)."
    )
    spec = batch["torch"]
    assert spec.startswith("=="), (
        f"torch must be pinned with '==' (got {spec!r}); a floating torch would "
        "land an untested build on the next Batch run."
    )
    assert "--extra-index-url" in contents and "download.pytorch.org/whl/" in contents, (
        "src/batch/requirements.txt must carry the pytorch CUDA --extra-index-url "
        "so the +cuXXX torch wheel resolves."
    )

    assert "flask" not in batch, "src/batch/requirements.txt should not include flask."
    assert "gunicorn" not in batch, "src/batch/requirements.txt should not include gunicorn."


def test_top_level_requirements_includes_runtime_essentials():
    """Belt-and-suspenders: requirements.txt must carry the packages the
    serving + training paths depend on. A future trim that drops one of
    these would crash the Flask container at import time.
    """
    top = _parse_requirements(REQUIREMENTS_TXT)
    for essential in ("numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "flask"):
        assert essential in top, f"{essential!r} missing from requirements.txt"
