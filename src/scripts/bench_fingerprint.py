"""Content fingerprints for the pre-PR benchmark gate (B2).

A *fingerprint* identifies the exact code a benchmark ran against, so the
gate in ``.claude/hooks/pre-pr.sh`` can accept evidence by **content
identity** instead of the old fragile mtime comparison (which broke on
``rebase``/``stash pop``/``checkout`` and never proved the benchmark ran on
the shipped code — the recorded ``git_hash`` was decorative).

Manifest (version ``v1``): for position P, the sorted list of
``(path, git_blob_sha)`` pairs over every **tracked** file under
``src/{pos}/`` plus the global set (``src/shared/``, ``src/data/``,
``src/features/``, ``src/config.py``, ``src/__init__.py``). The fingerprint
is ``sha256("v1\\n" + one "path\\0sha\\n" line per pair)``.

Two sources, identical output for identical content:

- ``worktree`` (benchmark writers): the code that actually trained —
  ``git ls-files --cached`` for the tracked set, ``git hash-object
  --stdin-paths`` over on-disk content, so dirty edits are captured.
- ``head`` (the gate): the code the PR ships — blob SHAs straight from
  ``git ls-tree -r HEAD`` (no content hashing needed).

Deliberately excluded, mirroring ``scope_positions.compute_benchmark_scope``'s
exemptions: ``src/batch/**`` and ``requirements.txt``. The soundness
invariant — every path that can scope position P into the gate is inside P's
manifest — is pinned by ``tests/scripts/test_bench_fingerprint.py``.

Pure stdlib (like ``scope_positions``) so hooks can run it with vanilla
``python3`` (3.9+, the macOS CLT floor) before any venv exists.

Known limitations (both verified absent from this repo's ``src/`` tree and
acceptable-by-construction — the failure direction is a fingerprint MISMATCH,
i.e. a required re-benchmark, never a false accept): a file committed with
CRLF content under ``core.autocrlf`` hashes differently in worktree vs head
mode (``hash-object`` applies clean filters; the committed blob keeps CRLF),
and a tracked symlink is hashed as its link-target string at HEAD but as the
target's content in worktree mode.

CLI (debugging):
    python3 -m src.scripts.bench_fingerprint [--head] QB RB ...
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from collections.abc import Iterable

FINGERPRINT_VERSION = "v1"

# Must cover every path scope_positions._BENCH_SHARED_REGEX can match (the
# soundness invariant above). Bump FINGERPRINT_VERSION when this set changes —
# old fingerprints must never falsely match a differently-defined manifest.
GLOBAL_PATHS: tuple[str, ...] = (
    "src/shared",
    "src/data",
    "src/features",
    "src/config.py",
    "src/__init__.py",
)


def position_paths(pos: str) -> tuple[str, ...]:
    """The manifest path set for one position: its dir + the global set."""
    return (f"src/{pos.lower()}",) + GLOBAL_PATHS


def _git(repo_root: str, args: list[str], input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", repo_root, *args],
        check=True,
        capture_output=True,
        input=input_bytes,
    ).stdout


def head_manifest(paths: Iterable[str], repo_root: str = ".") -> list[tuple[str, str]]:
    """(path, blob_sha) pairs at HEAD — blob SHAs read from the tree, no hashing."""
    out = _git(repo_root, ["ls-tree", "-r", "-z", "HEAD", "--", *paths])
    entries: list[tuple[str, str]] = []
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        _mode, otype, sha = meta.split()
        if otype != b"blob":
            continue
        entries.append((path.decode(), sha.decode()))
    return sorted(entries)


def worktree_manifest(paths: Iterable[str], repo_root: str = ".") -> list[tuple[str, str]]:
    """(path, blob_sha) pairs for the TRACKED set hashed from on-disk content.

    Captures dirty (uncommitted) edits — the code that actually trained.
    Tracked-but-deleted files are dropped; untracked new files are absent from
    both modes (commit them before benchmarking so the evidence covers them).
    ``git hash-object --stdin-paths`` reads newline-separated paths; repo
    paths contain no newlines, and git runs with cwd=repo_root so relative
    paths resolve there.
    """
    out = _git(repo_root, ["ls-files", "--cached", "-z", "--", *paths])
    tracked = [p.decode() for p in out.split(b"\0") if p]
    present = [p for p in tracked if os.path.exists(os.path.join(repo_root, p))]
    if not present:
        return []
    stdin = ("\n".join(present) + "\n").encode()
    hashes = _git(repo_root, ["hash-object", "--stdin-paths"], input_bytes=stdin).decode().split()
    if len(hashes) != len(present):  # 3.9-safe strict-zip (vanilla-python3 contract)
        raise RuntimeError(f"hash-object returned {len(hashes)} hashes for {len(present)} paths")
    return sorted(zip(present, hashes))  # noqa: B905 - strict= kwarg is 3.10+; length-checked above


def fingerprint_from_manifest(manifest: Iterable[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    h.update((FINGERPRINT_VERSION + "\n").encode())
    for path, sha in manifest:
        h.update(f"{path}\0{sha}\n".encode())
    return h.hexdigest()


def position_fingerprint(pos: str, repo_root: str = ".", *, source: str = "worktree") -> str:
    """Fingerprint one position. ``source`` is ``"worktree"`` or ``"head"``."""
    paths = position_paths(pos)
    if source == "head":
        manifest = head_manifest(paths, repo_root)
    elif source == "worktree":
        manifest = worktree_manifest(paths, repo_root)
    else:  # pragma: no cover - programmer error
        raise ValueError(f"unknown fingerprint source: {source!r}")
    return fingerprint_from_manifest(manifest)


def collect_code_fingerprints(
    positions: Iterable[str], repo_root: str = "."
) -> dict[str, str] | None:
    """Writer entrypoint: worktree fingerprints for a benchmark history entry.

    Fail-open: any git failure (not a repo, git missing, detached weirdness)
    returns ``None`` with one printed warning — the benchmark run must never
    die over provenance metadata; writers simply omit the key (the gate then
    treats the entry as legacy).
    """
    try:
        return {pos: position_fingerprint(pos, repo_root, source="worktree") for pos in positions}
    except Exception as e:  # noqa: BLE001 - deliberate fail-open boundary
        print(f"WARNING: code fingerprints unavailable ({e}); history entry will omit them")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Print per-position benchmark code fingerprints")
    parser.add_argument("positions", nargs="+", help="Positions, e.g. QB RB")
    parser.add_argument("--head", action="store_true", help="Hash HEAD instead of the worktree")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    source = "head" if args.head else "worktree"
    for pos in args.positions:
        fp = position_fingerprint(pos.upper(), args.repo_root, source=source)
        sys.stdout.write(f"{pos.upper()}\t{fp}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
