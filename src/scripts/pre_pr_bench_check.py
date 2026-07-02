"""The pre-PR benchmark gate's brain (B2), invoked by .claude/hooks/pre-pr.sh.

Two stdin-driven subcommands (paths newline-separated, like scope_positions):

``inert --base SHA``
    AST-equivalence tier: prints each input ``.py`` path whose ``base..HEAD``
    change is provably inert — comments / docstrings / formatting only —
    determined by parsing both versions with THIS interpreter, stripping
    docstrings, and comparing ``ast.dump`` (default args exclude attributes,
    so line numbers are invisible). Any parse/``git show`` failure means "not
    inert" (conservative). Single-interpreter comparison, so the known
    cross-version instability of ``ast.dump`` never applies.

``evaluate``
    Scope the input files via ``scope_positions.compute_benchmark_scope``,
    compute HEAD fingerprints (the code the PR ships), and check the three
    evidence tiers per required position:

      1. **Fingerprint** (primary): some ``benchmark_history/*.json`` entry
         has ``code_fingerprints[P]`` equal to the current HEAD fingerprint
         AND lists P in ``positions``.
      2. **Legacy mtime** (self-retiring): only while NO history entry
         carries a fingerprint for P — entry file mtime newer than the
         changed gate-relevant files' mtimes. Retires forever once the first
         fingerprinted benchmark of P lands.
      3. **outputs/models mtime** (permanent, warns): ``{pos}/outputs/models``
         newer than the changed files — accepts a bare ``run_pipeline`` run,
         nudging toward ``benchmark P --no-sync`` for fingerprinted evidence.

    Shared-path changes require evidence on at least ONE position (shared
    code runs the same path for every position — a structural regression
    shows anywhere; the hook's risky-token filter is the safety net for
    partial-effect changes). Exempt paths are reported, never silently
    dropped. Uncommitted edits to gated paths produce a non-blocking warning
    (they don't ship in the PR, so they can't be gated on).

Protocol: on a clean run exit 0 with stdout line 1 ``PASS`` or ``FAIL`` and
human-readable detail lines after. Any crash exits nonzero — the hook then
warns loudly and FAILS OPEN (parity with its missing-jq behavior).

Pure stdlib + sibling ``src.scripts`` imports only.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import subprocess
import sys

from src.scripts.bench_fingerprint import position_fingerprint, position_paths
from src.scripts.scope_positions import (
    _BENCH_SHARED_REGEX,
    ALL_POSITIONS,
    compute_benchmark_scope,
)

HISTORY_DIR = "benchmark_history"  # top level only; tuning/ + ablations/ subdirs excluded


# --------------------------------------------------------------------------
# inert tier
# --------------------------------------------------------------------------


def _strip_docstrings(tree: ast.Module) -> ast.Module:
    for node in ast.walk(tree):
        # Tuple form, NOT a PEP 604 union — isinstance unions are 3.10+ and
        # this module's contract is vanilla python3 (macOS CLT ships 3.9).
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:]
    return tree


def is_inert(base_src: str, head_src: str) -> bool:
    """True iff the two sources are AST-identical after docstring stripping."""
    try:
        base_tree = ast.parse(base_src)
        head_tree = ast.parse(head_src)
    except SyntaxError:
        return False
    return ast.dump(_strip_docstrings(base_tree)) == ast.dump(_strip_docstrings(head_tree))


def _git_show(ref: str, path: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "show", f"{ref}:{path}"], check=True, capture_output=True
        ).stdout
    except subprocess.CalledProcessError:
        return None  # new/deleted at that ref -> not inert
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return None


def cmd_inert(base: str, files: list[str]) -> int:
    for f in files:
        if not f.endswith(".py"):
            continue
        base_src = _git_show(base, f)
        head_src = _git_show("HEAD", f)
        if base_src is None or head_src is None:
            continue
        if is_inert(base_src, head_src):
            sys.stdout.write(f + "\n")
    return 0


# --------------------------------------------------------------------------
# evaluate tier
# --------------------------------------------------------------------------


def _load_history_entries() -> list[dict]:
    entries = []
    for path in sorted(glob.glob(os.path.join(HISTORY_DIR, "*.json"))):
        try:
            with open(path) as fh:
                data = json.load(fh)
            mtime = os.path.getmtime(path)  # inside the try: the file can
            # vanish between glob and stat (concurrent benchmark/cleanup) and
            # a crash here would fail the whole gate open
        except (OSError, json.JSONDecodeError):
            continue  # corrupt/unreadable/vanished entry proves nothing
        entries.append(
            {
                "positions": data.get("positions") or [],
                "code_fingerprints": data.get("code_fingerprints") or {},
                "mtime": mtime,
            }
        )
    return entries


def _ref_ts(pos: str, files: list[str]) -> float | None:
    """Max mtime over the changed files relevant to ``pos``.

    ``None`` when no relevant file exists on disk (a deletion-only change):
    the mtime tiers have no timestamp to anchor on, so they must NOT accept —
    a 0.0 anchor would let arbitrarily stale evidence pass. The fingerprint
    tier handles deletions exactly (the HEAD manifest simply lacks the file).
    """
    prefix = f"src/{pos.lower()}/"
    relevant = [f for f in files if f.startswith(prefix) or _BENCH_SHARED_REGEX.match(f)]
    times = [os.path.getmtime(f) for f in relevant if os.path.exists(f)]
    return max(times) if times else None


def cmd_evaluate(files: list[str]) -> int:
    scope = compute_benchmark_scope(files)
    notes: list[str] = []
    if scope["exempt"]:
        notes.append(
            "note: exempt from the local benchmark gate (Batch/deps — not exercised by the local benchmark path): "
            + ", ".join(scope["exempt"])
        )

    required = list(scope["positions"])
    if not required and not scope["shared"]:
        sys.stdout.write("PASS\n" + "".join(n + "\n" for n in notes))
        return 0

    check_set = list(ALL_POSITIONS) if scope["shared"] else required
    current_fp = {p: position_fingerprint(p, source="head") for p in check_set}
    entries = _load_history_entries()

    def fingerprint_match(p: str) -> bool:
        return any(
            e["code_fingerprints"].get(p) == current_fp[p] and p in e["positions"] for e in entries
        )

    def fingerprint_era(p: str) -> bool:
        return any(p in e["code_fingerprints"] for e in entries)

    def legacy_mtime(p: str) -> bool:
        if fingerprint_era(p):
            return False  # self-retired: fingerprinted evidence exists for P
        ts = _ref_ts(p, files)
        if ts is None:
            return False  # deletion-only change: no anchor, only tier 1 counts
        return any(e["mtime"] > ts and p in e["positions"] for e in entries)

    def outputs_mtime(p: str) -> bool:
        ts = _ref_ts(p, files)
        if ts is None:
            return False  # deletion-only change: no anchor, only tier 1 counts
        d = os.path.join(p.lower(), "outputs", "models")
        if not os.path.isdir(d):
            return False
        # Newest mtime across the artifact FILES only: in-place overwrites
        # (torch.save/joblib.dump to fixed names) never bump the dirent, so a
        # bare dir mtime goes permanently stale on a warm box — and seeding
        # from the dirent would let an EMPTY dir (crashed run) or a stale-file
        # deletion count as evidence.
        newest = 0.0
        for root, _dirs, names in os.walk(d):
            for n in names:
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, n)))
                except OSError:
                    continue
        return newest > ts

    def accepted(p: str) -> str | None:
        if fingerprint_match(p):
            return "fingerprint"
        if legacy_mtime(p):
            return "legacy-mtime"
        if outputs_mtime(p):
            return "outputs-mtime"
        return None

    missing: list[str] = []
    for p in required:
        tier = accepted(p)
        if tier is None:
            missing.append(p)
        elif tier != "fingerprint":
            notes.append(
                f"note: {p} accepted via {tier} evidence — run "
                f"'python -m src.benchmarking.benchmark {p} --no-sync' once to record "
                f"fingerprinted evidence and retire the mtime fallback"
            )

    shared_ok = True
    if scope["shared"] and not missing:
        shared_ok = any(accepted(p) for p in ALL_POSITIONS)

    # Uncommitted edits to gated paths can't be gated (the PR ships HEAD).
    all_paths = sorted({path for p in ALL_POSITIONS for path in position_paths(p)})
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *all_paths],
            check=True,
            capture_output=True,
        ).stdout.decode()
    except subprocess.CalledProcessError:
        dirty = ""
    if dirty.strip():
        notes.append(
            "warning: uncommitted changes under gated pipeline paths — they will NOT "
            "ship in this PR and are not covered by this gate:\n  "
            + "\n  ".join(dirty.strip().splitlines())
        )

    if missing:
        cmd = f"python -m src.benchmarking.benchmark {' '.join(missing)} --no-sync"
        lines = [
            "FAIL",
            f"benchmark evidence missing for changed position(s): {', '.join(missing)}",
            f"  fix: {cmd}   (~1-2 min/position on a warm box)",
            "  or per position: "
            + "; ".join(f"python -m src.{p.lower()}.run_pipeline" for p in missing),
            "  evidence = a benchmark_history entry whose code_fingerprints match this HEAD",
            "  (commit your pipeline edits BEFORE benchmarking — evidence is matched",
            "   against committed HEAD content, so a dirty-tree run won't count)",
        ]
        sys.stdout.write("\n".join(lines) + "\n" + "".join(n + "\n" for n in notes))
        return 0
    if not shared_ok:
        lines = [
            "FAIL",
            "shared pipeline files changed but no position has benchmark evidence for this HEAD",
            "  fix (any one position suffices; K is fastest): "
            "python -m src.benchmarking.benchmark K --no-sync",
        ]
        sys.stdout.write("\n".join(lines) + "\n" + "".join(n + "\n" for n in notes))
        return 0

    sys.stdout.write("PASS\n" + "".join(n + "\n" for n in notes))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-PR benchmark gate checks")
    sub = parser.add_subparsers(dest="command", required=True)
    p_inert = sub.add_parser("inert", help="print base..HEAD AST-inert files from stdin list")
    p_inert.add_argument("--base", required=True)
    sub.add_parser("evaluate", help="PASS/FAIL benchmark-evidence verdict for stdin file list")
    args = parser.parse_args()
    files = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    if args.command == "inert":
        return cmd_inert(args.base, files)
    return cmd_evaluate(files)


if __name__ == "__main__":
    sys.exit(main())
