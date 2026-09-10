"""Check or repair this interpreter's macOS OpenMP runtime (see SETUP.md).

Use only the standard library here: importing the platform/model helpers would
load the conflicting native libraries before the environment can be repaired.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PROBE = """
import json
import torch, sklearn, lightgbm
from threadpoolctl import threadpool_info
print(json.dumps([p['filepath'] for p in threadpool_info()
                  if p['user_api'] == 'openmp']))
"""


def runtime_paths() -> set[Path]:
    """Inspect a fresh interpreter without temporary loader workarounds."""
    env = os.environ.copy()
    for key in ("DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "KMP_DUPLICATE_LIB_OK"):
        env.pop(key, None)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return {Path(path).resolve() for path in json.loads(result.stdout)}


def vendored_runtimes() -> list[Path]:
    """Find project-package libomp copies owned by the selected environment."""
    prefix = Path(sys.prefix).resolve()
    paths = {prefix / "lib" / "libomp.dylib"}
    for package in ("torch", "scikit-learn", "lightgbm"):
        dist = importlib.metadata.distribution(package)
        for entry in dist.files or ():
            if entry.name == "libomp.dylib":
                path = Path(dist.locate_file(entry))
                paths.add(path.parent.resolve() / path.name)
    # Resolve the parent, not the file: an already-repaired file is a symlink
    # outside the environment. Never rewrite inherited system-site-packages.
    return sorted(
        path
        for path in paths
        if path.parent.resolve().is_relative_to(prefix) and (path.is_file() or path.is_symlink())
    )


def verify_runtime(library: Path) -> None:
    paths = runtime_paths()
    if paths != {library.resolve()}:
        found = ", ".join(map(str, sorted(paths))) or "none"
        raise RuntimeError(f"Expected one OpenMP runtime at {library}; loaded: {found}")


def repair(library: Path) -> Path | None:
    """Relink bundled copies, preserving originals and rolling back on failure."""
    paths = [path for path in vendored_runtimes() if path.resolve() != library.resolve()]
    if not paths:
        verify_runtime(library)
        return None
    prefix = Path(sys.prefix).resolve()
    backup_root = prefix / ".openmp-backups"
    backup_root.mkdir(exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(prefix="repair-", dir=backup_root))
    changed = []
    try:
        for path in paths:
            backup = backup_dir / path.relative_to(prefix)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup, follow_symlinks=False)
            # Replace the directory entry, never edit the binary in place:
            # uv can hardlink package files to its cache or another environment.
            with tempfile.TemporaryDirectory(prefix=".openmp-", dir=path.parent) as tmp:
                link = Path(tmp) / path.name
                link.symlink_to(library)
                os.replace(link, path)
            changed.append((path, backup))
        verify_runtime(library)
    except Exception:
        for path, backup in reversed(changed):
            with tempfile.TemporaryDirectory(prefix=".openmp-", dir=path.parent) as tmp:
                original = Path(tmp) / path.name
                shutil.copy2(backup, original, follow_symlinks=False)
                os.replace(original, path)
        raise
    return backup_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="repair (default: check only)")
    parser.add_argument("--library", type=Path, help="override Homebrew's libomp.dylib")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        print("OpenMP repair is macOS-only; no changes made.")
        return 0
    try:
        library = args.library
        if library is None:
            result = subprocess.run(
                ["brew", "--prefix", "libomp"],
                text=True,
                capture_output=True,
                check=True,
                timeout=30,
            )
            library = Path(result.stdout.strip()) / "lib" / "libomp.dylib"
        # Keep Homebrew's stable opt path in symlinks so upgrades don't strand
        # them on a deleted Cellar version. Resolve only when comparing paths.
        library = library.absolute()
        if not library.is_file() or library.name != "libomp.dylib":
            raise RuntimeError(f"OpenMP library not found: {library}; run brew install libomp")
        if args.apply:
            backup = repair(library)
            if backup:
                print(f"Original libraries saved in {backup}")
        else:
            verify_runtime(library)
        print(f"Verified one OpenMP runtime for {sys.executable}: {library.resolve()}")
        return 0
    except (
        OSError,
        RuntimeError,
        ValueError,
        importlib.metadata.PackageNotFoundError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"OpenMP check failed: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr[-2000:], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
