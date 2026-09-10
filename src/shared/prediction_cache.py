"""Immutable prediction-cache generations and single-object S3 transport.

Readers resolve the local pointer once. Writers finish a private generation
before atomically replacing it. Old generations remain until task replacement
so another worker's open/read-in-progress generation is never removed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

REQUIRED_FILES = frozenset({"predictions.parquet", "metrics.json", "fingerprint.json"})
OPTIONAL_FILES = frozenset({"snapshot.json"})
BUNDLE_NAME = "cache.tar.gz"
_MANIFEST = "generation.json"
_POINTER = "current.json"
_MAX_BYTES = 256 * 1024 * 1024


def _manifest_bytes(files: dict[str, bytes]) -> bytes:
    if not REQUIRED_FILES.issubset(files) or set(files) - REQUIRED_FILES - OPTIONAL_FILES:
        raise ValueError("Incomplete or unexpected prediction cache members")
    return json.dumps(
        {
            "version": 1,
            "files": {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())},
        },
        sort_keys=True,
    ).encode()


def current_generation(cache_dir: str | Path) -> Path | None:
    """Resolve a committed directory; legacy loose files are never trusted."""
    root = Path(cache_dir)
    try:
        pointer = json.loads((root / _POINTER).read_bytes())
        generation = pointer["generation"]
        if not isinstance(generation, str) or not re.fullmatch(r"[a-f0-9]{64}", generation):
            return None
        directory = root / "generations" / generation
        manifest = (directory / _MANIFEST).read_bytes()
        if hashlib.sha256(manifest).hexdigest() != generation:
            return None
        return directory
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_generation(cache_dir: str | Path) -> tuple[Path, dict[str, bytes]]:
    """Read and verify one immutable generation, even if the pointer changes."""
    directory = current_generation(cache_dir)
    if directory is None:
        raise ValueError("No committed prediction cache generation")
    manifest = (directory / _MANIFEST).read_bytes()
    metadata = json.loads(manifest)
    files = {}
    for name in metadata.get("files", {}):
        if name not in REQUIRED_FILES | OPTIONAL_FILES:
            raise ValueError("Unexpected prediction cache member")
        files[name] = (directory / name).read_bytes()
    if _manifest_bytes(files) != manifest:
        raise ValueError("Prediction cache checksum mismatch")
    return directory, files


def publish_generation(cache_dir: str | Path, files: dict[str, bytes]) -> Path:
    """Commit complete immutable bytes, then switch the pointer in one rename."""
    root = Path(cache_dir)
    parent = root / "generations"
    parent.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_bytes(files)
    generation = hashlib.sha256(manifest).hexdigest()
    directory = parent / generation
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=parent))
    pointer_tmp = None
    try:
        for name, content in files.items():
            (staging / name).write_bytes(content)
        (staging / _MANIFEST).write_bytes(manifest)
        try:
            os.rename(staging, directory)
        except OSError:
            # Identical concurrent writers converge on the same content ID.
            # Other rename errors must leave the previously committed pointer.
            if not directory.is_dir() or (directory / _MANIFEST).read_bytes() != manifest:
                raise
        with tempfile.NamedTemporaryFile(dir=root, prefix=".current-", delete=False) as stream:
            pointer_tmp = Path(stream.name)
            stream.write(json.dumps({"generation": generation}).encode())
        os.replace(pointer_tmp, root / _POINTER)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if pointer_tmp is not None:
            pointer_tmp.unlink(missing_ok=True)
    return directory


def bundle_generation(cache_dir: str | Path) -> tuple[str, bytes]:
    """Serialize one verified generation; all S3 readers fetch this one object."""
    directory, files = read_generation(cache_dir)
    files[_MANIFEST] = _manifest_bytes(files)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return directory.name, buffer.getvalue()


def install_bundle(cache_dir: str | Path, data: bytes) -> Path:
    """Validate the entire S3 bundle before touching the current pointer."""
    files = {}
    size = 0
    allowed = REQUIRED_FILES | OPTIONAL_FILES | {_MANIFEST}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            size += member.size
            if (
                member.name not in allowed
                or member.name in files
                or not member.isfile()
                or size > _MAX_BYTES
            ):
                raise ValueError("Invalid prediction cache bundle member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Missing prediction cache member")
            files[member.name] = stream.read()
    manifest = files.pop(_MANIFEST, None)
    if manifest != _manifest_bytes(files):
        raise ValueError("Prediction cache bundle checksum mismatch")
    return publish_generation(cache_dir, files)
