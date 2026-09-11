"""Content-verified raw inputs and splits, published with one atomic pointer.

A refresh seals the files after all producers have finished. Publication refuses
unsealed/modified inputs, verifies every uploaded object, then advances current.
Readers resolve current once and stage/verify the entire requested set before
installing any file. Older unversioned data is an explicit operator opt-in only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
SEAL_NAME = "release-inputs.json"
SPLIT_NAMES = ("train.parquet", "val.parquet", "test.parquet")


# Keep refresh-splits.yml's push paths aligned with this compatibility contract.
# These sources create raw caches, baked split features, or reference cohorts.
DATA_PRODUCER_PATHS = (
    "src/data",
    "src/features",
    "src/config.py",
    "src/qb/config.py",
    "src/rb/config.py",
    "src/wr/config.py",
    "src/te/config.py",
    "src/k/data.py",
    "src/k/config.py",
    "src/dst/data.py",
    "src/dst/config.py",
    "src/shared/comparison_scoring.py",
    "src/shared/comparison_truth.py",
    "src/shared/evaluation_cohorts.py",
    "src/shared/expert_eligibility.py",
    "src/shared/weather_features.py",
    "src/contracts/feature_names.py",
    "src/shared/team_box_score.py",
    "src/training/context.py",
    "requirements.txt",
    "src/scripts/build_evaluation_reference.py",
    "src/analysis/analysis_expert_comparison.py",
    "src/analysis/sleeper_loader.py",
    ".github/workflows/refresh-splits.yml",
)


def data_producer_hashes(root: Path) -> dict[str, str]:
    paths = set()
    for name in DATA_PRODUCER_PATHS:
        path = root / name
        paths.update(path.rglob("*.py") if path.is_dir() else [path])
    return {str(p.relative_to(root)): _hash(p) for p in sorted(paths) if p.is_file()}


def producer_fingerprint(hashes: dict[str, str]) -> str:
    """Stable index key for a data recipe, independent of docs/model-only commits."""
    return hashlib.sha256(_json_bytes(hashes)).hexdigest()


def _object_missing(error: Exception) -> bool:
    response = getattr(error, "response", {})
    return isinstance(error, FileNotFoundError) or response.get("Error", {}).get("Code") in {
        "404",
        "NoSuchKey",
        "NotFound",
    }


def resolve_compatible_release(s3, bucket: str, expected: dict[str, str], prefix="data"):
    """Select the latest immutable snapshot of exactly this producer recipe.

    A separate pointer per producer keeps A reachable after global current moves
    to B. Refreshing unchanged producer A advances only A's pointer. The fallback
    permits existing current-only snapshots during the index migration.
    """
    prefix = prefix.rstrip("/")
    index = f"{prefix}/by-producer/{producer_fingerprint(expected)}/manifest.json"
    try:
        pointer = json.loads(s3.get_object(Bucket=bucket, Key=index)["Body"].read())
    except Exception as error:
        if not _object_missing(error):
            raise
        pointer = json.loads(
            s3.get_object(Bucket=bucket, Key=f"{prefix}/manifest.json")["Body"].read()
        )
    if pointer.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported producer data index schema")
    selected_id = pointer.get("release_id", "")
    if not re.fullmatch(r"[0-9a-f]{64}", str(selected_id)):
        raise ValueError("Invalid producer data release ID")
    selected, manifest = resolve_release(s3, bucket, prefix, selected_id)
    mismatch = [
        name
        for name, digest in expected.items()
        if manifest.get("producer", {}).get(name) != digest
    ]
    if mismatch:
        raise DataReleaseError("producer mismatch: " + ", ".join(mismatch[:6]))
    return selected, manifest


class DataReleaseError(RuntimeError):
    """A pinned historical input cannot be rebuilt or mixed with live data."""


_live_cache_roots: dict[Path, int] = {}
_replay_cache_roots: dict[Path, int] = {}
_live_cache_lock = threading.Lock()


@contextmanager
def live_source_cache(cache_dir: str | Path):
    """Permit existing live-builder fetches only inside its separate cache.

    The allowlist is directory-scoped and shared by producer worker threads.
    Historical directories carrying a release marker remain immutable even
    inside this context; callers must supply their separate temporary cache.
    """
    root = Path(cache_dir).resolve()
    if (root / ".release.json").is_file():
        raise DataReleaseError(
            "Live source fetches require a cache separate from historical release data"
        )
    with _live_cache_lock:
        _live_cache_roots[root] = _live_cache_roots.get(root, 0) + 1
    try:
        from src.data.providers.snapshot import live_provider_sources

        with live_provider_sources():
            yield
    finally:
        with _live_cache_lock:
            _live_cache_roots[root] -= 1
            if not _live_cache_roots[root]:
                del _live_cache_roots[root]


@contextmanager
def require_cached_sources(cache_dir: str | Path):
    """Verify replay without allowing a failed source to recover mid-build.

    Directory scope reaches the loader's worker threads without changing the
    process environment or constraining unrelated live-cache directories.
    """
    root = Path(cache_dir).resolve()
    with _live_cache_lock:
        _replay_cache_roots[root] = _replay_cache_roots.get(root, 0) + 1
    try:
        yield
    finally:
        with _live_cache_lock:
            _replay_cache_roots[root] -= 1
            if not _replay_cache_roots[root]:
                del _replay_cache_roots[root]


def verify_historical_loader_inputs(cache_dir: str | Path, seasons=None) -> None:
    """Replay every unconditional/conditional historical loader dependency.

    Optional fetch failures are deliberately not cached by their producers. A
    release must reject that transient state, since pinned readers cannot later
    reproduce its empty fallback. Valid cached empty frames remain valid, and
    missing years within a cached source (notably 2012 snaps) remain unknown.
    """
    from src.config import SEASONS
    from src.data.loader import load_raw_data

    with require_cached_sources(cache_dir):
        load_raw_data(list(SEASONS if seasons is None else seasons), cache_dir=str(cache_dir))


def assert_source_fetch_allowed(cache_path: str | Path) -> None:
    """Reject cache regeneration inside a selected immutable historical release.

    The marker also covers local offline analyses that hydrate then launch in a
    new process without FF_DATA_RELEASE. Rebuilds belong in a clean directory;
    removing a seal is not an automatic fallback to unrecorded live inputs.
    """
    selected = os.environ.get("FF_DATA_RELEASE", "")
    path = Path(cache_path).resolve()
    marker = path.parent / ".release.json"
    with _live_cache_lock:
        allowed_live = any(path.is_relative_to(root) for root in _live_cache_roots)
        cache_only = any(path.is_relative_to(root) for root in _replay_cache_roots)
    if selected == "legacy" and not cache_only:
        return
    if cache_only or marker.is_file() or (selected and not allowed_live):
        raise DataReleaseError(
            f"Historical data release requires cache {cache_path}; it is missing or incompatible. "
            "Rebuild the release with current producers in a clean data directory; "
            "refusing to mix a sealed snapshot with newly fetched inputs."
        )


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _record(path: Path) -> dict:
    return {"sha256": _hash(path), "bytes": path.stat().st_size}


def _producer_hashes(root: Path) -> dict:
    # Source hashes describe the producer, independently of a possibly dirty
    # checkout's Git SHA. Runtime images need not have Git installed.
    paths = {
        root / "src/config.py",
        root / "requirements.txt",
        root / "src/scripts/build_evaluation_reference.py",
    }
    for folder in ("src/data", "src/features", "src/shared", "src/analysis"):
        paths.update((root / folder).glob("*.py"))
    for position in ("qb", "rb", "wr", "te", "k", "dst"):
        paths.update((root / "src" / position).glob("*.py"))
    return {
        **{str(p.relative_to(root)): _hash(p) for p in sorted(paths) if p.is_file()},
        **data_producer_hashes(root),
    }


def _input_files(raw_dir: Path, splits_dir: Path) -> dict[str, Path]:
    files = {f"splits/{name}": splits_dir / name for name in SPLIT_NAMES}
    # JSON includes producer/cache-version sidecars. No models or large PBP
    # archives are included; these are only the existing derived raw caches.
    files.update(
        {
            f"raw/{p.name}": p
            for p in sorted(raw_dir.iterdir())
            if p.is_file() and p.suffix in {".parquet", ".json"} and not p.name.startswith(".")
        }
    )
    # Provider transport responses belong to the same sealed release. Keep
    # other nested directories (live overlays/quarantine) out of history.
    providers = raw_dir / "provider_sources"
    if providers.is_dir():
        if any(
            path.is_file() and not re.fullmatch(r"[0-9a-f]{64}\.(json|parquet)", path.name)
            for path in providers.iterdir()
        ):
            raise ValueError("Unsafe provider snapshot path")
        files.update(
            {
                f"raw/provider_sources/{path.name}": path
                for path in sorted(providers.iterdir())
                if path.is_file() and path.suffix in {".json", ".parquet"}
            }
        )
    if not any(name.startswith("raw/") for name in files):
        raise ValueError("Cannot publish splits without their raw dependencies")
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return files


def _coverage(files: dict[str, Path]) -> dict:
    import pyarrow.parquet as pq

    from src.config import SEASONS

    coverage = {}
    for name, path in files.items():
        if path.suffix != ".parquet":
            continue
        parquet = pq.ParquetFile(path)
        item = {"rows": parquet.metadata.num_rows}
        if "season" in parquet.schema.names:
            present = sorted(
                set(parquet.read(columns=["season"]).column(0).drop_null().to_pylist())
            )
            item["seasons_present"] = present
            # Missing 2012 snaps are unavailable context, not fabricated zeros.
            # K's narrower scope is visible here alongside the producer config.
            item["global_seasons_absent"] = sorted(set(SEASONS) - set(present))
        coverage[name] = item
    return coverage


def seal_inputs(*, raw_dir="data/raw", splits_dir="data/splits", repo_root=".") -> dict:
    """Record the completed build's inputs/outputs; call only in its producer.

    This is intentionally separate from upload_data: uploading arbitrary old
    splits alongside newly refreshed raw caches must never silently bless them.
    """
    raw, splits, root = Path(raw_dir), Path(splits_dir), Path(repo_root)
    verify_historical_loader_inputs(raw)
    from src.data.providers.snapshot import verify_provider_snapshot_files

    verify_provider_snapshot_files(raw / "provider_sources")
    files = _input_files(raw, splits)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "data_producer_sha256": producer_fingerprint(data_producer_hashes(root)),
        "producer": _producer_hashes(root),
        "runtime": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("pandas", "pyarrow", "nflreadpy")},
        },
        "git_sha": os.environ.get("GITHUB_SHA", os.environ.get("FF_TRAIN_GIT_SHA", "")),
        "files": {name: _record(path) for name, path in files.items()},
        "coverage": _coverage(files),
    }
    _atomic_json(splits / SEAL_NAME, manifest)
    return manifest


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".release-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes(value))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_release(
    s3,
    bucket: str,
    *,
    raw_dir="data/raw",
    splits_dir="data/splits",
    repo_root=".",
    prefix="data",
    force=False,
) -> str:
    """Upload immutable inputs, then atomically publish their manifest pointer.

    ``force`` is accepted for old callers; every publication uploads and verifies
    the complete snapshot. No unversioned compatibility mirror is written.
    """
    splits = Path(splits_dir)
    seal = splits / SEAL_NAME
    if not seal.is_file():
        raise RuntimeError("Unsealed training inputs: rebuild with refresh-splits before uploading")
    manifest = json.loads(seal.read_text())
    files = _input_files(Path(raw_dir), splits)
    actual = {name: _record(path) for name, path in files.items()}
    if (
        manifest.get("files") != actual
        or manifest.get("producer") != _producer_hashes(Path(repo_root))
        or manifest.get("data_producer_sha256")
        != producer_fingerprint(data_producer_hashes(Path(repo_root)))
    ):
        raise RuntimeError(
            "Training inputs or producer changed after build; rebuild splits before upload"
        )
    release_id = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    base = f"{prefix.rstrip('/')}/releases/{release_id}"
    # Copy first: a concurrent local producer cannot change bytes between the
    # manifest hash and upload; recheck each staged byte before any S3 write.
    with tempfile.TemporaryDirectory(prefix="ff-data-publish-") as directory:
        staged = Path(directory)
        for name, path in files.items():
            dest = staged / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            if _record(dest) != actual[name]:
                raise RuntimeError(f"Training input changed during snapshot: {name}")

        def upload(name):
            key = f"{base}/{name}"
            s3.upload_file(str(staged / name), bucket, key)
            # Verify remote content, not multipart ETags or caller metadata.
            body = s3.get_object(Bucket=bucket, Key=key)["Body"]
            h, size = hashlib.sha256(), 0
            try:
                for chunk in iter(lambda: body.read(1024 * 1024), b""):
                    h.update(chunk)
                    size += len(chunk)
            finally:
                body.close()
            if {"sha256": h.hexdigest(), "bytes": size} != actual[name]:
                raise RuntimeError(f"Uploaded training input failed checksum: {key}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(upload, files))
    body = _json_bytes(manifest)
    s3.put_object(
        Bucket=bucket, Key=f"{base}/manifest.json", Body=body, ContentType="application/json"
    )
    # Publish the producer-specific pointer before global current. Readers for
    # older code continue to find their verified recipe after a newer promotion.
    recipe = manifest["data_producer_sha256"]
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix.rstrip('/')}/by-producer/{recipe}/manifest.json",
        Body=_json_bytes({"schema_version": SCHEMA_VERSION, "release_id": release_id}),
        ContentType="application/json",
    )
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix.rstrip('/')}/manifest.json",
        Body=_json_bytes({"schema_version": SCHEMA_VERSION, "release_id": release_id}),
        ContentType="application/json",
    )
    print(f"Published data release {release_id} ({len(files)} verified files)")
    return release_id


def resolve_release(
    s3, bucket: str, prefix="data", release_id: str | None = None
) -> tuple[str, dict]:
    """Resolve one immutable manifest; never fall back on a corrupt/missing one."""
    prefix = prefix.rstrip("/")
    release_id = release_id or os.environ.get("FF_DATA_RELEASE")
    if not release_id:
        pointer = json.loads(
            s3.get_object(Bucket=bucket, Key=f"{prefix}/manifest.json")["Body"].read()
        )
        if pointer.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported training data pointer schema")
        release_id = pointer.get("release_id", "")
    if not re.fullmatch(r"[0-9a-f]{64}", release_id):
        raise ValueError("FF_DATA_RELEASE must be a published 64-character release hash")
    body = s3.get_object(Bucket=bucket, Key=f"{prefix}/releases/{release_id}/manifest.json")[
        "Body"
    ].read()
    if hashlib.sha256(body).hexdigest() != release_id:
        raise ValueError("Training data manifest checksum mismatch")
    manifest = json.loads(body)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported training data manifest schema")
    files = manifest.get("files", {})
    if not {f"splits/{n}" for n in SPLIT_NAMES}.issubset(files) or not any(
        n.startswith("raw/") for n in files
    ):
        raise ValueError("Training data manifest lacks raw dependencies or splits")
    for name, info in files.items():
        parts = PurePosixPath(name).parts
        if (
            not parts
            or str(PurePosixPath(name)) != name
            or parts[0] not in {"raw", "splits"}
            or any(part in {".", ".."} for part in parts)
            or "\\" in name
            or not (
                len(parts) == 2
                or (
                    len(parts) == 3
                    and parts[:2] == ("raw", "provider_sources")
                    and re.fullmatch(r"[0-9a-f]{64}\.(json|parquet)", parts[2])
                )
            )
        ):
            raise ValueError(f"Unsafe training data path: {name}")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", str(info.get("sha256", "")))
            or not isinstance(info.get("bytes"), int)
            or info["bytes"] < 0
        ):
            raise ValueError(f"Invalid training data checksum: {name}")
    return release_id, manifest


_MANAGED_RAW_PREFIXES = (
    "weekly_",
    "rosters_",
    "rosters_weekly_",
    "schedules_",
    "snap_counts_",
    "injuries_",
    "depth_charts_",
    "redzone_",
    "team_stats_",
    "contracts_",
    "qbr_",
    "ff_opportunity_",
    "kicker_",
    "dst_scoring_",
    "player_id_bridge_",
    "player_metadata_",
    "nflcom_projections_",
    "sleeper_projections_",
    "espn_projections_",
    "weekly_evaluation_reference_",
)


def _validate_release_directories(roots: dict[str, Path], files: dict, selected: str) -> None:
    """Keep every nested write inside the caller's selected data directories.

    The root itself may be an explicitly supplied alias or an EC2 bind mount.
    Descendant symlinks are not owned by this release and must not be followed
    for either installation or quarantine. Check the whole request before any
    destination mutation, including when no provider files occur in the release.
    """
    directories = {name: {Path()} for name in roots}
    for name in files:
        part, relative = name.split("/", 1)
        directories[part].add(Path(relative).parent)
    directories["raw"].update(
        {
            Path("provider_sources"),
            Path(".quarantine") / selected,
            Path(".quarantine") / selected / "provider_sources",
        }
    )
    for part, relative_paths in directories.items():
        root = roots[part]
        if root.exists() and not root.is_dir():
            raise ValueError(f"Training data destination is not a directory: {root}")
        for relative in relative_paths:
            current = root
            for component in relative.parts:
                current = current / component
                if current.is_symlink():
                    raise ValueError(f"Refusing a symlinked release directory: {current}")
                if current.exists() and not current.is_dir():
                    raise ValueError(f"Training data destination is not a directory: {current}")


def _quarantine_unlisted_raw(raw: Path, files: dict, prior_raw: set[str], selected: str) -> None:
    """Remove only previously released or known producer caches from cache hits."""
    if not raw.is_dir():
        return
    expected = {name.split("/", 1)[1] for name in files if name.startswith("raw/")}
    for path in list(raw.iterdir()) + list((raw / "provider_sources").glob("*")):
        relative = path.relative_to(raw).as_posix()
        managed = (
            relative in prior_raw
            or relative.startswith("provider_sources/")
            or (
                path.suffix in {".parquet", ".json", ".etag"}
                and path.name.startswith(_MANAGED_RAW_PREFIXES)
            )
        )
        if not path.is_file() or relative in expected or not managed:
            continue
        quarantine = raw / ".quarantine" / selected / Path(relative).parent
        quarantine.mkdir(parents=True, exist_ok=True)
        os.replace(path, quarantine / path.name)


def download_release(
    s3, bucket: str, *, raw_dir="data/raw", splits_dir="data/splits", prefix="data", release_id=None
) -> dict:
    """Hydrate one release before training/serving; old files survive failed fetches.

    Local readers must wait for this bootstrap to finish. Raw and split paths can
    be existing EC2 bind mounts, so installation uses atomic per-file replacements
    after all downloads pass, followed by the provenance marker as the commit.
    """
    release_id, manifest = resolve_release(s3, bucket, prefix, release_id)
    roots = {"raw": Path(raw_dir), "splits": Path(splits_dir)}
    files = manifest["files"]
    _validate_release_directories(roots, files, release_id)
    prior_seal = roots["splits"] / SEAL_NAME
    prior_raw = set()
    if prior_seal.is_file():
        prior_raw = {
            name.split("/", 1)[1]
            for name in json.loads(prior_seal.read_text()).get("files", {})
            if name.startswith("raw/")
        }
    with tempfile.TemporaryDirectory(prefix="ff-data-download-") as directory:
        stage = Path(directory)

        def download(item):
            name, info = item
            part, leaf = name.split("/", 1)
            current = roots[part] / leaf
            if current.is_file() and _record(current) == info:
                return
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(
                bucket, f"{prefix.rstrip('/')}/releases/{release_id}/{name}", str(destination)
            )
            if _record(destination) != info:
                raise ValueError(f"Training input checksum mismatch: {name}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(download, files.items()))
        # Remote downloads can take time; recheck before touching live roots.
        _validate_release_directories(roots, files, release_id)
        _quarantine_unlisted_raw(roots["raw"], files, prior_raw, release_id)
        for name in files:
            staged = stage / name
            if not staged.exists():
                continue
            part, leaf = name.split("/", 1)
            destination = roots[part] / leaf
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Cross-filesystem safe, and compatible with mounted directories.
            fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".release-")
            os.close(fd)
            try:
                shutil.copyfile(staged, temporary)
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        _atomic_json(roots["splits"] / SEAL_NAME, manifest)
        _atomic_json(
            roots["raw"] / ".release.json",
            {
                "release_id": release_id,
                "provider_sources": (
                    "captured"
                    if any(name.startswith("raw/provider_sources/") for name in files)
                    else "derived_only"
                ),
            },
        )
    return {
        "release_id": release_id,
        "files": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "failed": [],
    }


def prewarm_training_dependencies() -> None:
    """Materialize K/DST inputs and the shared evaluation reference before sealing."""
    from src.config import CACHE_DIR, SEASONS, TEST_SEASONS
    from src.data.dst_scoring import load_dst_scoring_events
    from src.data.external_sources import _seasons_cache_signature
    from src.data.identity import load_player_id_bridge
    from src.data.loader import load_team_week_stats
    from src.k.config import POSITION_CONFIG
    from src.k.data import (
        load_data,
        reconstruct_kicker_kicks_from_pbp,
        reconstruct_kicker_weekly_from_pbp,
    )
    from src.scripts.build_evaluation_reference import write_reference
    from src.shared.evaluation_cohorts import REFERENCE_FILENAME

    # Check the exact dependencies that produced the baked splits before any
    # later producer can mask an earlier optional-source failure by retrying it.
    verify_historical_loader_inputs(CACHE_DIR, SEASONS)
    load_player_id_bridge(CACHE_DIR)
    load_team_week_stats(SEASONS)
    scoring_events = load_dst_scoring_events(SEASONS)
    from src.dst.data import build_data as build_dst_data

    build_dst_data(scoring_events=scoring_events, allow_scoring_fetch=False)
    kicker_seasons = list(POSITION_CONFIG.seasons)
    reconstruct_kicker_weekly_from_pbp([s for s in kicker_seasons if s <= 2024])
    reconstruct_kicker_kicks_from_pbp(kicker_seasons)
    load_data()  # also materialize modern-weekly PBP backfill inputs
    reference = write_reference(TEST_SEASONS, upload=False)
    expected = {
        (position, season)
        for position in ("QB", "RB", "WR", "TE", "K", "DST")
        for season in TEST_SEASONS
    }
    available = set(zip(reference["position"], reference["season"], strict=True))
    if expected - available:
        raise DataReleaseError(
            f"Evaluation reference unavailable for {sorted(expected - available)}; refusing partial release"
        )
    historical = [s for s in kicker_seasons if s <= 2024]
    signature = _seasons_cache_signature(SEASONS)
    required = [
        "player_id_bridge_v2.parquet",
        REFERENCE_FILENAME,
        f"team_stats_{signature}.parquet",
        f"dst_scoring_pbp_v1_{signature}.parquet",
        f"kicker_kicks_pbp_{kicker_seasons[0]}_{kicker_seasons[-1]}.parquet",
        *[f"kicker_backfill_pbp_v1_{s}.parquet" for s in kicker_seasons if s >= 2025],
    ]
    if historical:
        required.append(f"kicker_pbp_{historical[0]}_{historical[-1]}.parquet")
    missing = [name for name in required if not (Path(CACHE_DIR) / name).is_file()]
    if missing:
        raise RuntimeError(f"Cannot seal a partial training dependency build: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--release", default=None)
    parser.add_argument("--prefix", default="data")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="data/splits")
    args = parser.parse_args()
    import boto3

    result = download_release(
        boto3.client("s3"),
        args.bucket,
        raw_dir=args.raw_dir,
        splits_dir=args.splits_dir,
        prefix=args.prefix,
        release_id=args.release,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
