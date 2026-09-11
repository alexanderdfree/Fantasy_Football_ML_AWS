"""Initialize missing serving manifests, or verify them without changing S3.

Seeding is an operator action for a fresh bucket. All local candidates must
pass the production artifact smoke test before the first write. Existing
manifests are verified through the serving consumer and never replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path

from src.shared.model_sync import POSITIONS, _resolve_manifest_extract, load_manifest


def _check_models(position: str, model_dir: Path) -> None:
    from src.shared.smoke_test import run_smoke_test

    metrics = model_dir / "benchmark_metrics.json"
    if not metrics.is_file() or not isinstance(json.loads(metrics.read_text()), dict):
        raise RuntimeError(f"{position}: missing or invalid benchmark_metrics.json in {model_dir}")
    run_smoke_test(position, model_dir)


def verify_models(s3, bucket: str, prefix: str, positions=POSITIONS) -> list[dict]:
    """Read the same manifest/fallback and extract path used at container boot.

    The additional CPU smoke test rejects parseable archives whose model
    weights/scalers cannot load or predict with this checkout's serving code.
    Temporary extraction never changes local serving artifacts or S3.
    """
    verified = []
    with tempfile.TemporaryDirectory(prefix="ff-seed-verify-") as temp:
        for position in positions:
            dest = Path(temp) / position
            result = _resolve_manifest_extract(s3, bucket, prefix, position, dest)
            _check_models(position, dest)
            verified.append(result)
    return verified


def _register_source(s3, bucket: str, prefix: str, source_sha: str, root: Path) -> None:
    from src.artifacts.source import register_source

    register_source(s3, bucket, prefix, source_sha, repo=root)


def _upload_initial_artifact(s3, bucket, prefix, position, archive, source_sha):
    from src.artifacts.publication import initialize_seed

    return initialize_seed(s3, bucket, prefix, position, archive.read_bytes(), source_sha)


def seed_models(
    s3, bucket: str, prefix: str, root: Path, source_sha: str, positions=POSITIONS
) -> list[dict]:
    """Validate the complete request before publishing any missing positions.

    The producer's initialize-only conditional manifest write also protects a
    training publication that races this preflight. Verification at the end
    checks whichever complete artifact won that race.
    """
    pending = []
    for position in positions:
        if load_manifest(s3, bucket, prefix, position) is not None:
            verify_models(s3, bucket, prefix, (position,))
        else:
            model_dir = root / "src" / position.lower() / "outputs" / "models"
            _check_models(position, model_dir)
            pending.append((position, model_dir))
    if pending:
        from src.artifacts.publication import validate_seed
        from src.artifacts.source import image_source_sha

        if image_source_sha(root=root) != source_sha:
            raise RuntimeError("Seed source must match the actual operator checkout/image")
        # Freeze and validate all candidate bytes before the first source record
        # or model upload. A changing local directory cannot invalidate preflight.
        with tempfile.TemporaryDirectory(prefix="ff-seed-candidates-") as temp:
            archives = []
            for position, model_dir in pending:
                archive = Path(temp) / f"{position}.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    for path in sorted(model_dir.iterdir()):
                        tar.add(path, arcname=path.name)
                validate_seed(position, archive.read_bytes(), source_sha)
                archives.append((position, archive))
            _register_source(s3, bucket, prefix, source_sha, root)
            for position, archive in archives:
                _upload_initial_artifact(s3, bucket, prefix, position, archive, source_sha)
    return verify_models(s3, bucket, prefix, positions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only", action="store_true", help="Only download and verify; no S3 writes"
    )
    parser.add_argument("--bucket", default="ff-predictor-training")
    parser.add_argument("--prefix", default="models")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    prefix = args.prefix.strip("/")
    if not prefix:
        parser.error("--prefix must not be empty")
    import boto3

    try:
        s3 = boto3.client("s3", region_name=args.region)
        if args.verify_only:
            result = verify_models(s3, args.bucket, prefix)
        else:
            from src.artifacts.source import image_source_sha

            source_sha = image_source_sha(root=root)
            result = seed_models(s3, args.bucket, prefix, root, source_sha)
    except Exception as exc:
        print(f"[seed-s3] ERROR: {exc}")
        return 1
    for entry in result:
        print(f"[seed-s3] Verified {entry['pos']}: s3://{args.bucket}/{entry['key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
