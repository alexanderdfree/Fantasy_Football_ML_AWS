"""Sync position model tarballs from S3 to local model_dir at container boot.

Opt-in via FF_MODEL_S3_BUCKET env var. Unset/empty -> no-op (dev, tests).

Reads s3://{bucket}/{prefix}/{POS}/model.tar.gz (prefix defaults to "models",
matching batch/train.py:upload_artifacts) and extracts into
{repo_root}/{POS}/outputs/models/ so shared.registry's literal model_dir
paths resolve to the freshly-synced files.

Fail-loud: any S3 or tar error raises, so gunicorn --preload aborts before
binding :8000 and ECS marks the task unhealthy, blocking a broken rollout.
"""
from __future__ import annotations
import concurrent.futures
import io
import os
import tarfile
import time
from pathlib import Path

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
_ENV_BUCKET = "FF_MODEL_S3_BUCKET"
_ENV_PREFIX = "FF_MODEL_S3_PREFIX"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _extract_tarball(data: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if dest_resolved not in target.parents and target != dest_resolved:
                raise RuntimeError(f"Tarball escape attempt: {member.name}")
        tar.extractall(dest, filter="data")


def _sync_one(s3_client, bucket: str, prefix: str, pos: str, root: Path) -> dict:
    key = f"{prefix}/{pos}/model.tar.gz"
    dest = root / pos / "outputs" / "models"
    t0 = time.time()
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    _extract_tarball(data, dest)
    return {"pos": pos, "bytes": len(data), "secs": round(time.time() - t0, 2)}


def sync_models_from_s3() -> dict | None:
    """Download+extract all six position tarballs in parallel.

    Returns a summary dict, or None if FF_MODEL_S3_BUCKET is unset/empty.
    """
    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    if not bucket:
        print(f"[model_sync] {_ENV_BUCKET} unset — skipping S3 sync, using on-disk models.")
        return None

    prefix = os.environ.get(_ENV_PREFIX, "models").strip("/")
    root = _repo_root()
    import boto3
    s3 = boto3.client("s3")

    print(f"[model_sync] syncing s3://{bucket}/{prefix}/{{POS}}/model.tar.gz -> {root}")
    t0 = time.time()
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(POSITIONS)) as pool:
        futs = [pool.submit(_sync_one, s3, bucket, prefix, pos, root) for pos in POSITIONS]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    total = round(time.time() - t0, 2)
    print(f"[model_sync] done in {total}s: {results}")
    return {"total_secs": total, "positions": results}
