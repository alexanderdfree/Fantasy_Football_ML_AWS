"""Sync position model tarballs + inference data from S3 at container boot.

Opt-in via FF_MODEL_S3_BUCKET env var. Unset/empty -> no-op (dev, tests).

`sync_models_from_s3` reads s3://{bucket}/{prefix}/{POS}/model.tar.gz (prefix
defaults to "models", matching batch/train.py:upload_artifacts) and extracts
into {repo_root}/{POS}/outputs/models/ so shared.registry's literal model_dir
paths resolve to the freshly-synced files.

`sync_data_from_s3` pulls the splits + raw weekly parquets that inference
needs (K reconstructs kicker stats from data/raw/, all positions read
schedules for weather features). These were previously baked into the
Docker image via the deploy workflow; fetching at boot shrinks the image
and decouples deploy.yml from data changes.

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

_SPLIT_KEYS = ("train.parquet", "val.parquet", "test.parquet")
_RAW_PREFIX = "data/raw/"
_RAW_EXCLUDE_SUFFIX = "_2023_2023.parquet"


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


def _download_file(s3_client, bucket: str, key: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    dest.write_bytes(data)
    return {"key": key, "bytes": len(data), "secs": round(time.time() - t0, 2)}


def sync_data_from_s3() -> dict | None:
    """Download inference data parquets from S3 in parallel.

    Pulls s3://{bucket}/data/{train,val,test}.parquet into data/splits/ and
    every data/raw/*.parquet except the 2023-only duplicates already covered
    by the 2012-2025 range files. Returns a summary dict, or None if
    FF_MODEL_S3_BUCKET is unset/empty.
    """
    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    if not bucket:
        print(f"[data_sync] {_ENV_BUCKET} unset — skipping S3 sync, using on-disk data.")
        return None

    root = _repo_root()
    import boto3

    s3 = boto3.client("s3")

    jobs: list[tuple[str, Path]] = [
        (f"data/{name}", root / "data" / "splits" / name) for name in _SPLIT_KEYS
    ]

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet") or key.endswith(_RAW_EXCLUDE_SUFFIX):
                continue
            jobs.append((key, root / key))

    print(f"[data_sync] syncing {len(jobs)} files from s3://{bucket}/data/ -> {root / 'data'}")
    t0 = time.time()
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        futs = [pool.submit(_download_file, s3, bucket, key, dest) for key, dest in jobs]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    total = round(time.time() - t0, 2)
    total_bytes = sum(r["bytes"] for r in results)
    print(f"[data_sync] done in {total}s, {total_bytes / 1e6:.1f} MB across {len(results)} files")
    return {"total_secs": total, "total_bytes": total_bytes, "files": len(results)}
