"""AWS Batch training entry point.

Batch runs this as: python src/batch/train.py --position RB --seed 42

Environment variables set via job definition / container overrides:
  TRAINING_DATA_DIR  = /opt/ml/input/data/training/
  MODEL_OUTPUT_DIR   = /opt/ml/model/
  LOG_EVERY          = 1
  S3_BUCKET          = ff-predictor-training
  S3_DATA_PREFIX     = data
"""

import argparse
import contextlib
import copy
import datetime
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

# Ensure project root is on path (baked into /opt/ml/code/ in the container)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import boto3
import pandas as pd
import torch

from src.shared.artifact_gc import prune as _gc_prune
from src.shared.core_pool import ENV_ADDR, ENV_POS, lease_cores, start_coordinator
from src.shared.model_sync import (
    build_manifest,
    load_manifest,
    manifest_key,
    new_history_key,
    write_manifest,
)
from src.shared.platform_detect import detect_platform
from src.shared.registry import (
    ALL_POSITIONS,
    INFERENCE_REGISTRY,
    accepts_dataframes,
    get_config,
    get_runner,
    is_cpu_only,
)
from src.shared.smoke_test import SmokeTestFailed, run_smoke_test
from src.shared.utils import cuda_graph_enabled, cuda_graph_full_enabled, seed_everything
from src.shared.utils import timed as _timed

SPLIT_BRANCHES = {"nn", "cpu"}
SPLIT_ROOT_PREFIX = "split-runs"


def _download_if_stale(s3, bucket, key, local_path):
    """Download s3://bucket/key to local_path, skipping if ETag matches cache.

    Writes a sidecar `{local_path}.etag` file with the remote ETag. On the
    next call, compare the remote ETag to the sidecar; if equal, skip the
    download. Set env var FF_FORCE_REFRESH=1 to force a fresh download.
    Falls through to unconditional download on any head_object failure.
    """
    sidecar = local_path + ".etag"
    try:
        remote_etag = s3.head_object(Bucket=bucket, Key=key)["ETag"]
    except Exception as e:
        print(f"[cache] head_object failed for s3://{bucket}/{key}: {e} — falling back to download")
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        s3.download_file(bucket, key, local_path)
        return
    if (
        os.environ.get("FF_FORCE_REFRESH") != "1"
        and os.path.exists(local_path)
        and os.path.exists(sidecar)
    ):
        with open(sidecar) as f:
            if f.read().strip() == remote_etag:
                print(f"[cache] hit: s3://{bucket}/{key}")
                return
    print(f"[cache] miss: s3://{bucket}/{key} -> {local_path}")
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    s3.download_file(bucket, key, local_path)
    with open(sidecar, "w") as f:
        f.write(remote_etag)


def _read_parquet_cached(parquet_path: str) -> "pd.DataFrame":
    """Read a parquet split, falling back to a Feather cache when fresh.

    The EC2 entrypoint runs each position as a fresh Python invocation, so
    six positions re-parse the same train/val/test parquets. A Feather (Arrow
    IPC) snapshot of the materialised DataFrame loads faster than re-decoding
    the parquet columns, so the second through sixth invocation can skip the
    parse cost. Feather over pickle: same pyarrow dependency we already use
    for parquet, but the file format is not code-executing on read so it
    can't widen the blast radius if the data dir is ever compromised.

    Cache file lives next to the parquet as ``{path}.feather``. We treat it
    as fresh iff its mtime is greater than the parquet's — ``_download_if_stale``
    bumps the parquet's mtime whenever it pulls bytes from S3, so a remote
    refresh automatically invalidates the local cache.

    Writes go through ``tempfile.NamedTemporaryFile`` + ``os.replace`` (mirrors
    the pattern in ``src.shared.benchmark_utils.append_to_history``) so a
    process killed mid-write can't leave a half-baked file that a later
    invocation reads as a stale cache hit. Any read failure (different
    pyarrow version, partial write that somehow slipped past the rename,
    etc.) falls through to the parquet parse silently and triggers a
    self-heal rewrite.
    """
    cache_path = parquet_path + ".feather"
    try:
        if os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(
            parquet_path
        ):
            print(f"[parquet-cache] hit: {cache_path}")
            return pd.read_feather(cache_path)
    except Exception as e:
        print(f"[parquet-cache] cache unreadable ({e}); falling back to parquet parse")
    print(f"[parquet-cache] miss: parsing {parquet_path}")
    df = pd.read_parquet(parquet_path)
    tmp_path = None
    try:
        cache_dir = os.path.dirname(cache_path) or "."
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_dir,
            prefix=os.path.basename(cache_path) + ".",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = tmp_file.name
        df.to_feather(tmp_path)
        os.replace(tmp_path, cache_path)
        tmp_path = None
    except Exception as e:
        print(f"[parquet-cache] failed to write {cache_path}: {e}")
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
    return df


def _start_nvidia_smi_sidecar(csv_path: str) -> subprocess.Popen | None:
    """Start a background nvidia-smi sampler writing GPU stats to ``csv_path``.

    No-op when CUDA isn't visible to torch (CPU-only positions, dev box without
    NVIDIA driver, --dry-run). Returns the Popen handle so the caller can
    terminate it; returns None when sampling was skipped so the caller's
    finally-block stop call is also a no-op.
    """
    if not torch.cuda.is_available():
        return None
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw",
        "--format=csv,nounits",
        "-lms",
        "500",
    ]
    # The file handle has to outlive this function — the Popen child writes to
    # it until terminate(). _stop_nvidia_smi_sidecar() closes it transitively
    # by killing the writer; the OS reclaims the descriptor on process exit.
    f = open(csv_path, "w")  # noqa: SIM115
    try:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        f.close()
        print("[gpu-profile] nvidia-smi not on PATH; skipping sidecar")
        return None
    print(f"[gpu-profile] sampling to {csv_path} (pid={proc.pid})")
    return proc


def _stop_nvidia_smi_sidecar(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    print(f"[gpu-profile] sidecar pid={proc.pid} stopped")


def _assert_gpu(position: str, *, force: bool = False):
    """Log GPU status and fail fast if REQUIRE_GPU=1 and CUDA is unavailable.

    This catches the silent-CPU-on-GPU-billed-instance failure mode where
    the Batch job definition forgets `resourceRequirements: [{type: GPU, ...}]`.

    For positions flagged ``cpu_only`` (K, DST) we skip the REQUIRE_GPU
    assertion so they can still run on a CPU box — but K/DST now train an
    attention NN and DO use CUDA when it's available; the flag only relaxes the
    hard GPU requirement, it does not mean the pipeline avoids CUDA.
    """
    available = torch.cuda.is_available()
    print(f"[gpu] torch.cuda.is_available() = {available}")
    print(f"[gpu] torch.version.cuda        = {torch.version.cuda}")
    print(f"[gpu] torch.__version__         = {torch.__version__}")
    if available:
        print(f"[gpu] device count              = {torch.cuda.device_count()}")
        print(f"[gpu] device 0 name             = {torch.cuda.get_device_name(0)}")
    if is_cpu_only(position) and not force:
        print(f"[gpu] {position} is CPU-only; skipping REQUIRE_GPU assertion")
        return
    require_gpu = os.environ.get("REQUIRE_GPU", "1") == "1"
    if require_gpu and not available:
        raise RuntimeError(
            "REQUIRE_GPU=1 but torch.cuda.is_available() is False. "
            "Check the Batch job definition's resourceRequirements for GPU=1 "
            "and the compute environment's ECS GPU-optimized AMI."
        )


def download_data(s3_bucket, s3_prefix, local_dir):
    """Download training parquet files from S3 to the container."""
    from concurrent.futures import ThreadPoolExecutor

    s3 = boto3.client("s3")
    os.makedirs(local_dir, exist_ok=True)
    names = ("train.parquet", "val.parquet", "test.parquet")

    def _download_one(name):
        s3_key = f"{s3_prefix}/{name}"
        local_path = os.path.join(local_dir, name)
        _download_if_stale(s3, s3_bucket, s3_key, local_path)

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        for _ in pool.map(_download_one, names):
            pass
    print("Data download complete.")


def sync_raw_data(s3_bucket):
    """Sync s3://{bucket}/data/raw/*.parquet into the container's data/raw/.

    Needed by src/shared/weather_features._load_schedules() (all positions during
    feature engineering) and by K/DST's self-contained loaders (src.k.data,
    src.dst.data). CACHE_DIR="data/raw" in src/config.py resolves relative to
    the container WORKDIR=/opt/ml/code. .dockerignore excludes data/ so these
    parquets aren't baked into the image.
    """
    s3 = boto3.client("s3")
    os.makedirs("data/raw", exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix="data/raw/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            local_path = key
            _download_if_stale(s3, s3_bucket, key, local_path)


def _validate_remote_tarball(s3_client, bucket: str, key: str, position: str) -> None:
    """Re-download the just-uploaded tarball and confirm it's structurally
    sound — reopenable, contains ``benchmark_metrics.json`` (parseable), and
    includes the NN weight + scaler files the inference registry expects.

    Runs AFTER the versioned upload and BEFORE the manifest write, so a
    corrupted or truncated upload can't be promoted to ``current``. Any
    raise here leaves the old manifest in place and the site keeps serving
    the previous good artifact.
    """
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = {m.name for m in tar.getmembers()}
        bench = "benchmark_metrics.json"
        if bench not in members:
            raise RuntimeError(
                f"{position}: uploaded tarball s3://{bucket}/{key} is missing "
                f"{bench}. Contents: {sorted(members)}"
            )
        # Parseability — catches a zero-byte or malformed metrics file that
        # slipped past _extract_metrics but would then crash benchmark readers.
        extracted = tar.extractfile(bench)
        if extracted is None:
            raise RuntimeError(f"{position}: {bench} in s3://{bucket}/{key} is not a regular file")
        try:
            json.loads(extracted.read())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"{position}: {bench} in s3://{bucket}/{key} is not valid JSON: {e}"
            ) from e

    reg = INFERENCE_REGISTRY[position]
    required = {reg["nn_file"], "nn_scaler.pkl", "nn_scaler_meta.json"}
    if reg.get("train_attention_nn") and reg.get("attn_nn_file"):
        required.update(
            {
                reg["attn_nn_file"],
                "attention_nn_scaler.pkl",
                "attention_nn_scaler_meta.json",
            }
        )
    # Ridge and LightGBM save into per-target subdirs or with dispatching file
    # names; leaving those out of the strict allowlist avoids false positives
    # on legitimate layouts. The NN weight + scaler pair is the canonical
    # "successful train" signal.
    missing = required - members
    if missing:
        raise RuntimeError(
            f"{position}: uploaded tarball s3://{bucket}/{key} is missing "
            f"required files: {sorted(missing)}. Contents: {sorted(members)}"
        )


def _try_smoke_test(position: str, model_dir: str) -> bool:
    """Run the post-upload load+predict smoke test. Returns True on pass,
    False on any failure. Never raises — a failure must NOT abort the upload
    (the artifact is still useful in ``current``/``history`` for forensics);
    it only pins the manifest's ``stable`` pointer to the previous good run.

    The producer is responsible for passing the boolean result through to
    ``build_manifest(..., smoke_passed=...)``.
    """
    try:
        run_smoke_test(position, model_dir)
    except SmokeTestFailed as e:
        print(
            f"[smoke_test] {position}: FAIL ({e}) — stable pointer NOT advanced",
            flush=True,
        )
        return False
    except Exception as e:
        # An unexpected exception during smoke test (import error, OOM, etc.)
        # is treated as a smoke-test failure rather than aborting the upload —
        # blocking promotion is the safe default. Logged loudly for triage.
        print(
            f"[smoke_test] {position}: UNEXPECTED {e!r} — stable pointer NOT advanced",
            flush=True,
        )
        return False
    print(f"[smoke_test] {position}: PASS", flush=True)
    return True


def upload_artifacts(s3_bucket, position, model_dir):
    """Tar, upload to a versioned history key, validate, smoke-test, atomically
    promote the manifest.

    Order (each step raises on failure unless noted):
      1. Structural check of ``model_dir`` (fast-fail before S3 round-trips).
      2. Build tarball, hash it, pick timestamped + sha7 history key.
      3. Upload to ``history/{ts}-{sha7}/model.tar.gz``.
      4. Re-download + validate (reopenable, expected files present).
      5. Run load+predict smoke test on the local ``model_dir`` (non-fatal —
         only gates whether the new manifest's ``stable`` pointer advances).
      6. Read old ``manifest.json`` (None on first run).
      7. Write new ``manifest.json`` with ``current=new, previous=old.current``
         and ``stable`` advanced iff smoke test passed — **this write is the
         atomic promotion**. Any earlier raise leaves the old manifest in
         place and the site keeps serving the previous good artifact.
      8. Best-effort retention prune (failure is non-fatal). The artifact
         pointed to by ``stable`` is exempted from pruning.

    Note: the legacy ``models/{POS}/model.tar.gz`` mirror is no longer written
    here. Two parallel train-batch runs writing the same legacy key were
    last-write-wins; the manifest's atomic single-PUT promotion is the only
    artifact pointer needed. Consumers — serving via
    ``src.shared.model_sync._sync_one`` and CI benchmark aggregation via
    ``src.batch.benchmark.download_metrics`` — both read the manifest now.
    """
    if not os.path.isdir(model_dir):
        raise RuntimeError(
            f"Model directory {model_dir} does not exist — pipeline did not produce artifacts."
        )
    items = os.listdir(model_dir)
    if not items:
        raise RuntimeError(
            f"Model directory {model_dir} is empty — refusing to upload an "
            "empty tarball. Pipeline likely returned None or failed silently."
        )
    if "benchmark_metrics.json" not in items:
        raise RuntimeError(
            f"benchmark_metrics.json not found in {model_dir}. Contents: {sorted(items)}"
        )

    s3 = boto3.client("s3")
    # Mirrors src.shared.model_sync's consumer-side env read so producer/consumer
    # paths can't drift. Default "models" matches the legacy layout.
    s3_prefix = os.environ.get("FF_MODEL_S3_PREFIX", "models").strip("/")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            for item in items:
                full_path = os.path.join(model_dir, item)
                tar.add(full_path, arcname=item)

        tar_bytes = os.path.getsize(tmp_path)
        with open(tmp_path, "rb") as f:
            sha7 = hashlib.sha256(f.read()).hexdigest()[:7]
        ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        new_key = new_history_key(s3_prefix, position, ts, sha7)

        print(f"Uploading artifacts to s3://{s3_bucket}/{new_key}")
        s3.upload_file(tmp_path, s3_bucket, new_key)

        print(f"Validating uploaded tarball at s3://{s3_bucket}/{new_key}")
        _validate_remote_tarball(s3, s3_bucket, new_key, position)

        smoke_passed = _try_smoke_test(position, model_dir)

        old_manifest = load_manifest(s3, s3_bucket, s3_prefix, position)
        new_manifest = build_manifest(
            new_key=new_key,
            sha7=sha7,
            bytes_=tar_bytes,
            uploaded_at=ts,
            old_manifest=old_manifest,
            smoke_passed=smoke_passed,
        )
        write_manifest(s3, s3_bucket, s3_prefix, position, new_manifest)
        print(f"Promoted s3://{s3_bucket}/{manifest_key(s3_prefix, position)}")

        try:
            deleted = _gc_prune(s3, s3_bucket, s3_prefix, position, new_manifest)
            if deleted:
                print(f"Pruned {len(deleted)} old history entries.")
        except Exception as e:
            # Retention failure is recoverable — next successful run will
            # re-prune. Don't let it mask the upload success.
            print(f"WARNING: retention prune failed (non-fatal): {e!r}")

        print("Artifact upload complete.")
    finally:
        os.unlink(tmp_path)


def _hardware_metadata() -> dict:
    """Runtime GPU facts for the History-tab hardware label.

    Stamped into ``benchmark_metrics.json`` so ``src/batch/benchmark.py`` builds
    the instance label from what the job ACTUALLY ran on — ``gpu_name`` and
    whether CUDA-graph capture was active (``cuda_graph_active``) — instead of a
    hardcoded workflow string that silently drifts when the compute environment
    changes (e.g. T4/g4dn -> L4/g6, or A10G/g5 fallback). ``sm`` records the compute capability that
    gates capture, so a reader can see *why* it was on/off (sm_75 < sm_80 -> off).

    On a non-CUDA box (CPU-only positions in a CPU container, dry-run, dev/CI)
    ``gpu_name``/``sm`` are ``None`` and ``cuda_graph_active`` is ``False``;
    benchmark.py treats a run with no GPU-bearing position as "no metadata" and
    falls back to its ``--instance-type`` argument.
    """
    info = detect_platform()
    return {
        "gpu_name": info.gpu_name,
        "sm": info.sm,
        "cuda_graph_active": cuda_graph_enabled(),
        # Records the capture SCOPE so the History tab marks the 2026-06-15
        # full-step rebaseline discontinuity (full-step is the prod default on
        # sm_80+; ADR-0017). model-only-graphed history predates this flag.
        "cuda_graph_full_active": cuda_graph_full_enabled(),
    }


def _extract_metrics(position, result):
    """Extract JSON-serializable benchmark metrics from pipeline result."""
    metrics: dict = {"position": position}

    # Stamp the image's commit SHA into the per-position artifact. launch.py
    # forwards FF_TRAIN_GIT_SHA from train-batch.yml; benchmark.py reads each
    # position's recorded SHA and warns when they diverge inside a single
    # train-batch roll-up (the lingering "two parallel runs hit the same
    # position's manifest in flight" race that Layer A's revision pinning
    # closes at submit time but can still surface if jobs interleave).
    git_sha = os.environ.get("FF_TRAIN_GIT_SHA", "").strip()
    if git_sha:
        metrics["git_sha"] = git_sha

    for model_key in ["ridge", "elasticnet", "nn", "attn_nn", "lgbm"]:
        m_key = f"{model_key}_metrics"
        r_key = f"{model_key}_ranking"
        m = result.get(m_key)
        if not m:
            continue
        metrics[m_key] = {
            "total": {
                k: (round(v, 4) if isinstance(v, (int, float)) else v)
                for k, v in m["total"].items()
            },
        }
        for t in m:
            if t != "total":
                metrics[m_key][t] = {
                    k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in m[t].items()
                }
        if r_key in result:
            ranking = result[r_key]
            metrics[r_key] = {
                "season_avg_hit_rate": round(ranking["season_avg_hit_rate"], 4),
            }
            if "season_avg_spearman" in ranking:
                metrics[r_key]["season_avg_spearman"] = round(ranking["season_avg_spearman"], 4)

    return metrics


def _dry_run_artifacts(
    position: str,
    model_dir: str,
    seed: int,
    t_total: float,
    phase_seconds: dict[str, float],
) -> None:
    """Write minimal stub artifacts for --dry-run mode.

    Exercises the post-training side of main() (artifact layout, metric
    serialization, non-None result guard) without invoking the heavy per-
    position pipeline. This lets the CLI be smoke-tested end-to-end in
    under a second with no S3 / data / GPU dependencies.
    """
    os.makedirs(model_dir, exist_ok=True)
    # Stub model file so model_dir is non-empty (upload_artifacts invariant).
    stub_path = os.path.join(model_dir, f"{position.lower()}_model.stub")
    with open(stub_path, "w") as f:
        f.write(f"dry-run stub for {position} (seed={seed})\n")
    metrics = {
        "position": position,
        "dry_run": True,
        "seed": seed,
        "ridge_metrics": {"total": {"mae": 0.0, "r2": 0.0}},
        "elapsed_sec": round(time.monotonic() - t_total, 1),
        "phase_seconds": phase_seconds,
    }
    with open(os.path.join(model_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[dry-run] Wrote stub artifacts to {model_dir}")


def _replace_model_dir_contents(src: str, dst: str) -> None:
    """Replace dst's contents with src's.

    On EC2, dst is /opt/ml/model — a bind-mount from /opt/ff/scratch/model
    that persists across ff-train invocations. We cannot rmtree the mount
    point (rmdir on a mount fails; rmtree with ignore_errors leaves an
    empty dir that then trips copytree's "dst must not exist" check). So
    clear the mount's contents in place, then copytree with
    dirs_exist_ok=True. Without the clear step, sequential ff-train calls
    would accumulate every prior position's artifacts into dst — including
    PCAs fit for the wrong feature count that then crash inference.
    """
    for name in os.listdir(dst):
        child = os.path.join(dst, name)
        if os.path.isdir(child) and not os.path.islink(child):
            shutil.rmtree(child)
        else:
            os.remove(child)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _split_root_prefix() -> str:
    return os.environ.get("FF_SPLIT_S3_PREFIX", SPLIT_ROOT_PREFIX).strip("/") or SPLIT_ROOT_PREFIX


def _split_key(split_run_id: str, position: str, branch: str, name: str) -> str:
    return f"{_split_root_prefix()}/{split_run_id}/{position}/{branch}/{name}"


def _branch_config(position: str, branch: str) -> dict:
    """Build a partial-training config for one split branch."""
    if branch not in SPLIT_BRANCHES:
        raise ValueError(f"Unsupported split branch: {branch}")
    cfg = copy.deepcopy(get_config(position))
    cfg["_artifact_branch"] = branch
    # Split mode intentionally publishes only the served model families.
    cfg["train_elasticnet"] = False
    cfg["train_tabpfn"] = False
    if branch == "nn":
        cfg["train_base_nn"] = True
        cfg["train_attention_nn"] = True
        cfg["train_ridge"] = False
        cfg["train_lightgbm"] = False
    else:
        cfg["train_base_nn"] = False
        cfg["train_attention_nn"] = False
        cfg["train_ridge"] = True
        cfg["train_lightgbm"] = True
    return cfg


def _available_core_ids() -> list[int]:
    try:
        cores = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = list(range(os.cpu_count() or 1))
    if not cores:
        raise RuntimeError("No CPU cores visible to the training container")
    return cores


@contextlib.contextmanager
def _dynamic_cpu_core_pool(position: str):
    """Activate the work-conserving 4-core pool for one CPU split job."""
    visible_cores = _available_core_ids()
    requested = int(os.environ.get("FF_CPU_BRANCH_CORES", str(len(visible_cores))))
    if requested < 1:
        raise RuntimeError(f"FF_CPU_BRANCH_CORES must be >= 1, got {requested}")
    if len(visible_cores) < requested:
        raise RuntimeError(
            f"CPU split branch requested {requested} cores but only "
            f"{len(visible_cores)} are visible: {visible_cores}"
        )
    cores = visible_cores[:requested]
    socket_dir = tempfile.mkdtemp(prefix="ff-core-pool-")
    addr, set_active_count, stop = start_coordinator(cores, socket_dir)
    set_active_count(1)

    managed_env = {
        ENV_ADDR: addr,
        ENV_POS: position,
        "FF_DEVICE": "cpu",
        "LGBM_N_JOBS": "1",
        "LOKY_MAX_CPU_COUNT": str(requested),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    previous = {key: os.environ.get(key) for key in managed_env}
    os.environ.update(managed_env)
    try:
        with lease_cores("probe_preflight", default=0) as granted:
            if granted != requested:
                raise RuntimeError(
                    f"CPU core-pool preflight leased {granted} cores; expected {requested}"
                )
        print(f"[split-cpu] core pool active for {position}: cores={cores}, addr={addr}")
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        stop()
        shutil.rmtree(socket_dir, ignore_errors=True)


def _tar_directory(directory: str, tmp_path: str) -> None:
    items = os.listdir(directory)
    if not items:
        raise RuntimeError(f"{directory} is empty; refusing to create split artifact")
    with tarfile.open(tmp_path, "w:gz") as tar:
        for item in items:
            full_path = os.path.join(directory, item)
            tar.add(full_path, arcname=item)


def _validate_split_dir(position: str, branch: str, directory: str) -> None:
    if not os.path.isdir(directory):
        raise RuntimeError(f"{branch} split artifact directory missing: {directory}")
    required = {"benchmark_metrics.json", "split_branch.json"}
    reg = INFERENCE_REGISTRY[position]
    if branch == "nn":
        required.update({reg["nn_file"], "nn_scaler.pkl", "nn_scaler_meta.json"})
        if reg.get("train_attention_nn") and reg.get("attn_nn_file"):
            required.update(
                {
                    reg["attn_nn_file"],
                    "attention_nn_scaler.pkl",
                    "attention_nn_scaler_meta.json",
                }
            )
    elif branch == "cpu":
        required.update(set(reg["targets"]))
        if reg.get("train_lightgbm", False):
            required.add("lightgbm")
    else:
        raise ValueError(f"Unsupported split branch: {branch}")

    missing = [
        name for name in sorted(required) if not os.path.exists(os.path.join(directory, name))
    ]
    if missing:
        raise RuntimeError(
            f"{position} {branch} split artifact is incomplete; missing {missing}. "
            f"Contents: {sorted(os.listdir(directory))}"
        )


def _write_split_branch_metadata(
    *,
    position: str,
    branch: str,
    split_run_id: str,
    seed: int,
    model_dir: str,
) -> None:
    doc = {
        "schema_version": 1,
        "position": position,
        "branch": branch,
        "split_run_id": split_run_id,
        "seed": seed,
        "git_sha": os.environ.get("FF_TRAIN_GIT_SHA", "").strip(),
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    with open(os.path.join(model_dir, "split_branch.json"), "w") as f:
        json.dump(doc, f, indent=2)


def _upload_split_branch_artifacts(
    s3_bucket: str,
    position: str,
    branch: str,
    split_run_id: str,
    model_dir: str,
    seed: int,
) -> None:
    """Upload a branch artifact to split staging without touching prod manifests."""
    _validate_split_dir(position, branch, model_dir)
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        _tar_directory(model_dir, tmp_path)
        with open(tmp_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        sha7 = digest[:7]
        size = os.path.getsize(tmp_path)
        uploaded_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        tar_key = _split_key(split_run_id, position, branch, "model.tar.gz")
        manifest_key_ = _split_key(split_run_id, position, branch, "manifest.json")
        print(f"[split-{branch}] Uploading staged artifact to s3://{s3_bucket}/{tar_key}")
        s3.upload_file(tmp_path, s3_bucket, tar_key)
        manifest = {
            "schema_version": 1,
            "split_run_id": split_run_id,
            "position": position,
            "branch": branch,
            "seed": seed,
            "git_sha": os.environ.get("FF_TRAIN_GIT_SHA", "").strip(),
            "key": tar_key,
            "sha256": digest,
            "sha7": sha7,
            "bytes": size,
            "uploaded_at": uploaded_at,
        }
        s3.put_object(
            Bucket=s3_bucket,
            Key=manifest_key_,
            Body=json.dumps(manifest, indent=2).encode(),
            ContentType="application/json",
        )
        print(f"[split-{branch}] Wrote manifest s3://{s3_bucket}/{manifest_key_}")
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)


def _load_split_manifest(s3, bucket: str, split_run_id: str, position: str, branch: str) -> dict:
    key = _split_key(split_run_id, position, branch, "manifest.json")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except Exception as e:
        raise RuntimeError(f"Missing split manifest s3://{bucket}/{key}: {e}") from e
    try:
        manifest = json.loads(obj["Body"].read())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Split manifest s3://{bucket}/{key} is invalid JSON: {e}") from e
    expected = {"split_run_id": split_run_id, "position": position, "branch": branch}
    mismatches = {
        name: (manifest.get(name), value)
        for name, value in expected.items()
        if manifest.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"Split manifest s3://{bucket}/{key} mismatch: {mismatches}")
    return manifest


def _download_split_branch_artifacts(
    s3,
    bucket: str,
    split_run_id: str,
    position: str,
    branch: str,
    expected_git_sha: str,
    parent_dir: str,
) -> tuple[str, dict]:
    manifest = _load_split_manifest(s3, bucket, split_run_id, position, branch)
    git_sha = str(manifest.get("git_sha") or "")
    if expected_git_sha and git_sha != expected_git_sha:
        raise RuntimeError(
            f"{position} {branch} split artifact SHA mismatch: "
            f"manifest={git_sha!r}, expected={expected_git_sha!r}"
        )
    key = manifest.get("key")
    if not key:
        raise RuntimeError(f"{position} {branch} split manifest has no artifact key")

    tmp_tar = os.path.join(parent_dir, f"{branch}.tar.gz")
    out_dir = os.path.join(parent_dir, branch)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[split-merge] Downloading {branch} artifact s3://{bucket}/{key}")
    s3.download_file(bucket, key, tmp_tar)
    with open(tmp_tar, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if manifest.get("sha256") and digest != manifest["sha256"]:
        raise RuntimeError(
            f"{position} {branch} split artifact checksum mismatch: "
            f"downloaded={digest}, manifest={manifest['sha256']}"
        )
    if manifest.get("bytes") and os.path.getsize(tmp_tar) != int(manifest["bytes"]):
        raise RuntimeError(
            f"{position} {branch} split artifact size mismatch: "
            f"downloaded={os.path.getsize(tmp_tar)}, manifest={manifest['bytes']}"
        )
    try:
        with tarfile.open(tmp_tar, "r:gz") as tar:
            tar.extractall(out_dir, filter="data")
    except tarfile.TarError as e:
        raise RuntimeError(f"{position} {branch} split artifact is not a valid tarball") from e
    _validate_split_dir(position, branch, out_dir)

    with open(os.path.join(out_dir, "split_branch.json")) as f:
        branch_doc = json.load(f)
    for name, value in {
        "split_run_id": split_run_id,
        "position": position,
        "branch": branch,
    }.items():
        if branch_doc.get(name) != value:
            raise RuntimeError(
                f"{position} {branch} split_branch.json mismatch for {name}: "
                f"{branch_doc.get(name)!r} != {value!r}"
            )
    return out_dir, manifest


def _copy_merge_artifacts(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    skipped = {"benchmark_metrics.json", "split_branch.json"}
    for name in os.listdir(src):
        if name in skipped:
            continue
        source = os.path.join(src, name)
        target = os.path.join(dst, name)
        if os.path.exists(target):
            raise RuntimeError(f"Split merge artifact conflict at {target}")
        if os.path.isdir(source) and not os.path.islink(source):
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def _read_json_file(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _merged_split_metrics(
    position: str,
    split_run_id: str,
    nn_metrics: dict,
    cpu_metrics: dict,
    phase_seconds: dict[str, float],
    t_total: float,
) -> dict:
    """Combine the two branch ``benchmark_metrics.json`` payloads into the
    merged artifact's metrics dict.

    Pure (no S3 / filesystem) so the shape the merge job publishes — including
    the ``*_ranking`` blocks behind every ``{model}_top12`` in
    benchmark_history, which the split branches silently dropped from
    2026-06-11 until the pipeline-side ranking attach — is pinned by unit
    tests (tests/batch/test_train.py).
    """
    metrics = {
        "position": position,
        "split_merged": True,
        "split_run_id": split_run_id,
        "seed": nn_metrics.get("seed", cpu_metrics.get("seed")),
        "elapsed_sec": round(time.monotonic() - t_total, 1),
        "phase_seconds": dict(phase_seconds),
    }
    for branch, branch_metrics in (("nn", nn_metrics), ("cpu", cpu_metrics)):
        for key, value in branch_metrics.items():
            if key.endswith("_metrics") or key.endswith("_ranking"):
                if key in metrics:
                    raise RuntimeError(f"Duplicate metric key during split merge: {key}")
                metrics[key] = value
        for phase, secs in (branch_metrics.get("phase_seconds") or {}).items():
            metrics["phase_seconds"][f"split.{branch}.{phase}"] = secs
        if branch_metrics.get("elapsed_sec") is not None:
            metrics["phase_seconds"][f"split.{branch}.elapsed_sec"] = branch_metrics["elapsed_sec"]

    # The NN branch carries the GPU/capture facts (Ridge/LGBM run CPU-only).
    # Derive the keys FROM _hardware_metadata (values still come from nn_metrics)
    # so a new field can't silently drop out of the merged artifact — exactly
    # how cuda_graph_full_active (the 2026-06-15 full-step rebaseline marker,
    # ADR-0017) was lost from production History rows until 2026-06-19. We use
    # only its .keys(); the merge job's own platform is irrelevant here.
    for key in ("git_sha", *_hardware_metadata()):
        if key in nn_metrics:
            metrics[key] = nn_metrics[key]
    return metrics


def _merge_split_artifacts(
    s3_bucket: str,
    position: str,
    split_run_id: str,
    model_dir: str,
    phase_seconds: dict[str, float],
    t_total: float,
) -> None:
    """Download staged CPU+NN tarballs, merge them, and promote the full artifact."""
    expected_git_sha = os.environ.get("FF_TRAIN_GIT_SHA", "").strip()
    s3 = boto3.client("s3")
    with tempfile.TemporaryDirectory(prefix=f"ff-split-merge-{position}-") as tmpdir:
        nn_dir, nn_manifest = _download_split_branch_artifacts(
            s3, s3_bucket, split_run_id, position, "nn", expected_git_sha, tmpdir
        )
        cpu_dir, cpu_manifest = _download_split_branch_artifacts(
            s3, s3_bucket, split_run_id, position, "cpu", expected_git_sha, tmpdir
        )
        if (
            nn_manifest.get("git_sha")
            and cpu_manifest.get("git_sha")
            and nn_manifest["git_sha"] != cpu_manifest["git_sha"]
        ):
            raise RuntimeError(
                f"{position} split artifact SHA mismatch between branches: "
                f"nn={nn_manifest['git_sha']}, cpu={cpu_manifest['git_sha']}"
            )
        _replace_model_dir_contents(nn_dir, model_dir)
        # Drop branch-only metadata copied by the first replace before adding CPU files.
        for name in ("benchmark_metrics.json", "split_branch.json"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(model_dir, name))
        _copy_merge_artifacts(cpu_dir, model_dir)

        nn_metrics = _read_json_file(os.path.join(nn_dir, "benchmark_metrics.json"))
        cpu_metrics = _read_json_file(os.path.join(cpu_dir, "benchmark_metrics.json"))

    metrics = _merged_split_metrics(
        position, split_run_id, nn_metrics, cpu_metrics, phase_seconds, t_total
    )

    metrics_path = os.path.join(model_dir, "benchmark_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[split-merge] Wrote merged metrics to {metrics_path}")

    upload_artifacts(s3_bucket, position, model_dir)


def _run_rb_gate_ablation(train_df, val_df, test_df, seed: int) -> None:
    """Container-side RB TD-head ablation runner.

    Delegates to ``src.tuning.ablate_rb_gate.VARIANTS`` and ``print_summary``
    so this path stays in sync with the operator-CLI version (running
    ``python -m src.tuning.ablate_rb_gate`` locally). Previously this
    function carried a hand-rolled fork that knew only 3 variants (A/B/C)
    with the old ">=0.05 pt/game keep-gate" rule, while ``ablate_rb_gate``
    grew to 6 variants (A/B/C/D/E/Bf) and a "lowest count-target MAE sum"
    decision rule. Two sources of truth shipped from one workflow trigger.

    The dataframes are passed through to avoid the ~30s/variant re-download
    that ``run(seed=...)`` would trigger inside the container.
    """
    from src.rb.run_pipeline import CONFIG, run
    from src.tuning.ablate_rb_gate import VARIANTS, print_summary

    rows: list[dict] = []
    for variant in sorted(VARIANTS):
        label, fn = VARIANTS[variant]
        print(f"\n{'=' * 72}")
        print(f"Variant {variant}: {label}")
        print(f"{'=' * 72}", flush=True)
        result = run(train_df, val_df, test_df, seed=seed, config=fn(CONFIG))
        attn = result.get("attn_nn_metrics") or result.get("metrics", {}).get("attn_nn")
        if attn is None:
            raise RuntimeError(
                f"Variant {variant}: could not find attn_nn_metrics in result keys "
                f"{sorted(result.keys())}"
            )
        rows.append(
            {
                "variant": variant,
                "label": label,
                "seed": seed,
                "fp_mae": attn["total"]["mae"],
                "fp_rmse": attn["total"]["rmse"],
                "rushing_tds_mae": attn["rushing_tds"]["mae"],
                "receiving_tds_mae": attn["receiving_tds"]["mae"],
                "fumbles_lost_mae": attn["fumbles_lost"]["mae"],
                "receptions_mae": attn["receptions"]["mae"],
                "rushing_yards_mae": attn["rushing_yards"]["mae"],
                "receiving_yards_mae": attn["receiving_yards"]["mae"],
                "count_target_mae_sum": (
                    attn["rushing_tds"]["mae"]
                    + attn["receiving_tds"]["mae"]
                    + attn["fumbles_lost"]["mae"]
                ),
                "gate_aucs": {
                    t: attn[t].get("gate_auc")
                    for t in attn
                    if isinstance(attn.get(t), dict) and "gate_auc" in attn[t]
                },
            }
        )

    print_summary(rows)


def _run_scheduler_type_ablation(pos, seeds, s3_bucket, frames=None):
    """Container-side LR-scheduler-type A/B runner.

    Delegates to ``src.tuning.ablate_scheduler_type.run_position`` so this path
    stays in sync with the operator CLI (``python -m
    src.tuning.ablate_scheduler_type``). Runs the 3-way (onecycle / cosine /
    plateau) A/B for one position across ``seeds`` and uploads the result JSON to
    ``s3://{s3_bucket}/ablate_scheduler/{POS}/result.json``. No model artifact
    upload — this is a diagnostic. ``frames`` (train/val/test) is passed through
    for QB/RB/WR/TE to avoid a re-read per variant; K/DST pass ``None`` and let
    their self-contained loaders run. ``FF_CUDA_GRAPH=0`` (set by the launcher)
    keeps the A/B bit-comparable and eager.
    """
    from src.tuning.ablate_scheduler_type import run_position

    result = run_position(pos, seeds, frames=frames)
    key = f"ablate_scheduler/{pos.upper()}/result.json"
    boto3.client("s3").put_object(
        Bucket=s3_bucket, Key=key, Body=json.dumps(result, indent=2).encode()
    )
    print(f"Uploaded scheduler-type ablation result to s3://{s3_bucket}/{key}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", required=True, choices=ALL_POSITIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        default=None,
        help="(--ablation scheduler-type only) comma-separated seeds "
        "(default: the single --seed value).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip S3 download/upload and the real pipeline. Writes stub "
        "artifacts so main() can be smoke-tested end-to-end without "
        "AWS credentials or training data.",
    )
    parser.add_argument(
        "--ablation",
        choices=["rb-gate", "scheduler-type"],
        default=None,
        help="Run a named ablation instead of a standard training run. "
        "'rb-gate' requires --position RB; runs the six-variant TD-gate "
        "ablation and prints the decision table. 'scheduler-type' runs the "
        "LR-scheduler-type A/B (onecycle/cosine/plateau) for --position over "
        "--seeds and uploads the per-position result JSON to "
        "s3://$S3_BUCKET/ablate_scheduler/{pos}/result.json. Skips model upload.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the WR attention NN batch-size sweep (see "
        "src.scripts.batch_size_sweep). Requires --position WR. Downloads "
        "splits from S3 then delegates to run_sweep(); skips S3 upload.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "tune"],
        default="train",
        help=(
            "Dispatch mode. 'train' (default) is the existing per-position "
            "training path with S3 artifact upload. 'tune' delegates to "
            "src.tuning.tune_nn for Optuna NN hyperparameter search; consumes "
            "--n-trials/--timeout and forwards --checkpoint-s3 so the SQLite "
            "study DB round-trips to s3://$S3_BUCKET/tune_nn/{pos}/. Mutually "
            "exclusive with --ablation/--sweep/--dry-run."
        ),
    )
    parser.add_argument(
        "--branch",
        choices=["full", "nn", "cpu", "merge"],
        default="full",
        help=(
            "Split-training branch. 'full' preserves the existing monolithic "
            "path. 'nn' stages base+attention NN artifacts, 'cpu' stages "
            "Ridge+LightGBM artifacts, and 'merge' combines staged artifacts "
            "then publishes the normal manifest."
        ),
    )
    parser.add_argument(
        "--split-run-id",
        default=os.environ.get("FF_SPLIT_RUN_ID"),
        help="Required for --branch nn/cpu/merge; namespaces staged split artifacts.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="(--mode=tune only) Number of Optuna trials per position.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="(--mode=tune only) Per-position wall-clock cap in seconds.",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=["thread", "mps", "auto"],
        default=None,
        help="(--mode=tune only) Trial concurrency backend forwarded to src.tuning.tune_nn.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="(--mode=tune only) Concurrent trial workers forwarded to src.tuning.tune_nn.",
    )
    args = parser.parse_args()

    pos = args.position
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else [args.seed]
    if args.branch != "full":
        if args.mode == "tune":
            parser.error("--mode=tune is mutually exclusive with --branch")
        if args.ablation or args.sweep:
            parser.error("--branch is mutually exclusive with --ablation/--sweep")
        if not args.split_run_id and not args.dry_run:
            parser.error("--split-run-id is required for --branch nn/cpu/merge")
    if args.ablation == "rb-gate" and pos != "RB":
        parser.error("--ablation rb-gate requires --position RB")
    if args.sweep and pos != "WR":
        parser.error("--sweep requires --position WR")
    if args.mode == "tune":
        # Tune mode runs Optuna over the attention NN — no artifact upload, no
        # ablation, no sweep. Reject conflicting flags up front so the Batch
        # job fails fast instead of running half a pipeline.
        if args.ablation or args.sweep or args.dry_run:
            parser.error("--mode=tune is mutually exclusive with --ablation/--sweep/--dry-run")
        # Forward to src.tuning.tune_nn's CLI. We replace sys.argv rather than
        # constructing the parser dance — keeps tune_nn's behaviour identical
        # whether the operator runs it directly or through this dispatcher.
        # --checkpoint-s3 is always on because Batch runs are Spot-resilient.
        from src.tuning import tune_nn

        tune_argv = ["tune_nn", pos, "--checkpoint-s3", "--seed", str(args.seed)]
        if args.n_trials is not None:
            tune_argv += ["--n-trials", str(args.n_trials)]
        if args.timeout is not None:
            tune_argv += ["--timeout", str(args.timeout)]
        if args.parallel_backend is not None:
            tune_argv += ["--parallel-backend", args.parallel_backend]
        if args.n_jobs is not None:
            tune_argv += ["--n-jobs", str(args.n_jobs)]
        # _assert_gpu and seed_everything below are training-path setup; tune
        # mode handles its own seeding inside tune_nn.main(). But REQUIRE_GPU
        # still matters — attention training is the bulk of each trial. Run
        # _assert_gpu here so a misconfigured GPU job fails before Optuna
        # spends 10 minutes per CPU-bound trial.
        _assert_gpu(pos, force=True)
        sys.argv = tune_argv
        tune_nn.main()
        return

    # Print build fingerprint so stale container images are immediately obvious.
    _fingerprint_file = os.path.join(os.path.dirname(__file__), "train.py")
    with open(_fingerprint_file, "rb") as _f:
        _hash = hashlib.sha256(_f.read()).hexdigest()[:12]
    print(f"[src/batch/train.py] build fingerprint: {_hash}")

    _t_total = time.monotonic()
    phase_seconds: dict[str, float] = {}
    # Skip the GPU assertion in dry-run — local/CI smoke tests rarely have CUDA.
    if args.dry_run:
        print(f"[dry-run] skipping _assert_gpu for {pos}")
    else:
        with _timed("assert_gpu", store=phase_seconds):
            if args.branch in {"cpu", "merge"}:
                print(f"[gpu] branch={args.branch}; skipping REQUIRE_GPU assertion")
            elif args.branch == "nn":
                _assert_gpu(pos, force=True)
            else:
                _assert_gpu(pos)
    seed_everything(args.seed)

    s3_bucket = os.environ.get("S3_BUCKET", "ff-predictor-training")
    s3_prefix = os.environ.get("S3_DATA_PREFIX", "data")
    data_dir = os.environ.get("TRAINING_DATA_DIR", "/opt/ml/input/data/training")
    model_dir = os.environ.get("MODEL_OUTPUT_DIR", "/opt/ml/model")
    # LOG_EVERY is consumed directly by src.shared.pipeline._resolve_nn_log_every()
    # so we don't need to inject it into cfg from here. Historically we
    # monkey-patched run_pipeline, but that only worked if callers used
    # `import src.shared.pipeline as pipeline_mod; pipeline_mod.run_pipeline(...)`.
    # All position runners use `from src.shared.pipeline import run_pipeline`, so
    # the patch was dead code. Env-var resolution sidesteps the issue.

    os.makedirs(model_dir, exist_ok=True)

    if args.dry_run:
        # Stub out S3 and the pipeline — we still exercise arg parsing,
        # seed setup, model-dir setup, metrics serialization, and the
        # skip-S3 code path.
        _dry_run_artifacts(pos, model_dir, args.seed, _t_total, phase_seconds)
        print(f"[dry-run] Completed for {pos}; skipping S3 upload.")
        return

    if args.branch == "merge":
        with _timed("merge_split_artifacts", store=phase_seconds):
            _merge_split_artifacts(
                s3_bucket,
                pos,
                args.split_run_id,
                model_dir,
                phase_seconds,
                _t_total,
            )
        print(f"[timing] total={time.monotonic() - _t_total:.1f}", flush=True)
        return

    run_fn = get_runner(pos)
    branch_cfg = _branch_config(pos, args.branch) if args.branch in SPLIT_BRANCHES else None

    # data/raw/*.parquet is needed for weather features (all positions) and
    # for K/DST's self-contained data loaders. Sync before branching.
    with _timed("sync_raw_data", store=phase_seconds):
        sync_raw_data(s3_bucket)

    gpu_profile_csv = f"/tmp/gpu_profile_{pos}.csv"

    if accepts_dataframes(pos):
        # Download train/val/test splits from S3 into the container
        with _timed("download_data", store=phase_seconds):
            download_data(s3_bucket, s3_prefix, data_dir)
        with _timed("read_parquets", store=phase_seconds):
            train_df = _read_parquet_cached(os.path.join(data_dir, "train.parquet"))
            val_df = _read_parquet_cached(os.path.join(data_dir, "val.parquet"))
            test_df = _read_parquet_cached(os.path.join(data_dir, "test.parquet"))
            print(f"Loaded data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        if args.ablation == "rb-gate":
            # Ablation runs the pipeline 3x with config overrides and prints
            # a decision table. No S3 artifact upload — this is a diagnostic
            # run, not a shipping build.
            with _timed("run_ablation", store=phase_seconds):
                _run_rb_gate_ablation(train_df, val_df, test_df, seed=args.seed)
            print(f"[timing] total={time.monotonic() - _t_total:.1f}", flush=True)
            return
        if args.ablation == "scheduler-type":
            with _timed("run_ablation", store=phase_seconds):
                _run_scheduler_type_ablation(
                    pos, seeds, s3_bucket, frames=(train_df, val_df, test_df)
                )
            print(f"[timing] total={time.monotonic() - _t_total:.1f}", flush=True)
            return
        if args.sweep:
            # Sweep runs the WR pipeline N times with attn_batch_size overrides
            # and prints a wall-clock table. No S3 artifact upload — diagnostic.
            from src.scripts.batch_size_sweep import run_sweep

            sweep_sidecar = _start_nvidia_smi_sidecar(gpu_profile_csv)
            try:
                with _timed("run_sweep", store=phase_seconds):
                    run_sweep(train_df, val_df, test_df, seed=args.seed)
            finally:
                _stop_nvidia_smi_sidecar(sweep_sidecar)
            print(f"[timing] total={time.monotonic() - _t_total:.1f}", flush=True)
            return
        gpu_sidecar = _start_nvidia_smi_sidecar(gpu_profile_csv) if args.branch != "cpu" else None
        try:
            with _timed("run_pipeline", store=phase_seconds):
                if branch_cfg is None:
                    result = run_fn(train_df, val_df, test_df, seed=args.seed)
                elif args.branch == "cpu":
                    with _dynamic_cpu_core_pool(pos):
                        result = run_fn(
                            train_df,
                            val_df,
                            test_df,
                            seed=args.seed,
                            config=branch_cfg,
                        )
                else:
                    result = run_fn(
                        train_df,
                        val_df,
                        test_df,
                        seed=args.seed,
                        config=branch_cfg,
                    )
        finally:
            _stop_nvidia_smi_sidecar(gpu_sidecar)
    else:
        # K/DST: self-contained raw-data loaders. They still use the
        # sync_raw_data() pull above, but they intentionally do not consume the
        # train/val/test split parquet artifacts used by QB/RB/WR/TE.
        if args.ablation == "scheduler-type":
            with _timed("run_ablation", store=phase_seconds):
                _run_scheduler_type_ablation(pos, seeds, s3_bucket, frames=None)
            print(f"[timing] total={time.monotonic() - _t_total:.1f}", flush=True)
            return
        gpu_sidecar = _start_nvidia_smi_sidecar(gpu_profile_csv) if args.branch != "cpu" else None
        try:
            with _timed("run_pipeline", store=phase_seconds):
                if branch_cfg is None:
                    result = run_fn(seed=args.seed)
                elif args.branch == "cpu":
                    with _dynamic_cpu_core_pool(pos):
                        result = run_fn(seed=args.seed, config=branch_cfg)
                else:
                    result = run_fn(seed=args.seed, config=branch_cfg)
        finally:
            _stop_nvidia_smi_sidecar(gpu_sidecar)

    if result is None:
        raise RuntimeError(
            f"Pipeline for {pos} returned None — cannot extract metrics. "
            "Refusing to upload incomplete artifacts."
        )

    # Copy model artifacts to output dir FIRST so a later metrics write cannot
    # be clobbered by a same-named file under src_model_dir.
    src_model_dir = os.path.join(pos.lower(), "outputs", "models")
    if os.path.isdir(src_model_dir):
        print(f"Copying model artifacts from {src_model_dir} to {model_dir}")
        with _timed("copy_artifacts", store=phase_seconds):
            _replace_model_dir_contents(src_model_dir, model_dir)
    else:
        raise RuntimeError(f"No model directory found at {src_model_dir}; refusing upload.")

    # Save benchmark metrics as JSON (after artifacts so it can't be overwritten).
    # upload_artifacts() requires benchmark_metrics.json, so this must come
    # before the upload.
    metrics = _extract_metrics(pos, result)
    if args.branch in SPLIT_BRANCHES:
        metrics["split_branch"] = args.branch
        metrics["split_run_id"] = args.split_run_id
        metrics["seed"] = args.seed
    # Fold the inner pipeline breakdown (ridge_tune, nn_train, attn_nn_train,
    # lgbm_train, etc.) into the outer phase dict under a ``pipeline.`` prefix
    # so persisted metrics distinguish "data sync + S3 + outer wrap" from the
    # phases that actually dominate the GPU box.
    inner_phases = result.get("phase_seconds", {}) if isinstance(result, dict) else {}
    for phase, secs in inner_phases.items():
        phase_seconds[f"pipeline.{phase}"] = secs
    # Record end-to-end elapsed and the per-phase breakdown so the run
    # written under benchmark_history/ by src/batch/benchmark.py --download-only
    # carries timing. elapsed_sec captures everything from seeding through
    # the S3 upload, matching local benchmark.py's wrap around run_one().
    metrics["elapsed_sec"] = round(time.monotonic() - _t_total, 1)
    metrics["phase_seconds"] = phase_seconds
    # Stamp the GPU the job ran on + whether CUDA-graph capture was active so
    # benchmark.py derives the History-tab hardware label at runtime instead of
    # a hardcoded workflow string (auto-tracks a T4->L4 CE migration).
    metrics.update(_hardware_metadata())
    metrics_path = os.path.join(model_dir, "benchmark_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved benchmark metrics to {metrics_path}")

    # Copy the sidecar CSV into model_dir after metrics write and before upload
    # so the tarball contains both artifacts and benchmark_metrics.json.
    if os.path.exists(gpu_profile_csv):
        shutil.copy2(gpu_profile_csv, os.path.join(model_dir, f"gpu_profile_{pos}.csv"))

    if args.branch in SPLIT_BRANCHES:
        _write_split_branch_metadata(
            position=pos,
            branch=args.branch,
            split_run_id=args.split_run_id,
            seed=args.seed,
            model_dir=model_dir,
        )
        with _timed("upload_split_branch_artifacts", store=phase_seconds):
            _upload_split_branch_artifacts(
                s3_bucket,
                pos,
                args.branch,
                args.split_run_id,
                model_dir,
                args.seed,
            )
        print(f"[timing] total={time.monotonic() - _t_total:.1f}", flush=True)
        return

    # Upload artifacts to S3 (raises if model_dir is empty or metrics missing)
    with _timed("upload_artifacts", store=phase_seconds):
        upload_artifacts(s3_bucket, pos, model_dir)

    # upload_artifacts ran after the metrics write, so its duration lives in
    # phase_seconds but is not reflected in metrics["phase_seconds"] for this
    # run. That's fine — the metrics file is already in the tarball.
    print(f"[timing] total={time.monotonic() - _t_total:.1f}", flush=True)


if __name__ == "__main__":
    main()
