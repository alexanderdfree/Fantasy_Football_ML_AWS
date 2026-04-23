"""AWS Batch training entry point.

Batch runs this as: python batch/train.py --position RB --seed 42

Environment variables set via job definition / container overrides:
  TRAINING_DATA_DIR  = /opt/ml/input/data/training/
  MODEL_OUTPUT_DIR   = /opt/ml/model/
  LOG_EVERY          = 1
  S3_BUCKET          = ff-predictor-training
  S3_DATA_PREFIX     = data
"""

import argparse
import datetime
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time

# Ensure project root is on path (baked into /opt/ml/code/ in the container)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import boto3
import pandas as pd
import torch

from shared.artifact_gc import prune as _gc_prune
from shared.model_sync import (
    build_manifest,
    legacy_model_key,
    load_manifest,
    manifest_key,
    new_history_key,
    write_manifest,
)
from shared.registry import (
    ALL_POSITIONS,
    INFERENCE_REGISTRY,
    accepts_dataframes,
    get_runner,
    is_cpu_only,
)
from shared.utils import seed_everything
from shared.utils import timed as _timed


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


def _assert_gpu(position: str):
    """Log GPU status and fail fast if REQUIRE_GPU=1 and CUDA is unavailable.

    This catches the silent-CPU-on-GPU-billed-instance failure mode where
    the Batch job definition forgets `resourceRequirements: [{type: GPU, ...}]`.

    For CPU-only positions (K, DST) we don't enforce REQUIRE_GPU even if the
    env var is set — those pipelines never touch CUDA.
    """
    available = torch.cuda.is_available()
    print(f"[gpu] torch.cuda.is_available() = {available}")
    print(f"[gpu] torch.version.cuda        = {torch.version.cuda}")
    print(f"[gpu] torch.__version__         = {torch.__version__}")
    if available:
        print(f"[gpu] device count              = {torch.cuda.device_count()}")
        print(f"[gpu] device 0 name             = {torch.cuda.get_device_name(0)}")
    if is_cpu_only(position):
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

    Needed by shared/weather_features._load_schedules() (all positions during
    feature engineering) and by K/DST's self-contained loaders (k_data,
    dst_data). CACHE_DIR="data/raw" in src/config.py resolves relative to
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


def upload_artifacts(s3_bucket, position, model_dir):
    """Tar, upload to a versioned history key, validate, atomically promote,
    then mirror to the legacy key for pre-manifest consumers.

    Order (each step raises on failure):
      1. Structural check of ``model_dir`` (fast-fail before S3 round-trips).
      2. Build tarball, hash it, pick timestamped + sha7 history key.
      3. Upload to ``history/{ts}-{sha7}/model.tar.gz``.
      4. Re-download + validate (reopenable, expected files present).
      5. Read old ``manifest.json`` (None on first run).
      6. Write new ``manifest.json`` with ``current=new, previous=old.current``
         — **this write is the atomic promotion**. Any earlier failure leaves
         the old manifest in place and the site keeps serving the previous
         good artifact.
      7. Overwrite legacy ``model.tar.gz`` mirror (pre-manifest compat).
      8. Best-effort retention prune (failure is non-fatal).
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
    # Mirrors shared.model_sync's consumer-side env read so producer/consumer
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

        old_manifest = load_manifest(s3, s3_bucket, s3_prefix, position)
        new_manifest = build_manifest(
            new_key=new_key,
            sha7=sha7,
            bytes_=tar_bytes,
            uploaded_at=ts,
            old_manifest=old_manifest,
        )
        write_manifest(s3, s3_bucket, s3_prefix, position, new_manifest)
        print(f"Promoted s3://{s3_bucket}/{manifest_key(s3_prefix, position)}")

        # Legacy mirror goes LAST — pre-manifest consumers see the same bytes
        # as the freshly-promoted current. Written last so a failure between
        # steps 6 and 7 leaves the site on the (working) new current with a
        # stale legacy; the other direction would be worse.
        legacy_k = legacy_model_key(s3_prefix, position)
        s3.upload_file(tmp_path, s3_bucket, legacy_k)
        print(f"Updated legacy mirror s3://{s3_bucket}/{legacy_k}")

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


def _extract_metrics(position, result):
    """Extract JSON-serializable benchmark metrics from pipeline result."""
    metrics = {"position": position}

    for model_key in ["ridge", "nn", "attn_nn", "lgbm"]:
        m_key = f"{model_key}_metrics"
        r_key = f"{model_key}_ranking"
        if m_key not in result:
            continue
        m = result[m_key]
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


def _run_rb_gate_ablation(train_df, val_df, test_df, seed: int) -> None:
    """Three-way ablation on RB TD heads: Huber+gate vs Poisson+no-gate vs
    Poisson+gate. Prints a decision table; skips S3 upload.

    Mirrors ``scripts/ablate_rb_gate.py`` but runs inside the container so
    the downloaded splits are reused across variants (cuts ~30s × 3 of repeated
    data-prep). The shipping RB config (variant B) ships Poisson+no-gate;
    this function answers whether that was the right call by comparing
    fantasy-point MAE and per-target TD MAE across the three variants.
    """
    import copy

    from RB.run_rb_pipeline import RB_CONFIG, run_rb_pipeline

    def _variant_a(cfg: dict) -> dict:
        """Pre-PR-2 baseline: Huber + gate on both TD heads."""
        cfg = copy.deepcopy(cfg)
        cfg["head_losses"] = {
            "rushing_tds": "huber",
            "receiving_tds": "huber",
            "rushing_yards": "huber",
            "receiving_yards": "huber",
            "receptions": "huber",
            "fumbles_lost": "huber",
        }
        cfg["gated_targets"] = ["rushing_tds", "receiving_tds"]
        cfg["loss_weights"] = {
            "rushing_tds": 4.0,
            "receiving_tds": 4.0,
            "rushing_yards": 0.133,
            "receiving_yards": 0.133,
            "receptions": 1.0,
            "fumbles_lost": 4.0,
        }
        cfg["huber_deltas"] = {
            "rushing_tds": 0.5,
            "receiving_tds": 0.5,
            "rushing_yards": 15.0,
            "receiving_yards": 15.0,
            "receptions": 2.0,
            "fumbles_lost": 0.5,
        }
        cfg["nn_head_hidden_overrides"] = {"rushing_tds": 64, "receiving_tds": 64}
        return cfg

    def _variant_b(cfg: dict) -> dict:
        """Pre-TD-gate-restoration (PR #96 shipping): Poisson NLL on TDs with
        no gate on them. Explicitly forces ``gated_targets=["receptions"]`` so
        this variant stays meaningful even as the live RB_CONFIG's list evolves."""
        cfg = copy.deepcopy(cfg)
        cfg["gated_targets"] = ["receptions"]
        return cfg

    def _variant_c(cfg: dict) -> dict:
        """Current shipping: Poisson NLL on TDs + BCE gate on each TD head on top
        of the reception hurdle."""
        cfg = copy.deepcopy(cfg)
        cfg["gated_targets"] = ["receptions", "rushing_tds", "receiving_tds"]
        return cfg

    variants = [
        ("A", "Huber + gate on TDs (pre-PR-2 baseline)", _variant_a),
        ("B", "Poisson NLL, no TD gate (PR #96 config)", _variant_b),
        ("C", "Poisson NLL + gate on TDs (current shipping)", _variant_c),
    ]

    rows: list[dict] = []
    for name, label, fn in variants:
        print(f"\n{'=' * 72}\nVariant {name}: {label}\n{'=' * 72}", flush=True)
        result = run_rb_pipeline(train_df, val_df, test_df, seed=seed, config=fn(RB_CONFIG))
        attn = result.get("attn_nn_metrics")
        if attn is None:
            raise RuntimeError(f"Variant {name}: attn_nn_metrics missing from pipeline result")
        row = {
            "variant": name,
            "label": label,
            "fp_mae": attn["total"]["mae"],
            "fp_rmse": attn["total"]["rmse"],
            "rushing_tds_mae": attn["rushing_tds"]["mae"],
            "receiving_tds_mae": attn["receiving_tds"]["mae"],
            "receptions_mae": attn["receptions"]["mae"],
            "gate_aucs": {
                t: attn[t].get("gate_auc")
                for t in attn
                if isinstance(attn.get(t), dict) and "gate_auc" in attn[t]
            },
        }
        rows.append(row)

    print(f"\n{'=' * 72}\nRB TD-gate ablation — summary\n{'=' * 72}")
    print(
        f"{'Var':<4}{'FP MAE':>10}{'FP RMSE':>10}{'Rush TD MAE':>14}"
        f"{'Rec TD MAE':>14}{'Rec MAE':>10}"
    )
    print("-" * 62)
    for r in rows:
        print(
            f"{r['variant']:<4}{r['fp_mae']:>10.3f}{r['fp_rmse']:>10.3f}"
            f"{r['rushing_tds_mae']:>14.3f}{r['receiving_tds_mae']:>14.3f}"
            f"{r['receptions_mae']:>10.3f}"
        )
    if any(r["gate_aucs"] for r in rows):
        print("\nGate AUCs (attention NN, gated targets only):")
        for r in rows:
            if r["gate_aucs"]:
                auc_str = ", ".join(
                    f"{t}={auc:.3f}" if auc is not None else f"{t}=n/a"
                    for t, auc in r["gate_aucs"].items()
                )
                print(f"  {r['variant']}: {auc_str}")

    by_var = {r["variant"]: r for r in rows}
    a, b, c = by_var["A"]["fp_mae"], by_var["B"]["fp_mae"], by_var["C"]["fp_mae"]
    margin_a = b - a
    margin_c = b - c
    print(f"\nFP-MAE margin vs B (positive = gate helps): A={margin_a:+.3f}, C={margin_c:+.3f}")
    if max(margin_a, margin_c) >= 0.05:
        print("Decision: keep a gate on TDs — exceeds 0.05 pt/game threshold.")
    else:
        print("Decision: drop gate on TDs (variant B wins) — below 0.05 pt/game threshold.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", required=True, choices=ALL_POSITIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip S3 download/upload and the real pipeline. Writes stub "
        "artifacts so main() can be smoke-tested end-to-end without "
        "AWS credentials or training data.",
    )
    parser.add_argument(
        "--ablation",
        choices=["rb-gate"],
        default=None,
        help="Run a named ablation instead of a standard training run. "
        "'rb-gate' requires --position RB; runs the three-way TD-gate "
        "ablation and prints the decision table. Skips S3 upload.",
    )
    args = parser.parse_args()

    pos = args.position
    if args.ablation == "rb-gate" and pos != "RB":
        parser.error("--ablation rb-gate requires --position RB")

    # Print build fingerprint so stale container images are immediately obvious.
    _fingerprint_file = os.path.join(os.path.dirname(__file__), "train.py")
    with open(_fingerprint_file, "rb") as _f:
        _hash = hashlib.sha256(_f.read()).hexdigest()[:12]
    print(f"[batch/train.py] build fingerprint: {_hash}")

    _t_total = time.monotonic()
    phase_seconds: dict[str, float] = {}
    # Skip the GPU assertion in dry-run — local/CI smoke tests rarely have CUDA.
    if args.dry_run:
        print(f"[dry-run] skipping _assert_gpu for {pos}")
    else:
        with _timed("assert_gpu", store=phase_seconds):
            _assert_gpu(pos)
    seed_everything(args.seed)

    s3_bucket = os.environ.get("S3_BUCKET", "ff-predictor-training")
    s3_prefix = os.environ.get("S3_DATA_PREFIX", "data")
    data_dir = os.environ.get("TRAINING_DATA_DIR", "/opt/ml/input/data/training")
    model_dir = os.environ.get("MODEL_OUTPUT_DIR", "/opt/ml/model")
    # LOG_EVERY is consumed directly by shared.pipeline._resolve_nn_log_every()
    # so we don't need to inject it into cfg from here. Historically we
    # monkey-patched run_pipeline, but that only worked if callers used
    # `import shared.pipeline as pipeline_mod; pipeline_mod.run_pipeline(...)`.
    # All position runners use `from shared.pipeline import run_pipeline`, so
    # the patch was dead code. Env-var resolution sidesteps the issue.

    os.makedirs(model_dir, exist_ok=True)

    if args.dry_run:
        # Stub out S3 and the pipeline — we still exercise arg parsing,
        # seed setup, model-dir setup, metrics serialization, and the
        # skip-S3 code path.
        _dry_run_artifacts(pos, model_dir, args.seed, _t_total, phase_seconds)
        print(f"[dry-run] Completed for {pos}; skipping S3 upload.")
        return

    run_fn = get_runner(pos)

    # data/raw/*.parquet is needed for weather features (all positions) and
    # for K/DST's self-contained data loaders. Sync before branching.
    with _timed("sync_raw_data", store=phase_seconds):
        sync_raw_data(s3_bucket)

    if accepts_dataframes(pos):
        # Download train/val/test splits from S3 into the container
        with _timed("download_data", store=phase_seconds):
            download_data(s3_bucket, s3_prefix, data_dir)
        with _timed("read_parquets", store=phase_seconds):
            train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
            val_df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
            test_df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
            print(f"Loaded data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        if args.ablation == "rb-gate":
            # Ablation runs the pipeline 3x with config overrides and prints
            # a decision table. No S3 artifact upload — this is a diagnostic
            # run, not a shipping build.
            with _timed("run_ablation", store=phase_seconds):
                _run_rb_gate_ablation(train_df, val_df, test_df, seed=args.seed)
            print(f"[timing] total={time.monotonic() - _t_total:.1f}", flush=True)
            return
        with _timed("run_pipeline", store=phase_seconds):
            result = run_fn(train_df, val_df, test_df, seed=args.seed)
    else:
        # K/DST: self-contained data loading
        with _timed("run_pipeline", store=phase_seconds):
            result = run_fn(seed=args.seed)

    # Copy model artifacts to output dir FIRST so a later metrics write cannot
    # be clobbered by a same-named file under src_model_dir.
    src_model_dir = os.path.join(pos, "outputs", "models")
    if os.path.isdir(src_model_dir):
        print(f"Copying model artifacts from {src_model_dir} to {model_dir}")
        with _timed("copy_artifacts", store=phase_seconds):
            _replace_model_dir_contents(src_model_dir, model_dir)
    else:
        print(f"WARNING: No model directory found at {src_model_dir}")

    # Save benchmark metrics as JSON (after artifacts so it can't be overwritten).
    # upload_artifacts() requires benchmark_metrics.json, so this must come
    # before the upload.
    if result is None:
        raise RuntimeError(
            f"Pipeline for {pos} returned None — cannot extract metrics. "
            "Refusing to upload incomplete artifacts."
        )
    metrics = _extract_metrics(pos, result)
    # Record end-to-end elapsed and the per-phase breakdown so the row
    # appended to benchmark_history.json by batch/benchmark.py --download-only
    # carries timing. elapsed_sec captures everything from seeding through
    # the S3 upload, matching local benchmark.py's wrap around run_one().
    metrics["elapsed_sec"] = round(time.monotonic() - _t_total, 1)
    metrics["phase_seconds"] = phase_seconds
    metrics_path = os.path.join(model_dir, "benchmark_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved benchmark metrics to {metrics_path}")

    # Upload artifacts to S3 (raises if model_dir is empty or metrics missing)
    with _timed("upload_artifacts", store=phase_seconds):
        upload_artifacts(s3_bucket, pos, model_dir)

    # upload_artifacts ran after the metrics write, so its duration lives in
    # phase_seconds but is not reflected in metrics["phase_seconds"] for this
    # run. That's fine — the metrics file is already in the tarball.
    print(f"[timing] total={time.monotonic() - _t_total:.1f}", flush=True)


if __name__ == "__main__":
    main()
