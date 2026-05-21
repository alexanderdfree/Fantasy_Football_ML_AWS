"""Summarize the nvidia-smi sidecar CSVs into a Markdown table.

The sidecar in :mod:`src.batch.train` samples GPU stats every 500ms during
``run_pipeline()`` for each position and folds the CSV into the position's
model tarball as ``gpu_profile_{POS}.csv``. This script reads one or many of
those CSVs and prints a comparison table.

Usage:
    # Single local CSV (post-extraction from a tarball)
    python -m src.scripts.analyze_gpu_profile --csv /tmp/gpu_profile_WR.csv

    # Directory of CSVs (one per position)
    python -m src.scripts.analyze_gpu_profile --csv-dir /tmp/profiles/

    # Download each position's latest model.tar.gz from S3 and extract
    python -m src.scripts.analyze_gpu_profile --s3 \
        --positions QB RB WR TE K DST \
        --s3-bucket ff-predictor-training

Caveat: the sampling window covers the *entire* ``run_pipeline`` call —
ridge tuning + regular NN + attention NN + lightgbm. ``util_avg`` is a lower
bound on the attention NN's utilization; ``util_peak`` reflects the busiest
phase (usually the attention NN). For per-phase resolution, run the WR
batch-size sweep (:mod:`src.scripts.batch_size_sweep`) which isolates the
attention NN training and reports per-epoch wall-clock.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _read_profile_csv(path: str) -> dict[str, list[float]]:
    """Parse nvidia-smi --query-gpu=...,utilization.gpu,...,memory.used,... output.

    The query string in :func:`src.batch.train._start_nvidia_smi_sidecar` fixes
    column order: timestamp, utilization.gpu, utilization.memory, memory.used,
    memory.free, temperature.gpu, power.draw. ``--format=csv,nounits`` strips
    unit suffixes (no ``%``, ``MiB``, etc.).
    """
    util_gpu: list[float] = []
    mem_used_gb: list[float] = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 4:
                continue
            try:
                util_gpu.append(float(row[1].strip()))
                mem_used_gb.append(float(row[3].strip()) / 1024.0)
            except ValueError:
                continue
    return {"util_gpu": util_gpu, "mem_used_gb": mem_used_gb}


def _percentile(arr: list[float], p: float) -> float:
    if not arr:
        return 0.0
    s = sorted(arr)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _stats(parsed: dict[str, list[float]]) -> dict[str, float]:
    util = parsed["util_gpu"]
    mem = parsed["mem_used_gb"]
    return {
        "n_samples": len(util),
        "util_avg": sum(util) / len(util) if util else 0.0,
        "util_p50": _percentile(util, 50),
        "util_p95": _percentile(util, 95),
        "util_peak": max(util) if util else 0.0,
        "mem_avg_gb": sum(mem) / len(mem) if mem else 0.0,
        "mem_peak_gb": max(mem) if mem else 0.0,
    }


def _print_markdown_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    if not rows:
        return
    print()
    print(
        "| Position | n_samples | util_avg | util_p50 | util_p95 | util_peak | "
        "mem_avg (GB) | mem_peak (GB) |"
    )
    print(
        "|----------|----------:|---------:|---------:|---------:|----------:|"
        "-------------:|--------------:|"
    )
    for pos, s in rows:
        print(
            f"| {pos:<8} | {int(s['n_samples']):>9d} | "
            f"{s['util_avg']:>7.1f}% | {s['util_p50']:>7.1f}% | "
            f"{s['util_p95']:>7.1f}% | {s['util_peak']:>8.1f}% | "
            f"{s['mem_avg_gb']:>11.2f} | {s['mem_peak_gb']:>12.2f} |"
        )
    print()
    # Decision-rule reminder from the investigation plan.
    overall_avg = sum(s["util_avg"] for _, s in rows) / len(rows)
    if overall_avg < 30.0:
        verdict = (
            f"avg util across positions = {overall_avg:.1f}% (< 30%) — "
            "T4 underutilized; packing follow-up is worth investigating"
        )
    elif overall_avg > 60.0:
        verdict = (
            f"avg util across positions = {overall_avg:.1f}% (> 60%) — "
            "T4 is well utilized; no packing follow-up needed"
        )
    else:
        verdict = (
            f"avg util across positions = {overall_avg:.1f}% (ambiguous: 30%–60%) — "
            "consider a torch.profiler trace before deciding"
        )
    print(f"verdict: {verdict}")


def _extract_csv_from_tarball(tar_bytes: bytes, position: str, dest_path: str) -> bool:
    """Extract gpu_profile_{position}.csv from a model.tar.gz blob to dest_path.

    Returns False if the CSV isn't in the tarball (e.g., trained before the
    sidecar landed) — caller prints a note and skips that position.
    """
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        target = f"gpu_profile_{position}.csv"
        try:
            member = tar.getmember(target)
        except KeyError:
            return False
        src = tar.extractfile(member)
        if src is None:
            return False
        with open(dest_path, "wb") as out:
            out.write(src.read())
    return True


def _download_from_s3(positions: list[str], bucket: str) -> list[tuple[str, str]]:
    """Download each position's current model.tar.gz from S3 and extract its CSV.

    Resolves the artifact key via ``models/{POS}/manifest.json`` (same path
    serving and ``benchmark.py`` use). The legacy ``models/{POS}/model.tar.gz``
    mirror was removed in the parallel-train-batch race fix and is no longer
    updated by the producer, so reading it would yield stale data.

    Returns a list of (position, csv_path) pairs for positions whose tarball
    contained the profile CSV. Positions trained before the sidecar landed are
    skipped with a warning.
    """
    import boto3

    from src.shared.model_sync import load_manifest

    s3 = boto3.client("s3")
    s3_prefix = os.environ.get("FF_MODEL_S3_PREFIX", "models").strip("/")
    tmpdir = tempfile.mkdtemp(prefix="gpu-profile-")
    pairs: list[tuple[str, str]] = []
    for pos in positions:
        manifest = load_manifest(s3, bucket, s3_prefix, pos)
        if manifest is None:
            print(f"  [s3] {pos}: no manifest — skipping")
            continue
        entry = manifest.get("current") or manifest.get("stable") or manifest.get("previous")
        if not entry or not entry.get("key"):
            print(f"  [s3] {pos}: manifest has no resolvable artifact entry — skipping")
            continue
        key = entry["key"]
        print(f"[s3] s3://{bucket}/{key}")
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        dest = os.path.join(tmpdir, f"gpu_profile_{pos}.csv")
        if _extract_csv_from_tarball(body, pos, dest):
            pairs.append((pos, dest))
        else:
            print(f"  no gpu_profile_{pos}.csv in tarball — skipping")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--csv", help="Path to a single gpu_profile_{POS}.csv")
    src_group.add_argument("--csv-dir", help="Directory containing gpu_profile_{POS}.csv files")
    src_group.add_argument(
        "--s3",
        action="store_true",
        help="Download each --positions tarball from --s3-bucket and extract its CSV",
    )
    parser.add_argument("--positions", nargs="+", help="Positions for --s3 mode")
    parser.add_argument("--s3-bucket", default="ff-predictor-training")
    args = parser.parse_args()

    pairs: list[tuple[str, str]] = []
    if args.csv:
        name = os.path.basename(args.csv)
        pos = (
            name.removeprefix("gpu_profile_").removesuffix(".csv")
            if name.startswith("gpu_profile_")
            else "GPU"
        )
        pairs.append((pos, args.csv))
    elif args.csv_dir:
        for fname in sorted(os.listdir(args.csv_dir)):
            if not fname.startswith("gpu_profile_") or not fname.endswith(".csv"):
                continue
            pos = fname.removeprefix("gpu_profile_").removesuffix(".csv")
            pairs.append((pos, os.path.join(args.csv_dir, fname)))
    else:
        if not args.positions:
            parser.error("--s3 requires --positions")
        pairs = _download_from_s3(args.positions, args.s3_bucket)

    if not pairs:
        print("No GPU profile CSVs found.")
        return 1

    rows = [(pos, _stats(_read_profile_csv(path))) for pos, path in pairs]
    _print_markdown_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
