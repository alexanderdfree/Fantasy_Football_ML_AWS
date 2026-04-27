"""Launch AWS Batch training for all positions and collect benchmark metrics.

Runs the same pipelines as benchmark.py but on AWS Batch GPU instances
(g4dn.xlarge Spot).  Downloads benchmark_metrics.json from each job's model
artifacts and prints a unified comparison table.

Usage:
    python batch/benchmark.py                          # all 6 positions
    python batch/benchmark.py --positions RB WR QB     # subset
    python batch/benchmark.py --note "attention + LGBM on GPU"
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from batch.launch import (
    ALL_POSITIONS,
    AWS_REGION,
    S3_BUCKET,
    submit_job,
    upload_data,
    wait_for_jobs,
)
from shared.benchmark_utils import (
    append_to_history,
    get_git_hash,
    print_comparison_table,
    summarize_pipeline_result,
    utc_now_iso,
)

RESULTS_FILE = "benchmark_results.json"
HISTORY_DIR = "benchmark_history"


def download_metrics(positions):
    """Download benchmark_metrics.json from each position's model artifacts."""
    s3 = boto3.client("s3", region_name=AWS_REGION)

    def _fetch_one(pos):
        s3_key = f"models/{pos}/model.tar.gz"
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
            print(f"[{pos}] Downloading s3://{S3_BUCKET}/{s3_key} ...")
            try:
                s3.download_file(S3_BUCKET, s3_key, tmp.name)
            except Exception as e:
                print(f"[{pos}] No artifacts found: {e}")
                return pos, None
            with tarfile.open(tmp.name, "r:gz") as tar:
                try:
                    member = tar.getmember("benchmark_metrics.json")
                    f = tar.extractfile(member)
                    print(f"[{pos}] Metrics loaded")
                    return pos, json.loads(f.read())
                except KeyError:
                    print(f"[{pos}] WARNING: benchmark_metrics.json not found in artifacts")
                    return pos, None

    all_metrics = {}
    with ThreadPoolExecutor(max_workers=max(1, len(positions))) as pool:
        for pos, metrics in pool.map(_fetch_one, positions):
            if metrics is not None:
                all_metrics[pos] = metrics
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="AWS Batch benchmark")
    parser.add_argument(
        "--positions",
        nargs="+",
        default=ALL_POSITIONS,
        choices=ALL_POSITIONS,
        help="Positions to benchmark",
    )
    parser.add_argument("--note", default="", help="Describe what changed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Skip launching jobs; download metrics from latest artifacts",
    )
    parser.add_argument(
        "--backend",
        choices=["batch", "ec2"],
        default="batch",
        help="Backend label recorded in benchmark_history/",
    )
    parser.add_argument(
        "--instance-type",
        default="g4dn.xlarge (Spot)",
        help="Instance-type label recorded in benchmark_history/",
    )
    args = parser.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(project_root)

    if not args.download_only:
        # Upload data
        print("Uploading data splits to S3...")
        upload_data(S3_BUCKET)

        # Submit all jobs in parallel (mirrors batch/launch.py:main)
        total_t0 = time.time()
        print(f"Submitting {len(args.positions)} benchmark jobs: {args.positions}")
        job_ids = {}
        with ThreadPoolExecutor(max_workers=len(args.positions)) as pool:
            futures = {pool.submit(submit_job, pos, args.seed): pos for pos in args.positions}
            for future in as_completed(futures):
                pos = futures[future]
                try:
                    pos, job_id = future.result()
                    job_ids[pos] = job_id
                except Exception as e:
                    print(f"[{pos}] FAILED to submit: {e}")

        # Wait for completion. wait_for_jobs now returns (status, stopped_at_ms).
        results = wait_for_jobs(job_ids)
        total_elapsed = time.time() - total_t0
        print(f"\nAll jobs completed in {total_elapsed:.0f}s wall time")

        failed = [p for p, (status, _) in results.items() if status == "FAILED"]
        if failed:
            print(f"Failed positions: {failed}")

    # Download metrics
    print("\nDownloading benchmark metrics...")
    all_metrics = download_metrics(args.positions)

    if not all_metrics:
        print("No metrics found. Exiting.")
        return

    # Build summaries
    summaries = []
    for pos in args.positions:
        if pos in all_metrics:
            summaries.append(summarize_pipeline_result(pos, all_metrics[pos]))

    print_comparison_table(
        summaries,
        header="AWS Batch Benchmark Results (MAE / R2)",
        show_time=False,
    )

    with open(RESULTS_FILE, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    git_hash = get_git_hash()
    now = utc_now_iso()
    append_to_history(
        HISTORY_DIR,
        {
            "run_id": f"{now}_{git_hash}",
            "timestamp": now,
            "git_hash": git_hash,
            "note": args.note or f"AWS {args.backend} benchmark",
            "backend": args.backend,
            "instance_type": args.instance_type,
            "positions": [s["position"] for s in summaries],
            "results": summaries,
        },
    )


if __name__ == "__main__":
    main()
