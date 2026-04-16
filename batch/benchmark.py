"""Launch AWS Batch training for all positions and collect benchmark metrics.

Runs the same pipelines as benchmark_nn.py but on AWS Batch GPU instances
(g4dn.xlarge Spot).  Downloads benchmark_metrics.json from each job's model
artifacts and prints a unified comparison table.

Usage:
    python batch/benchmark.py                          # all 6 positions
    python batch/benchmark.py --positions RB WR QB     # subset
    python batch/benchmark.py --note "attention + LGBM on GPU"
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from batch.launch import (
    S3_BUCKET,
    JOB_QUEUE,
    JOB_DEFINITION,
    ALL_POSITIONS,
    AWS_REGION,
    POLL_INTERVAL_SECONDS,
    TERMINAL_STATES,
    upload_data,
    submit_job,
    wait_for_jobs,
)

RESULTS_FILE = "benchmark_results.json"
HISTORY_FILE = "benchmark_history.json"


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def download_metrics(positions):
    """Download benchmark_metrics.json from each position's model artifacts."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    all_metrics = {}

    for pos in positions:
        s3_key = f"models/{pos}/model.tar.gz"

        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
            print(f"[{pos}] Downloading s3://{S3_BUCKET}/{s3_key} ...")
            try:
                s3.download_file(S3_BUCKET, s3_key, tmp.name)
            except Exception as e:
                print(f"[{pos}] No artifacts found: {e}")
                continue

            with tarfile.open(tmp.name, "r:gz") as tar:
                try:
                    member = tar.getmember("benchmark_metrics.json")
                    f = tar.extractfile(member)
                    metrics = json.loads(f.read())
                    all_metrics[pos] = metrics
                    print(f"[{pos}] Metrics loaded")
                except KeyError:
                    print(f"[{pos}] WARNING: benchmark_metrics.json not found in artifacts")

    return all_metrics


def summarize(metrics):
    """Convert raw metrics dict into a summary row."""
    pos = metrics["position"]
    ridge = metrics["ridge_metrics"]["total"]
    nn = metrics["nn_metrics"]["total"]

    summary = {
        "position": pos,
        "ridge_mae": ridge["mae"],
        "ridge_r2": ridge["r2"],
        "nn_mae": nn["mae"],
        "nn_r2": nn["r2"],
        "nn_wins_mae": nn["mae"] < ridge["mae"],
        "nn_per_target": {
            t: {"mae": round(v["mae"], 3), "r2": round(v["r2"], 3)}
            for t, v in metrics["nn_metrics"].items() if t != "total"
        },
        "ridge_per_target": {
            t: {"mae": round(v["mae"], 3), "r2": round(v["r2"], 3)}
            for t, v in metrics["ridge_metrics"].items() if t != "total"
        },
        "ridge_top12": metrics.get("ridge_ranking", {}).get("season_avg_hit_rate", 0),
        "nn_top12": metrics.get("nn_ranking", {}).get("season_avg_hit_rate", 0),
    }

    if "attn_nn_metrics" in metrics:
        attn = metrics["attn_nn_metrics"]["total"]
        summary["attn_nn_mae"] = attn["mae"]
        summary["attn_nn_r2"] = attn["r2"]
        summary["attn_nn_per_target"] = {
            t: {"mae": round(v["mae"], 3), "r2": round(v["r2"], 3)}
            for t, v in metrics["attn_nn_metrics"].items() if t != "total"
        }
        summary["attn_nn_top12"] = metrics.get("attn_nn_ranking", {}).get(
            "season_avg_hit_rate", 0
        )

    if "lgbm_metrics" in metrics:
        lgbm = metrics["lgbm_metrics"]["total"]
        summary["lgbm_mae"] = lgbm["mae"]
        summary["lgbm_r2"] = lgbm["r2"]
        summary["lgbm_per_target"] = {
            t: {"mae": round(v["mae"], 3), "r2": round(v["r2"], 3)}
            for t, v in metrics["lgbm_metrics"].items() if t != "total"
        }
        summary["lgbm_top12"] = metrics.get("lgbm_ranking", {}).get(
            "season_avg_hit_rate", 0
        )

    return summary


def print_table(summaries):
    """Print benchmark comparison table."""
    has_attn = any("attn_nn_mae" in s for s in summaries)
    has_lgbm = any("lgbm_mae" in s for s in summaries)

    cols = ["Pos", "Ridge", "NN"]
    if has_attn:
        cols.append("Attn NN")
    if has_lgbm:
        cols.append("LightGBM")
    cols.append("Best")

    width = 12 * len(cols) + 5
    print(f"\n{'=' * width}")
    print("AWS Batch Benchmark Results (MAE / R2)")
    print(f"{'=' * width}")

    print(f"\n{'':>5}", end="")
    for c in cols:
        print(f"{c:>12}", end="")
    print()
    print("-" * width)

    for s in summaries:
        models = {"Ridge": s["ridge_mae"], "NN": s["nn_mae"]}
        if "attn_nn_mae" in s:
            models["Attn NN"] = s["attn_nn_mae"]
        if "lgbm_mae" in s:
            models["LightGBM"] = s["lgbm_mae"]
        best_name = min(models, key=models.get)

        print(f"{s['position']:>5}", end="")
        print(f"{s['ridge_mae']:>12.3f}", end="")
        print(f"{s['nn_mae']:>12.3f}", end="")
        if has_attn:
            print(f"{s.get('attn_nn_mae', float('nan')):>12.3f}", end="")
        if has_lgbm:
            print(f"{s.get('lgbm_mae', float('nan')):>12.3f}", end="")
        print(f"{best_name:>12}", end="")
        print()

    print(f"{'=' * width}")

    # R2
    print(f"\n{'R-squared':>5}", end="")
    for c in cols[:-1]:
        print(f"{c:>12}", end="")
    print()
    print("-" * width)
    for s in summaries:
        models = {"Ridge": s["ridge_r2"], "NN": s["nn_r2"]}
        if "attn_nn_r2" in s:
            models["Attn NN"] = s["attn_nn_r2"]
        if "lgbm_r2" in s:
            models["LightGBM"] = s["lgbm_r2"]
        best_name = max(models, key=models.get)
        print(f"{s['position']:>5}", end="")
        print(f"{s['ridge_r2']:>12.3f}", end="")
        print(f"{s['nn_r2']:>12.3f}", end="")
        if has_attn:
            print(f"{s.get('attn_nn_r2', float('nan')):>12.3f}", end="")
        if has_lgbm:
            print(f"{s.get('lgbm_r2', float('nan')):>12.3f}", end="")
        print(f"{best_name:>12}", end="")
        print()
    print(f"{'=' * width}")

    # Top-12 hit rate
    print(f"\n{'Top-12 Hit Rate':>5}", end="")
    for c in cols[:-1]:
        print(f"{c:>12}", end="")
    print()
    print("-" * width)
    for s in summaries:
        models = {"Ridge": s["ridge_top12"], "NN": s["nn_top12"]}
        if "attn_nn_top12" in s:
            models["Attn NN"] = s["attn_nn_top12"]
        if "lgbm_top12" in s:
            models["LightGBM"] = s["lgbm_top12"]
        best_name = max(models, key=models.get)
        print(f"{s['position']:>5}", end="")
        print(f"{s['ridge_top12']:>12.3f}", end="")
        print(f"{s['nn_top12']:>12.3f}", end="")
        if has_attn:
            print(f"{s.get('attn_nn_top12', 0):>12.3f}", end="")
        if has_lgbm:
            print(f"{s.get('lgbm_top12', 0):>12.3f}", end="")
        print(f"{best_name:>12}", end="")
        print()
    print(f"{'=' * width}")

    # -- Per-target breakdown (all models) --
    tgt_w, col_w = 20, 9
    pt_hdr = f"  {'Target':<{tgt_w}} {'Ridge':>{col_w}} {'NN':>{col_w}}"
    if has_attn:
        pt_hdr += f" {'Attn NN':>{col_w}}"
    if has_lgbm:
        pt_hdr += f" {'LightGBM':>{col_w}}"
    pt_hdr += f" {'Best':>{col_w}}"

    for metric_key, label, higher_better in [("mae", "Per-Target MAE", False),
                                              ("r2", "Per-Target R\u00b2", True)]:
        print(f"\n{label}")
        print("=" * len(pt_hdr))
        for s in summaries:
            print(f"\n  {s['position']}")
            print(pt_hdr)
            print("  " + "-" * (len(pt_hdr) - 2))
            targets = list(s.get("nn_per_target",
                                 s.get("ridge_per_target", {})).keys())
            for t in targets:
                models = {}
                for mname, key in [("Ridge", "ridge_per_target"),
                                    ("NN", "nn_per_target"),
                                    ("Attn NN", "attn_nn_per_target"),
                                    ("LightGBM", "lgbm_per_target")]:
                    if key in s and t in s[key]:
                        models[mname] = s[key][t][metric_key]
                if not models:
                    continue
                best = (max if higher_better else min)(models, key=models.get)
                line = f"  {t:<{tgt_w}}"
                line += f" {models.get('Ridge', float('nan')):>{col_w}.3f}"
                line += f" {models.get('NN', float('nan')):>{col_w}.3f}"
                if has_attn:
                    line += f" {models.get('Attn NN', float('nan')):>{col_w}.3f}"
                if has_lgbm:
                    line += f" {models.get('LightGBM', float('nan')):>{col_w}.3f}"
                line += f" {best:>{col_w}}"
                print(line)
        print("=" * len(pt_hdr))


def append_to_history(run_entry):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            history = json.load(f)
    else:
        history = []
    history.append(run_entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nRun appended to {HISTORY_FILE}")


def main():
    parser = argparse.ArgumentParser(description="AWS Batch benchmark")
    parser.add_argument(
        "--positions", nargs="+", default=ALL_POSITIONS,
        choices=ALL_POSITIONS, help="Positions to benchmark",
    )
    parser.add_argument("--note", default="", help="Describe what changed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--download-only", action="store_true",
        help="Skip launching jobs; download metrics from latest artifacts",
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
            futures = {
                pool.submit(submit_job, pos, args.seed): pos
                for pos in args.positions
            }
            for future in as_completed(futures):
                pos = futures[future]
                try:
                    pos, job_id = future.result()
                    job_ids[pos] = job_id
                except Exception as e:
                    print(f"[{pos}] FAILED to submit: {e}")

        # Wait for completion
        results = wait_for_jobs(job_ids)
        total_elapsed = time.time() - total_t0
        print(f"\nAll jobs completed in {total_elapsed:.0f}s wall time")

        failed = [p for p, s in results.items() if s == "FAILED"]
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
            summaries.append(summarize(all_metrics[pos]))

    print_table(summaries)

    with open(RESULTS_FILE, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    git_hash = get_git_hash()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    append_to_history({
        "run_id": f"{now}_{git_hash}",
        "timestamp": now,
        "git_hash": git_hash,
        "note": args.note or "AWS Batch benchmark",
        "backend": "batch",
        "instance_type": "g4dn.xlarge (Spot)",
        "positions": [s["position"] for s in summaries],
        "results": summaries,
    })


if __name__ == "__main__":
    main()
