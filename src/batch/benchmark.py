"""Launch AWS Batch training for all positions and collect benchmark metrics.

Runs the same pipelines as benchmark.py but on AWS Batch GPU instances
(g6.xlarge Spot).  Downloads benchmark_metrics.json from each job's model
artifacts and prints a unified comparison table.

Usage:
    python src/batch/benchmark.py                          # all 6 positions
    python src/batch/benchmark.py --positions RB WR QB     # subset
    python src/batch/benchmark.py --note "attention + LGBM on GPU"
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.batch.launch import (
    ALL_POSITIONS,
    AWS_REGION,
    S3_BUCKET,
    WAIT_TIMEOUT_SECONDS,
    submit_job,
    upload_data,
    wait_for_jobs,
)
from src.shared.benchmark_utils import (
    append_to_history,
    get_git_hash,
    print_comparison_table,
    summarize_pipeline_result,
    utc_now_iso,
)
from src.shared.model_sync import load_manifest

RESULTS_FILE = "benchmark_results.json"
HISTORY_DIR = "benchmark_history"

# Same default as src/batch/train.py — the producer side writes manifest +
# history keys under ``${S3_PREFIX}/${POS}/...``; env override stays available
# for any future bucket-layout migration.
S3_PREFIX = os.environ.get("FF_MODEL_S3_PREFIX", "models")

# Repo root, so record_benchmark_run() resolves HISTORY_DIR / RESULTS_FILE
# independent of cwd. main() chdirs here, but src/batch/launch.py's auto-append
# calls record_benchmark_run() without chdir-ing.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def download_metrics(positions):
    """Download benchmark_metrics.json from each position's model artifacts.

    Resolves the per-position artifact via ``models/{POS}/manifest.json`` rather
    than the legacy ``models/{POS}/model.tar.gz`` mirror. Two parallel
    train-batch runs writing the same position's legacy key were last-write-
    wins; the manifest's ``current`` entry is an atomic single-PUT promotion
    paired with a versioned ``models/{POS}/history/{ts}-{sha7}/model.tar.gz``
    key, so each consumer reads exactly the artifact the producer's manifest
    write committed to.

    Falls through ``current`` -> ``stable`` -> ``previous`` (the same chain
    src/shared/model_sync.py uses on the serving side). Manifest-absent and
    "all entries failed" both surface a per-position WARNING and skip that
    position in the aggregated output rather than raising — one bad position
    shouldn't kill a six-position benchmark roll-up. The crucial thing is
    that we no longer silently fall back to the legacy ``model.tar.gz`` key,
    which was the source of the cross-run pollution this layer fixes.
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)

    def _fetch_one(pos):
        manifest = load_manifest(s3, S3_BUCKET, S3_PREFIX, pos)
        if manifest is None:
            # Treated as soft error here (returns None metrics, lets the rest
            # of the aggregation proceed) rather than raising — the caller
            # already prints a per-position WARNING when metrics are missing,
            # and one stale position shouldn't kill a six-position aggregation.
            print(
                f"[{pos}] WARNING: no manifest at s3://{S3_BUCKET}/{S3_PREFIX}/{pos}/manifest.json"
            )
            return pos, None

        # Resolve in priority order; serving uses the same chain in
        # src/shared/model_sync.py::_sync_one.
        tried = []
        for label in ("current", "stable", "previous"):
            entry = manifest.get(label)
            if not entry or not entry.get("key"):
                continue
            s3_key = entry["key"]
            tried.append((label, s3_key))
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
                print(f"[{pos}] Downloading s3://{S3_BUCKET}/{s3_key} (source={label}) ...")
                try:
                    s3.download_file(S3_BUCKET, s3_key, tmp.name)
                except Exception as e:
                    print(f"[{pos}] {label} download FAILED: {e!r} — falling through")
                    continue
                try:
                    with tarfile.open(tmp.name, "r:gz") as tar:
                        member = tar.getmember("benchmark_metrics.json")
                        f = tar.extractfile(member)
                        print(f"[{pos}] Metrics loaded (source={label})")
                        return pos, json.loads(f.read())
                except (KeyError, tarfile.TarError) as e:
                    print(f"[{pos}] {label} tarball unreadable: {e!r} — falling through")
                    continue
        print(f"[{pos}] WARNING: all manifest entries failed to yield metrics. tried={tried!r}")
        return pos, None

    all_metrics = {}
    with ThreadPoolExecutor(max_workers=max(1, len(positions))) as pool:
        for pos, metrics in pool.map(_fetch_one, positions):
            if metrics is not None:
                all_metrics[pos] = metrics
    return all_metrics


def find_git_sha_divergence(all_metrics: dict, expected_sha: str | None) -> list[tuple[str, str]]:
    """Return (pos, recorded_sha_short) pairs whose ``git_sha`` (stamped by
    src/batch/train.py from FF_TRAIN_GIT_SHA) doesn't match ``expected_sha``.

    Empty when ``expected_sha`` is falsy (workflow_dispatch / local) or when
    every position with a recorded SHA matches. Positions without a recorded
    SHA (pre-PR-288 artifacts, or jobs where the env var wasn't forwarded)
    are skipped rather than flagged — silence here just means "no positive
    coherency signal," not "divergence detected."
    """
    if not expected_sha:
        return []
    expected_short = expected_sha[:7]
    return [
        (pos, (metrics.get("git_sha") or "")[:7])
        for pos, metrics in all_metrics.items()
        if metrics.get("git_sha") and (metrics.get("git_sha") or "")[:7] != expected_short
    ]


def record_benchmark_run(
    positions,
    *,
    backend="batch",
    instance_type="g6.xlarge (Spot)",
    note="",
    pr_number=None,
    git_hash=None,
):
    """Aggregate already-trained artifacts into one benchmark_history row.

    Downloads ``benchmark_metrics.json`` for ``positions`` (via each manifest),
    prints the comparison table, writes ``benchmark_history/{run_id}.json``, and
    mirrors it to S3. Returns the written path, or ``None`` if no metrics were
    resolvable.

    Shared by ``main()`` (CLI / CI) and ``src/batch/launch.py``'s standalone
    auto-append so both go through exactly one code path. ``HISTORY_DIR`` /
    ``RESULTS_FILE`` are resolved against the repo root when relative, so the
    function is correct regardless of the caller's cwd.
    """
    print("\nDownloading benchmark metrics...")
    all_metrics = download_metrics(positions)

    if not all_metrics:
        print("No metrics found. Skipping benchmark history append.")
        return None

    # Per-position git_sha coherency check (see find_git_sha_divergence for
    # rationale). Defense-in-depth against the lingering manifest-write race
    # after Layer A's job-def revision pinning.
    expected_sha = ((git_hash or get_git_hash() or "")[:7]) or None
    diverged = find_git_sha_divergence(all_metrics, expected_sha)
    if diverged:
        print(f"\nWARNING: git_sha divergence across positions (expected {expected_sha}):")
        for pos, recorded in diverged:
            print(f"  {pos}: trained image at {recorded}")
        print(
            "  Investigate whether two train-batch.yml runs overlapped on "
            "this run's S3 writes. The model artifacts are still each "
            "internally consistent (Layer A guarantees per-job image pinning), "
            "but the run is heterogeneous and shouldn't be compared as a unit."
        )
    elif expected_sha:
        with_sha = [p for p, m in all_metrics.items() if m.get("git_sha")]
        if with_sha:
            print(f"\ngit_sha coherent at {expected_sha} across {len(with_sha)} positions")

    # Build summaries
    summaries = []
    for pos in positions:
        if pos in all_metrics:
            summaries.append(summarize_pipeline_result(pos, all_metrics[pos]))

    print_comparison_table(
        summaries,
        header="AWS Batch Benchmark Results (MAE / R2)",
        show_time=False,
    )

    results_path = (
        RESULTS_FILE if os.path.isabs(RESULTS_FILE) else os.path.join(_REPO_ROOT, RESULTS_FILE)
    )
    with open(results_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Truncate to 7 chars so a CI-supplied full 40-char SHA matches the
    # short-SHA convention ``get_git_hash()`` already returns.
    git_short = (git_hash or get_git_hash())[:7]
    now = utc_now_iso()
    history_dir = (
        HISTORY_DIR if os.path.isabs(HISTORY_DIR) else os.path.join(_REPO_ROOT, HISTORY_DIR)
    )
    written_path = append_to_history(
        history_dir,
        {
            "run_id": f"{now}_{git_short}",
            "timestamp": now,
            "git_hash": git_short,
            "note": note or f"AWS {backend} benchmark",
            "backend": backend,
            "instance_type": instance_type,
            "positions": [s["position"] for s in summaries],
            "results": summaries,
        },
        pr_number=pr_number,
    )

    # Mirror the new file to S3 so the serving container can pull it at boot
    # via sync_benchmark_history_from_s3 — git auto-commit is preserved for
    # auditability but is not on the inference data path. Env-gated to match
    # the model/data sync pattern in src/shared/model_sync.py.
    _maybe_upload_to_s3(written_path)
    return written_path


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
        "--wait-timeout",
        type=int,
        default=None,
        help=(
            f"Override wait-for-jobs timeout in seconds "
            f"(default: {WAIT_TIMEOUT_SECONDS}, matching src/batch/launch.py)."
        ),
    )
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
        default="g6.xlarge (Spot)",
        help="Instance-type label recorded in benchmark_history/",
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        default=None,
        help=(
            "Pull-request number associated with this run. CI fills this in "
            "from `gh api commits/{sha}/pulls`; embedded as a top-level field "
            "in the JSON so the serving UI can link rows to GitHub."
        ),
    )
    parser.add_argument(
        "--git-hash",
        default=None,
        help=(
            "Override the recorded ``git_hash`` with the SHA the trained "
            "docker image was built from. Defaults to the workspace HEAD via "
            "``get_git_hash()``. CI passes this explicitly because the "
            "workflow workspace can advance past the image-build SHA when "
            "subsequent PRs merge while the Batch job is queued — recording "
            "the workspace HEAD would mislabel runs and break MAE comparison "
            "across consecutive benchmarks."
        ),
    )
    args = parser.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    os.chdir(project_root)

    if not args.download_only:
        # Upload data
        print("Uploading data splits to S3...")
        upload_data(S3_BUCKET)

        # Submit all jobs in parallel (mirrors src/batch/launch.py:main)
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
        # Forward ``--wait-timeout`` only when the operator passed an explicit
        # override — leaving the default-None case as a bare ``wait_for_jobs(
        # job_ids)`` call preserves the historical signature for tests that
        # stub the function with a single-arg lambda.
        if args.wait_timeout is not None:
            results = wait_for_jobs(job_ids, timeout_seconds=args.wait_timeout)
        else:
            results = wait_for_jobs(job_ids)
        total_elapsed = time.time() - total_t0
        print(f"\nAll jobs completed in {total_elapsed:.0f}s wall time")

        failed = [p for p, (status, _) in results.items() if status == "FAILED"]
        if failed:
            print(f"Failed positions: {failed}")

    # Download metrics, build the comparison table, and record the run (writes
    # benchmark_history/{run_id}.json + S3 mirror). Shared with
    # src/batch/launch.py's standalone auto-append via record_benchmark_run.
    record_benchmark_run(
        args.positions,
        backend=args.backend,
        instance_type=args.instance_type,
        note=args.note,
        pr_number=args.pr_number,
        git_hash=args.git_hash,
    )


def _maybe_upload_to_s3(local_path: str) -> None:
    """Mirror one benchmark JSON to S3 under ``{prefix}/benchmark_history/``.

    Note on layout: the key sits under the same ``FF_MODEL_S3_PREFIX`` as
    model tarballs (``s3://{bucket}/models/...``). That keeps producer and
    consumer aligned to a single prefix env var, but it means a lifecycle
    policy on ``models/*`` would silently expire benchmark history. If we
    ever add such a policy, move benchmark uploads to a dedicated top-level
    prefix (e.g. ``benchmark_history/``) and update sync_benchmark_history_
    from_s3 in lockstep. Cheap to migrate while the history is small.
    """
    bucket = os.environ.get("FF_MODEL_S3_BUCKET", "").strip()
    if not bucket:
        return
    prefix = os.environ.get("FF_MODEL_S3_PREFIX", "models").strip("/")
    key = f"{prefix}/benchmark_history/{os.path.basename(local_path)}"
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(local_path, bucket, key)
    print(f"Uploaded benchmark to s3://{bucket}/{key}")


if __name__ == "__main__":
    main()
