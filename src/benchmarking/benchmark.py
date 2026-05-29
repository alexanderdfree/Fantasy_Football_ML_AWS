"""Benchmark script: runs the QB, RB, WR, TE, K, DST pipelines and prints a comparison table.

Usage:
    python benchmark.py                          # run all 6 positions
    python benchmark.py RB                       # run one position
    python benchmark.py --note "tuned WR dropout" # annotate the run
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.shared.benchmark_utils import (
    append_to_history,
    get_git_hash,
    print_comparison_table,
    print_history_comparison,
    summarize_pipeline_result,
    utc_now_iso,
)

RESULTS_FILE = "benchmark_results.json"
HISTORY_DIR = "benchmark_history"


def collect_global_config():
    from src.config import (
        NN_BATCH_SIZE,
        NN_DROPOUT,
        NN_EPOCHS,
        NN_LR,
        NN_PATIENCE,
        TEST_SEASONS,
        TRAIN_SEASONS,
        VAL_SEASONS,
    )

    return {
        "train_seasons": TRAIN_SEASONS,
        "val_seasons": VAL_SEASONS,
        "test_seasons": TEST_SEASONS,
        "nn_epochs": NN_EPOCHS,
        "nn_batch_size": NN_BATCH_SIZE,
        "nn_patience": NN_PATIENCE,
        "nn_dropout": NN_DROPOUT,
        "nn_lr": NN_LR,
    }


def collect_pos_config(pos):
    import importlib

    mod = importlib.import_module(f"src.{pos.lower()}.config")
    prefix = f"{pos}_"
    return {
        k[len(prefix) :].lower(): v
        for k, v in vars(mod).items()
        if k.startswith(prefix) and not k.endswith("FEATURES") and k != f"{prefix}RIDGE_ALPHA_GRIDS"
    }


def run_one(position, cv=False):
    """Run a single position pipeline and return its metrics dict."""
    from src.shared.registry import get_cv_runner, get_runner
    from src.shared.utils import seed_everything

    seed_everything(42)
    runner = get_cv_runner(position) if cv else get_runner(position)
    return runner()


def _maybe_upload_to_s3(local_path: str) -> None:
    """Mirror one local benchmark JSON to S3 so the run reaches the website's History tab.

    The serving container downloads ``s3://{bucket}/{prefix}/benchmark_history/*.json`` at
    boot (``src/shared/model_sync.py::sync_benchmark_history_from_s3``) and serves it via
    ``/api/benchmark_history``; uploading here is what makes a *local* run eventually appear
    on the site. Env-gated on ``FF_MODEL_S3_BUCKET`` (no-op for pure-local dev without a
    bucket configured) and writes the same ``{prefix}/benchmark_history/{basename}`` key the
    cloud path uses, so producer and consumer stay aligned.

    Mirror of ``src/batch/benchmark.py::_maybe_upload_to_s3`` — deliberately kept here rather
    than lifted to the otherwise-natural ``src/shared/benchmark_utils.py``, because any edit
    under ``src/shared/`` fires the 6-position GPU retrain in
    ``src/scripts/scope_positions.py`` and this is a serving/tooling change that touches no
    model artifact. Two intentional differences from the batch copy:

    1. **Best-effort:** a network/credential failure warns and returns rather than crashing
       the run — the local ``benchmark_history/{run_id}.json`` is already durably written by
       ``append_to_history`` (tmp + ``os.replace``), so the result is never lost. The batch
       copy lets the exception propagate because CI wants a hard failure signal; a dev's
       local benchmark must not die just because S3 is unreachable.
    2. **Lazy ``import boto3``** inside the function, so the no-bucket path has no boto3
       dependency at all.

    Do not "fix" #1 to match the batch copy.
    """
    bucket = os.environ.get("FF_MODEL_S3_BUCKET", "").strip()
    if not bucket:
        print(
            "FF_MODEL_S3_BUCKET unset — skipping cloud sync; this run won't appear on the "
            "website (set FF_MODEL_S3_BUCKET + AWS creds to enable, or pass --no-sync to silence)."
        )
        return
    prefix = os.environ.get("FF_MODEL_S3_PREFIX", "models").strip("/")
    key = f"{prefix}/benchmark_history/{os.path.basename(local_path)}"
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    try:
        import boto3

        s3 = boto3.client("s3", region_name=region)
        s3.upload_file(local_path, bucket, key)
        print(f"Uploaded benchmark to s3://{bucket}/{key}")
    except Exception as exc:  # noqa: BLE001 — network/credential boundary, see CLAUDE.md
        print(f"WARNING: benchmark S3 sync failed ({exc}); local JSON kept at {local_path}")


def _significance_block(position, result):
    """Compact within-season bootstrap CI on the best model's MAE gaps, for history fold-in.

    Schema-safe: the website History tab (``src/serving/app.py::_benchmark_row``) and
    ``print_history_comparison`` read only known keys, so this extra ``significance`` key is
    ignored by old readers. Returns None if the result lacks per-row test predictions.
    """
    test_df = result.get("test_df")
    if test_df is None:
        return None
    from src.analysis.significance import (
        compact_significance,
        paired_bootstrap,
        pred_columns_from_test_df,
    )

    pred_cols = pred_columns_from_test_df(test_df)
    if "Ridge" not in pred_cols:
        return None
    boot = paired_bootstrap(test_df, pred_cols, n_boot=2000)
    return compact_significance(boot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark NN pipelines")
    parser.add_argument(
        "positions",
        nargs="*",
        default=["QB", "RB", "WR", "TE", "K", "DST"],
        help="Positions to benchmark (e.g. RB QB)",
    )
    parser.add_argument("--note", default="", help="Describe what changed in this run")
    parser.add_argument("--cv", action="store_true", help="Use expanding-window cross-validation")
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip the S3 mirror of this run (local benchmark_history/ is still written). "
        "Use for throwaway/experimental runs you don't want on the website's History tab.",
    )
    parser.add_argument(
        "--significance",
        action="store_true",
        help="Attach a within-season paired-bootstrap CI on the best model's MAE gap vs Ridge "
        "and vs the baseline (src/analysis/significance.py). Single-split only; ignored for --cv.",
    )
    args = parser.parse_args()

    positions = args.positions
    summaries = []
    for pos in positions:
        t0 = time.time()
        mode = "CV" if args.cv else "SINGLE-SPLIT"
        print(f"\n{'#' * 60}")
        print(f"# BENCHMARKING {pos} ({mode})")
        print(f"{'#' * 60}")
        result = run_one(pos, cv=args.cv)
        elapsed = time.time() - t0
        s = summarize_pipeline_result(pos, result)
        s["elapsed_sec"] = round(elapsed, 1)
        if args.significance and not args.cv:
            sig_block = _significance_block(pos, result)
            if sig_block is not None:
                s["significance"] = sig_block
        summaries.append(s)
        print(f"\n  [{pos}] Completed in {elapsed:.1f}s")

    print_comparison_table(summaries, header="MAE Comparison (test set)", show_time=True)

    # Save latest results (backwards compat)
    with open(RESULTS_FILE, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Append to history
    git_hash = get_git_hash()
    now = utc_now_iso()
    written_path = append_to_history(
        HISTORY_DIR,
        {
            "run_id": f"{now}_{git_hash}",
            "timestamp": now,
            "git_hash": git_hash,
            "note": args.note,
            "positions": positions,
            "config": {
                "global": collect_global_config(),
                **{p.lower(): collect_pos_config(p) for p in positions},
            },
            "results": summaries,
        },
    )

    if not args.no_sync:
        _maybe_upload_to_s3(written_path)

    print_history_comparison(HISTORY_DIR, summaries, exclude_path=written_path)
