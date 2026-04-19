"""Benchmark script: runs RB, QB, WR pipelines and prints a comparison table.

Usage:
    python benchmark_nn.py                          # run all 3 positions
    python benchmark_nn.py RB                       # run one position
    python benchmark_nn.py --note "tuned WR dropout" # annotate the run
"""

import sys, os, time, json, argparse, datetime
sys.path.insert(0, os.path.dirname(__file__))

from shared.benchmark_utils import (
    append_to_history, get_git_hash, print_comparison_table,
    print_history_comparison, summarize_pipeline_result,
)

RESULTS_FILE = "benchmark_results.json"
HISTORY_FILE = "benchmark_history.json"


def collect_global_config():
    from src.config import (
        TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
        NN_EPOCHS, NN_BATCH_SIZE, NN_PATIENCE, NN_DROPOUT, NN_LR,
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
    mod = importlib.import_module(f"{pos}.{pos.lower()}_config")
    prefix = f"{pos}_"
    return {k[len(prefix):].lower(): v
            for k, v in vars(mod).items()
            if k.startswith(prefix)
            and not k.endswith("FEATURES")
            and k != f"{prefix}RIDGE_ALPHA_GRIDS"}


def run_one(position, cv=False):
    """Run a single position pipeline and return its metrics dict."""
    from shared.registry import get_runner, get_cv_runner
    from shared.utils import seed_everything
    seed_everything(42)
    runner = get_cv_runner(position) if cv else get_runner(position)
    return runner()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark NN pipelines")
    parser.add_argument("positions", nargs="*", default=["QB", "RB", "WR", "TE", "K", "DST"],
                        help="Positions to benchmark (e.g. RB QB)")
    parser.add_argument("--note", default="", help="Describe what changed in this run")
    parser.add_argument("--cv", action="store_true",
                        help="Use expanding-window cross-validation")
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
        summaries.append(s)
        print(f"\n  [{pos}] Completed in {elapsed:.1f}s")

    print_comparison_table(summaries, header="MAE Comparison (test set)", show_time=True)

    # Save latest results (backwards compat)
    with open(RESULTS_FILE, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Append to history
    git_hash = get_git_hash()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    append_to_history(HISTORY_FILE, {
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
    })

    print_history_comparison(HISTORY_FILE, summaries)
