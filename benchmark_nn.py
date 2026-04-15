"""Benchmark script: runs RB, QB, WR pipelines and prints a comparison table.

Usage:
    python benchmark_nn.py                          # run all 3 positions
    python benchmark_nn.py RB                       # run one position
    python benchmark_nn.py --note "tuned WR dropout" # annotate the run
"""

import sys, os, time, json, argparse, subprocess, datetime
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

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
            and k != f"{prefix}RIDGE_ALPHAS"}


def append_to_history(run_entry):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            history = json.load(f)
    else:
        history = []
    history.append(run_entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Run appended to {HISTORY_FILE}")

def run_one(position):
    """Run a single position pipeline and return its metrics dict."""
    np.random.seed(42)
    torch.manual_seed(42)

    if position == "RB":
        from RB.run_rb_pipeline import run_rb_pipeline
        return run_rb_pipeline()
    elif position == "QB":
        from QB.run_qb_pipeline import run_qb_pipeline
        return run_qb_pipeline()
    elif position == "WR":
        from WR.run_wr_pipeline import run_wr_pipeline
        return run_wr_pipeline()
    else:
        raise ValueError(f"Unknown position: {position}")


def summarize(position, result):
    """Extract the key metrics we care about."""
    ridge = result["ridge_metrics"]["total"]
    nn = result["nn_metrics"]["total"]
    return {
        "position": position,
        "ridge_mae": round(ridge["mae"], 3),
        "ridge_r2":  round(ridge["r2"], 3),
        "nn_mae":    round(nn["mae"], 3),
        "nn_r2":     round(nn["r2"], 3),
        "nn_wins_mae": nn["mae"] < ridge["mae"],
        # Per-target NN MAE
        "nn_per_target": {
            t: round(result["nn_metrics"][t]["mae"], 3)
            for t in result["nn_metrics"] if t != "total"
        },
        "ridge_per_target": {
            t: round(result["ridge_metrics"][t]["mae"], 3)
            for t in result["ridge_metrics"] if t != "total"
        },
        "ridge_top12": round(result["ridge_ranking"]["season_avg_hit_rate"], 3),
        "nn_top12":    round(result["nn_ranking"]["season_avg_hit_rate"], 3),
    }


def print_table(summaries):
    print("\n" + "=" * 72)
    print(f"{'Pos':<5} {'Ridge MAE':>10} {'NN MAE':>10} {'Delta':>8} {'Ridge R2':>9} {'NN R2':>9} {'NN Wins?':>9}")
    print("-" * 72)
    for s in summaries:
        delta = s["nn_mae"] - s["ridge_mae"]
        marker = "YES" if s["nn_wins_mae"] else "no"
        print(f"{s['position']:<5} {s['ridge_mae']:>10.3f} {s['nn_mae']:>10.3f} {delta:>+8.3f} {s['ridge_r2']:>9.3f} {s['nn_r2']:>9.3f} {marker:>9}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark NN pipelines")
    parser.add_argument("positions", nargs="*", default=["RB", "QB", "WR"],
                        help="Positions to benchmark (e.g. RB QB)")
    parser.add_argument("--note", default="", help="Describe what changed in this run")
    args = parser.parse_args()

    positions = args.positions
    summaries = []
    for pos in positions:
        t0 = time.time()
        print(f"\n{'#' * 60}")
        print(f"# BENCHMARKING {pos}")
        print(f"{'#' * 60}")
        result = run_one(pos)
        elapsed = time.time() - t0
        s = summarize(pos, result)
        s["elapsed_sec"] = round(elapsed, 1)
        summaries.append(s)
        print(f"\n  [{pos}] Completed in {elapsed:.1f}s")

    print_table(summaries)

    # Save latest results (backwards compat)
    with open(RESULTS_FILE, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Append to history
    git_hash = get_git_hash()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    append_to_history({
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
