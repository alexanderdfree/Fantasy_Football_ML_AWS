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
            and k != f"{prefix}RIDGE_ALPHA_GRIDS"}


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

def run_one(position, cv=False):
    """Run a single position pipeline and return its metrics dict."""
    np.random.seed(42)
    torch.manual_seed(42)

    if position == "RB":
        from RB.run_rb_pipeline import run_rb_pipeline, run_rb_cv_pipeline
        return run_rb_cv_pipeline() if cv else run_rb_pipeline()
    elif position == "QB":
        from QB.run_qb_pipeline import run_qb_pipeline, run_qb_cv_pipeline
        return run_qb_cv_pipeline() if cv else run_qb_pipeline()
    elif position == "WR":
        from WR.run_wr_pipeline import run_wr_pipeline, run_wr_cv_pipeline
        return run_wr_cv_pipeline() if cv else run_wr_pipeline()
    elif position == "TE":
        from TE.run_te_pipeline import run_te_pipeline
        if cv:
            raise ValueError("TE CV pipeline not implemented yet")
        return run_te_pipeline()
    elif position == "K":
        from K.run_k_pipeline import run_k_pipeline
        if cv:
            raise ValueError("K CV pipeline not implemented yet")
        return run_k_pipeline()
    elif position == "DST":
        from DST.run_dst_pipeline import run_dst_pipeline
        if cv:
            raise ValueError("DST CV pipeline not implemented yet")
        return run_dst_pipeline()
    else:
        raise ValueError(f"Unknown position: {position}")


def summarize(position, result):
    """Extract the key metrics we care about."""
    ridge = result["ridge_metrics"]["total"]
    nn = result["nn_metrics"]["total"]
    summary = {
        "position": position,
        "ridge_mae": round(ridge["mae"], 3),
        "ridge_r2":  round(ridge["r2"], 3),
        "nn_mae":    round(nn["mae"], 3),
        "nn_r2":     round(nn["r2"], 3),
        "nn_wins_mae": nn["mae"] < ridge["mae"],
        # Per-target metrics (MAE + R2)
        "nn_per_target": {
            t: {"mae": round(result["nn_metrics"][t]["mae"], 3),
                "r2": round(result["nn_metrics"][t]["r2"], 3)}
            for t in result["nn_metrics"] if t != "total"
        },
        "ridge_per_target": {
            t: {"mae": round(result["ridge_metrics"][t]["mae"], 3),
                "r2": round(result["ridge_metrics"][t]["r2"], 3)}
            for t in result["ridge_metrics"] if t != "total"
        },
        "ridge_top12": round(result["ridge_ranking"]["season_avg_hit_rate"], 3),
        "nn_top12":    round(result["nn_ranking"]["season_avg_hit_rate"], 3),
    }
    # Attention NN metrics (if trained)
    if "attn_nn_metrics" in result:
        attn = result["attn_nn_metrics"]["total"]
        summary["attn_nn_mae"] = round(attn["mae"], 3)
        summary["attn_nn_r2"] = round(attn["r2"], 3)
        summary["attn_nn_per_target"] = {
            t: {"mae": round(result["attn_nn_metrics"][t]["mae"], 3),
                "r2": round(result["attn_nn_metrics"][t]["r2"], 3)}
            for t in result["attn_nn_metrics"] if t != "total"
        }
        summary["attn_nn_top12"] = round(result["attn_nn_ranking"]["season_avg_hit_rate"], 3)
    # LightGBM metrics (if trained)
    if "lgbm_metrics" in result:
        lgbm = result["lgbm_metrics"]["total"]
        summary["lgbm_mae"] = round(lgbm["mae"], 3)
        summary["lgbm_r2"] = round(lgbm["r2"], 3)
        summary["lgbm_per_target"] = {
            t: {"mae": round(result["lgbm_metrics"][t]["mae"], 3),
                "r2": round(result["lgbm_metrics"][t]["r2"], 3)}
            for t in result["lgbm_metrics"] if t != "total"
        }
        summary["lgbm_top12"] = round(result["lgbm_ranking"]["season_avg_hit_rate"], 3)
    if "cv_metrics" in result:
        cv = result["cv_metrics"]
        summary["cv_ridge_mae_mean"] = round(cv["ridge"]["total"]["mae_mean"], 3)
        summary["cv_ridge_mae_std"] = round(cv["ridge"]["total"]["mae_std"], 3)
        summary["cv_nn_mae_mean"] = round(cv["nn"]["total"]["mae_mean"], 3)
        summary["cv_nn_mae_std"] = round(cv["nn"]["total"]["mae_std"], 3)
        summary["best_cv_alpha"] = result["best_cv_alpha"]
    return summary


def _best_model(s):
    """Return (name, mae) of the best model for a summary row."""
    models = {"Ridge": s["ridge_mae"], "NN": s["nn_mae"]}
    if "attn_nn_mae" in s:
        models["Attn"] = s["attn_nn_mae"]
    if "lgbm_mae" in s:
        models["LGBM"] = s["lgbm_mae"]
    best = min(models, key=models.get)
    return best, models[best]


def print_table(summaries):
    has_cv = any("cv_ridge_mae_mean" in s for s in summaries)
    has_attn = any("attn_nn_mae" in s for s in summaries)
    has_lgbm = any("lgbm_mae" in s for s in summaries)

    # -- MAE comparison table --
    hdr = f"{'Pos':<5} {'Ridge':>9} {'NN':>9}"
    if has_attn:
        hdr += f" {'Attn NN':>9}"
    if has_lgbm:
        hdr += f" {'LGBM':>9}"
    hdr += f" {'Best':>9} {'Time':>8}"
    w = len(hdr)

    print(f"\n{'=' * w}")
    print("MAE Comparison (test set)")
    print(f"{'=' * w}")
    print(hdr)
    print("-" * w)
    for s in summaries:
        best_name, _ = _best_model(s)
        line = f"{s['position']:<5} {s['ridge_mae']:>9.3f} {s['nn_mae']:>9.3f}"
        if has_attn:
            line += f" {s.get('attn_nn_mae', float('nan')):>9.3f}"
        if has_lgbm:
            line += f" {s.get('lgbm_mae', float('nan')):>9.3f}"
        line += f" {best_name:>9}"
        line += f" {s.get('elapsed_sec', 0):>7.0f}s"
        print(line)
    print("=" * w)

    # -- R2 comparison --
    print(f"\n{'R-squared':>5}")
    print("-" * w)
    for s in summaries:
        models = {"Ridge": s["ridge_r2"], "NN": s["nn_r2"]}
        if "attn_nn_r2" in s:
            models["Attn"] = s["attn_nn_r2"]
        if "lgbm_r2" in s:
            models["LGBM"] = s["lgbm_r2"]
        best = max(models, key=models.get)
        line = f"{s['position']:<5} {s['ridge_r2']:>9.3f} {s['nn_r2']:>9.3f}"
        if has_attn:
            line += f" {s.get('attn_nn_r2', float('nan')):>9.3f}"
        if has_lgbm:
            line += f" {s.get('lgbm_r2', float('nan')):>9.3f}"
        line += f" {best:>9}"
        print(line)
    print("=" * w)

    # -- Top-12 hit rate --
    print(f"\n{'Top-12 Hit Rate':>5}")
    print("-" * w)
    for s in summaries:
        models = {"Ridge": s["ridge_top12"], "NN": s["nn_top12"]}
        if "attn_nn_top12" in s:
            models["Attn"] = s["attn_nn_top12"]
        if "lgbm_top12" in s:
            models["LGBM"] = s["lgbm_top12"]
        best = max(models, key=models.get)
        line = f"{s['position']:<5} {s['ridge_top12']:>9.3f} {s['nn_top12']:>9.3f}"
        if has_attn:
            line += f" {s.get('attn_nn_top12', 0):>9.3f}"
        if has_lgbm:
            line += f" {s.get('lgbm_top12', 0):>9.3f}"
        line += f" {best:>9}"
        print(line)
    print("=" * w)

    # -- Per-target breakdown (all models) --
    tgt_w, col_w = 20, 9
    pt_hdr = f"  {'Target':<{tgt_w}} {'Ridge':>{col_w}} {'NN':>{col_w}}"
    if has_attn:
        pt_hdr += f" {'Attn NN':>{col_w}}"
    if has_lgbm:
        pt_hdr += f" {'LGBM':>{col_w}}"
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
                                    ("Attn", "attn_nn_per_target"),
                                    ("LGBM", "lgbm_per_target")]:
                    if key in s and t in s[key]:
                        models[mname] = s[key][t][metric_key]
                if not models:
                    continue
                best = (max if higher_better else min)(models, key=models.get)
                line = f"  {t:<{tgt_w}}"
                line += f" {models.get('Ridge', float('nan')):>{col_w}.3f}"
                line += f" {models.get('NN', float('nan')):>{col_w}.3f}"
                if has_attn:
                    line += f" {models.get('Attn', float('nan')):>{col_w}.3f}"
                if has_lgbm:
                    line += f" {models.get('LGBM', float('nan')):>{col_w}.3f}"
                line += f" {best:>{col_w}}"
                print(line)
        print("=" * len(pt_hdr))

    if has_cv:
        print(f"\n{'=' * 72}")
        print("Cross-Validation Metrics (mean +/- std across 4 folds)")
        print("=" * 72)
        print(f"{'Pos':<5} {'Ridge MAE':>20} {'NN MAE':>20} {'Best Alpha':>12}")
        print("-" * 60)
        for s in summaries:
            if "cv_ridge_mae_mean" in s:
                print(f"{s['position']:<5} "
                      f"{s['cv_ridge_mae_mean']:>8.3f} +/- {s['cv_ridge_mae_std']:<6.3f} "
                      f"{s['cv_nn_mae_mean']:>8.3f} +/- {s['cv_nn_mae_std']:<6.3f} "
                      f"{s['best_cv_alpha']:>10.2f}")
        print("=" * 72)


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
