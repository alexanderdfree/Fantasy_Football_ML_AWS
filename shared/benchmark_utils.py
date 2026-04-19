"""Shared helpers for local (benchmark_nn.py) and AWS Batch (batch/benchmark.py)
benchmark scripts. Consolidates summary-row construction, comparison-table
printing, git-hash capture, and history append.
"""
import datetime
import json
import os
import subprocess


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def append_to_history(history_file: str, run_entry: dict) -> None:
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file) as f:
                history = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            quarantine = f"{history_file}.corrupt-{ts}"
            os.rename(history_file, quarantine)
            print(f"WARNING: {history_file} is corrupt ({e}); moved to {quarantine}. Starting fresh.")
    history.append(run_entry)
    tmp = f"{history_file}.tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2, default=_json_default)
    os.replace(tmp, history_file)
    print(f"Run appended to {history_file}")


def _json_default(obj):
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _per_target(metrics: dict, exclude="total") -> dict:
    return {
        t: {"mae": round(v["mae"], 3), "r2": round(v["r2"], 3)}
        for t, v in metrics.items() if t != exclude
    }


def summarize_pipeline_result(position: str, result: dict) -> dict:
    """Extract a flat summary row from a position pipeline result dict.

    Used by both local (benchmark_nn.py with in-memory result) and AWS Batch
    (batch/benchmark.py with parsed benchmark_metrics.json) — the nested shape
    is identical.
    """
    ridge = result["ridge_metrics"]["total"]
    nn = result["nn_metrics"]["total"]
    summary = {
        "position": position,
        "ridge_mae": round(ridge["mae"], 3),
        "ridge_r2": round(ridge["r2"], 3),
        "nn_mae": round(nn["mae"], 3),
        "nn_r2": round(nn["r2"], 3),
        "nn_wins_mae": nn["mae"] < ridge["mae"],
        "nn_per_target": _per_target(result["nn_metrics"]),
        "ridge_per_target": _per_target(result["ridge_metrics"]),
        "ridge_top12": round(result.get("ridge_ranking", {}).get("season_avg_hit_rate", 0), 3),
        "nn_top12": round(result.get("nn_ranking", {}).get("season_avg_hit_rate", 0), 3),
    }
    if "attn_nn_metrics" in result:
        attn = result["attn_nn_metrics"]["total"]
        summary["attn_nn_mae"] = round(attn["mae"], 3)
        summary["attn_nn_r2"] = round(attn["r2"], 3)
        summary["attn_nn_per_target"] = _per_target(result["attn_nn_metrics"])
        summary["attn_nn_top12"] = round(
            result.get("attn_nn_ranking", {}).get("season_avg_hit_rate", 0), 3,
        )
    if "lgbm_metrics" in result:
        lgbm = result["lgbm_metrics"]["total"]
        summary["lgbm_mae"] = round(lgbm["mae"], 3)
        summary["lgbm_r2"] = round(lgbm["r2"], 3)
        summary["lgbm_per_target"] = _per_target(result["lgbm_metrics"])
        summary["lgbm_top12"] = round(
            result.get("lgbm_ranking", {}).get("season_avg_hit_rate", 0), 3,
        )
    if "cv_metrics" in result:
        cv = result["cv_metrics"]
        summary["cv_ridge_mae_mean"] = round(cv["ridge"]["total"]["mae_mean"], 3)
        summary["cv_ridge_mae_std"] = round(cv["ridge"]["total"]["mae_std"], 3)
        summary["cv_nn_mae_mean"] = round(cv["nn"]["total"]["mae_mean"], 3)
        summary["cv_nn_mae_std"] = round(cv["nn"]["total"]["mae_std"], 3)
        summary["best_cv_alpha"] = result["best_cv_alpha"]
    return summary


def _best_model_mae(s: dict) -> tuple[str, float]:
    models = {"Ridge": s["ridge_mae"], "NN": s["nn_mae"]}
    if "attn_nn_mae" in s:
        models["Attn"] = s["attn_nn_mae"]
    if "lgbm_mae" in s:
        models["LGBM"] = s["lgbm_mae"]
    best = min(models, key=models.get)
    return best, models[best]


def print_comparison_table(summaries: list, *, header: str, show_time: bool = True) -> None:
    """Print MAE / R² / Top-12 / per-target comparison tables."""
    has_cv = any("cv_ridge_mae_mean" in s for s in summaries)
    has_attn = any("attn_nn_mae" in s for s in summaries)
    has_lgbm = any("lgbm_mae" in s for s in summaries)

    hdr = f"{'Pos':<5} {'Ridge':>9} {'NN':>9}"
    if has_attn:
        hdr += f" {'Attn NN':>9}"
    if has_lgbm:
        hdr += f" {'LGBM':>9}"
    hdr += f" {'Best':>9}"
    if show_time:
        hdr += f" {'Time':>8}"
    w = len(hdr)

    print(f"\n{'=' * w}")
    print(header)
    print(f"{'=' * w}")
    print(hdr)
    print("-" * w)
    for s in summaries:
        best_name, _ = _best_model_mae(s)
        line = f"{s['position']:<5} {s['ridge_mae']:>9.3f} {s['nn_mae']:>9.3f}"
        if has_attn:
            line += f" {s.get('attn_nn_mae', float('nan')):>9.3f}"
        if has_lgbm:
            line += f" {s.get('lgbm_mae', float('nan')):>9.3f}"
        line += f" {best_name:>9}"
        if show_time:
            line += f" {s.get('elapsed_sec', 0):>7.0f}s"
        print(line)
    print("=" * w)

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

    tgt_w, col_w = 20, 9
    pt_hdr = f"  {'Target':<{tgt_w}} {'Ridge':>{col_w}} {'NN':>{col_w}}"
    if has_attn:
        pt_hdr += f" {'Attn NN':>{col_w}}"
    if has_lgbm:
        pt_hdr += f" {'LGBM':>{col_w}}"
    pt_hdr += f" {'Best':>{col_w}}"

    for metric_key, label, higher_better in [
        ("mae", "Per-Target MAE", False),
        ("r2", "Per-Target R\u00b2", True),
    ]:
        print(f"\n{label}")
        print("=" * len(pt_hdr))
        for s in summaries:
            print(f"\n  {s['position']}")
            print(pt_hdr)
            print("  " + "-" * (len(pt_hdr) - 2))
            targets = list(s.get("nn_per_target", s.get("ridge_per_target", {})).keys())
            for t in targets:
                models = {}
                for mname, key in [
                    ("Ridge", "ridge_per_target"),
                    ("NN", "nn_per_target"),
                    ("Attn", "attn_nn_per_target"),
                    ("LGBM", "lgbm_per_target"),
                ]:
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


def print_history_comparison(
    history_file: str, summaries: list, *, last_n: int = 5,
) -> None:
    """Print per-position tables comparing the new run vs. the last N history runs.

    Reads {history_file}, filters to entries that recorded each position, and
    prints one table per position with timestamp, git hash, note, and the
    same MAE/top-12 columns used by summarize_pipeline_result.
    """
    if not os.path.exists(history_file):
        return
    try:
        with open(history_file) as f:
            history = json.load(f)
    except (json.JSONDecodeError, ValueError):
        print(f"(could not read {history_file} for comparison)")
        return

    def _fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "  \u2014  "

    for new in summaries:
        pos = new["position"]
        rows = []
        for entry in history[:-1]:   # exclude the just-appended new run
            for s in entry.get("results", []):
                if s.get("position") == pos:
                    rows.append((entry.get("timestamp", "?")[:10],
                                 entry.get("git_hash", "?"),
                                 (entry.get("note") or "")[:38],
                                 s))
                    break
        rows = rows[-last_n:]

        hdr = (f"{'Date':<11} {'Hash':<9} {'Note':<40} "
               f"{'Ridge':>7} {'NN':>7} {'Attn':>7} {'LGBM':>7} {'Top12':>7}")
        print(f"\n{'=' * len(hdr)}")
        print(f"{pos} history (last {len(rows)} runs + this run)")
        print("=" * len(hdr))
        print(hdr)
        print("-" * len(hdr))
        for date, h, note, s in rows:
            print(f"{date:<11} {h:<9} {note:<40} "
                  f"{_fmt(s.get('ridge_mae')):>7} {_fmt(s.get('nn_mae')):>7} "
                  f"{_fmt(s.get('attn_nn_mae')):>7} {_fmt(s.get('lgbm_mae')):>7} "
                  f"{_fmt(s.get('nn_top12')):>7}")
        print(f"{'> NEW':<11} {'':<9} {'(this run)':<40} "
              f"{_fmt(new.get('ridge_mae')):>7} {_fmt(new.get('nn_mae')):>7} "
              f"{_fmt(new.get('attn_nn_mae')):>7} {_fmt(new.get('lgbm_mae')):>7} "
              f"{_fmt(new.get('nn_top12')):>7}")
        print("=" * len(hdr))
