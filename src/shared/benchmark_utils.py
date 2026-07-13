"""Shared helpers for local (benchmark.py) and AWS Batch (src/batch/benchmark.py)
benchmark scripts. Consolidates summary-row construction, comparison-table
printing, git-hash capture, and history append.
"""

import datetime
import json
import os
import subprocess


def utc_now_iso() -> str:
    """ISO8601 UTC timestamp with seconds precision (no timezone marker).

    Used as the timestamp prefix in run_ids so local benchmarks and CI runs
    sort consistently regardless of the operator's local timezone. The
    marker is omitted to keep the format identical to existing migrated
    filenames (which were unmarked but already produced under UTC by CI).
    """
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def get_git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _run_filename(run_entry: dict) -> str:
    # Colons in ISO timestamps are kept inside JSON bodies but stripped from
    # filenames so the directory plays nicely with shell globs and `find`.
    stem = run_entry.get("run_id") or run_entry.get("timestamp") or "run"
    return stem.replace(":", "-") + ".json"


def append_to_history(
    history_dir: str,
    run_entry: dict,
    *,
    pr_number: int | None = None,
) -> str:
    """Record one run by writing a standalone JSON file under ``history_dir``.

    Each run lives in its own ``{run_id}.json`` file (with ``:`` sanitized to
    ``-``) so the history can grow indefinitely without bloating a single
    file. Writes go through ``{path}.tmp`` + ``os.replace`` for crash safety.

    Returns the written path so callers can hand it to
    ``print_history_comparison`` as ``exclude_path`` — that avoids inferring
    "the just-written run" from filename sort order, which is fragile under
    clock skew or mixed timezones.

    ``pr_number`` (when supplied by CI) is recorded as a top-level field so
    the serving UI can deep-link rows to GitHub. Old files without the field
    are read as ``None`` — backward compatible.
    """
    os.makedirs(history_dir, exist_ok=True)
    if pr_number is not None:
        run_entry = {**run_entry, "pr_number": int(pr_number)}
    path = os.path.join(history_dir, _run_filename(run_entry))
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(run_entry, f, indent=2, default=_json_default)
    os.replace(tmp, path)
    print(f"Run written to {path}")
    return path


def _json_default(obj):
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _per_target(metrics: dict, exclude="total") -> dict:
    out: dict[str, dict] = {}
    for t, v in metrics.items():
        if t == exclude:
            continue
        entry = {"mae": round(v["mae"], 3), "r2": round(v["r2"], 3)}
        # rmse is additive: real pipeline metrics (compute_metrics) always carry
        # it, but synthetic test fixtures and the dry-run stub omit it — guard so
        # those stay {mae, r2} (and the exact-equality tests keep passing) while
        # production runs gain a per-target rmse for the History tab's toggle.
        if v.get("rmse") is not None:
            entry["rmse"] = round(v["rmse"], 3)
        out[t] = entry
    return out


def _rmse_field(prefix: str, total: dict) -> dict:
    """``{prefix}_rmse`` as a one-key dict, or empty when the block has no rmse.

    Mirrors ``_per_target``'s additive guard so summaries built from synthetic
    fixtures / the dry-run stub (no rmse) stay byte-identical, while production
    summaries gain ``{model}_rmse`` for the History tab's MAE/RMSE toggle.
    """
    rmse = total.get("rmse")
    return {f"{prefix}_rmse": round(rmse, 3)} if rmse is not None else {}


def _top12_value(result: dict, ranking_key: str) -> float | None:
    """``season_avg_hit_rate`` rounded to 3 dp, or ``None`` when ``result``
    carries no ranking block under ``ranking_key``.

    ``None`` (JSON ``null``) instead of a silent ``0``: the Batch split
    nn/cpu/merge path returned pipeline results without ``*_ranking`` from the
    2026-06-11 BATCH_SPLIT_ACTIVE flip (ADR-0019) until the ranking attach in
    ``src/shared/pipeline.py``'s short-circuit, and the old ``0`` default made
    every merged history row look like a real 0.000 hit rate for a month. A
    missing ranking must be visibly missing.
    """
    rate = (result.get(ranking_key) or {}).get("season_avg_hit_rate")
    return None if rate is None else round(rate, 3)


def summarize_pipeline_result(position: str, result: dict) -> dict:
    """Extract a flat summary row from a position pipeline result dict.

    Used by both local (benchmark.py with in-memory result) and AWS Batch
    (src/batch/benchmark.py with parsed benchmark_metrics.json) — the nested shape
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
        "ridge_top12": _top12_value(result, "ridge_ranking"),
        "nn_top12": _top12_value(result, "nn_ranking"),
    }
    summary.update(_rmse_field("ridge", ridge))
    summary.update(_rmse_field("nn", nn))
    if "elasticnet_metrics" in result:
        enet = result["elasticnet_metrics"]["total"]
        summary["elasticnet_mae"] = round(enet["mae"], 3)
        summary["elasticnet_r2"] = round(enet["r2"], 3)
        summary.update(_rmse_field("elasticnet", enet))
        summary["elasticnet_per_target"] = _per_target(result["elasticnet_metrics"])
        summary["elasticnet_top12"] = _top12_value(result, "elasticnet_ranking")
    if "attn_nn_metrics" in result:
        attn = result["attn_nn_metrics"]["total"]
        summary["attn_nn_mae"] = round(attn["mae"], 3)
        summary["attn_nn_r2"] = round(attn["r2"], 3)
        summary.update(_rmse_field("attn_nn", attn))
        summary["attn_nn_per_target"] = _per_target(result["attn_nn_metrics"])
        summary["attn_nn_top12"] = _top12_value(result, "attn_nn_ranking")
    if "lgbm_metrics" in result:
        lgbm = result["lgbm_metrics"]["total"]
        summary["lgbm_mae"] = round(lgbm["mae"], 3)
        summary["lgbm_r2"] = round(lgbm["r2"], 3)
        summary.update(_rmse_field("lgbm", lgbm))
        summary["lgbm_per_target"] = _per_target(result["lgbm_metrics"])
        summary["lgbm_top12"] = _top12_value(result, "lgbm_ranking")
    if "cv_metrics" in result:
        cv = result["cv_metrics"]
        summary["cv_ridge_mae_mean"] = round(cv["ridge"]["total"]["mae_mean"], 3)
        summary["cv_ridge_mae_std"] = round(cv["ridge"]["total"]["mae_std"], 3)
        summary["cv_nn_mae_mean"] = round(cv["nn"]["total"]["mae_mean"], 3)
        summary["cv_nn_mae_std"] = round(cv["nn"]["total"]["mae_std"], 3)
        # ``run_cv_pipeline`` returns per-target alphas under ``best_cv_alphas``
        # (plural dict). The old code read ``best_cv_alpha`` (singular scalar) —
        # a key that hasn't existed since the per-target rename, so every
        # ``benchmark --cv`` run KeyError'd here. Store the dict; the printer
        # renders a compact distinct-value summary.
        summary["best_cv_alphas"] = result["best_cv_alphas"]
    # EC2 path: src/batch/train.py writes these into benchmark_metrics.json so the
    # row appended by src/batch/benchmark.py --download-only carries timing. Local
    # benchmark.py sets elapsed_sec on the summary directly so these no-op for
    # it (result is the in-memory pipeline dict, not the parsed JSON file).
    if "elapsed_sec" in result:
        summary["elapsed_sec"] = result["elapsed_sec"]
    if "phase_seconds" in result:
        summary["phase_seconds"] = result["phase_seconds"]
    return summary


def _best_model_mae(s: dict) -> tuple[str, float]:
    models = {"Ridge": s["ridge_mae"], "NN": s["nn_mae"]}
    if "elasticnet_mae" in s:
        models["ENet"] = s["elasticnet_mae"]
    if "attn_nn_mae" in s:
        models["Attn"] = s["attn_nn_mae"]
    if "lgbm_mae" in s:
        models["LGBM"] = s["lgbm_mae"]
    best = min(models, key=models.get)
    return best, models[best]


def print_comparison_table(summaries: list, *, header: str, show_time: bool = True) -> None:
    """Print MAE / R² / Top-12 / per-target comparison tables."""
    has_cv = any("cv_ridge_mae_mean" in s for s in summaries)
    has_enet = any("elasticnet_mae" in s for s in summaries)
    has_attn = any("attn_nn_mae" in s for s in summaries)
    has_lgbm = any("lgbm_mae" in s for s in summaries)

    hdr = f"{'Pos':<5} {'Ridge':>9} {'NN':>9}"
    if has_enet:
        hdr += f" {'ENet':>9}"
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
        if has_enet:
            line += f" {s.get('elasticnet_mae', float('nan')):>9.3f}"
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
        if "elasticnet_r2" in s:
            models["ENet"] = s["elasticnet_r2"]
        if "attn_nn_r2" in s:
            models["Attn"] = s["attn_nn_r2"]
        if "lgbm_r2" in s:
            models["LGBM"] = s["lgbm_r2"]
        best = max(models, key=models.get)
        line = f"{s['position']:<5} {s['ridge_r2']:>9.3f} {s['nn_r2']:>9.3f}"
        if has_enet:
            line += f" {s.get('elasticnet_r2', float('nan')):>9.3f}"
        if has_attn:
            line += f" {s.get('attn_nn_r2', float('nan')):>9.3f}"
        if has_lgbm:
            line += f" {s.get('lgbm_r2', float('nan')):>9.3f}"
        line += f" {best:>9}"
        print(line)
    print("=" * w)

    print(f"\n{'Top-12 Hit Rate':>5}")
    print("-" * w)
    dash = "—"

    def _t12(v) -> str:
        # None = the run recorded no ranking block for that model (e.g. a
        # pre-fix split-path artifact) — render an em-dash, never a fake 0.000.
        return f"{v:>9.3f}" if isinstance(v, (int, float)) else f"{dash:>9}"

    for s in summaries:
        models = {"Ridge": s.get("ridge_top12"), "NN": s.get("nn_top12")}
        if has_enet:
            models["ENet"] = s.get("elasticnet_top12")
        if has_attn:
            models["Attn"] = s.get("attn_nn_top12")
        if has_lgbm:
            models["LGBM"] = s.get("lgbm_top12")
        numeric = {k: v for k, v in models.items() if isinstance(v, (int, float))}
        best = max(numeric, key=numeric.get) if numeric else dash
        line = f"{s['position']:<5} {_t12(models['Ridge'])} {_t12(models['NN'])}"
        if has_enet:
            line += f" {_t12(models['ENet'])}"
        if has_attn:
            line += f" {_t12(models['Attn'])}"
        if has_lgbm:
            line += f" {_t12(models['LGBM'])}"
        line += f" {best:>9}"
        print(line)
    print("=" * w)

    tgt_w, col_w = 20, 9
    pt_hdr = f"  {'Target':<{tgt_w}} {'Ridge':>{col_w}} {'NN':>{col_w}}"
    if has_enet:
        pt_hdr += f" {'ENet':>{col_w}}"
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
                    ("ENet", "elasticnet_per_target"),
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
                if has_enet:
                    line += f" {models.get('ENet', float('nan')):>{col_w}.3f}"
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
        print(f"{'Pos':<5} {'Ridge MAE':>20} {'NN MAE':>20} {'Best Alphas':>18}")
        print("-" * 66)
        for s in summaries:
            if "cv_ridge_mae_mean" in s:
                # Per-target alphas -> compact distinct-value summary (the column
                # used to format a single scalar, which broke after the
                # per-target rename).
                alphas = s.get("best_cv_alphas", {})
                if isinstance(alphas, dict) and alphas:
                    alpha_str = ",".join(f"{a:g}" for a in sorted(set(alphas.values())))
                else:
                    alpha_str = str(alphas)
                print(
                    f"{s['position']:<5} "
                    f"{s['cv_ridge_mae_mean']:>8.3f} +/- {s['cv_ridge_mae_std']:<6.3f} "
                    f"{s['cv_nn_mae_mean']:>8.3f} +/- {s['cv_nn_mae_std']:<6.3f} "
                    f"{alpha_str:>16}"
                )
        print("=" * 72)


def print_history_comparison(
    history_dir: str,
    summaries: list,
    *,
    exclude_path: str | None = None,
    last_n: int = 5,
) -> None:
    """Print per-position tables comparing the new run vs. the last N history runs.

    Reads every ``*.json`` file at the top level of ``history_dir`` (so any
    ``ablations/`` subdir is naturally excluded), filters to entries that
    recorded each position, and prints one table per position with timestamp,
    git hash, note, and MAE/top-12 columns. Filenames are ISO-timestamp-
    prefixed, so lexical sort orders display chronologically.

    ``exclude_path`` (typically the value returned by ``append_to_history``)
    drops the just-written run from the historical rows so it doesn't
    duplicate the explicit ``> NEW`` row built from ``summaries``. The
    comparison no longer infers "latest filename = new run", which broke
    when local-time and CI-UTC timestamps mixed.
    """
    if not os.path.isdir(history_dir):
        return
    exclude_filename = os.path.basename(exclude_path) if exclude_path else None
    files = sorted(
        f
        for f in os.listdir(history_dir)
        if f.endswith(".json")
        and f != exclude_filename
        and os.path.isfile(os.path.join(history_dir, f))
    )
    history = []
    for fn in files:
        path = os.path.join(history_dir, fn)
        try:
            with open(path) as f:
                history.append(json.load(f))
        except (json.JSONDecodeError, ValueError):
            print(f"(could not read {path} for comparison)")

    def _fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "  \u2014  "

    for new in summaries:
        pos = new["position"]
        rows = []
        for entry in history:
            for s in entry.get("results", []):
                if s.get("position") == pos:
                    rows.append(
                        (
                            entry.get("timestamp", "?")[:10],
                            entry.get("git_hash", "?"),
                            (entry.get("note") or "")[:38],
                            s,
                        )
                    )
                    break
        rows = rows[-last_n:]

        hdr = (
            f"{'Date':<11} {'Hash':<9} {'Note':<40} "
            f"{'Ridge':>7} {'NN':>7} {'Attn':>7} {'LGBM':>7} {'Top12':>7}"
        )
        print(f"\n{'=' * len(hdr)}")
        print(f"{pos} history (last {len(rows)} runs + this run)")
        print("=" * len(hdr))
        print(hdr)
        print("-" * len(hdr))
        for date, h, note, s in rows:
            print(
                f"{date:<11} {h:<9} {note:<40} "
                f"{_fmt(s.get('ridge_mae')):>7} {_fmt(s.get('nn_mae')):>7} "
                f"{_fmt(s.get('attn_nn_mae')):>7} {_fmt(s.get('lgbm_mae')):>7} "
                f"{_fmt(s.get('nn_top12')):>7}"
            )
        print(
            f"{'> NEW':<11} {'':<9} {'(this run)':<40} "
            f"{_fmt(new.get('ridge_mae')):>7} {_fmt(new.get('nn_mae')):>7} "
            f"{_fmt(new.get('attn_nn_mae')):>7} {_fmt(new.get('lgbm_mae')):>7} "
            f"{_fmt(new.get('nn_top12')):>7}"
        )
        print("=" * len(hdr))
