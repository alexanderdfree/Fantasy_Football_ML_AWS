"""Benchmark script: runs the QB, RB, WR, TE, K, DST pipelines and prints a comparison table.

By default this autodetects concurrency: on a many-core CUDA box (e.g. the 9950X3D / RTX
5080 dev machine) it fans the requested positions out in parallel via
``src.benchmarking.parallel_train`` — the measured-optimal local regime, since the small
attention NN is GPU launch-bound and stacking processes fills the GPU's idle gaps
(``todo/gpu_launch_bound_levers.md``). On any other host it runs the positions sequentially.
Pass ``-j N`` to cap concurrency or ``--sequential`` to force the in-process loop.

Usage:
    python benchmark.py                          # all 6; parallel on a capable box, else sequential
    python benchmark.py RB                       # run one position (always sequential)
    python benchmark.py -j 3                      # cap concurrency at 3
    python benchmark.py --sequential             # force the in-process sequential loop
    python benchmark.py --note "tuned WR dropout" # annotate the run
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.scripts.bench_fingerprint import collect_code_fingerprints
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
    cfg = {
        k[len(prefix) :].lower(): v
        for k, v in vars(mod).items()
        if k.startswith(prefix) and not k.endswith("FEATURES") and k != f"{prefix}RIDGE_ALPHA_GRIDS"
    }
    # The loss machinery now lives on POSITION_CONFIG (not module-level
    # constants), so surface the loss-objective knobs explicitly — this is what
    # records a Huber<->MSE objective switch in the benchmark history entry.
    pc = getattr(mod, "POSITION_CONFIG", None)
    if pc is not None:
        cfg["lgbm_objective"] = getattr(pc, "lgbm_objective", None)
        cfg["head_losses"] = dict(getattr(pc, "head_losses", {}) or {})
        cfg["loss_weights"] = dict(getattr(pc, "loss_weights", {}) or {})
    return cfg


def run_one(position):
    """Run a single position pipeline and return its metrics dict.

    The single-split path only; ``--cv`` was collapsed into ``--rolling-origin``
    (PR #719), so the rolling-origin branch in ``main`` calls
    ``run_rolling_origin`` and never reaches here with a CV request — the old
    ``cv`` parameter (which dispatched to each position's ``run_cv``) was
    unreachable and has been removed.
    """
    from src.shared.registry import get_runner
    from src.shared.utils import seed_everything

    seed_everything(42)
    runner = get_runner(position)
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

    Schema-safe: the website History tab (``src/serving/benchmark_history.py::_benchmark_row``) and
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


def _elite_top24_mask(test_df):
    """Rows belonging to the per-season top-24 players by prior-season mean FP.

    A-priori (prior-season) proxy, so the cohort is leakage-free and stable
    within a season; mirrors ``cohort_analysis.label_scoring_tier_rows``
    semantics without reloading splits. Top-24 is over distinct players, then
    every row of those players counts.
    """
    import pandas as pd

    col = "prior_season_mean_fantasy_points"
    mask = pd.Series(False, index=test_df.index)
    for _, sub in test_df.groupby("season"):
        top = set(sub.drop_duplicates("player_id").nlargest(24, col)["player_id"])
        mask.loc[sub.index] = sub["player_id"].isin(top)
    return mask


# Tracked cohorts (#1102/#1106 ask; anomaly clusters from #1141). Each entry is
# (name, required columns, mask fn). Column-guarded so positions lacking a
# feature degrade by omitting that cohort: RB deliberately drops
# is_returning_from_absence (multicollinearity, see src/rb/config.py), and the
# K/DST frames lack the skill-position contextual columns.
_COHORT_SPECS = (
    ("week1", ("week",), lambda df: df["week"] == 1),
    (
        "returning",
        ("is_returning_from_absence",),
        lambda df: df["is_returning_from_absence"] == 1,
    ),
    ("questionable", ("game_status",), lambda df: df["game_status"] < 1.0),
    ("inheritor", ("inherited_opportunity",), lambda df: df["inherited_opportunity"] > 0),
    (
        "elite_top24",
        ("prior_season_mean_fantasy_points", "player_id", "season"),
        _elite_top24_mask,
    ),
)


def _cohorts_block(position, result):
    """Per-cohort, per-model fantasy-point bias/MAE for history fold-in.

    The tracked subgroup metrics #1102/#1106 asked for: week-1 cold start,
    returners, Questionable designations, vacancy inheritors (the spot-start
    cohort), and a-priori elite players — so feature work on any of these has
    a recorded before/after baseline in every benchmark run. ``bias`` is
    mean(pred - actual): negative = under-prediction. Schema-safe like
    ``significance``: History-tab readers ignore unknown keys. Returns None if
    the result lacks per-row test predictions.
    """
    test_df = result.get("test_df")
    if test_df is None or "fantasy_points" not in getattr(test_df, "columns", ()):
        return None
    from src.analysis.cohort_analysis import available_models, per_model_metrics

    models = available_models(test_df)
    if not models:
        return None
    block = {}
    for name, required, mask_fn in _COHORT_SPECS:
        if any(col not in test_df.columns for col in required):
            continue
        sub = test_df[mask_fn(test_df)]
        if len(sub) == 0:
            block[name] = {"n": 0, "models": {}}
            continue
        metrics = per_model_metrics(sub, models)
        block[name] = {
            "n": int(len(sub)),
            "models": {
                model: {"bias": round(float(v["bias"]), 3), "mae": round(float(v["mae"]), 3)}
                for model, v in metrics.items()
            },
        }
    return block or None


# --- Rolling-origin (walk-forward) multi-season TEST evaluation ---------------
# Score each origin in ROLLING_ORIGIN_TEST_SEASONS (train [..T-2] / val T-1 /
# test T) and report per-model MAE/R2/top-12 as mean±std across origins. The
# final origin (test=2025) reproduces the production single split, so a
# rolling-origin run is directly comparable to a normal benchmark. Headline /
# operator-invoked only (~N_origins x the single-split cost), not per-PR.

_RO_MODELS = ("ridge", "nn", "elasticnet", "attn_nn", "lgbm")


def _load_full_featured_frame():
    """Reconstruct the full featured frame from the 3 on-disk split parquets.

    Every engineered column is split-independent (see ``rolling_origin_folds``),
    so concatenating train+val+test and re-slicing by season is leakage-free and
    avoids rebuilding features.
    """
    import pandas as pd

    from src.config import SPLITS_DIR
    from src.shared.pipeline import _read_split

    return pd.concat(
        [
            _read_split(f"{SPLITS_DIR}/train.parquet"),
            _read_split(f"{SPLITS_DIR}/val.parquet"),
            _read_split(f"{SPLITS_DIR}/test.parquet"),
        ],
        ignore_index=True,
    )


def _slice_origin(df, test_season, min_train_season=None):
    """Season-slice a self-loaded (K/DST) frame into (train, val, test) for one origin.

    Mirrors ``rolling_origin_folds`` but without the skill-split ``season_type`` /
    ``snap_pct`` handling (K/DST team/kicker frames carry neither). The train
    floor defaults to production's first train season (``TRAIN_SEASONS[0]``, 2013)
    so the final origin reproduces the production split rather than silently
    training on the context-only 2012 season.
    """
    if min_train_season is None:
        from src.config import TRAIN_SEASONS

        min_train_season = TRAIN_SEASONS[0]
    val_season = test_season - 1
    train_seasons = list(range(min_train_season, val_season))
    tr = df[df["season"].isin(train_seasons)].copy()
    va = df[df["season"] == val_season].copy()
    te = df[df["season"] == test_season].copy()
    return tr, va, te


def _self_load_full_frame_and_cfg(position):
    """Build a self-loading position's (K/DST) full featured frame + runtime cfg once.

    Mirrors each position's ``run()`` data assembly so per-origin slices feed the
    exact same pipeline; K additionally needs the ``attn_history_builder_fn``
    closure (captures ``kicks_df``) injected into a copied cfg.
    """
    if position == "DST":
        from src.dst.data import build_data
        from src.dst.features import compute_features
        from src.dst.run_pipeline import CONFIG
        from src.dst.targets import compute_targets

        df = build_data()
        df = compute_targets(df)
        compute_features(df)
        return df, CONFIG
    if position == "K":
        from src.k.config import POSITION_CONFIG
        from src.k.data import load_data, load_kicks
        from src.k.features import compute_features
        from src.k.run_pipeline import CONFIG, _build_kick_history_closure
        from src.k.targets import compute_targets

        df = load_data()
        df = compute_targets(df)
        compute_features(df)
        kicks_df = load_kicks(df)
        cfg = dict(CONFIG)
        cfg.setdefault("attn_kick_stats", POSITION_CONFIG.attn_kick_stats)
        cfg.setdefault("attn_max_games", POSITION_CONFIG.attn_max_games)
        cfg.setdefault("attn_max_kicks_per_game", POSITION_CONFIG.attn_max_kicks_per_game)
        cfg["attn_history_builder_fn"] = _build_kick_history_closure(cfg, kicks_df)
        return df, cfg
    raise ValueError(f"_self_load_full_frame_and_cfg: unexpected position {position!r}")


def _rolling_origin_inputs(position):
    """Return ``[(test_season, train_df, val_df, test_df), ...]`` for one position.

    QB/RB/WR/TE re-slice the on-disk splits via ``rolling_origin_folds``; K/DST
    self-load their frame and season-slice it. The second element of the returned
    tuple is the cfg to pass to ``run_pipeline`` for self-loading positions
    (``None`` for dataframe positions, which go through ``get_runner``).
    """
    from src.config import ROLLING_ORIGIN_TEST_SEASONS
    from src.data.split import rolling_origin_folds
    from src.shared.registry import accepts_dataframes

    if accepts_dataframes(position):
        full_df = _load_full_featured_frame()
        folds = rolling_origin_folds(full_df)
        origins = [(ROLLING_ORIGIN_TEST_SEASONS[i], tr, va, te) for (i, tr, va, te) in folds]
        return origins, None

    full_df, cfg = _self_load_full_frame_and_cfg(position)
    origins = [(ts, *_slice_origin(full_df, ts)) for ts in ROLLING_ORIGIN_TEST_SEASONS]
    return origins, cfg


def _score_origin(position, train_df, val_df, test_df, cfg, seed=42):
    """Train + score one origin; return its flat ``summarize_pipeline_result`` summary.

    Factored out so the rolling-origin driver's iteration/aggregation can be
    unit-tested by monkeypatching this single call.
    """
    from src.shared.registry import accepts_dataframes, get_runner
    from src.shared.utils import seed_everything

    seed_everything(seed)
    if accepts_dataframes(position):
        result = get_runner(position)(train_df=train_df, val_df=val_df, test_df=test_df, seed=seed)
    else:
        from src.shared.pipeline import run_pipeline

        result = run_pipeline(position, cfg, train_df, val_df, test_df, seed)
    return summarize_pipeline_result(position, result)


def _mean_std(xs):
    import statistics

    vals = [x for x in xs if x is not None]
    if not vals:
        return None, None
    mean = round(statistics.mean(vals), 4)
    std = round(statistics.stdev(vals), 4) if len(vals) >= 2 else 0.0
    return mean, std


def _aggregate_rolling_origin(per_origin):
    """Build the additive ``rolling_origin`` block from ``[(test_season, summary), ...]``.

    Aggregates per-model MAE/R2/top-12 as mean±std across origins (only for models
    present in every origin) and records each origin's flat per-model numbers.
    """
    summaries = [s for _, s in per_origin]
    aggregate = {}
    for m in _RO_MODELS:
        if not all(f"{m}_mae" in s for s in summaries):
            continue
        mae_mean, mae_std = _mean_std([s.get(f"{m}_mae") for s in summaries])
        r2_mean, r2_std = _mean_std([s.get(f"{m}_r2") for s in summaries])
        top_mean, top_std = _mean_std([s.get(f"{m}_top12") for s in summaries])
        aggregate[m] = {
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "top12_mean": top_mean,
            "top12_std": top_std,
        }
    per_origin_rows = []
    for ts, s in per_origin:
        row = {"test_season": ts}
        for m in _RO_MODELS:
            for suff in ("mae", "r2", "top12"):
                key = f"{m}_{suff}"
                if key in s:
                    row[key] = s[key]
        per_origin_rows.append(row)
    return {
        "test_seasons": [ts for ts, _ in per_origin],
        "n_origins": len(per_origin),
        "aggregate": aggregate,
        "per_origin": per_origin_rows,
    }


def score_one_origin(position, test_season, seed=42):
    """Train + score a SINGLE rolling origin; return ``(test_season, summary)``.

    The (position × origin) unit the flattened parallel orchestrator dispatches:
    it loads the position's folds, selects the origin whose test season ==
    ``test_season``, and scores just that one. ``run_rolling_origin`` keeps the
    in-process loop (loading the frame once); this is the per-cell entry point so
    a single position's origins can fan out concurrently.
    """
    origins, cfg = _rolling_origin_inputs(position)
    for ts, tr, va, te in origins:
        if ts == test_season:
            return ts, _score_origin(position, tr, va, te, cfg, seed)
    available = [ts for ts, *_ in origins]
    raise ValueError(
        f"score_one_origin: test_season {test_season} is not a rolling origin for "
        f"{position} (have {available})"
    )


def finalize_rolling_origin(position, per_origin):
    """Assemble one position's per-origin summaries into its final benchmark record.

    ``per_origin`` is ``[(test_season, summary), ...]``. Returns the production
    origin's (test == ``TEST_SEASONS[0]``) flat summary augmented with a
    ``rolling_origin`` mean±std block — so a rolling-origin run stays comparable
    to a normal single-split run in the History tab. Same result as the tail of the
    old ``run_rolling_origin``, factored out so the parallel merge can call it after
    gathering per-cell summaries.
    """
    from src.config import TEST_SEASONS

    prod_season = TEST_SEASONS[0]
    final = next((dict(s) for ts, s in per_origin if ts == prod_season), dict(per_origin[-1][1]))
    final["rolling_origin"] = _aggregate_rolling_origin(per_origin)
    return final


def run_rolling_origin(position):
    """Score all rolling origins for one position; return the production-origin summary
    augmented with a ``rolling_origin`` block.

    The returned dict keeps the flat keys (``ridge_mae`` etc.) of the production
    origin (test == ``TEST_SEASONS[0]``), so a rolling-origin run is comparable to
    a normal single-split run in the History tab; the mean±std lives under
    ``summary["rolling_origin"]``.

    The in-process ``-j 1`` / non-CUDA fallback (and the path tests monkeypatch):
    loads the position's frame ONCE via ``_rolling_origin_inputs`` and loops the
    origins, then defers to ``finalize_rolling_origin`` for the production-origin
    selection + aggregation.
    """
    origins, cfg = _rolling_origin_inputs(position)
    per_origin = []
    for test_season, tr, va, te in origins:
        summary = _score_origin(position, tr, va, te, cfg)
        per_origin.append((test_season, summary))
    return finalize_rolling_origin(position, per_origin)


def _print_rolling_origin_table(summaries):
    print(f"\n{'=' * 72}")
    print("Rolling-Origin TEST metrics (MAE mean +/- std across origins)")
    print("=" * 72)
    for s in summaries:
        ro = s.get("rolling_origin")
        if not ro:
            continue
        seasons = ",".join(str(x) for x in ro["test_seasons"])
        print(f"\n{s['position']} — {ro['n_origins']} origins (test seasons {seasons}):")
        for model, agg in ro["aggregate"].items():
            std = agg["mae_std"] if agg["mae_std"] is not None else 0.0
            print(f"    {model:<12} MAE {agg['mae_mean']:>7.3f} +/- {std:<6.3f}")
    print("=" * 72)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark NN pipelines")
    parser.add_argument(
        "positions",
        nargs="*",
        default=["QB", "RB", "WR", "TE", "K", "DST"],
        help="Positions to benchmark (e.g. RB QB)",
    )
    parser.add_argument("--note", default="", help="Describe what changed in this run")
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Deprecated benchmark-reporting alias for --rolling-origin.",
    )
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
    parser.add_argument(
        "--rolling-origin",
        action="store_true",
        help="Walk-forward multi-season TEST eval over ROLLING_ORIGIN_TEST_SEASONS (train "
        "[..T-2] / val T-1 / test T per origin); reports per-model MAE/R2/top-12 as mean±std. "
        "~N_origins x the single-split cost — headline eval, not per-PR. Overrides --cv.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Max positions to train concurrently. Default: autodetect — all positions on a "
        "many-core CUDA box (the measured-optimal regime, see todo/gpu_launch_bound_levers.md), "
        "else sequential. Reuses the parallel runner (src/benchmarking/parallel_train.py).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force the in-process sequential loop (equivalent to -j 1), bypassing autodetect.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan (core-pool layout + dispatch order) and launch nothing.",
    )
    args = parser.parse_args(argv)
    rolling_origin_mode = args.rolling_origin or args.cv
    if args.cv and not args.rolling_origin:
        print("DEPRECATED: benchmark --cv now aliases --rolling-origin walk-forward reporting.")

    positions = args.positions

    # Number of training runs this invocation dispatches: one per position for a
    # single-split run, but ``len(ROLLING_ORIGIN_TEST_SEASONS)`` per position for a
    # rolling-origin run (each origin is its own train+score). The orchestrator
    # flattens rolling-origin work into a (position × origin) grid, so concurrency is
    # sized on cells, not positions.
    from src.config import ROLLING_ORIGIN_TEST_SEASONS

    n_origins = len(ROLLING_ORIGIN_TEST_SEASONS) if rolling_origin_mode else 1
    n_cells = len(positions) * n_origins

    # Concurrency dispatch. By default autodetect: on a many-core CUDA box the
    # measured-optimal regime is to saturate the GPU's idle gaps with parallel training
    # runs (the GPU is launch-bound — todo/gpu_launch_bound_levers.md), elsewhere
    # sequential. ``--sequential``/``-j 1`` forces the in-process loop below. The parallel
    # engine lives in ``parallel_train`` (which imports from this module), so these imports
    # are function-local to avoid a circular import at load time. For rolling-origin we cap
    # at the GPU-saturating slot count (``_default_jobs(6)``), NOT the position count, so a
    # single-position rolling-origin run still fans its origins out concurrently.
    if args.sequential:
        jobs = 1
    elif args.jobs is not None:
        jobs = args.jobs
    else:
        from src.benchmarking.parallel_train import _default_jobs

        if rolling_origin_mode:
            slots = _default_jobs(6)  # 6 on a capable CUDA box, 1 off-CUDA
            jobs = min(slots, n_cells) if slots > 1 else 1
        else:
            jobs = _default_jobs(len(positions))

    if jobs > 1 and n_cells > 1:
        from src.benchmarking import parallel_train

        passthrough = []
        if rolling_origin_mode:
            passthrough.append("--rolling-origin")
        if args.significance:
            passthrough.append("--significance")
        return parallel_train.orchestrate(
            positions,
            jobs,
            passthrough,
            args.note,
            args.no_sync,
            args.dry_run,
            rolling_origin=rolling_origin_mode,
        )

    if args.dry_run:
        print(f"[dry-run] would run sequentially (-j 1): {positions}")
        return 0

    # Fingerprint the code BEFORE training so a mid-run edit can't be
    # laundered into benchmark evidence for code that never trained.
    code_fps = collect_code_fingerprints(positions)

    summaries = []
    for pos in positions:
        t0 = time.time()
        mode = "ROLLING-ORIGIN" if rolling_origin_mode else "SINGLE-SPLIT"
        print(f"\n{'#' * 60}")
        print(f"# BENCHMARKING {pos} ({mode})")
        print(f"{'#' * 60}")
        if rolling_origin_mode:
            s = run_rolling_origin(pos)
        else:
            result = run_one(pos)
            s = summarize_pipeline_result(pos, result)
            if args.significance and not args.cv:
                sig_block = _significance_block(pos, result)
                if sig_block is not None:
                    s["significance"] = sig_block
            cohorts = _cohorts_block(pos, result)
            if cohorts is not None:
                s["cohorts"] = cohorts
        elapsed = time.time() - t0
        s["elapsed_sec"] = round(elapsed, 1)
        summaries.append(s)
        print(f"\n  [{pos}] Completed in {elapsed:.1f}s")

    if rolling_origin_mode:
        _print_rolling_origin_table(summaries)
    print_comparison_table(summaries, header="MAE Comparison (test set)", show_time=True)

    # Save latest results (backwards compat)
    with open(RESULTS_FILE, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Append to history
    git_hash = get_git_hash()
    now = utc_now_iso()
    entry = {
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
    }
    if code_fps:
        # Re-fingerprint after training: an edit that landed mid-run and
        # persisted to run end means some positions trained different code
        # than the snapshot — omit rather than record laundered evidence.
        # (Two-point check: an intra-run edit reverted before run end is not
        # detectable here.)
        if collect_code_fingerprints(positions) == code_fps:
            entry["code_fingerprints"] = code_fps
        else:
            print("WARNING: gated code changed during the run; omitting code_fingerprints")
    # Additive marker so the History tab can badge walk-forward runs; old
    # readers (and single-split runs) simply lack the key.
    if rolling_origin_mode:
        entry["mode"] = "rolling_origin"
    written_path = append_to_history(HISTORY_DIR, entry)

    if not args.no_sync:
        _maybe_upload_to_s3(written_path)

    print_history_comparison(HISTORY_DIR, summaries, exclude_path=written_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
