"""CUDA graph vs eager MAE/GradScaler analysis for the local RTX 5080 path.

This is the runnable companion to the CUDA-graph benchmarkability notes in
``todo/gpu_launch_bound_levers.md``. It runs the real position attention
pipeline under a paired seed and compares:

* graph-vs-graph reproducibility (same code/settings twice),
* eager-vs-graph MAE drift,
* the BatchNorm warmup snapshot/restore rule-out,
* optional fixed-loss-scale behaviour.

The script defaults to RB because the original CUDA-graph speed/regression
measurement used RB. It disables unrelated model branches by default so the
attention NN is the only trained model, but it does not shrink data, heads, loss
configuration, or the attention architecture.

Usage:
    source scripts/wsl-env.sh
    .venv/bin/python -m src.analysis.cuda_graph_gradscale
    .venv/bin/python -m src.analysis.cuda_graph_gradscale --include-fixed-scale
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.shared.benchmark_utils import append_to_history, get_git_hash, utc_now_iso  # noqa: E402

ANALYSIS_NAME = "cuda_graph_gradscale"
HISTORY_DIR = Path("benchmark_history") / "ablations"
TRACE_DIR = HISTORY_DIR / ANALYSIS_NAME

TRACE_ENV = "FF_GRADSCALER_TRACE_PATH"
TRACE_LABEL_ENV = "FF_GRADSCALER_TRACE_LABEL"
GRAPH_ENV = "FF_CUDA_GRAPH"
GRAPH_RESTORE_BN_ENV = "FF_CUDA_GRAPH_RESTORE_BN"
FIXED_SCALE_ENV = "FF_AMP_FIXED_SCALE"
INIT_SCALE_ENV = "FF_AMP_INIT_SCALE"

VARIANT_ENVS: dict[str, dict[str, str]] = {
    "graph_a": {GRAPH_ENV: "1"},
    "graph_b": {GRAPH_ENV: "1"},
    "eager": {GRAPH_ENV: "0"},
    "graph_restore_bn": {GRAPH_ENV: "1", GRAPH_RESTORE_BN_ENV: "1"},
    "graph_fixed_scale": {GRAPH_ENV: "1", FIXED_SCALE_ENV: "1"},
    "graph_fixed_scale_restore_bn": {
        GRAPH_ENV: "1",
        FIXED_SCALE_ENV: "1",
        GRAPH_RESTORE_BN_ENV: "1",
    },
    "eager_fixed_scale": {GRAPH_ENV: "0", FIXED_SCALE_ENV: "1"},
}
DEFAULT_VARIANTS = ["graph_a", "graph_b", "eager", "graph_restore_bn"]

_MANAGED_ENVS = {
    TRACE_ENV,
    TRACE_LABEL_ENV,
    GRAPH_ENV,
    GRAPH_RESTORE_BN_ENV,
    FIXED_SCALE_ENV,
    INIT_SCALE_ENV,
    "FF_DEVICE",
    "FF_DETERMINISTIC",
    "FF_FORCE_DROPOUT_ZERO",
    "FF_NN_FIXED_EPOCHS",
}


@contextmanager
def _patched_env(updates: dict[str, str]):
    old = {k: os.environ.get(k) for k in _MANAGED_ENVS | set(updates)}
    try:
        for key in _MANAGED_ENVS:
            os.environ.pop(key, None)
        for key, val in updates.items():
            os.environ[key] = val
        yield
    finally:
        for key, val in old.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def _make_attention_cfg(base_cfg: dict[str, Any], *, attn_only: bool) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["train_attention_nn"] = True
    cfg["train_lightgbm"] = False
    cfg["train_elasticnet"] = False
    if attn_only:
        cfg["train_ridge"] = False
        cfg["train_base_nn"] = False
    return cfg


def _metric_payload(result: dict[str, Any], targets: list[str]) -> dict[str, Any]:
    metrics = result["attn_nn_metrics"]
    payload = {
        "total_mae": float(metrics["total"]["mae"]),
        "total_r2": float(metrics["total"]["r2"]),
        "targets": {t: float(metrics[t]["mae"]) for t in targets},
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    payload["metrics_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_trace(path: Path) -> dict[str, Any]:
    rows = _trace_rows(path)
    steps = [r for r in rows if r.get("kind") == "step"]
    first_skip = next((r["step"] for r in steps if r.get("skipped")), None)
    first_change = next((r["step"] for r in steps if r.get("scale_changed")), None)
    scales = [r.get("scale") for r in steps]
    next_scales = [r.get("next_scale") for r in steps]
    return {
        "path": str(path),
        "n_steps": len(steps),
        "first_skip_step": first_skip,
        "first_scale_change_step": first_change,
        "initial_scale": scales[0] if scales else None,
        "final_scale": next_scales[-1] if next_scales else None,
        "n_scale_changes": sum(1 for r in steps if r.get("scale_changed")),
        "n_skipped": sum(1 for r in steps if r.get("skipped")),
    }


def first_trace_diff(left: Path, right: Path) -> dict[str, Any] | None:
    l_steps = [r for r in _trace_rows(left) if r.get("kind") == "step"]
    r_steps = [r for r in _trace_rows(right) if r.get("kind") == "step"]
    keys = ("scale", "next_scale", "skipped", "scale_changed")
    for idx, (l_row, r_row) in enumerate(zip(l_steps, r_steps, strict=False)):
        if any(l_row.get(k) != r_row.get(k) for k in keys):
            return {
                "index": idx,
                "left_step": l_row.get("step"),
                "right_step": r_row.get("step"),
                "left": {k: l_row.get(k) for k in keys},
                "right": {k: r_row.get(k) for k in keys},
            }
    if len(l_steps) != len(r_steps):
        return {"length_mismatch": [len(l_steps), len(r_steps)]}
    return None


def _run_variant(
    variant: str,
    *,
    run_fn,
    base_cfg: dict[str, Any],
    targets: list[str],
    position: str,
    seed: int,
    trace_dir: Path,
    attn_only: bool,
    common_env: dict[str, str],
    fixed_variant_env: dict[str, str],
) -> dict[str, Any]:
    trace_path = trace_dir / f"{position.lower()}_{variant}_seed{seed}.jsonl"
    variant_env = VARIANT_ENVS[variant]
    is_fixed_scale = variant_env.get(FIXED_SCALE_ENV) == "1"
    env = {
        **common_env,
        **variant_env,
        **(fixed_variant_env if is_fixed_scale else {}),
        TRACE_ENV: str(trace_path),
        TRACE_LABEL_ENV: variant,
        "FF_DEVICE": "cuda",
    }
    cfg = _make_attention_cfg(base_cfg, attn_only=attn_only)

    print(f"\n{'=' * 78}")
    print(f"{position} {variant} seed={seed}")
    print(f"{'=' * 78}")
    t0 = time.perf_counter()
    try:
        with _patched_env(env):
            result = run_fn(seed=seed, config=cfg)
    except Exception as exc:  # noqa: BLE001 - analysis variant should not hide sibling results
        elapsed = time.perf_counter() - t0
        trace = summarize_trace(trace_path)
        row = {
            "variant": variant,
            "seed": seed,
            "elapsed_sec": round(elapsed, 3),
            "error": repr(exc),
            "trace": trace,
        }
        print(f"{variant}: FAILED after {elapsed:.1f}s steps={trace['n_steps']} error={exc!r}")
        return row
    elapsed = time.perf_counter() - t0

    metrics = _metric_payload(result, targets)
    trace = summarize_trace(trace_path)
    row = {
        "variant": variant,
        "seed": seed,
        "elapsed_sec": round(elapsed, 3),
        "phase_seconds": result.get("phase_seconds", {}),
        "metrics": metrics,
        "trace": trace,
    }
    print(
        f"{variant}: attn total MAE={metrics['total_mae']:.6f} "
        f"steps={trace['n_steps']} first_scale_change={trace['first_scale_change_step']} "
        f"first_skip={trace['first_skip_step']} elapsed={elapsed:.1f}s"
    )
    return row


def _fmt_delta(a: float, b: float) -> str:
    return f"{b - a:+.6f}"


def print_summary(rows: list[dict[str, Any]]) -> None:
    by = {r["variant"]: r for r in rows}
    print(f"\n{'=' * 78}")
    print("CUDA graph / GradScaler analysis summary")
    print(f"{'=' * 78}")
    print(f"{'variant':<18}{'MAE':>12}{'Δ vs graph_a':>16}{'steps':>9}{'scale changes':>15}")
    base = by.get("graph_a")
    base_mae = base["metrics"]["total_mae"] if base and "metrics" in base else None
    for row in rows:
        if "metrics" not in row:
            print(
                f"{row['variant']:<18}{'FAILED':>12}{'n/a':>16}"
                f"{row['trace']['n_steps']:>9}{row['trace']['n_scale_changes']:>15}"
            )
            print(f"  error: {row.get('error')}")
            continue
        mae = row["metrics"]["total_mae"]
        delta = _fmt_delta(base_mae, mae) if base_mae is not None else "n/a"
        print(
            f"{row['variant']:<18}{mae:>12.6f}{delta:>16}"
            f"{row['trace']['n_steps']:>9}{row['trace']['n_scale_changes']:>15}"
        )

    if {"graph_a", "graph_b"}.issubset(by) and "metrics" in by["graph_a"]:
        g0 = by["graph_a"]["metrics"]
        g1 = by["graph_b"].get("metrics")
        metric_equal = bool(g1 and g0["metrics_sha256"] == g1["metrics_sha256"])
        trace_diff = first_trace_diff(
            Path(by["graph_a"]["trace"]["path"]),
            Path(by["graph_b"]["trace"]["path"]),
        )
        print("\nGraph-vs-graph sentinel:")
        print(f"  metrics identical: {metric_equal}")
        print(f"  first scale-schedule diff: {trace_diff or 'none'}")

    if (
        {"eager", "graph_a"}.issubset(by)
        and "metrics" in by["eager"]
        and "metrics" in by["graph_a"]
    ):
        trace_diff = first_trace_diff(
            Path(by["eager"]["trace"]["path"]),
            Path(by["graph_a"]["trace"]["path"]),
        )
        print("\nEager-vs-graph:")
        print(
            "  total MAE delta graph-eager: "
            f"{by['graph_a']['metrics']['total_mae'] - by['eager']['metrics']['total_mae']:+.6f}"
        )
        print(f"  first scale-schedule diff: {trace_diff or 'none'}")

    if (
        {"graph_a", "graph_restore_bn"}.issubset(by)
        and "metrics" in by["graph_a"]
        and "metrics" in by["graph_restore_bn"]
    ):
        print("\nBN warmup snapshot/restore:")
        print(
            "  total MAE delta restore-stock_graph: "
            f"{by['graph_restore_bn']['metrics']['total_mae'] - by['graph_a']['metrics']['total_mae']:+.6f}"
        )

    metric_rows = [r for r in rows if "metrics" in r]
    if len(metric_rows) > 1:
        vals = [r["metrics"]["total_mae"] for r in metric_rows]
        print(f"\nMAE range across variants: {min(vals):.6f} - {max(vals):.6f}")
        if len(vals) > 2:
            print(f"MAE sample std across variants: {statistics.stdev(vals):.6f}")


def _write_history(
    rows: list[dict[str, Any]],
    *,
    position: str,
    seed: int,
    settings: dict[str, Any],
) -> str:
    now = utc_now_iso()
    git_hash = get_git_hash()
    entry = {
        "run_id": f"{now}_{git_hash}_{ANALYSIS_NAME}_{position.lower()}",
        "timestamp": now,
        "git_hash": git_hash,
        "kind": "analysis",
        "name": ANALYSIS_NAME,
        "position": position,
        "seed": seed,
        "settings": settings,
        "variants": rows,
    }
    return append_to_history(str(HISTORY_DIR), entry)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--position", default="RB", help="Position to run (default: RB)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANT_ENVS),
        default=None,
        help="Variants to run. Default: graph_a graph_b eager graph_restore_bn.",
    )
    parser.add_argument(
        "--include-fixed-scale",
        action="store_true",
        help=(
            "Append graph_fixed_scale, graph_fixed_scale_restore_bn, "
            "and eager_fixed_scale to the default variant set."
        ),
    )
    parser.add_argument(
        "--fixed-scale-init",
        type=float,
        default=None,
        help="Set FF_AMP_INIT_SCALE for fixed-scale variants only.",
    )
    parser.add_argument(
        "--attn-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip Ridge/base-NN/LightGBM and train only the attention NN (default: true).",
    )
    parser.add_argument(
        "--dropout-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set FF_FORCE_DROPOUT_ZERO=1 to isolate graph/scaler drift (default: true).",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set FF_DETERMINISTIC=1 to disable CUDA autotuner/TF32 (default: true).",
    )
    parser.add_argument(
        "--fixed-epochs",
        type=int,
        default=0,
        help="Set FF_NN_FIXED_EPOCHS=N for every variant; 0 keeps normal early stopping.",
    )
    parser.add_argument("--no-history", action="store_true", help="Do not write history JSON.")
    args = parser.parse_args()

    variants = args.variants or list(DEFAULT_VARIANTS)
    if args.include_fixed_scale:
        for variant in ("graph_fixed_scale", "graph_fixed_scale_restore_bn", "eager_fixed_scale"):
            if variant not in variants:
                variants.append(variant)

    common_env: dict[str, str] = {}
    if args.dropout_zero:
        common_env["FF_FORCE_DROPOUT_ZERO"] = "1"
    if args.deterministic:
        common_env["FF_DETERMINISTIC"] = "1"
    if args.fixed_epochs > 0:
        common_env["FF_NN_FIXED_EPOCHS"] = str(args.fixed_epochs)
    fixed_variant_env: dict[str, str] = {}
    if args.fixed_scale_init is not None:
        fixed_variant_env[INIT_SCALE_ENV] = str(args.fixed_scale_init)

    position = args.position.upper()
    mod = importlib.import_module(f"src.{position.lower()}.run_pipeline")
    base_cfg, run_fn = mod.CONFIG, mod.run
    targets = base_cfg["targets"]

    run_id = f"{utc_now_iso().replace(':', '-')}_{get_git_hash()}_{position.lower()}"
    trace_dir = TRACE_DIR / run_id
    trace_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _run_variant(
            variant,
            run_fn=run_fn,
            base_cfg=base_cfg,
            targets=targets,
            position=position,
            seed=args.seed,
            trace_dir=trace_dir,
            attn_only=args.attn_only,
            common_env=common_env,
            fixed_variant_env=fixed_variant_env,
        )
        for variant in variants
    ]
    print_summary(rows)

    settings = {
        "variants": variants,
        "attn_only": args.attn_only,
        "dropout_zero": args.dropout_zero,
        "deterministic": args.deterministic,
        "fixed_epochs": args.fixed_epochs,
        "fixed_scale_init": args.fixed_scale_init,
        "trace_dir": str(trace_dir),
    }
    if not args.no_history:
        _write_history(rows, position=position, seed=args.seed, settings=settings)


if __name__ == "__main__":
    main()
