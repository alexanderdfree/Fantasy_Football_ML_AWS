"""Ablation: the seven default-OFF attention-architecture extensions (PRs #109–121).

On 2026-04-23 a coordinated 7-PR series landed opt-in attention extensions into
``src/shared/neural_net.py``. Each ships behind a config flag that defaults OFF
(zero-init / no-op so the baseline is byte-identical when disabled) with full
unit tests — but **no position config enables any of them**, so none has ever
been run through the real pipeline or benchmarked. They are tested scaffolding,
not evaluated features. This script settles "which are actually worth turning
on" empirically, the way every other model change in this project is judged:
the real position pipeline at production config, multi-seed, with a Ridge
data-identity sentinel and a mean±std decision table.

The seven flags (each is read by the attention factory in neural_net.py via
``cfg.get(...)`` at both the flat ``:1225-1242`` and nested-K ``:1278-1291`` call
sites — so toggling one cfg key is sufficient to run it; no plumbing change):

    PR #109  attn_learn_temperature             per-target learned softmax temp
    PR #112  attn_history_dropout               drop whole games from history
    PR #115  attn_use_swiglu_encoder            SwiGLU game encoder (only bites
                                                when attn_encoder_hidden_dim > 0)
    PR #116  attn_entropy_coeff                 attention entropy regulariser
    PR #117  attn_use_alibi_bias                ALiBi time-decay positional bias
    PR #120  attn_self_layers                   Pre-LN self-attention block
    PR #121  attn_condition_queries_on_static   opponent/context-conditioned query

Prior (from theory + this project's stop-rules — confirm against the table, do
not pre-judge):
  * Tier 1 (domain-motivated): alibi — recency dominates fantasy output; a
    monotonic games-ago decay is a strong, parameter-free prior. It overlaps the
    default-ON positional encoding, so it is tested both stacked on PE (``alibi``)
    and replacing PE (``alibi_only``).
  * Tier 2 (cheap regularisers, small-N positions benefit most): seqdrop, temp,
    swiglu.
  * Tier 3 (needs tuning / redundancy check): entropy (coeff needs a sweep;
    better as a tune_nn axis), condq (likely redundant with the opp-defense
    attention branch).
  * Tier 4 (skeptical — collides with the "larger regressed on 15K-sample
    positions" stop-rule): selfattn. Expect a regression; one run to confirm,
    then delete as dead code.

Method (mirrors ablate_backbone_norm.py):
  * Baseline = all extensions OFF (production attention) — the never-benchmarked
    reference; always run so every flag gets a paired per-seed delta.
  * Each variant deep-copies the production config and toggles exactly one flag
    (``alibi_only`` also turns PE off). NN stays at PRODUCTION values — no
    epoch/data shrink; a reduced proxy can flip the sign of a small effect.
  * LightGBM is disabled (``train_lightgbm=False``) — it is attention-free, so
    dropping it saves wall-clock without touching the measured quantity. Ridge
    and the base NN stay ON (disabling them makes run_pipeline bail to a minimal
    result that drops the metrics we read).
  * Ridge is attention-free → its FP MAE MUST be identical across variants for a
    given seed. That equality is the data-identity sentinel; a mismatch means the
    runs did not see the same data/seed and the deltas are meaningless.

STACKED PORT: for the GPU-default N=24 stacked-seed regime use the ab_harness spec
``src.tuning.ab_attn_arch`` (``python -m src.tuning.launch_ab --spec src.tuning.ab_attn_arch``).
This eager script stays for the per-target attention-MAE decision table and the ``entropy`` arm,
both dropped from the stacked port (per-target arrays aren't surfaced under stacking, and
``attn_entropy_coeff`` is a vmap side-channel reject).

Usage:
    python -m src.tuning.ablate_attn_arch                          # RB, seed 42, all flags
    python -m src.tuning.ablate_attn_arch --seeds 42,7,123,5       # multi-seed (>=8 advised)
    python -m src.tuning.ablate_attn_arch --variants alibi         # baseline + one flag
    python -m src.tuning.ablate_attn_arch --variants alibi,seqdrop,temp,swiglu  # subset
    python -m src.tuning.ablate_attn_arch --position QB            # different position
    python -m src.tuning.ablate_attn_arch --dry-run                # show plan, no training
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
from typing import Any

from src.shared.registry import get_config, get_runner
from src.tuning.ablation_runner import (
    AblationJob,
    AblationResult,
    fmt_mean_std,
    format_dry_run_table,
    parse_seed_list,
    resolve_max_workers,
    run_grid,
    select_variants,
    write_history,
)

ABLATION_NAME = "attn_arch"
DEFAULT_LOG_DIR = f"logs/ablations/{ABLATION_NAME}"
DEFAULT_POSITION = "RB"

# A mean Δ FP MAE smaller than this (or within seed noise) counts as "no
# meaningful difference" — ~0.5% of a typical ~4 pt/game baseline.
FLAT_NOISE_THRESHOLD = 0.02

BASELINE = "baseline"

# variant key -> (human label, {cfg key: value} override applied on top of the
# production config). The baseline override is empty — it is the all-OFF
# reference. Floats use a real value (a flag like attn_history_dropout is a
# probability/coeff, not a bool); attn_self_layers is an int count.
VARIANTS: dict[str, tuple[str, dict]] = {
    BASELINE: ("All extensions OFF (production attention)", {}),
    "temp": ("#109 per-target learned softmax temperature", {"attn_learn_temperature": True}),
    "seqdrop": ("#112 game-history sequence dropout (p=0.1)", {"attn_history_dropout": 0.1}),
    "swiglu": ("#115 SwiGLU game encoder", {"attn_use_swiglu_encoder": True}),
    "entropy": ("#116 attention entropy regulariser (coeff=0.01)", {"attn_entropy_coeff": 0.01}),
    "alibi": (
        "#117 ALiBi time-decay bias (stacked on positional enc.)",
        {"attn_use_alibi_bias": True},
    ),
    "alibi_only": (
        "#117 ALiBi replacing positional encoding",
        {"attn_use_alibi_bias": True, "attn_positional_encoding": False},
    ),
    "selfattn": ("#120 Pre-LN self-attention block (1 layer)", {"attn_self_layers": 1}),
    "condq": ("#121 context-conditioned queries", {"attn_condition_queries_on_static": True}),
}
FLAG_VARIANTS = [k for k in VARIANTS if k != BASELINE]

# Default variant set for --variants "all" and dry-run: baseline + all flags.
DEFAULT_VARIANTS = [BASELINE] + FLAG_VARIANTS

# Every cfg key any variant is allowed to touch. Guards against a typo silently
# producing a no-op variant (the failure mode the project explicitly warns about
# — an unread cfg key falls through to its OFF default with no error).
KNOWN_FLAG_KEYS = {
    "attn_learn_temperature",
    "attn_history_dropout",
    "attn_use_swiglu_encoder",
    "attn_entropy_coeff",
    "attn_use_alibi_bias",
    "attn_positional_encoding",
    "attn_self_layers",
    "attn_condition_queries_on_static",
}


def _make_cfg(base_cfg: dict, overrides: dict) -> dict:
    """Deep-copy the production config, disable LightGBM, apply the variant flag(s).

    LightGBM is attention-free and invariant to these flags, so dropping it saves
    wall-clock without touching the measured quantity. Everything else stays at
    production values — no epoch/data shrink (a reduced proxy can flip the sign of
    a small effect).
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["train_lightgbm"] = False
    cfg.update(overrides)
    return cfg


def _extract_run_payload(
    result: dict[str, Any],
    *,
    targets: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Extract metrics/timings from a pipeline result dict into an AblationJob payload."""
    attn = result.get("attn_nn_metrics")
    base = result.get("nn_metrics")
    ridge = result.get("ridge_metrics")
    if attn is None or base is None or ridge is None:
        raise RuntimeError(
            f"missing metrics (attn/base/ridge) in result keys {sorted(result.keys())}"
        )
    metrics = {
        "attn_fp_mae": float(attn["total"]["mae"]),
        "base_fp_mae": float(base["total"]["mae"]),
        "ridge_fp_mae": float(ridge["total"]["mae"]),
        "attn_targets": {t: float(attn[t]["mae"]) for t in targets},
    }
    timings: dict[str, Any] = {}
    return {"metrics": metrics, "timings": timings, "metadata": metadata}


def _execute_attn_arch_job(job: AblationJob) -> dict[str, Any]:
    """Run one attn-arch ablation job and return a metrics/timings/metadata payload."""
    label, overrides = VARIANTS[job.variant]
    cfg = _make_cfg(job.base_cfg, overrides)
    run_fn = get_runner(job.position)
    result = run_fn(seed=job.seed, config=cfg)
    run_metadata = {"variant_label": label, "overrides": dict(overrides), **job.metadata}
    return _extract_run_payload(result, targets=cfg["targets"], metadata=run_metadata)


def _build_jobs(
    *,
    position: str,
    seeds: list[int],
    variants: list[str],
) -> list[AblationJob]:
    base_cfg = get_config(position)
    jobs: list[AblationJob] = []
    for seed in seeds:
        for variant_key in variants:
            label, _ = VARIANTS[variant_key]
            jobs.append(
                AblationJob(
                    position=position,
                    seed=seed,
                    variant=variant_key,
                    label=label,
                    run_fn=_execute_attn_arch_job,
                    base_cfg=base_cfg,
                    metadata={"run_kind": "experiment"},
                )
            )
    return jobs


# ---------------------------------------------------------------------------
# Summary / verdict helpers — these operate on raw dicts (as before) so
# they are callable from tests without constructing AblationResult objects.
# ---------------------------------------------------------------------------


def _vals(by: dict, variant: str, seeds: list[int], key: str) -> list[float]:
    return [by[(variant, s)]["metrics"][key] for s in seeds if (variant, s) in by]


def _paired_deltas(by: dict, variant: str, seeds: list[int], key: str) -> list[float]:
    """Per-seed (variant - baseline) for seeds where both ran — a paired delta
    (same seed, same data) has far less variance than comparing the two means."""
    return [
        by[(variant, s)]["metrics"][key] - by[(BASELINE, s)]["metrics"][key]
        for s in seeds
        if (variant, s) in by and (BASELINE, s) in by
    ]


def print_summary(
    results: list[AblationResult], targets: list[str], variants_run: list[str]
) -> bool:
    """Print the decision table. Returns the data-identity sentinel result."""
    # Build a lookup: (variant, seed) -> flat metrics dict (preserving the old dict shape
    # for the helper functions above).
    rows = [
        {"variant": r.variant, "seed": r.seed, "metrics": r.metrics}
        for r in results
        if r.error is None
    ]
    seeds = sorted({r["seed"] for r in rows})
    by = {(r["variant"], r["seed"]): r for r in rows}
    flags = [v for v in variants_run if v != BASELINE]
    have_baseline = bool(seeds) and all((BASELINE, s) in by for s in seeds)

    print(f"\n{'=' * 90}")
    print("Attention-architecture ablation — seven default-OFF extensions (PRs #109–121)")
    print(f"seeds: {seeds}   variants: {variants_run}")
    print(f"{'=' * 90}")

    # --- Headline: attention-NN FP MAE per variant + paired Δ vs baseline ---
    print("\nAttention NN — FP MAE (mean±std across seeds), Δ vs baseline (paired):")
    print(f"  {'variant':<12}{'FP MAE':>16}{'Δ vs baseline':>20}  label")
    print("  " + "-" * 86)
    for v in variants_run:
        abs_str = fmt_mean_std(_vals(by, v, seeds, "attn_fp_mae"))
        if v == BASELINE:
            delta_str = "—"
        else:
            delta_str = fmt_mean_std(_paired_deltas(by, v, seeds, "attn_fp_mae"))
        print(f"  {v:<12}{abs_str:>16}{delta_str:>20}  {VARIANTS[v][0]}")

    # --- Base MLP NN (attention-free) — should also be ~flat across variants ---
    print("\nBase NN — FP MAE (mean±std):  ", end="")
    print(
        "  ".join(f"{v}={fmt_mean_std(_vals(by, v, seeds, 'base_fp_mae'))}" for v in variants_run)
    )

    # --- Data-identity sentinel: Ridge is attention-free → identical per seed ---
    print("\nData-identity sentinel (Ridge FP MAE — must be identical across variants):")
    sentinel_ok = bool(seeds)
    for s in seeds:
        vals = [by[(v, s)]["metrics"]["ridge_fp_mae"] for v in variants_run if (v, s) in by]
        if len(vals) < 2:
            continue
        spread = max(vals) - min(vals)
        ok = spread < 1e-9
        sentinel_ok = sentinel_ok and ok
        flag = "OK" if ok else "*** MISMATCH — variants saw different data/seed ***"
        print(f"  seed {s}: ridge={vals[0]:.6f}  max spread={spread:.2e}  {flag}")

    # --- Per-target attention-NN MAE Δ vs baseline (mean across seeds) ---
    if flags:
        print("\nPer-target attention-NN MAE Δ vs baseline (mean across seeds; - = better):")
        for v in flags:
            parts = []
            for t in targets:
                vv = [
                    by[(v, s)]["metrics"]["attn_targets"][t]
                    for s in seeds
                    if (v, s) in by and t in by[(v, s)]["metrics"]["attn_targets"]
                ]
                bb = [
                    by[(BASELINE, s)]["metrics"]["attn_targets"][t]
                    for s in seeds
                    if (BASELINE, s) in by and t in by[(BASELINE, s)]["metrics"]["attn_targets"]
                ]
                if vv and bb:
                    parts.append(f"{t}={statistics.mean(vv) - statistics.mean(bb):+.4f}")
            if parts:
                print(f"  {v:<12} " + "  ".join(parts))

    _verdict(seeds, by, flags, sentinel_ok, have_baseline)
    return sentinel_ok


def _verdict(
    seeds: list[int], by: dict, flags: list[str], sentinel_ok: bool, have_baseline: bool
) -> None:
    print(f"\n{'-' * 90}")
    if not seeds or not flags or not have_baseline:
        print(
            "VERDICT: incomplete — need the baseline and at least one flag variant on shared seeds."
        )
        return
    if not sentinel_ok:
        print(
            "VERDICT: SENTINEL FAILED — Ridge MAE differs across variants, so this is not a clean "
            "comparison. Fix data/seed handling before trusting any flag delta."
        )
        return

    # Rank flags by mean paired Δ FP MAE (most negative = biggest improvement).
    scored = []
    for v in flags:
        deltas = _paired_deltas(by, v, seeds, "attn_fp_mae")
        if deltas:
            mean_d = statistics.mean(deltas)
            sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
            scored.append((v, mean_d, sd, len(deltas)))
    scored.sort(key=lambda x: x[1])

    print(f"VERDICT (attention NN FP MAE, headline) over {len(seeds)} seed(s):")
    single_seed = len(seeds) < 2
    for v, mean_d, sd, _n in scored:
        if abs(mean_d) < FLAT_NOISE_THRESHOLD or abs(mean_d) <= sd:
            tag = "FLAT (within noise)"
        elif mean_d < 0:
            tag = f"PROMISING — improves {abs(mean_d):.4f} (> noise {sd:.4f})"
        else:
            tag = f"REGRESSION — worsens {mean_d:.4f} (> noise {sd:.4f})"
        print(f"  {v:<12} Δ={mean_d:+.4f}±{sd:.4f}  {tag}")
    if single_seed:
        print(
            "  Directional only — single seed. Re-run with ≥8 seeds (project floor for small NN "
            "deltas) before promoting any flag; judge by mean±std, never one seed."
        )
    else:
        promising = [
            v for v, m, sd, _ in scored if m < 0 and abs(m) >= FLAT_NOISE_THRESHOLD and abs(m) > sd
        ]
        if promising:
            print(
                f"  Next: confirm {promising} on a 2nd position (QB = small-sample/overfit stress "
                "test), then enable in that POSITION_CONFIG + update ADR-0004 (accepts a retrain)."
            )
        else:
            print(
                "  No flag beats baseline beyond seed noise — leave them OFF (or delete the dead "
                "code) and document this result."
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--position", default=DEFAULT_POSITION, help="Position to ablate (default: RB)"
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated seeds (default: 42; ≥8 advised for a verdict)",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help=(
            "Comma-separated subset of variants (baseline always included), or 'all'. "
            f"Choices: {sorted(VARIANTS)}. Default: all."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without training")
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Skip writing results to benchmark_history/ablations/",
    )
    parser.add_argument(
        "--max-workers",
        default="auto",
        help=(
            "Process workers. Use an integer or 'auto' "
            "(default: local many-core CUDA -> 6, otherwise 1; pass 1 for clean timing)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=f"Directory for per-job logs (default: {DEFAULT_LOG_DIR})",
    )
    args = parser.parse_args(argv)

    try:
        seeds = parse_seed_list(args.seeds)
        variants = select_variants(args.variants, VARIANTS, DEFAULT_VARIANTS)
    except ValueError as exc:
        parser.error(str(exc))

    if BASELINE not in variants:
        # Ensure baseline is always present for paired deltas.
        variants = [BASELINE] + [v for v in variants if v != BASELINE]

    position = args.position.upper()
    jobs = _build_jobs(position=position, seeds=seeds, variants=variants)

    try:
        max_workers = resolve_max_workers(args.max_workers, job_count=len(jobs))
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(format_dry_run_table(jobs))
        print(f"\nExperiment workers: {max_workers} ({args.max_workers})")
        print(f"Job logs: {args.log_dir}")
        return

    results = run_grid(jobs, max_workers=max_workers, log_dir=args.log_dir, progress=True)

    # Targets come from the position's base config (same across variants).
    base_cfg = get_config(position)
    targets = base_cfg["targets"]
    print_summary(results, targets, variants)

    if not args.no_history:
        write_history(
            ABLATION_NAME,
            results,
            metadata={
                "position": position,
                "seeds": seeds,
                "variants_run": variants,
                "max_workers": max_workers,
                "log_dir": args.log_dir,
            },
        )

    errors = [r for r in results if r.error]
    if errors:
        for r in errors:
            print(f"ERROR {r.position} seed={r.seed} variant={r.variant}: {r.error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
