"""Ablation / A-B: which LR-scheduler *type* is best per position?

Three scheduler types are wired into ``src/shared/pipeline.py::_build_scheduler``:
``onecycle`` (OneCycleLR, stepped per-batch), ``cosine_warm_restarts``
(CosineAnnealingWarmRestarts, stepped per-epoch), and ``plateau``
(ReduceLROnPlateau, stepped per-epoch on the val metric). Production wires only
two of them through ``PositionConfig``: QB/RB/WR/DST use
``cosine_warm_restarts`` and TE/K use ``onecycle``. ``plateau`` is fully
supported by the training loop (``src/shared/training.py`` steps it on
``avg_val_loss``) but has no ``PositionConfig`` fields, so no position can
currently *select* it. This script settles the type choice empirically.

For each (position, scheduler_type) it runs the REAL position pipeline once under
a fixed seed and identical splits, swapping only ``scheduler_type`` (+ that
type's hyperparameters) and prints a side-by-side decision table.

The scheduler is the ONLY thing that differs between a position's variants:
  * Ridge / ElasticNet are scheduler-free, so their FP MAE MUST be identical
    across variants for a given seed — asserted as a *data-identity sentinel*
    (a mismatch means the variants did not see the same data/seed, so the NN
    deltas are meaningless).
  * LightGBM is scheduler-free too, so it is disabled (``train_lightgbm=False``)
    to save wall-clock without affecting the measured quantity (the NN deltas).
  * NN configs stay at PRODUCTION values — no epoch/data shrink. A reduced proxy
    can flip the sign of a small effect (see project memory).

Both the base MLP NN (``nn_metrics``) and the production attention NN
(``attn_nn_metrics``) build their optimizer/scheduler through the same
``_build_scheduler``, so deltas are reported for both; the attention NN is the
headline.

Hyperparameters per type are ASYMMETRIC by design — the comparison asks "should
this position switch *type*?", so the production type keeps its tuned settings
and the alternatives use canonical settings:
  * The variant matching the position's PRODUCTION ``scheduler_type`` uses that
    position's own tuned hyperparameters (already in ``CONFIG``).
  * ``cosine_warm_restarts`` (canonical): T_0=40, T_mult=2, eta_min=1e-5 — the
    QB/RB/WR production setting.
  * ``onecycle`` (canonical): max_lr = 4 x nn_lr (the TE production ratio),
    pct_start=0.3.
  * ``plateau`` (always canonical — no production position selects it):
    factor=0.5, patience=8 (several reductions fit inside the 25-35 early-stop
    window).

Read the results as: an UNTUNED alternative beating the tuned production type is
STRONG evidence to switch; the production type winning is WEAK evidence (it had
the tuned home-field advantage). Single-seed NN MAE is noisy — the project
default is >=3 seeds, reported mean±std; treat a delta inside the seed band as
flat.

CUDA graphs (autodetect-ON for sm_80+, PR #874/#889) are NOT numerically inert
(~0.5% worst-target trajectory drift). For a bit-comparable eager A/B set
``FF_CUDA_GRAPH=0``; the Batch launcher (``launch_ablate_scheduler.py``) does
this so the scheduler deltas are not polluted by graph-capture non-determinism.

STACKED PORT: for the GPU-default N=24 stacked-seed regime use the ab_harness spec
``src.tuning.ab_scheduler_type`` (``python -m src.tuning.launch_ab --spec
src.tuning.ab_scheduler_type``); it compares onecycle vs cosine for the attention NN with
canonical params on both arms. This eager script stays for the ``plateau`` arm (dropped from the
stacked port — ``train_stacked`` rejects ``ReduceLROnPlateau``), the cross-position rollup, and
the ``src/batch/train.py --ablation scheduler-type`` entrypoint.

Usage (local):
    python -m src.tuning.ablate_scheduler_type                      # all six, seed 42
    python -m src.tuning.ablate_scheduler_type --positions k dst    # subset
    python -m src.tuning.ablate_scheduler_type --only onecycle cosine_warm_restarts
    python -m src.tuning.ablate_scheduler_type --seeds 42,43,44     # multi-seed
    python -m src.tuning.ablate_scheduler_type --dry-run            # print the plan

In the cloud this module is driven per-position by
``src/batch/train.py --ablation scheduler-type`` (one Spot job per position),
which calls :func:`run_position` with the container's pre-loaded frames and
uploads the returned dict to ``s3://$S3_BUCKET/ablate_scheduler/{pos}/result.json``.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.shared.benchmark_utils import (  # noqa: E402
    append_to_history,
    get_git_hash,
    utc_now_iso,
)

ABLATION_NAME = "scheduler_type"
HISTORY_DIR = "benchmark_history"
DEFAULT_POSITIONS = ["qb", "rb", "wr", "te", "k", "dst"]
SCHEDULERS = ("onecycle", "cosine_warm_restarts", "plateau")
SHORT = {"onecycle": "1cycle", "cosine_warm_restarts": "cosine", "plateau": "plateau"}

# Scheduler-type-specific cfg keys (read the production params; inject the
# swapped-in type's params cleanly).
_TYPE_KEYS = {
    "cosine_warm_restarts": ("cosine_t0", "cosine_t_mult", "cosine_eta_min"),
    "onecycle": ("onecycle_max_lr", "onecycle_pct_start"),
    "plateau": ("plateau_factor", "plateau_patience"),
}

# A mean Δ FP MAE smaller than this (or within seed noise) counts as "no
# meaningful difference" — ~0.5% of a typical ~4 pt/game baseline.
FLAT_NOISE_THRESHOLD = 0.02


def _canonical_params(cfg: dict, sched_type: str) -> dict:
    """Canonical hyperparameters for a scheduler type not native to this position.

    ``onecycle`` peaks at 4x the position's base ``nn_lr`` (the TE production
    ratio); the other two carry the family-standard settings the tuned positions
    use. These are deliberately NOT per-position-tuned — see the module docstring
    on reading an untuned-alternative win vs a tuned-production win.
    """
    if sched_type == "cosine_warm_restarts":
        return {"cosine_t0": 40, "cosine_t_mult": 2, "cosine_eta_min": 1e-5}
    if sched_type == "onecycle":
        return {"onecycle_max_lr": 4.0 * float(cfg["nn_lr"]), "onecycle_pct_start": 0.3}
    if sched_type == "plateau":
        return {"plateau_factor": 0.5, "plateau_patience": 8}
    raise ValueError(f"unknown scheduler type: {sched_type}")


def _scheduler_params(base_cfg: dict, sched_type: str) -> dict:
    """Params for ``sched_type``: the position's tuned set if it is production, else canonical.

    ``base_cfg`` must be the ORIGINAL production config (its ``scheduler_type``
    still names the production choice) — never a config already overwritten with
    the swapped-in type.
    """
    if sched_type == base_cfg.get("scheduler_type"):
        present = {k: base_cfg[k] for k in _TYPE_KEYS[sched_type] if k in base_cfg}
        if len(present) == len(_TYPE_KEYS[sched_type]):
            return present
        # Named as production but params absent (shouldn't happen for
        # cosine/onecycle; plateau is never production) — fall back to canonical.
    return _canonical_params(base_cfg, sched_type)


def _make_cfg(base_cfg: dict, sched_type: str) -> dict:
    """Production config with LightGBM off and the scheduler swapped to ``sched_type``."""
    params = _scheduler_params(base_cfg, sched_type)  # read BEFORE overwriting type
    cfg = copy.deepcopy(base_cfg)
    cfg["train_lightgbm"] = False  # scheduler-free; skip to save wall-clock
    # Drop any stale other-type keys so the active config is unambiguous in the
    # JSON history (``_build_scheduler`` only reads the active type's keys).
    for keys in _TYPE_KEYS.values():
        for k in keys:
            cfg.pop(k, None)
    cfg["scheduler_type"] = sched_type
    cfg.update(params)
    return cfg


def _load_position(pos: str):
    mod = importlib.import_module(f"src.{pos.lower()}.run_pipeline")
    return mod.run, mod.CONFIG


def _extract(
    result: dict, position: str, sched_type: str, seed: int, targets, is_prod: bool, params: dict
) -> dict:
    attn = result.get("attn_nn_metrics")
    base = result.get("nn_metrics")
    ridge = result.get("ridge_metrics")
    if attn is None or base is None or ridge is None:
        raise RuntimeError(
            f"{position}/{sched_type}: missing metrics (attn/base/ridge) in result keys "
            f"{sorted(result.keys())}"
        )
    return {
        "position": position.upper(),
        "scheduler_type": sched_type,
        "is_production": is_prod,
        "params": {k: float(v) for k, v in params.items()},
        "seed": seed,
        "attn_fp_mae": float(attn["total"]["mae"]),
        "base_fp_mae": float(base["total"]["mae"]),
        "ridge_fp_mae": float(ridge["total"]["mae"]),
        "attn_targets": {t: float(attn[t]["mae"]) for t in targets},
    }


def run_variant(
    position: str,
    sched_type: str,
    seed: int,
    run_fn,
    base_cfg: dict,
    targets,
    frames: tuple | None = None,
) -> dict:
    """Run one (position, scheduler_type, seed) pipeline variant and extract metrics.

    ``frames`` is an optional ``(train_df, val_df, test_df)`` tuple. The Batch
    container passes its pre-downloaded frames to avoid a re-read per variant;
    K/DST (self-contained loaders) and local runs pass ``None``.
    """
    is_prod = sched_type == base_cfg.get("scheduler_type")
    params = _scheduler_params(base_cfg, sched_type)
    cfg = _make_cfg(base_cfg, sched_type)
    tag = "  [PRODUCTION]" if is_prod else "  [alt / canonical params]"
    print(f"\n{'=' * 76}")
    print(f"{position.upper()} / {sched_type} (seed {seed}){tag}")
    print(f"  params={params}")
    print(f"{'=' * 76}", flush=True)
    if frames is not None:
        result = run_fn(*frames, seed=seed, config=cfg)
    else:
        result = run_fn(seed=seed, config=cfg)
    return _extract(result, position, sched_type, seed, targets, is_prod, params)


# --------------------------------------------------------------------------- #
# Aggregation + reporting
# --------------------------------------------------------------------------- #
def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), 0.0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def _fmt(mean: float, sd: float, multi: bool) -> str:
    return f"{mean:.4f}±{sd:.4f}" if multi else f"{mean:.4f}"


def _sentinel_ok(pos_rows: list[dict]) -> tuple[bool, list[str]]:
    """Ridge is scheduler-free → must match across scheduler types within each seed."""
    msgs: list[str] = []
    ok = True
    for s in sorted({r["seed"] for r in pos_rows}):
        vals = [r["ridge_fp_mae"] for r in pos_rows if r["seed"] == s]
        if len(vals) < 2:
            continue
        spread = max(vals) - min(vals)
        good = spread < 1e-9
        ok = ok and good
        flag = "OK" if good else "*** MISMATCH — variants saw different data/seed ***"
        msgs.append(
            f"  seed {s}: ridge_fp_mae spread={spread:.2e} over {len(vals)} variants {flag}"
        )
    return ok, msgs


def print_position_summary(position: str, pos_rows: list[dict], prod_type: str, targets) -> dict:
    scheds = [s for s in SCHEDULERS if any(r["scheduler_type"] == s for r in pos_rows)]
    seeds = sorted({r["seed"] for r in pos_rows})
    multi = len(seeds) > 1

    agg: dict[str, dict] = {}
    for s in scheds:
        srows = [r for r in pos_rows if r["scheduler_type"] == s]
        agg[s] = {
            "attn": _mean_std([r["attn_fp_mae"] for r in srows]),
            "base": _mean_std([r["base_fp_mae"] for r in srows]),
            "targets": {t: _mean_std([r["attn_targets"][t] for r in srows]) for t in targets},
        }

    print(f"\n{'#' * 92}")
    print(
        f"# {position.upper()} — LR-scheduler-type ablation"
        f"{'  (mean±std over seeds)' if multi else f'  (seed {seeds[0]})'}"
        f"   [production: {prod_type}]"
    )
    print(f"{'#' * 92}")

    print(f"\n  {'scheduler':<22}{'attn FP MAE':>18}{'base FP MAE':>18}")
    print("  " + "-" * 56)
    for s in scheds:
        am, asd = agg[s]["attn"]
        bm, bsd = agg[s]["base"]
        mark = "  <- production" if s == prod_type else ""
        print(f"  {s:<22}{_fmt(am, asd, multi):>18}{_fmt(bm, bsd, multi):>18}{mark}")

    print("\n  Data-identity sentinel (Ridge FP MAE — must match across types per seed):")
    sok, smsgs = _sentinel_ok(pos_rows)
    for m in smsgs:
        print(m)

    print("\n  Per-target attention-NN MAE (mean across seeds):")
    head = f"  {'target':<20}" + "".join(f"{SHORT[s]:>12}" for s in scheds)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for t in targets:
        cells = "".join(f"{agg[s]['targets'][t][0]:>12.4f}" for s in scheds)
        print(f"  {t:<20}{cells}")

    verdict = _position_verdict(position, agg, scheds, prod_type, multi, sok)
    return {
        "position": position.upper(),
        "production_type": prod_type,
        "seeds": seeds,
        "sentinel_ok": sok,
        "aggregated": {
            s: {
                "attn_fp_mae_mean": agg[s]["attn"][0],
                "attn_fp_mae_std": agg[s]["attn"][1],
                "base_fp_mae_mean": agg[s]["base"][0],
                "base_fp_mae_std": agg[s]["base"][1],
                "attn_targets_mean": {t: agg[s]["targets"][t][0] for t in targets},
            }
            for s in scheds
        },
        "verdict": verdict,
    }


def _position_verdict(
    position: str, agg: dict, scheds: list[str], prod_type: str, multi: bool, sentinel_ok: bool
) -> dict:
    print(f"\n  {'-' * 88}")
    if not sentinel_ok:
        print(
            "  VERDICT: SENTINEL FAILED — Ridge MAE differs across types; deltas are not a clean "
            "single-variable comparison. Fix data/seed handling before trusting them."
        )
        return {"winner": None, "note": "sentinel_failed"}

    attn_means = {s: agg[s]["attn"][0] for s in scheds}
    winner = min(attn_means, key=attn_means.get)
    prod_mae = attn_means.get(prod_type)
    win_mae = attn_means[winner]
    pooled_sd = (
        (agg[prod_type]["attn"][1] ** 2 + agg[winner]["attn"][1] ** 2) ** 0.5 if multi else 0.0
    )

    if winner == prod_type:
        alts = {s: m for s, m in attn_means.items() if s != prod_type}
        best_alt = min(alts, key=alts.get)
        margin = attn_means[best_alt] - prod_mae  # >0 => production ahead
        print(
            f"  VERDICT ({position.upper()}): production '{prod_type}' is best on attention FP MAE "
            f"({win_mae:.4f}); best alt '{best_alt}' +{margin:.4f}."
        )
        if margin < FLAT_NOISE_THRESHOLD or (multi and margin <= pooled_sd):
            print(
                "    FLAT: alternatives within noise — production not distinguishable, not beaten."
            )
            note = "production_best_flat"
        else:
            print(
                "    Production type leads beyond the flat threshold (alts untuned → weak, but "
                "production is not behind). Keep production."
            )
            note = "production_best"
    else:
        margin = prod_mae - win_mae  # >0 => alt better than production
        print(
            f"  VERDICT ({position.upper()}): alternative '{winner}' ({win_mae:.4f}) beats "
            f"production '{prod_type}' ({prod_mae:.4f}) by {margin:.4f} on attention FP MAE."
        )
        if margin < FLAT_NOISE_THRESHOLD or (multi and margin <= pooled_sd):
            print("    FLAT: gap within noise — not a real lead. Keep production.")
            note = "alt_ahead_flat"
        elif not multi:
            print(
                f"    DIRECTIONAL (single seed): an UNTUNED '{winner}' leading the tuned production "
                f"type is a strong candidate. CONFIRM with >=3 seeds before shipping."
            )
            note = "alt_candidate_single_seed"
        else:
            print(
                f"    '{winner}' leads by {margin:.4f} > pooled σ {pooled_sd:.4f} across seeds — "
                f"strong candidate (and UNTUNED). Tune it, then ship."
            )
            note = "alt_candidate_multiseed"
    return {
        "winner": winner,
        "winner_attn_fp_mae": win_mae,
        "production_attn_fp_mae": prod_mae,
        "margin_vs_production": float(prod_mae - win_mae),
        "pooled_std": float(pooled_sd),
        "note": note,
    }


def run_position(
    position: str,
    seeds: list[int],
    scheds: list[str] | None = None,
    *,
    frames: tuple | None = None,
    run_fn=None,
    base_cfg=None,
) -> dict:
    """Run the full scheduler-type A/B for ONE position and return a JSON-serializable dict.

    Used by both ``main`` (local, ``frames=None``) and the Batch container
    (``src/batch/train.py``, which passes pre-loaded ``frames`` for QB/RB/WR/TE
    and ``None`` for the self-contained K/DST loaders).
    """
    scheds = list(scheds) if scheds else list(SCHEDULERS)
    if run_fn is None or base_cfg is None:
        run_fn, base_cfg = _load_position(position)
    targets = base_cfg["targets"]
    prod_type = base_cfg.get("scheduler_type")
    rows = [
        run_variant(position, sched_type, seed, run_fn, base_cfg, targets, frames=frames)
        for seed in seeds
        for sched_type in scheds
    ]
    summary = print_position_summary(position, rows, prod_type, targets)
    return {"summary": summary, "rows": rows}


def print_cross_position(summaries: list[dict]) -> None:
    print(f"\n\n{'=' * 92}")
    print("CROSS-POSITION ROLLUP — attention-NN FP MAE by scheduler type (lower = better)")
    print(f"{'=' * 92}")
    header = (
        f"  {'position':<10}{'prod':<10}"
        + "".join(f"{SHORT[s]:>12}" for s in SCHEDULERS)
        + f"{'winner':>14}{'Δ vs prod':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    n_switch = 0
    for sm in summaries:
        agg = sm["aggregated"]
        cells = ""
        for s in SCHEDULERS:
            if s in agg:
                mark = "*" if s == sm["production_type"] else " "
                cells += f"{agg[s]['attn_fp_mae_mean']:>11.4f}{mark}"
            else:
                cells += f"{'-':>12}"
        v = sm["verdict"]
        winner = v.get("winner") or "n/a"
        margin = v.get("margin_vs_production")
        mstr = f"{margin:+.4f}" if margin is not None else "n/a"
        switch = winner not in (sm["production_type"], "n/a") and v.get("note", "").startswith(
            "alt_candidate"
        )
        if switch:
            n_switch += 1
        print(
            f"  {sm['position']:<10}{SHORT.get(sm['production_type'], sm['production_type']):<10}"
            f"{cells}{SHORT.get(winner, winner):>14}{mstr:>12}{'  <<' if switch else ''}"
        )
    print("  " + "-" * (len(header) - 2))
    print(
        "  ( * = production type ;  Δ vs prod = MAE(prod) − MAE(winner), + means winner better ;"
        "  << = candidate switch )"
    )
    print(f"\n  Candidate switches flagged: {n_switch}/{len(summaries)}.")
    if n_switch:
        print(
            "  Confirm any candidate switch with >=3 seeds (and tune the winner's params) before "
            "changing POSITION_CONFIG."
        )
    else:
        print("  No production scheduler-type choice is beaten beyond noise — choices stand.")


def _print_position_done(sm: dict, idx: int, total: int) -> None:
    """One-line per-position completion marker (easy to grep / stream live)."""
    v = sm["verdict"]
    winner = SHORT.get(v.get("winner"), v.get("winner") or "n/a")
    margin = v.get("margin_vs_production")
    mstr = f"{margin:+.4f}" if margin is not None else "n/a"
    prod = SHORT.get(sm["production_type"], sm["production_type"])
    flag = "" if v.get("winner") == sm["production_type"] else "  <-- candidate switch"
    sentinel = "ok" if sm.get("sentinel_ok") else "FAILED"
    print(
        f"\n[done] {sm['position']} ({idx}/{total})  prod={prod}  winner={winner}  "
        f"Δvs_prod={mstr}  sentinel={sentinel}{flag}\n",
        flush=True,
    )


def _write_history(summaries: list[dict], rows: list[dict], positions, seeds) -> str:
    now = utc_now_iso()
    git_hash = get_git_hash()
    run_id = f"{now}_{git_hash}_{ABLATION_NAME}"
    entry = {
        "run_id": run_id,
        "timestamp": now,
        "git_hash": git_hash,
        "kind": "ablation",
        "name": ABLATION_NAME,
        "positions": [p.upper() for p in positions],
        "seeds": seeds,
        "schedulers": list(SCHEDULERS),
        "summaries": summaries,
        "rows": rows,
        "note": (
            "Scheduler-TYPE A/B. Production type keeps tuned params; alternatives canonical "
            "(cosine T0=40/Tmult=2/eta_min=1e-5, onecycle max_lr=4*nn_lr/pct_start=0.3, "
            "plateau factor=0.5/patience=8). Ridge is the data-identity sentinel."
        ),
    }
    append_to_history(os.path.join(HISTORY_DIR, "ablations"), entry)
    return run_id


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--positions", nargs="+", default=DEFAULT_POSITIONS, help="positions (default: all six)"
    )
    p.add_argument("--seeds", default="42", help="comma-separated seeds (default: 42)")
    p.add_argument(
        "--only",
        nargs="+",
        choices=SCHEDULERS,
        default=list(SCHEDULERS),
        help="subset of scheduler types to compare (default: all three)",
    )
    p.add_argument("--dry-run", action="store_true", help="print the plan without training")
    p.add_argument("--no-history", action="store_true", help="skip benchmark_history/ablations/")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    scheds = [s for s in SCHEDULERS if s in args.only]
    positions = [p.lower() for p in args.positions]

    if args.dry_run:
        print(f"scheduler-type A/B  positions={[p.upper() for p in positions]}  seeds={seeds}")
        print(f"schedulers={scheds}")
        for pos in positions:
            _, cfg = _load_position(pos)
            prod = cfg.get("scheduler_type")
            print(
                f"\n  {pos.upper()}  (nn_lr={cfg.get('nn_lr')}, nn_epochs={cfg.get('nn_epochs')}, "
                f"production={prod})"
            )
            for s in scheds:
                params = _scheduler_params(cfg, s)
                tag = " [PROD]" if s == prod else "      "
                print(f"    {s:<22}{tag}  {params}")
        print(
            f"\nplanned pipeline runs: {len(positions) * len(scheds) * len(seeds)} "
            f"({len(positions)} pos x {len(scheds)} sched x {len(seeds)} seed)"
        )
        return

    all_rows: list[dict] = []
    summaries: list[dict] = []
    n_pos = len(positions)
    for idx, pos in enumerate(positions, start=1):
        res = run_position(pos, seeds, scheds)
        summaries.append(res["summary"])
        all_rows.extend(res["rows"])
        _print_position_done(res["summary"], idx, n_pos)

    print_cross_position(summaries)
    if not args.no_history:
        run_id = _write_history(summaries, all_rows, positions, seeds)
        print(f"\nwrote benchmark_history/ablations/  run_id={run_id}")


if __name__ == "__main__":
    main()
