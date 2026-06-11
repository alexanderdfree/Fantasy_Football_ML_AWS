"""Shared parallel A/B / ablation harness (device-autodetect, artifact-isolated).

The single place A/Bs and ablations run. It kills the hand-rolled
``for s in 42 123 7; do FF_DEVICE=cpu python … ; done`` anti-pattern (the
2026-06-08 role-inheritance A/B) that ran sequential cells on CPU with the GPU
and 14 cores idle **and** clobbered the served ``{pos}/outputs`` artifacts three
times (``run_pipeline`` hard-codes that save path). See
[todo/ab_harness_priority.md](../../todo/ab_harness_priority.md) and the
"A/Bs & ablations" operating lesson in AGENTS.md.

What it gives you over a bespoke loop:

1. **Parallel, gated on the platform.** The cell grid (positions × variants ×
   seeds) is fanned out by :func:`resolve_jobs` off :func:`detect_platform` +
   ``FF_DEVICE``: a CUDA A/B shares the one GPU ``~-j6`` (the model is
   GPU-launch-bound, not CPU-bound — #670, so do *not* pin to 1); a CPU A/B fans
   across the 16 *physical* cores (never the 32 SMT — LightGBM's SMT penalty is
   16-27×) with per-worker BLAS pinned to 1 and the core pool fair-sharing the
   joblib/LightGBM stages. ``FF_AB_JOBS`` / ``--jobs`` override; workers are
   ``nice``-d (the owner games on this box). It *composes* the existing
   ``parallel_train``/``core_pool`` primitives — it does not re-implement them,
   and nothing lands in ``src/shared/`` (that fires a 6-position retrain).

2. **Artifact isolation.** Every cell runs ``chdir``-ed into its own tmp dir with
   ``data/`` symlinked in, so all the hard-coded ``{pos}/outputs`` writes land in
   the tmp dir and the served artifacts are never touched (same lever as
   ``tests/_pipeline_e2e_utils.run_pipeline_in_tmp``). The feature cache is
   ``FF_FEATURE_CACHE_DISABLE``-d by default — the cache keys on data, not code,
   so a re-run could silently reuse a sibling variant's features and report a
   false ``Δ=0`` (the cache-confound footgun); each cell computes features once
   anyway, so there is nothing to lose.

3. **Aggregation you can trust.** mean±std across seeds per (position, variant,
   model, metric), Δ-vs-baseline, and the **Ridge-invariance sentinel**: a
   feature/frame variant *must* move Ridge MAE off the baseline (else the feature
   didn't take — the ``Δ=0`` smell); a pure loss/arch variant *must not*
   (else the "NN-only" change leaked into the data path). Declare the expectation
   on the :class:`Variant`; unset = report-only.

A *spec* is just a module exposing ``VARIANTS`` (and optionally ``POSITIONS`` /
``SEEDS`` / ``metric_fn`` / ``BASELINE``). The smallest possible one::

    # src/tuning/ab_myfeature.py
    from src.tuning.ab_harness import Variant, ab_main

    POSITIONS = ["RB", "WR"]
    SEEDS = [42, 123, 7]

    def _add_col(train, val, test):
        # PRE-KICKOFF / LEAKAGE-SAFE is YOUR job — the sentinel can't see a
        # feature-side leak (the role-inheritance season-mean leak inflated ~60%).
        for df in (train, val, test):
            df["my_feature"] = df["season"] - 2012  # known-before-kickoff
        return train, val, test

    def _whitelist(cfg):  # extend BOTH model paths
        base = cfg["get_feature_columns_fn"]
        cfg["get_feature_columns_fn"] = lambda: [*base(), "my_feature"]
        cfg["attn_static_features"] = [*cfg["attn_static_features"], "my_feature"]
        return cfg

    VARIANTS = [
        Variant("baseline"),  # identity
        Variant("+my_feature", cfg_mutator=_whitelist, frame_injector=_add_col,
                expect_ridge_identical=False),  # MUST move Ridge
    ]

    if __name__ == "__main__":
        ab_main(__spec__.name)   # `python -m src.tuning.ab_myfeature [--positions …] [-j N]`

``cfg_mutator`` and ``frame_injector`` are looked up by re-importing the spec in
each worker process, so they may be lambdas/closures — nothing is pickled across
the process boundary.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import inspect
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

ALL_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
DEFAULT_SEEDS = (42, 123, 7)  # 3-seed default for FP-MAE A/Bs (AGENTS.md)
RIDGE_MODEL = "Ridge"  # cohort_analysis.MODELS key
_RIDGE_TOL = 1e-9  # |ΔRidge MAE| below this == data-identical
_ENV_JOBS = "FF_AB_JOBS"
_ENV_NICE = "FF_AB_NICE"
_ENV_CACHE_DISABLE = "FF_FEATURE_CACHE_DISABLE"
_DEFAULT_NICE = 10  # be a polite background citizen — the owner games on this box


# --------------------------------------------------------------------------- #
# Variant / Cell
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    """One arm of an A/B.

    ``cfg_mutator`` takes the position's production config and returns a mutated
    config (it receives a deep copy, so an in-place mutator is safe and cannot
    contaminate sibling cells). ``frame_injector`` takes the *general* splits
    ``(train, val, test)`` — every position, pre-position-filter — and returns
    injected frames; ``None`` means "load splits internally" (byte-identical to a
    plain ``run()``). A feature A/B typically sets both: inject the column(s) and
    whitelist them into ``get_feature_columns_fn`` / ``attn_static_features``.

    ``expect_ridge_identical`` drives the sentinel:
      * ``True``  — assert Ridge MAE matches the baseline (a loss/arch/NN-only
        change must NOT touch the deterministic Ridge data path).
      * ``False`` — assert Ridge MAE differs from the baseline (a feature/frame
        change MUST move it; equality is the ``Δ=0`` "feature didn't take" smell).
      * ``None``  — report only, assert nothing (the default).
    """

    name: str
    cfg_mutator: Callable[[dict], dict] | None = None
    frame_injector: Callable[..., tuple] | None = None
    expect_ridge_identical: bool | None = None
    label: str = ""

    @property
    def is_baseline_shape(self) -> bool:
        """No config or data change ⇒ a candidate identity baseline."""
        return self.cfg_mutator is None and self.frame_injector is None


@dataclass(frozen=True)
class Cell:
    """One independent unit of work: a single isolated ``run(seed, config)``."""

    position: str
    variant: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.position}-{self.variant}-{self.seed}"


@dataclass
class Spec:
    """Resolved A/B specification (variants + grid + metric)."""

    variants: OrderedDict[str, Variant]
    baseline: str
    positions: list[str]
    seeds: list[int]
    metric_fn: Callable[[dict, str], dict]
    dotted: str | None = None  # importable module path; required for parallel mode
    name: str = "ab"


# --------------------------------------------------------------------------- #
# Default metric
# --------------------------------------------------------------------------- #
def default_metric_fn(result: dict, position: str) -> dict[str, dict[str, float]]:
    """Per-model MAE / signed bias / RMSE / n on ``result["test_df"]``.

    Reuses ``cohort_analysis.per_model_metrics`` (the same fantasy-point columns
    ``pred_{model}_total`` vs ``fantasy_points`` the rest of the project reports
    on) so every position — including K/DST, whose totals only exist post-pipeline
    on ``test_df`` — is covered uniformly. Override by defining ``metric_fn`` on
    the spec (e.g. a cohort/subgroup bias or ``rmse_gap_decomposition`` cut).
    """
    from src.analysis.cohort_analysis import available_models, per_model_metrics

    df = result["test_df"]
    return per_model_metrics(df, available_models(df))


# --------------------------------------------------------------------------- #
# Spec resolution
# --------------------------------------------------------------------------- #
def _coerce_variants(raw) -> OrderedDict[str, Variant]:
    out: OrderedDict[str, Variant] = OrderedDict()
    items = raw.values() if isinstance(raw, dict) else raw
    for v in items:
        if not isinstance(v, Variant):
            raise TypeError(f"spec VARIANTS must contain Variant objects, got {type(v)!r}")
        if v.name in out:
            raise ValueError(f"duplicate variant name {v.name!r}")
        out[v.name] = v
    if not out:
        raise ValueError("spec defines no variants")
    return out


def _pick_baseline(variants: OrderedDict[str, Variant], declared: str | None) -> str:
    if declared is not None:
        if declared not in variants:
            raise ValueError(f"BASELINE {declared!r} not among variants {list(variants)}")
        return declared
    for name, v in variants.items():
        if v.is_baseline_shape:
            return name
    return next(iter(variants))  # fall back to first


def resolve_spec(
    spec,
    *,
    positions: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    only: Sequence[str] | None = None,
) -> Spec:
    """Turn a spec module (or dotted path) + CLI overrides into a :class:`Spec`.

    ``spec`` may be an imported module/object or a dotted module path string. A
    string is required for parallel (subprocess) execution so each worker can
    re-import it; an object resolves only the in-process sequential path.
    """
    dotted: str | None = None
    if isinstance(spec, str):
        dotted = spec
        spec = importlib.import_module(spec)
    else:
        dotted = getattr(getattr(spec, "__spec__", None), "name", None) or getattr(
            spec, "__name__", None
        )
        if dotted in (None, "__main__"):
            dotted = None  # not importable by workers; sequential-only

    variants = _coerce_variants(spec.VARIANTS)
    if only:
        missing = [n for n in only if n not in variants]
        if missing:
            raise ValueError(f"--only names not in spec: {missing}")
        baseline = _pick_baseline(variants, getattr(spec, "BASELINE", None))
        keep = list(dict.fromkeys([baseline, *only]))  # always keep the baseline
        variants = OrderedDict((n, variants[n]) for n in keep)

    baseline = _pick_baseline(variants, getattr(spec, "BASELINE", None))
    pos = [p.upper() for p in (positions or getattr(spec, "POSITIONS", None) or [])]
    if not pos:
        raise ValueError("no positions: pass --positions or define POSITIONS in the spec")
    bad = [p for p in pos if p not in ALL_POSITIONS]
    if bad:
        raise ValueError(f"unknown positions {bad}; valid: {ALL_POSITIONS}")
    sds = [int(s) for s in (seeds or getattr(spec, "SEEDS", None) or DEFAULT_SEEDS)]
    metric_fn = getattr(spec, "metric_fn", default_metric_fn)
    name = getattr(spec, "AB_NAME", None) or (dotted or "ab").rsplit(".", 1)[-1]
    return Spec(variants, baseline, pos, sds, metric_fn, dotted, name)


def build_cells(spec: Spec) -> list[Cell]:
    """The full position × variant × seed grid (positions outermost)."""
    return [Cell(p, v, s) for p in spec.positions for v in spec.variants for s in spec.seeds]


# --------------------------------------------------------------------------- #
# Jobs autodetect
# --------------------------------------------------------------------------- #
def resolve_jobs(n_cells: int, requested: int | None = None, *, sequential: bool = False) -> int:
    """Concurrent-cell count, gated on ``FF_DEVICE`` + the detected platform.

    Precedence: ``--sequential`` → explicit ``requested`` → ``FF_AB_JOBS`` →
    autodetect. Autodetect (the AGENTS.md gate):
      * CUDA A/B (``cuda_enabled()``): ``min(cells, 6)`` — share the one GPU; the
        small model is GPU-launch-bound, so over-pinning to 1 wastes it (#670).
      * CPU A/B on a many-core box (``cpu_count ≥ 12``): one cell per physical
        core (``min(cells, n_physical)``); the core pool fair-shares the heavier
        joblib/LightGBM stages, BLAS stays pinned to 1.
      * MPS / small boxes: sequential (1) — MPS is unproven for this model and
        16 processes would thrash a single Mac GPU.
    """
    if sequential:
        return 1
    if requested is not None:
        return max(1, min(int(requested), n_cells))
    env = os.environ.get(_ENV_JOBS)
    if env:
        with contextlib.suppress(ValueError):
            return max(1, min(int(env), n_cells))

    from src.shared.platform_detect import detect_platform
    from src.shared.utils import cuda_enabled

    plat = detect_platform()
    if cuda_enabled():
        return max(1, min(n_cells, 6))
    if plat.backend == "mps":
        return 1
    if (plat.cpu_count or 0) >= 12:
        from src.benchmarking.parallel_train import physical_cores

        return max(1, min(n_cells, len(physical_cores())))
    return 1


# --------------------------------------------------------------------------- #
# Cell execution (shared by sequential + worker)
# --------------------------------------------------------------------------- #
def _apply_config(variant: Variant, base_cfg: dict) -> dict:
    """Deep-copy the base config, then apply the variant's mutator.

    The deep copy is taken *first* so even a mutator that edits in place (the
    common ``ablate_*`` shape) cannot contaminate the module-level ``CONFIG``
    shared across in-process cells. Module-level callables are atomic to
    ``deepcopy`` (the cache fingerprints them by qualname), so the copy is
    cache-key-stable.
    """
    cfg = copy.deepcopy(base_cfg)
    if variant.cfg_mutator is not None:
        cfg = variant.cfg_mutator(cfg)
    return cfg


def _load_general_splits():
    """Read the general train/val/test splits the way ``run_pipeline`` does.

    Called *inside* the isolated cwd so ``SPLITS_DIR`` (a relative path) resolves
    through the symlinked ``data/``. Returns position-unfiltered frames — a
    frame-injector that needs within-position grouping does that itself.
    """
    from src.shared.pipeline import SPLITS_DIR, _read_split

    return (
        _read_split(f"{SPLITS_DIR}/train.parquet"),
        _read_split(f"{SPLITS_DIR}/val.parquet"),
        _read_split(f"{SPLITS_DIR}/test.parquet"),
    )


def run_cell(
    cell: Cell,
    variant: Variant,
    metric_fn: Callable[[dict, str], dict],
    *,
    data_dir: str,
    run_fn: Callable | None = None,
) -> dict:
    """Run one isolated cell and return its small result dict.

    chdir into a private tmp dir with ``data/`` symlinked in, apply the variant
    (config + optional frame injection), run the position pipeline, compute the
    metric, then restore cwd and delete the tmp dir. The served ``{pos}/outputs``
    are never touched. ``run_fn`` is injectable for tests; production resolves
    ``src.{pos}.run_pipeline.run``.
    """
    pos = cell.position
    if run_fn is None:
        run_fn = importlib.import_module(f"src.{pos.lower()}.run_pipeline").run
        base_cfg = importlib.import_module(f"src.{pos.lower()}.run_pipeline").CONFIG
    else:
        base_cfg = importlib.import_module(f"src.{pos.lower()}.run_pipeline").CONFIG

    cfg = _apply_config(variant, base_cfg)
    orig_cwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp(prefix=f"ff-ab-{cell.key}-")
    try:
        os.chdir(tmp_dir)
        link = Path(tmp_dir) / "data"
        if not link.exists():
            link.symlink_to(data_dir, target_is_directory=True)

        # K/DST build their own splits inside run(seed, config) and take no
        # train/val/test args; the skill positions take (train_df, val_df,
        # test_df, seed, config) and support frame injection.
        accepts_frames = any(
            p not in ("seed", "config") for p in inspect.signature(run_fn).parameters
        )
        if variant.frame_injector is not None and not accepts_frames:
            raise ValueError(
                f"{pos} run() builds its own splits (no train/val/test args), so the "
                f"frame_injector variant {cell.variant!r} can't inject into it. Use a "
                f"cfg_mutator instead, or restrict the spec's POSITIONS to QB/RB/WR/TE."
            )

        if accepts_frames:
            train = val = test = None
            if variant.frame_injector is not None:
                train, val, test = variant.frame_injector(*_load_general_splits())
            result = run_fn(train, val, test, seed=cell.seed, config=cfg)
        else:
            result = run_fn(seed=cell.seed, config=cfg)
        metrics = metric_fn(result, pos)
        ridge = metrics.get(RIDGE_MODEL, {}).get("mae")
        return {
            "position": pos,
            "variant": cell.variant,
            "seed": cell.seed,
            "label": variant.label or cell.variant,
            "ok": True,
            "metrics": metrics,
            "ridge_mae": ridge,
            "error": None,
        }
    finally:
        os.chdir(orig_cwd)
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Stacked-seeds mode (opt-in): one (position, variant) GROUP trains all seeds
# at once via the vmap ensemble harness (src.tuning.ab_ensemble_seeds).
# Owner-sanctioned 2026-06-11 for comparative pipelines: ~4.5x per host thread,
# within-mode consistent; NOT seed-comparable to eager runs — never mix arms
# across modes in one comparison (see the ab_ensemble_seeds module banner).
# --------------------------------------------------------------------------- #
_STACKED_POSITIONS = ("QB", "RB", "WR", "TE")  # flat-history only (= ENSEMBLE_POSITIONS)
DEFAULT_STACKED_EPOCHS = 30  # the established fixed-epochs A/B-isolation regime


@dataclass(frozen=True)
class Group:
    """Stacked unit of work: one (position, variant) across ALL seeds."""

    position: str
    variant: str
    seeds: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.position}-{self.variant}-stacked{len(self.seeds)}"


def build_stacked_units(spec: Spec) -> tuple[list[Group], list[Cell]]:
    """Stacked groups for flat-history positions; eager per-seed cells for the
    rest (K's nested trainer / DST's own-splits run() are excluded from the
    vmap harness)."""
    groups = [
        Group(p, v, tuple(spec.seeds))
        for p in spec.positions
        if p in _STACKED_POSITIONS
        for v in spec.variants
    ]
    cells = [
        Cell(p, v, s)
        for p in spec.positions
        if p not in _STACKED_POSITIONS
        for v in spec.variants
        for s in spec.seeds
    ]
    return groups, cells


def run_group_stacked(
    group: Group,
    variant: Variant,
    metric_fn: Callable[[dict, str], dict],
    *,
    data_dir: str,
    stacked_epochs: int = DEFAULT_STACKED_EPOCHS,
    run_fn: Callable | None = None,
) -> list[dict]:
    """Run one stacked (position, variant) group; return one result per seed.

    Phase A: ONE real full run at ``seeds[0]`` in the DEFAULT regime with
    attention disabled — it supplies the non-attention models' predictions,
    the reference ``test_df``, and the Ridge sentinel value (so frame
    injection + the data-identity tell stay exactly as honest as eager mode).
    Phase B: capture all seeds + stacked attention training in the ensemble
    regime (LN/FP32/fixed-epochs; env restored on exit), then per-member test
    predictions overwrite ``pred_attn_nn_total`` on a copy of the reference
    ``test_df`` and ``metric_fn`` recomputes per-seed metrics.

    Contract notes: non-attention models repeat Phase-A values across seeds
    (std=0 by construction) — stacked mode measures ATTENTION deltas; custom
    ``metric_fn``s must derive from ``result["test_df"]`` (other result keys
    are passed through from Phase A and do not reflect the stacked arm).
    """
    from src.shared.aggregate_targets import predictions_to_fantasy_points
    from src.shared.utils import cuda_enabled
    from src.tuning.ab_ensemble_seeds import (
        capture_seeds,
        ensemble_env,
        predict_stacked,
        train_stacked,
    )

    pos = group.position
    mod = importlib.import_module(f"src.{pos.lower()}.run_pipeline")
    if run_fn is None:
        run_fn = mod.run
    base_cfg = mod.CONFIG
    cfg = _apply_config(variant, base_cfg)

    orig_cwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp(prefix=f"ff-ab-{group.key}-")
    try:
        os.chdir(tmp_dir)
        link = Path(tmp_dir) / "data"
        if not link.exists():
            link.symlink_to(data_dir, target_is_directory=True)

        frames = None
        if variant.frame_injector is not None:
            frames = variant.frame_injector(*_load_general_splits())

        cfg_a = copy.deepcopy(cfg)
        cfg_a["train_attention_nn"] = False  # Phase B supplies attention
        seed0 = group.seeds[0]
        if frames is not None:
            result0 = run_fn(
                frames[0].copy(), frames[1].copy(), frames[2].copy(), seed=seed0, config=cfg_a
            )
        else:
            result0 = run_fn(None, None, None, seed=seed0, config=cfg_a)
        # metric_fn is NOT called on the Phase-A result: its test_df has no
        # attention column yet, and a custom metric_fn may require it. Each
        # per-seed metrics_k below carries the (constant) Ridge value instead.

        import torch

        with ensemble_env(stacked_epochs):
            captures, test_capture = capture_seeds(
                pos, list(group.seeds), base_cfg=cfg, frames=frames
            )
            device = torch.device("cuda" if cuda_enabled() else "cpu")
            params, buffers, template = train_stacked(captures, cfg, device, stacked_epochs)
            member_preds = predict_stacked(template, params, buffers, test_capture, device)

        out = []
        for seed, preds in zip(group.seeds, member_preds, strict=True):
            df_k = result0["test_df"].copy()
            df_k["pred_attn_nn_total"] = predictions_to_fantasy_points(pos, preds)
            metrics_k = metric_fn({**result0, "test_df": df_k}, pos)
            out.append(
                {
                    "position": pos,
                    "variant": group.variant,
                    "seed": seed,
                    "label": variant.label or group.variant,
                    "ok": True,
                    "metrics": metrics_k,
                    "ridge_mae": metrics_k.get(RIDGE_MODEL, {}).get("mae"),
                    "error": None,
                    "stacked": True,
                }
            )
        return out
    finally:
        os.chdir(orig_cwd)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _group_failed(group: Group, variant: Variant, exc: BaseException) -> list[dict]:
    return [
        {
            "position": group.position,
            "variant": group.variant,
            "seed": s,
            "label": variant.label or group.variant,
            "ok": False,
            "metrics": {},
            "ridge_mae": None,
            "error": f"{type(exc).__name__}: {exc}",
            "stacked": True,
        }
        for s in group.seeds
    ]


# --------------------------------------------------------------------------- #
# Sequential (in-process) orchestration
# --------------------------------------------------------------------------- #
def _cell_failed(cell: Cell, variant: Variant, exc: BaseException) -> dict:
    return {
        "position": cell.position,
        "variant": cell.variant,
        "seed": cell.seed,
        "label": variant.label or cell.variant,
        "ok": False,
        "metrics": {},
        "ridge_mae": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_sequential(spec: Spec, cells: list[Cell], data_dir: str) -> list[dict]:
    """Run cells one at a time, in-process. The low-core fallback and the
    test/CI path; chdir isolation is naturally safe when only one cell runs at a
    time."""
    results = []
    for i, cell in enumerate(cells, 1):
        variant = spec.variants[cell.variant]
        print(f"[ab] cell {i}/{len(cells)}: {cell.key}", flush=True)
        try:
            results.append(run_cell(cell, variant, spec.metric_fn, data_dir=data_dir))
        except Exception as exc:  # noqa: BLE001 — one cell's failure must not sink the run
            print(f"[ab] cell {cell.key} FAILED: {exc}", file=sys.stderr, flush=True)
            results.append(_cell_failed(cell, variant, exc))
    return results


def run_sequential_stacked(
    spec: Spec, groups: list[Group], cells: list[Cell], data_dir: str, stacked_epochs: int
) -> list[dict]:
    """Stacked groups + leftover eager cells, one at a time, in-process."""
    results: list[dict] = []
    total = len(groups) + len(cells)
    for i, group in enumerate(groups, 1):
        variant = spec.variants[group.variant]
        print(f"[ab] group {i}/{total}: {group.key}", flush=True)
        try:
            results.extend(
                run_group_stacked(
                    group,
                    variant,
                    spec.metric_fn,
                    data_dir=data_dir,
                    stacked_epochs=stacked_epochs,
                )  # fmt: skip
            )
        except Exception as exc:  # noqa: BLE001 — one group's failure must not sink the run
            print(f"[ab] group {group.key} FAILED: {exc}", file=sys.stderr, flush=True)
            results.extend(_group_failed(group, variant, exc))
    for j, cell in enumerate(cells, len(groups) + 1):
        variant = spec.variants[cell.variant]
        print(f"[ab] cell {j}/{total}: {cell.key} (eager fallback)", flush=True)
        try:
            results.append(run_cell(cell, variant, spec.metric_fn, data_dir=data_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"[ab] cell {cell.key} FAILED: {exc}", file=sys.stderr, flush=True)
            results.append(_cell_failed(cell, variant, exc))
    return results


# --------------------------------------------------------------------------- #
# Parallel (subprocess-per-cell) orchestration
# --------------------------------------------------------------------------- #
def _worker_preexec(cores: list[int], nice: int):
    """preexec for a cell worker: pin to all physical cores (the core pool
    narrows per CPU stage) and lower priority so interactive use wins."""

    def _fn():
        with contextlib.suppress(AttributeError, OSError):
            os.sched_setaffinity(0, set(cores))
        with contextlib.suppress(OSError):
            os.nice(nice)

    return _fn


def _spawn_worker(key: str, argv: list[str], out_path: str, cores, nice, pool_addr, logdir):
    """Popen one worker with the harness env (cache-disable, core pool, BLAS=1)."""
    import subprocess

    from src.shared.core_pool import ENV_ADDR, ENV_POS

    env = dict(os.environ)
    # Cache-disable is inherited from run_ab (set per --feature-cache); default to
    # disabled only if a worker is somehow launched outside run_ab.
    env.setdefault(_ENV_CACHE_DISABLE, "1")
    env[ENV_POS] = key
    if pool_addr:
        env[ENV_ADDR] = pool_addr
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(k, "1")
    # FF_DEVICE is deliberately NOT forced — a CPU A/B on the 5080 box (FF_DEVICE=cpu)
    # must use the CPU pool, a CUDA A/B the GPU-launch-bound pool. The user/env decides.
    log_path = os.path.join(logdir, f"ab-{key}.log")
    logf = open(log_path, "w")  # noqa: SIM115 — owned by the orchestrator until the proc exits
    proc = subprocess.Popen(
        argv, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=os.getcwd(),
        preexec_fn=_worker_preexec(cores, nice),
    )  # fmt: skip
    return {
        "proc": proc,
        "logf": logf,
        "out": out_path,
        "log": log_path,
        "t0": time.time(),
    }


def _launch_worker(cell: Cell, spec: Spec, out_path: str, cores, nice, pool_addr, data_dir, logdir):
    argv = [
        sys.executable, "-m", "src.tuning.ab_harness", "--worker",
        "--spec", spec.dotted, "--position", cell.position,
        "--variant", cell.variant, "--seed", str(cell.seed),
        "--out", out_path, "--data-dir", data_dir,
    ]  # fmt: skip
    info = _spawn_worker(cell.key, argv, out_path, cores, nice, pool_addr, logdir)
    info["cell"] = cell
    return info


def run_parallel(spec: Spec, cells: list[Cell], jobs: int, data_dir: str) -> list[dict]:
    """Fan cells out as subprocess workers sharing a core pool.

    Mirrors ``parallel_train.orchestrate`` (queue → dispatch up to ``jobs`` →
    poll → reap → set_active_count) but the unit is a cell, not a position, and
    there is no benchmark/S3 recording. Each worker writes its small result JSON;
    the orchestrator reads them back.
    """
    if not spec.dotted:
        raise ValueError(
            "parallel mode needs an importable spec (a dotted module path); pass the spec "
            "as a string or run with --sequential"
        )
    from src.benchmarking.parallel_train import physical_cores
    from src.shared.core_pool import start_coordinator

    nice = int(os.environ.get(_ENV_NICE, _DEFAULT_NICE))
    phys = physical_cores()
    logdir = "logs"
    os.makedirs(logdir, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="ff-ab-pool-")
    pool_addr, set_active_count, pool_stop = start_coordinator(phys, tmpdir)

    queue = deque(cells)
    active: OrderedDict[str, dict] = OrderedDict()
    by_key: dict[str, dict] = {}
    print(
        f"[ab] {len(cells)} cells, -j {jobs}, {len(phys)} physical cores; core pool {pool_addr}; "
        f"nice {nice}; logs -> {logdir}/ab-<cell>.log",
        flush=True,
    )
    try:
        while queue or active:
            while queue and len(active) < jobs:
                cell = queue.popleft()
                out_path = os.path.join(tmpdir, f"{cell.key}.json")
                active[cell.key] = _launch_worker(
                    cell, spec, out_path, phys, nice, pool_addr, data_dir, logdir
                )
                print(f"[ab] launched {cell.key} (pid {active[cell.key]['proc'].pid})", flush=True)
            set_active_count(len(active))

            done = [k for k, info in active.items() if info["proc"].poll() is not None]
            if not done:
                time.sleep(0.4)
                continue
            for k in done:
                info = active.pop(k)
                info["logf"].close()
                rc = info["proc"].returncode
                elapsed = time.time() - info["t0"]
                by_key[k] = _collect_worker(info, rc, elapsed)
                tag = "ok" if by_key[k]["ok"] else f"FAILED (rc={rc})"
                print(f"[ab] {k} {tag} in {elapsed:.1f}s (log: {info['log']})", flush=True)
            set_active_count(len(active))
    finally:
        pool_stop()
    # Preserve grid order in the returned list.
    return [by_key[c.key] for c in cells]


def _collect_worker(info: dict, rc: int, elapsed: float) -> dict:
    cell: Cell = info["cell"]
    result = None
    # Read the JSON regardless of rc — a failed cell still writes its structured
    # error there (rc=1), and surfacing that beats a bare "rc=1, see log".
    try:
        with open(info["out"]) as f:
            result = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        result = None
    if result is None:
        return {
            "position": cell.position, "variant": cell.variant, "seed": cell.seed,
            "label": cell.variant, "ok": False, "metrics": {}, "ridge_mae": None,
            "error": f"worker rc={rc}, no result JSON; see {info['log']}",
            "elapsed_sec": round(elapsed, 1),
        }  # fmt: skip
    result["elapsed_sec"] = round(elapsed, 1)
    return result


def run_parallel_stacked(
    spec: Spec,
    groups: list[Group],
    cells: list[Cell],
    jobs: int,
    data_dir: str,
    stacked_epochs: int,
) -> list[dict]:
    """Fan stacked groups (+ leftover eager cells) out as subprocess workers.

    Same queue/poll/reap shape as :func:`run_parallel`; a group worker writes a
    JSON LIST (one result per seed), an eager cell worker the usual dict.
    """
    if not spec.dotted:
        raise ValueError(
            "parallel mode needs an importable spec (a dotted module path); pass the spec "
            "as a string or run with --sequential"
        )
    from src.benchmarking.parallel_train import physical_cores
    from src.shared.core_pool import start_coordinator

    nice = int(os.environ.get(_ENV_NICE, _DEFAULT_NICE))
    phys = physical_cores()
    logdir = "logs"
    os.makedirs(logdir, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="ff-ab-pool-")
    pool_addr, set_active_count, pool_stop = start_coordinator(phys, tmpdir)

    units: list[tuple[str, object]] = [("group", g) for g in groups]
    units += [("cell", c) for c in cells]
    queue = deque(units)
    active: OrderedDict[str, dict] = OrderedDict()
    by_key: dict[str, list[dict]] = {}
    print(
        f"[ab] {len(groups)} stacked groups + {len(cells)} eager cells, -j {jobs}, "
        f"{len(phys)} physical cores; core pool {pool_addr}; nice {nice}; "
        f"logs -> {logdir}/ab-<unit>.log",
        flush=True,
    )
    try:
        while queue or active:
            while queue and len(active) < jobs:
                kind, work = queue.popleft()
                out_path = os.path.join(tmpdir, f"{work.key}.json")
                if kind == "group":
                    argv = [
                        sys.executable, "-m", "src.tuning.ab_harness", "--worker-group",
                        "--spec", spec.dotted, "--position", work.position,
                        "--variant", work.variant,
                        "--seeds", *[str(s) for s in work.seeds],
                        "--stacked-epochs", str(stacked_epochs),
                        "--out", out_path, "--data-dir", data_dir,
                    ]  # fmt: skip
                else:
                    argv = [
                        sys.executable, "-m", "src.tuning.ab_harness", "--worker",
                        "--spec", spec.dotted, "--position", work.position,
                        "--variant", work.variant, "--seed", str(work.seed),
                        "--out", out_path, "--data-dir", data_dir,
                    ]  # fmt: skip
                info = _spawn_worker(work.key, argv, out_path, phys, nice, pool_addr, logdir)
                info["unit"] = (kind, work)
                active[work.key] = info
                print(f"[ab] launched {work.key} (pid {info['proc'].pid})", flush=True)
            set_active_count(len(active))

            done = [k for k, info in active.items() if info["proc"].poll() is not None]
            if not done:
                time.sleep(0.4)
                continue
            for k in done:
                info = active.pop(k)
                info["logf"].close()
                rc = info["proc"].returncode
                elapsed = time.time() - info["t0"]
                by_key[k] = _collect_unit(info, rc, elapsed)
                tag = "ok" if all(r.get("ok") for r in by_key[k]) else f"FAILED (rc={rc})"
                print(f"[ab] {k} {tag} in {elapsed:.1f}s (log: {info['log']})", flush=True)
            set_active_count(len(active))
    finally:
        pool_stop()
    ordered: list[dict] = []
    for _, work in units:
        ordered.extend(by_key[work.key])
    return ordered


def _collect_unit(info: dict, rc: int, elapsed: float) -> list[dict]:
    """Read a unit worker's JSON (list for groups, dict for cells) → flat list."""
    kind, work = info["unit"]
    payload = None
    try:
        with open(info["out"]) as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        payload = None
    if payload is None:
        err = f"worker rc={rc}, no result JSON; see {info['log']}"
        seeds = work.seeds if kind == "group" else (work.seed,)
        return [
            {
                "position": work.position,
                "variant": work.variant,
                "seed": s,
                "label": work.variant,
                "ok": False,
                "metrics": {},
                "ridge_mae": None,
                "error": err,
                "elapsed_sec": round(elapsed, 1),
            }  # fmt: skip
            for s in seeds
        ]
    results = payload if isinstance(payload, list) else [payload]
    for r in results:
        r["elapsed_sec"] = round(elapsed, 1)
    return results


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean_std(values: list[float]) -> tuple[float, float, int]:
    vals = [v for v in values if v is not None and v == v]  # drop None / NaN
    if not vals:
        return float("nan"), float("nan"), 0
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return mean, std, len(vals)


def aggregate(spec: Spec, results: list[dict]) -> dict:
    """mean±std + Δ-vs-baseline per (position, variant, model, metric) and the
    per-(position, seed) Ridge-invariance sentinel verdicts."""
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    # index: (pos, variant, seed) -> result
    idx = {(r["position"], r["variant"], r["seed"]): r for r in ok}

    table: dict = {}  # pos -> variant -> model -> metric -> {mean,std,n,delta}
    for pos in spec.positions:
        table[pos] = {}
        # models present anywhere for this position (union, baseline-first order)
        models: list[str] = []
        for v in spec.variants:
            for s in spec.seeds:
                r = idx.get((pos, v, s))
                if r:
                    for m in r["metrics"]:
                        if m not in models:
                            models.append(m)
        # Pass 1: means/std for every (variant, model, metric).
        for v in spec.variants:
            table[pos][v] = {}
            for model in models:
                metric_names: list[str] = []
                for s in spec.seeds:
                    r = idx.get((pos, v, s))
                    if r and model in r["metrics"]:
                        for mn in r["metrics"][model]:
                            if mn not in metric_names:
                                metric_names.append(mn)
                table[pos][v][model] = {}
                for mn in metric_names:
                    vals = [
                        idx[(pos, v, s)]["metrics"][model][mn]
                        for s in spec.seeds
                        if idx.get((pos, v, s)) and model in idx[(pos, v, s)]["metrics"]
                    ]
                    mean, std, n = _mean_std(vals)
                    table[pos][v][model][mn] = {"mean": mean, "std": std, "n": n}

        # Pass 2: Δ vs baseline — separate pass so variant *order* (baseline need
        # not be listed first) can't drop the delta.
        base = table[pos].get(spec.baseline, {})
        for v in spec.variants:
            if v == spec.baseline:
                continue
            for model, metrics in table[pos][v].items():
                for mn, cell in metrics.items():
                    bmean = base.get(model, {}).get(mn, {}).get("mean")
                    cell["delta"] = (
                        (cell["mean"] - bmean)
                        if bmean is not None and bmean == bmean and cell["mean"] == cell["mean"]
                        else None
                    )

    sentinel = _ridge_sentinel(spec, idx)
    return {"table": table, "sentinel": sentinel, "n_ok": len(ok), "failed": failed}


def _ridge_sentinel(spec: Spec, idx: dict) -> list[dict]:
    """Per (position, non-baseline variant, seed): ΔRidge MAE vs baseline +
    verdict against ``expect_ridge_identical``."""
    out = []
    for pos in spec.positions:
        for vname, variant in spec.variants.items():
            if vname == spec.baseline:
                continue
            for s in spec.seeds:
                b = idx.get((pos, spec.baseline, s))
                v = idx.get((pos, vname, s))
                if not b or not v or b.get("ridge_mae") is None or v.get("ridge_mae") is None:
                    continue
                delta = v["ridge_mae"] - b["ridge_mae"]
                identical = abs(delta) < _RIDGE_TOL
                status = "ok"
                if variant.expect_ridge_identical is True and not identical:
                    status = "VIOLATION: expected Ridge-identical (NN-only change leaked to data)"
                elif variant.expect_ridge_identical is False and identical:
                    status = "VIOLATION: Ridge Δ=0 — feature did not take (cache/injection bug?)"
                out.append(
                    {
                        "position": pos,
                        "variant": vname,
                        "seed": s,
                        "delta": delta,
                        "identical": identical,
                        "expect": variant.expect_ridge_identical,
                        "status": status,
                    }  # fmt: skip
                )
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(mean: float, std: float) -> str:
    if mean != mean:  # NaN
        return "   n/a   "
    return f"{mean:7.3f}±{std:5.3f}"


def print_report(spec: Spec, agg: dict, *, jobs: int, primary: str = "mae") -> None:
    table = agg["table"]
    for pos in spec.positions:
        print(f"\n{'=' * 84}")
        print(f"  {pos}  ·  baseline={spec.baseline}  ·  seeds={spec.seeds}  ·  -j {jobs}")
        print(f"{'=' * 84}")
        models = list({m for v in table[pos].values() for m in v})
        models.sort()
        for model in models:
            print(f"\n  [{model}]  metric={primary}")
            print(f"    {'variant':22}{primary + ' (mean±std)':>20}{'Δ vs base':>14}")
            print("    " + "-" * 56)
            for v in spec.variants:
                cell = table[pos].get(v, {}).get(model, {}).get(primary)
                if not cell:
                    continue
                label = spec.variants[v].label or v
                delta = cell.get("delta")
                dstr = "    —" if v == spec.baseline else (
                    f"{delta:+8.3f}" if isinstance(delta, float) and delta == delta else "    n/a"
                )  # fmt: skip
                print(f"    {label[:22]:22}{_fmt(cell['mean'], cell['std']):>20}{dstr:>14}")

    # Ridge-invariance sentinel
    sentinel = agg["sentinel"]
    if sentinel:
        print(f"\n{'-' * 84}")
        print("  Ridge-invariance sentinel (ΔRidge MAE vs baseline; data-identity tell)")
        print(f"{'-' * 84}")
        for s in sentinel:
            tell = "data-identical" if s["identical"] else "data changed"
            exp = {True: "expect=identical", False: "expect=differ", None: "report-only"}[
                s["expect"]
            ]
            mark = "  ⚠ " + s["status"] if s["status"] != "ok" else ""
            print(
                f"    {s['position']:4} {s['variant'][:20]:20} seed={s['seed']:<5} "
                f"Δ={s['delta']:+.5f}  {tell:14} ({exp}){mark}"
            )

    # No silent truncation: always say what ran / was dropped.
    failed = agg["failed"]
    print(f"\n[ab] {agg['n_ok']} cells ok, {len(failed)} failed; "
          f"feature cache DISABLED (FF_FEATURE_CACHE_DISABLE), artifacts isolated.")  # fmt: skip
    for r in failed:
        print(f"    FAILED {r['position']}-{r['variant']}-{r['seed']}: {r.get('error')}",
              file=sys.stderr)  # fmt: skip
    violations = [s for s in sentinel if s["status"] != "ok"]
    if violations:
        print(f"[ab] {len(violations)} Ridge-sentinel VIOLATION(s) — results suspect; see above.",
              file=sys.stderr)  # fmt: skip


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def run_ab(
    spec,
    *,
    positions: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    only: Sequence[str] | None = None,
    jobs: int | None = None,
    sequential: bool = False,
    feature_cache: bool = False,
    data_dir: str | None = None,
    stacked_seeds: bool = False,
    stacked_epochs: int = DEFAULT_STACKED_EPOCHS,
) -> dict:
    """Resolve the spec, run the grid (parallel or sequential), aggregate, print.

    Returns the aggregation dict. ``spec`` is a module/object (sequential) or a
    dotted path string (required for parallel). The feature cache is disabled by
    default for A/B correctness; ``feature_cache=True`` re-enables it.

    ``stacked_seeds=True`` switches the unit of work from one (position,
    variant, seed) cell to one (position, variant) GROUP whose attention NN
    trains all seeds at once via the vmap ensemble harness (~4.5× per host
    thread; see :func:`run_group_stacked` for regime + contract). Results stay
    per-seed, so aggregation, Δ-vs-baseline, and the Ridge sentinel are
    unchanged. Stacked results are within-mode consistent — never compare a
    stacked arm against an eager arm seed-by-seed.
    """
    resolved = resolve_spec(spec, positions=positions, seeds=seeds, only=only)
    data_dir = os.path.abspath(data_dir or "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data dir not found: {data_dir} (need data/splits/*.parquet)")

    if stacked_seeds:
        groups, leftover = build_stacked_units(resolved)
        if leftover:
            skipped = sorted({c.position for c in leftover})
            print(
                f"[ab] stacked mode: {skipped} excluded from vmap stacking "
                "(nested-history / own-splits run()); their cells run eager.",
                flush=True,
            )
        jobs = resolve_jobs(len(groups) + len(leftover), jobs, sequential=sequential)
    else:
        cells = build_cells(resolved)
        jobs = resolve_jobs(len(cells), jobs, sequential=sequential)

    # Disable the feature cache by default (it keys on data, not code → a sibling
    # variant could silently reuse features → false Δ=0). Set it *explicitly*
    # (not just when disabling) so subprocess workers inherit the right value and
    # ``--feature-cache`` is honoured in parallel mode too, not only sequential.
    prev_cache = os.environ.get(_ENV_CACHE_DISABLE)
    os.environ[_ENV_CACHE_DISABLE] = "0" if feature_cache else "1"
    try:
        if stacked_seeds:
            if jobs <= 1:
                results = run_sequential_stacked(
                    resolved, groups, leftover, data_dir, stacked_epochs
                )
            else:
                results = run_parallel_stacked(
                    resolved, groups, leftover, jobs, data_dir, stacked_epochs
                )
        elif jobs <= 1:
            results = run_sequential(resolved, cells, data_dir)
        else:
            results = run_parallel(resolved, cells, jobs, data_dir)
    finally:
        if prev_cache is None:
            os.environ.pop(_ENV_CACHE_DISABLE, None)
        else:
            os.environ[_ENV_CACHE_DISABLE] = prev_cache

    agg = aggregate(resolved, results)
    print_report(resolved, agg, jobs=jobs)
    if stacked_seeds:
        print(
            "[ab] stacked-seeds mode: attention = vmap ensemble (LN/FP32/"
            f"fixed-epochs={stacked_epochs}); non-attention models repeat "
            "Phase-A values across seeds (std=0). Compare stacked runs only "
            "against stacked runs."
        )
    return agg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Shared parallel A/B / ablation harness (device-autodetect, artifact-isolated)."
    )
    p.add_argument("--spec", help="Dotted module path of the A/B spec (e.g. src.tuning.ab_example)")
    p.add_argument("--positions", nargs="+", help="Override the spec's POSITIONS")
    p.add_argument("--seeds", type=int, nargs="+", help="Override the spec's SEEDS")
    p.add_argument("--only", nargs="+", help="Run only these variant names (baseline always kept)")
    p.add_argument(
        "-j", "--jobs", type=int, default=None, help="Concurrent cells (default: autodetect)"
    )
    p.add_argument(
        "--sequential", action="store_true", help="Force in-process sequential execution"
    )
    p.add_argument(
        "--feature-cache", action="store_true", help="Re-enable the feature cache (off by default)"
    )
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], help="Set FF_DEVICE for the run"
    )
    p.add_argument(
        "--stacked-seeds",
        action="store_true",
        help="Train each (position, variant)'s seeds as ONE vmap ensemble "
        "(~4.5x/thread; QB/RB/WR/TE only, others fall back to eager cells). "
        "Within-mode consistent — compare stacked runs only against stacked runs.",
    )
    p.add_argument(
        "--stacked-epochs",
        type=int,
        default=DEFAULT_STACKED_EPOCHS,
        help=f"Fixed epochs for stacked attention training (default {DEFAULT_STACKED_EPOCHS})",
    )
    p.add_argument("--list", action="store_true", help="Print the resolved grid + jobs and exit")
    # Internal worker invocation (one cell / one stacked group, spawned by the
    # orchestrator).
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-group", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--position", help=argparse.SUPPRESS)
    p.add_argument("--variant", help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    p.add_argument("--data-dir", help=argparse.SUPPRESS)
    return p


def _run_worker(args) -> int:
    """Execute exactly one cell in this fresh process and write its result JSON."""
    os.environ.setdefault(_ENV_CACHE_DISABLE, "1")
    # Pass this cell's position so resolve_spec is satisfied even for a spec that
    # defines no POSITIONS (relying on the orchestrator's --positions); the worker
    # runs exactly one (position, variant, seed) regardless.
    spec = resolve_spec(args.spec, positions=[args.position])
    variant = spec.variants[args.variant]
    cell = Cell(args.position.upper(), args.variant, int(args.seed))
    data_dir = os.path.abspath(args.data_dir or "data")
    try:
        result = run_cell(cell, variant, spec.metric_fn, data_dir=data_dir)
    except Exception as exc:  # noqa: BLE001 — report the failure via JSON, exit non-zero
        result = _cell_failed(cell, variant, exc)
    with open(args.out, "w") as f:
        json.dump(result, f)
    return 0 if result["ok"] else 1


def _run_worker_group(args) -> int:
    """Execute one stacked group in this fresh process; write a result LIST."""
    os.environ.setdefault(_ENV_CACHE_DISABLE, "1")
    spec = resolve_spec(args.spec, positions=[args.position])
    variant = spec.variants[args.variant]
    group = Group(args.position.upper(), args.variant, tuple(int(s) for s in args.seeds))
    data_dir = os.path.abspath(args.data_dir or "data")
    try:
        results = run_group_stacked(
            group,
            variant,
            spec.metric_fn,
            data_dir=data_dir,
            stacked_epochs=int(args.stacked_epochs),
        )
    except Exception as exc:  # noqa: BLE001 — report the failure via JSON, exit non-zero
        results = _group_failed(group, variant, exc)
    with open(args.out, "w") as f:
        json.dump(results, f)
    return 0 if all(r["ok"] for r in results) else 1


def main(argv: list[str] | None = None, *, default_spec: str | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.device:
        os.environ["FF_DEVICE"] = args.device

    if args.worker:
        return _run_worker(args)
    if args.worker_group:
        return _run_worker_group(args)

    spec_ref = args.spec or default_spec
    if not spec_ref:
        print("error: --spec <module> is required", file=sys.stderr)
        return 2

    if args.list:
        spec = resolve_spec(spec_ref, positions=args.positions, seeds=args.seeds, only=args.only)
        print(f"spec={spec.dotted or spec.name} baseline={spec.baseline}")
        print(f"variants={list(spec.variants)}")
        print(f"positions={spec.positions} seeds={spec.seeds}")
        if args.stacked_seeds:
            groups, leftover = build_stacked_units(spec)
            jobs = resolve_jobs(len(groups) + len(leftover), args.jobs, sequential=args.sequential)
            print(
                f"{len(groups)} stacked groups (epochs={args.stacked_epochs}) "
                f"+ {len(leftover)} eager cells, -j {jobs}"
            )
        else:
            cells = build_cells(spec)
            jobs = resolve_jobs(len(cells), args.jobs, sequential=args.sequential)
            print(f"{len(cells)} cells, -j {jobs}")
        return 0

    run_ab(
        spec_ref,
        positions=args.positions,
        seeds=args.seeds,
        only=args.only,
        jobs=args.jobs,
        sequential=args.sequential,
        feature_cache=args.feature_cache,
        stacked_seeds=args.stacked_seeds,
        stacked_epochs=args.stacked_epochs,
    )
    return 0


def ab_main(spec_dotted: str, argv: list[str] | None = None) -> int:
    """Convenience entry for a spec module's ``__main__`` (pass ``__spec__.name``)."""
    return main(argv, default_spec=spec_dotted)


if __name__ == "__main__":
    # Re-dispatch through the canonically-imported module. Running as ``-m`` loads
    # this file as ``__main__``; a spec's ``from src.tuning.ab_harness import
    # Variant`` imports it *again* as ``src.tuning.ab_harness`` — two distinct
    # ``Variant`` classes, so the worker's ``isinstance`` check would reject the
    # spec's variants. Delegating to the canonical ``main`` makes every identity
    # come from the one module. (The subprocess-only path unit tests can't see.)
    from src.tuning.ab_harness import main as _main

    sys.exit(_main())
