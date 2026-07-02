"""Parallel local trainer — fan all six positions across the box's physical cores.

``src.benchmarking.benchmark`` loops positions *sequentially* in one process; on a
many-core CUDA box (the WSL2 / RTX 5080 / 9950X3D dev machine) that wastes the GPU.
Each position only drives the GPU to ~20% util because the small attention model is
launch/host-bound, so running every position as its **own process** (own GIL) lets one
process's kernels run while another's Python preps its next batch — the GPU's idle gaps
fill in and total wall-clock drops ~3-4x.

The only real hazard is CPU oversubscription: each position's Ridge alpha-CV fans out
over ``joblib.Parallel(prefer="threads")`` (``src/shared/pipeline.py``) and LightGBM uses
``LGBM_N_JOBS`` threads, so six positions each grabbing all 16 cores thrash. This
orchestrator starts a **work-conserving core pool** (``src/shared/core_pool.py``): a
coordinator owns the box's physical cores and each position leases its fair share
(``ceil(total / active)``) for the duration of a CPU stage, releasing on stage exit. As
positions finish, the orchestrator lowers the active count and the survivors' next leases
widen — so the thread *count*, not just affinity, follows the work. (The old static
per-slice ``LGBM_N_JOBS`` was immutable post-spawn, so freed cores were stranded; this
fixes that.) When ``-j`` is below the position count, the next queued position dispatches
into the freed slot.

By default, outputs are identical to a sequential ``benchmark QB RB …`` run: each worker
trains its position once (which writes ``{pos}/outputs/`` artifacts via the shared
pipeline's unconditional save block) and emits its metrics summary to a unique temp file
— **no shared-file writes during the parallel phase**. The orchestrator then merges the
summaries into one ``benchmark_results.json`` + one ``benchmark_history/{run_id}.json``
entry and mirrors it to S3 (the website History tab), reusing ``benchmark.py``'s helpers.
``--rolling-origin`` flattens the run into a (position × origin) cell grid — each cell is
one origin's train+score, dispatched to the SAME work-conserving core pool — so a single
position's origins run concurrently and a full 6-position run stays GPU-saturated to the
end (no single-position tail). The orchestrator merges each position's per-origin
summaries via ``finalize_rolling_origin`` into one walk-forward history record, the same
shape as ``benchmark --rolling-origin``; deprecated ``--cv`` is an alias for that mode.

Usage::

    python -m src.benchmarking.parallel_train                 # all 6, -j auto
    python -m src.benchmarking.parallel_train QB RB WR        # subset
    python -m src.benchmarking.parallel_train -j 4            # cap concurrency
    python -m src.benchmarking.parallel_train --rolling-origin # walk-forward report
    python -m src.benchmarking.parallel_train --dry-run       # show the plan, launch nothing
    python -m src.benchmarking.parallel_train --no-sync       # don't touch the website

Prefer the ``scripts/train-local-parallel.sh`` wrapper, which sources ``scripts/wsl-env.sh``
(BLAS caps + ``FF_MODEL_S3_BUCKET`` for the History sync) before invoking this module.
"""

import argparse
import contextlib
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict, deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.benchmarking.benchmark import (  # noqa: E402 — after sys.path bootstrap
    HISTORY_DIR,
    RESULTS_FILE,
    _cohorts_block,
    _maybe_upload_to_s3,
    _print_rolling_origin_table,
    _significance_block,
    collect_global_config,
    collect_pos_config,
    finalize_rolling_origin,
    run_one,
    run_rolling_origin,
    score_one_origin,
)
from src.scripts.bench_fingerprint import collect_code_fingerprints  # noqa: E402
from src.shared.benchmark_utils import (  # noqa: E402
    append_to_history,
    get_git_hash,
    print_comparison_table,
    print_history_comparison,
    summarize_pipeline_result,
    utc_now_iso,
)
from src.shared.core_pool import ENV_ADDR, ENV_POS, start_coordinator  # noqa: E402

ALL_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")

# Fallback heaviest-first dispatch order, used only when ``benchmark_history`` carries no
# usable per-position ``elapsed_sec`` for ``_history_cost_order`` to derive a measured order
# from. A heuristic — it only seeds the ``-j < N`` dispatch order; the core pool balances
# core counts mid-run regardless of who starts first.
_COST_ORDER = ("WR", "RB", "QB", "TE", "DST", "K")


# --------------------------------------------------------------------------- cores


def physical_cores() -> list[int]:
    """Return one logical CPU id per physical core (first sibling of each core).

    LightGBM/BLAS regress under SMT (see ``scripts/wsl-env.sh``), so we pin to physical
    cores only — 16 on the 9950X3D, not the 32 logical. Falls back to the lower half of
    the logical ids if ``lscpu -p`` isn't parseable.
    """
    try:
        out = subprocess.run(
            ["lscpu", "-p=CPU,CORE"], capture_output=True, text=True, check=True
        ).stdout
        first_cpu_of_core: dict[int, int] = {}
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cpu_s, core_s = line.split(",")[:2]
            core = int(core_s)
            if core not in first_cpu_of_core:
                first_cpu_of_core[core] = int(cpu_s)
        cores = sorted(first_cpu_of_core.values())
        if cores:
            return cores
    except Exception:  # noqa: BLE001 — best-effort; fall through to the heuristic
        pass
    n = os.cpu_count() or 2
    return list(range(max(1, n // 2)))


def _split_cores(phys: list[int], n: int) -> list[list[int]]:
    """Partition ``phys`` into ``n`` contiguous chunks, the larger chunks first."""
    n = max(1, min(n, len(phys)))
    base, extra = divmod(len(phys), n)
    chunks, i = [], 0
    for k in range(n):
        size = base + (1 if k < extra else 0)
        chunks.append(phys[i : i + size])
        i += size
    return chunks


def _pin_self(cores: list[int]):
    """Return a ``preexec_fn`` that pins the forked child to ``cores`` before exec."""
    cset = set(cores)

    def _fn():
        # non-Linux or restricted — env thread-caps still bound the fan-out
        with contextlib.suppress(AttributeError, OSError):
            os.sched_setaffinity(0, cset)

    return _fn


def _history_cost_order(history_dir: str = HISTORY_DIR) -> list[str] | None:
    """Heaviest-first order from the most recent history entry with per-position timings.

    Reads the newest ``benchmark_history/*.json`` whose ``results`` carry ``elapsed_sec``
    and returns positions sorted slowest-first. ``None`` when no usable entry exists (the
    caller falls back to ``_COST_ORDER``). Parallel-run times are *contended* wall-clock, so
    this is a rough heavy-first signal — but it only seeds dispatch; the pool self-balances.
    """
    try:
        files = sorted(
            glob.glob(os.path.join(history_dir, "*.json")), key=os.path.getmtime, reverse=True
        )
    except OSError:
        return None
    for path in files:
        entry = _read_summary(path)
        results = (entry or {}).get("results")
        if not isinstance(results, list):
            continue
        timed = [
            (r.get("position"), r.get("elapsed_sec"))
            for r in results
            if r.get("position") and isinstance(r.get("elapsed_sec"), int | float)
        ]
        if len(timed) >= 2:
            return [p for p, _ in sorted(timed, key=lambda x: x[1], reverse=True)]
    return None


def _sort_by_cost(positions: list[str]) -> list[str]:
    """Order positions heaviest-first — measured ``elapsed_sec`` if history has it, else
    the static ``_COST_ORDER`` fallback."""
    order = _history_cost_order() or list(_COST_ORDER)
    rank = {p: i for i, p in enumerate(order)}
    return sorted(positions, key=lambda p: rank.get(p, len(order)))


def _default_jobs(n_positions: int) -> int:
    """Auto concurrency: all positions on a many-core CUDA box, else 1 (sequential)."""
    try:
        from src.shared.platform_detect import detect_platform

        plat = detect_platform()
        if plat.backend == "cuda" and (plat.cpu_count or 0) >= 12:
            return n_positions
        print(
            f"[parallel_train] {plat.backend} host with {plat.cpu_count} CPUs — defaulting "
            "to -j 1 (sequential). Pass -j N to force concurrency.",
            file=sys.stderr,
        )
        return 1
    except Exception:  # noqa: BLE001 — detection is best-effort
        return 1


# ----------------------------------------------------------------------- worker


def _run_worker(
    pos: str,
    summary_out: str,
    rolling_origin: bool,
    significance: bool,
    origin: str | None = None,
) -> int:
    """Train one work unit, write only its metrics summary to ``summary_out`` (JSON).

    The unit is ONE position for a single-split run; for a rolling-origin run it is
    ONE (position × origin) cell when ``origin`` (a test season) is supplied — the
    worker then scores just that origin and writes its single-origin summary, and the
    orchestrator merges the per-position origins via ``finalize_rolling_origin``. A
    rolling-origin worker WITHOUT an ``origin`` (the ``-j 1`` / non-CUDA fallback)
    scores every origin in-process via ``run_rolling_origin``.

    No ``benchmark_results.json`` / history / S3 writes here — those are the
    orchestrator's single-threaded merge step, so concurrent workers never collide.
    """
    if rolling_origin and origin is not None:
        _ts, summary = score_one_origin(pos, int(origin))
    elif rolling_origin:
        summary = run_rolling_origin(pos)
    else:
        result = run_one(pos)
        summary = summarize_pipeline_result(pos, result)
        if significance:
            sig = _significance_block(pos, result)
            if sig is not None:
                summary["significance"] = sig
        cohorts = _cohorts_block(pos, result)
        if cohorts is not None:
            summary["cohorts"] = cohorts
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    tag = f"{pos}:{origin}" if origin is not None else pos
    print(f"[{tag}] worker complete — summary -> {summary_out}")
    return 0


# ------------------------------------------------------------------- orchestrator


def _cell_slug(cell_key: str) -> str:
    """Filesystem-safe slug for a cell key (``"RB:2025"`` -> ``"RB-2025"``)."""
    return cell_key.replace(":", "-")


def _launch(cell_key, pos, origin, cores, tmpdir, logdir, passthrough, pool_addr):
    """Launch one work-unit worker (a position, or a (position × origin) cell).

    ``cell_key`` is the orchestrator's bookkeeping key (``"RB"`` for a single-split
    position, ``"RB:2025"`` for a rolling-origin cell); it names the summary/log files
    so concurrent cells of the SAME position never collide. ``pos`` is the real
    position (passed to ``--worker`` and the core-pool label); ``origin`` (a test
    season string or ``None``) becomes ``--origin`` so the worker scores one origin.
    """
    slug = _cell_slug(cell_key)
    summary_path = os.path.join(tmpdir, f"{slug}.json")
    log_path = os.path.join(logdir, f"local-train-{slug}.log")
    env = dict(os.environ)
    # Each CPU stage leases its thread count from the core pool at runtime (see
    # src/shared/core_pool.py), so we no longer freeze LGBM_N_JOBS/LOKY_MAX_CPU_COUNT per
    # slice — that was immutable post-spawn and stranded freed cores. The worker is pinned
    # to all physical cores at launch (``cores``); the pool narrows affinity per CPU stage.
    # BLAS stays single-threaded so the leased joblib/LightGBM axis owns the fan-out.
    env[ENV_ADDR] = pool_addr
    env[ENV_POS] = cell_key
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(k, "1")
    env.setdefault("FF_DEVICE", "cuda")
    argv = [
        sys.executable,
        "-m",
        "src.benchmarking.parallel_train",
        "--worker",
        pos,
        "--summary-out",
        summary_path,
        *passthrough,
    ]
    if origin is not None:
        argv += ["--origin", str(origin)]
    # Kept open for the subprocess's lifetime (closed in the orchestrator on exit), so a
    # `with` block can't own it here.
    logf = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        argv, env=env, stdout=logf, stderr=subprocess.STDOUT, preexec_fn=_pin_self(cores)
    )
    return {
        "cell_key": cell_key,
        "pos": pos,
        "origin": origin,
        "proc": proc,
        "cores": cores,
        "summary_path": summary_path,
        "log_path": log_path,
        "logf": logf,
        "t0": time.time(),
    }


def _read_summary(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _record_and_sync(
    summaries: list[dict],
    positions: list[str],
    note: str,
    no_sync: bool,
    total_wall_sec: float | None = None,
    rolling_origin: bool = False,
    code_fingerprints: dict[str, str] | None = None,
) -> None:
    """Mirror of ``benchmark.py``'s post-loop block: table, results file, history, S3.

    ``total_wall_sec`` is the orchestrator's measured end-to-end run time (launch of the
    first worker → last worker done); it is recorded on the history entry so the headline
    parallel-run wall-clock is captured automatically, without an external stopwatch. It
    differs from each position's ``elapsed_sec`` (which only sums to the total at ``-j``
    ≥ position count, where every worker launches at once)."""
    if rolling_origin:
        _print_rolling_origin_table(summaries)
    print_comparison_table(summaries, header="MAE Comparison (test set)", show_time=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    git_hash = get_git_hash()
    now = utc_now_iso()
    entry = {
        "run_id": f"{now}_{git_hash}",
        "timestamp": now,
        "git_hash": git_hash,
        "note": note,
        "positions": positions,
        "config": {
            "global": collect_global_config(),
            **{p.lower(): collect_pos_config(p) for p in positions},
        },
        "results": summaries,
        "total_wall_sec": total_wall_sec,
    }
    if code_fingerprints:
        # Only the positions actually recorded — a failed cell's fingerprint
        # must not masquerade as benchmark evidence for that position.
        recorded = {p: code_fingerprints[p] for p in positions if p in code_fingerprints}
        if recorded:
            entry["code_fingerprints"] = recorded
    if rolling_origin:
        entry["mode"] = "rolling_origin"
    written_path = append_to_history(HISTORY_DIR, entry)
    if not no_sync:
        _maybe_upload_to_s3(written_path)
    print_history_comparison(HISTORY_DIR, summaries, exclude_path=written_path)


def _build_cells(positions, rolling_origin):
    """Flatten the run into ``[(cell_key, pos, origin), ...]`` work units, heaviest-first.

    Single-split: one cell per position (``origin`` None, ``cell_key == pos``).
    Rolling-origin: one cell per (position × origin) — for each position
    (heaviest-first), its origins in reverse-chronological order so the latest year
    (the heaviest origin, training on the most seasons) dispatches first. The cell key
    is ``f"{pos}:{ts}"`` so concurrent cells of the same position never collide.
    """
    order = _sort_by_cost(positions)
    if not rolling_origin:
        return [(pos, pos, None) for pos in order]
    from src.config import ROLLING_ORIGIN_TEST_SEASONS

    return [
        (f"{pos}:{ts}", pos, ts) for pos in order for ts in reversed(ROLLING_ORIGIN_TEST_SEASONS)
    ]


def orchestrate(positions, jobs, passthrough, note, no_sync, dry_run, rolling_origin=False) -> int:
    phys = physical_cores()
    cells = _build_cells(positions, rolling_origin)
    cell_keys = [ck for ck, _, _ in cells]
    # Concurrency caps at the cell count: an active cell is one concurrent training run
    # (the core pool's fair-share is over cells, since each is its own GPU+CPU workload).
    jobs = max(1, min(jobs, len(cells)))
    # "6 positions" for a single-split run (one cell per position); "6 positions, 18
    # cells" for a rolling-origin run (each position × origin is its own cell).
    cells_desc = (
        f"{len(positions)} positions, {len(cells)} cells"
        if rolling_origin
        else f"{len(positions)} positions"
    )

    if dry_run:
        first = cell_keys[:jobs]
        init_cap = _split_cores(phys, len(first))  # indicative initial per-cell cap
        print(f"[dry-run] physical cores ({len(phys)}): {phys}")
        print(f"[dry-run] -j {jobs}; {cells_desc}; dispatch order {cell_keys}")
        print(
            f"[dry-run] core pool: each CPU stage leases up to ceil({len(phys)}/active) cores; "
            "the cap widens as cells finish"
        )
        for ck, chunk in zip(first, init_cap, strict=False):
            print(f"[dry-run]   {ck:<10} ~{len(chunk)} cores initially")
        if len(cell_keys) > jobs:
            print(f"[dry-run] queued (dispatched as cells finish): {cell_keys[jobs:]}")
        return 0

    # Fingerprint the code BEFORE dispatch so a mid-run edit can't be
    # laundered into benchmark evidence for code that never trained.
    code_fps = collect_code_fingerprints(positions)

    logdir = "logs"
    os.makedirs(logdir, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="ff-parallel-")
    pool_addr, set_active_count, pool_stop = start_coordinator(phys, tmpdir)
    queue = deque(cells)
    active: OrderedDict = OrderedDict()
    results: dict = {}  # cell_key -> summary | None

    print(
        f"[parallel_train] {cells_desc}, -j {jobs}, "
        f"{len(phys)} physical cores {phys}; core pool {pool_addr}; "
        f"logs -> {logdir}/local-train-<CELL>.log",
        flush=True,
    )

    run_t0 = time.time()
    try:
        while queue or active:
            if queue and len(active) < jobs:
                while queue and len(active) < jobs:
                    cell_key, pos, origin = queue.popleft()
                    active[cell_key] = _launch(
                        cell_key, pos, origin, phys, tmpdir, logdir, passthrough, pool_addr
                    )
                    print(f"[{cell_key}] launched (pid {active[cell_key]['proc'].pid})", flush=True)
                # Tell the pool how many cells now contend so its per-stage fair-share
                # cap (ceil(cores / active)) is right.
                set_active_count(len(active))

            finished = [ck for ck, info in active.items() if info["proc"].poll() is not None]
            if not finished:
                time.sleep(0.5)
                continue

            for cell_key in finished:
                info = active.pop(cell_key)
                info["logf"].close()
                rc = info["proc"].returncode
                elapsed = time.time() - info["t0"]
                summary = _read_summary(info["summary_path"]) if rc == 0 else None
                if summary is not None:
                    summary.setdefault("elapsed_sec", round(elapsed, 1))
                results[cell_key] = summary
                tag = "ok" if summary is not None else f"FAILED (rc={rc})"
                print(
                    f"[{cell_key}] {tag} in {elapsed:.1f}s  (log: {info['log_path']})", flush=True
                )
            # A finished cell lowers the active count, so survivors' next CPU-stage
            # leases widen — and the next queued cell dispatches into the freed slot.
            set_active_count(len(active))
    finally:
        pool_stop()

    total_wall_sec = round(time.time() - run_t0, 1)
    print(
        f"[parallel_train] total wall-clock: {total_wall_sec}s ({cells_desc}, -j {jobs})",
        flush=True,
    )

    summaries, ordered, failed = _merge_cell_results(positions, cells, results, rolling_origin)
    if not ordered:
        print("[parallel_train] all positions failed — nothing recorded.", file=sys.stderr)
        return 1
    if code_fps and collect_code_fingerprints(positions) != code_fps:
        # An edit landed mid-run (even if reverted before commit): some cells
        # trained different code than the snapshot — omit rather than record
        # laundered evidence.
        print(
            "[parallel_train] WARNING: gated code changed during the run; omitting code_fingerprints"
        )
        code_fps = None
    _record_and_sync(
        summaries,
        ordered,
        note,
        no_sync,
        total_wall_sec,
        rolling_origin=rolling_origin,
        code_fingerprints=code_fps,
    )
    if failed:
        print(f"\n[parallel_train] FAILED positions (not recorded): {failed}", file=sys.stderr)
        for p in failed:
            print(f"  see logs/local-train-{p}*.log", file=sys.stderr)
        return 2
    return 0


def _merge_cell_results(positions, cells, results, rolling_origin):
    """Fold per-cell summaries back into one summary per position.

    Returns ``(summaries, ordered_positions, failed_positions)`` in the ORIGINAL
    ``positions`` order. Single-split: each position is its own cell, so this is a
    straight pass-through. Rolling-origin: gather a position's per-origin summaries
    and call ``finalize_rolling_origin`` — but a position is "complete" (and recorded)
    only when EVERY origin landed; one missing origin fails the whole position rather
    than recording a partial mean±std.
    """
    if not rolling_origin:
        ordered = [p for p in positions if results.get(p) is not None]
        failed = [p for p in positions if results.get(p) is None]
        return [results[p] for p in ordered], ordered, failed

    from src.config import ROLLING_ORIGIN_TEST_SEASONS

    by_cell = {(pos, origin): results.get(ck) for ck, pos, origin in cells}
    summaries, ordered, failed = [], [], []
    for pos in positions:
        per_origin = [
            (ts, by_cell[(pos, ts)])
            for ts in ROLLING_ORIGIN_TEST_SEASONS
            if (pos, ts) in by_cell and by_cell[(pos, ts)] is not None
        ]
        if len(per_origin) == len(ROLLING_ORIGIN_TEST_SEASONS):
            summaries.append(finalize_rolling_origin(pos, per_origin))
            ordered.append(pos)
        else:
            failed.append(pos)
    return summaries, ordered, failed


# -------------------------------------------------------------------------- cli


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train all positions in parallel across the box's physical cores, then "
        "merge into one benchmark record and sync to the website History tab."
    )
    p.add_argument(
        "positions", nargs="*", default=list(ALL_POSITIONS), help="Positions (default: all 6)"
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Max concurrent positions (default: autodetect)",
    )
    p.add_argument("--note", default="", help="Describe what changed in this run")
    p.add_argument(
        "--cv",
        action="store_true",
        help="Deprecated local-benchmark alias for --rolling-origin.",
    )
    p.add_argument(
        "--rolling-origin",
        action="store_true",
        help="Walk-forward multi-season TEST eval; fans each position's origins out as "
        "(position × origin) cells and reports rolling-origin history.",
    )
    p.add_argument(
        "--significance", action="store_true", help="Attach paired-bootstrap CI (single-split only)"
    )
    p.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip the S3/History mirror (local files still written)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the plan and launch nothing")
    # Internal: a single-position worker invocation spawned by the orchestrator.
    p.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    p.add_argument("--summary-out", default=None, help=argparse.SUPPRESS)
    # Internal: when set, the worker scores ONE rolling origin (this test season)
    # rather than looping every origin in-process — the (position × origin) cell.
    p.add_argument("--origin", default=None, help=argparse.SUPPRESS)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    rolling_origin_mode = args.rolling_origin or args.cv
    if args.cv and not args.rolling_origin:
        print(
            "DEPRECATED: parallel_train --cv now aliases --rolling-origin walk-forward reporting."
        )

    if args.worker:
        return _run_worker(
            args.worker.upper(),
            args.summary_out,
            rolling_origin_mode,
            args.significance,
            args.origin,
        )

    requested = [p.upper() for p in (args.positions or ALL_POSITIONS)]
    unknown = [p for p in requested if p not in ALL_POSITIONS]
    if unknown:
        print(f"[parallel_train] ignoring unknown positions: {unknown}", file=sys.stderr)
    positions = [p for p in requested if p in ALL_POSITIONS]
    if not positions:
        print("[parallel_train] no valid positions given.", file=sys.stderr)
        return 1

    jobs = args.jobs if args.jobs else _default_jobs(len(positions))
    passthrough = []
    if rolling_origin_mode:
        passthrough.append("--rolling-origin")
    if args.significance:
        passthrough.append("--significance")
    return orchestrate(
        positions,
        jobs,
        passthrough,
        args.note,
        args.no_sync,
        args.dry_run,
        rolling_origin=rolling_origin_mode,
    )


if __name__ == "__main__":
    sys.exit(main())
