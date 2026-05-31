# Consolidate `--cv` reporting into `--rolling-origin`

> **Status:** Planned, not yet implemented. Parked as a priority note in
> [TODO.md](../TODO.md) (Open → *Consolidate `--cv` benchmark reporting into the clean
> rolling-origin walk-forward*). A future session can pick this up directly from this doc.
> Authored 2026-05-30.

## Context

The benchmark CLI has **two overlapping multi-season evaluation paths**, and the older one is
mildly optimistic and caps at 2024:

- **`--cv`** → [`run_cv_pipeline`](../src/shared/pipeline.py) drives
  [`expanding_window_folds`](../src/data/split.py) over `CV_VAL_SEASONS=[2021,2022,2023,2024]`.
  Per fold it trains on `[2012..V-1]` and **scores on the same season V it early-stops /
  selects on** (`val == eval`) → mildly optimistic. It never sees 2025 (2025 is the held-out
  production test season). Reports ridge/nn/lgbm only.
- **`--rolling-origin`** → [`run_rolling_origin`](../src/benchmarking/benchmark.py) drives
  [`rolling_origin_folds`](../src/data/split.py) over `ROLLING_ORIGIN_TEST_SEASONS=[2023,2024,2025]`.
  Per origin: train `[..T-2]` / val `T-1` / **test `T` (clean, never tuned on)**. Runs the full
  `run_pipeline`, so it covers **all** models, **includes 2025**, and its final origin is
  byte-identical to the production single split.

`--rolling-origin` is a **strict superset** of what `--cv` reports — clean (`val != eval`),
all models, includes 2025 — so the two are redundant and `--cv`'s numbers are the optimistic,
2024-capped subset. Both landed in the **same 2026-05-29 cycle** (ADR `Update history` lines
16–17), so consolidating is timely, not second-guessing stable code.

**Goal:** one multi-season eval-of-record (the clean walk-forward); stop emitting the optimistic
expanding-window numbers as a parallel benchmark mode — at **zero retrain cost**.

| | `--cv` (`run_cv_pipeline`) | `--rolling-origin` (`run_rolling_origin`) |
|---|---|---|
| Scored seasons | 2021–2024 (val) | 2023–2025 (test) |
| val vs eval | **val == eval** (optimistic) | **val != eval** (clean) |
| Models scored multi-season | ridge, nn, lgbm | **all** (ridge/nn/enet/attn/lgbm) |
| Includes 2025? | No | **Yes** (final origin = production split) |
| Retrain trigger? | in `src/shared/` | driver in `src/benchmarking/` (no trigger) |

## Key constraints discovered

- **Retrain triggers** ([`scope_positions.py`](../src/scripts/scope_positions.py), `_GLOBAL_REGEX`):
  `src/shared/`, `src/data/`, `src/config.py`, `src/{pos}/` → **6-position GPU retrain**.
  `src/benchmarking/` → **no retrain** (only the `shared` *test* shard).
- `run_cv_pipeline` / per-position `run_cv()` / `get_cv_runner` / `has_cv_runner` are consumed
  **only** by the benchmark `--cv` flag and unit tests — **not** by AWS Batch training (production
  trains via `run()`/`run_pipeline`). Changing the `--cv` flag's behavior does not affect trained
  artifacts or serving.
- The `cv_*` summary keys are read **only** by `benchmark_utils` (the stdout CV table +
  `print_history_comparison`); the serving app / History tab do **not** read them. Old history
  JSONs with `cv_*` keys still render; new runs write a `rolling_origin` block + `"mode":
  "rolling_origin"` marker (additive, already supported).
- Blast radius of changing `--cv`: **manual operator runs only**. `pre-pr.sh`, CI `tests.yml`, and
  the training workflows never invoke `--cv`.

## Recommended change (zero retrain, single PR)

Consolidate at the **benchmark surface** — `--cv` becomes a deprecated alias that runs the clean
rolling-origin walk-forward. The optimistic expanding-window reporting is no longer emitted as a
benchmark mode; `run_cv_pipeline`/`run_cv()` stay callable for ad-hoc use (they were just extended
to all six positions and are unit-tested — no reason to remove them).

### 1. `src/benchmarking/benchmark.py` — `__main__` dispatch (the core change)

In the `__main__` block:
- Compute `use_rolling = args.rolling_origin or args.cv`.
- If `args.cv and not args.rolling_origin`, print a one-line note: *"`--cv` now runs the clean
  rolling-origin walk-forward (val != eval, includes 2025); the expanding-window in-loop CV is
  deprecated as a benchmark mode. See ADR D1."*
- `mode = "ROLLING-ORIGIN" if use_rolling else "SINGLE-SPLIT"`.
- Per position: `if use_rolling: s = run_rolling_origin(pos)`, else the existing single-split branch
  (drop the now-vestigial `and not args.cv` guard on `--significance`, since `--cv` no longer
  reaches that branch).
- Set `entry["mode"] = "rolling_origin"` and call `_print_rolling_origin_table(summaries)` when
  `use_rolling` (not just `args.rolling_origin`).
- Update the `--cv` `help=` to describe it as a deprecated alias for `--rolling-origin`; tweak the
  module header usage comment and the `--significance` help (`ignored for --rolling-origin/--cv`).

Leave `run_one` as-is (its `cv=True` branch and the `get_cv_runner` import simply stop being
reached from `__main__`; tests that call it directly keep working).

### 2. `docs/ARCHITECTURE.md` — ADR D1 + Update history

Add an `Update history` line and extend the **D1 rolling-origin** paragraph: rolling-origin is now
the **multi-season eval-of-record**; `--cv` is a deprecated alias for it; `expanding_window_folds`
is retained **only** as the LightGBM tuner's fold generator ([`tune_lgbm.py`](../src/tuning/tune_lgbm.py)),
where `val == eval` is the correct tuning objective, not optimism.

**Files touched:** `src/benchmarking/benchmark.py`, `docs/ARCHITECTURE.md` → **no retrain
trigger**; not `[docs-only]` (benchmark.py is behavioral, so CI still runs the `shared` test shard).

### What is intentionally dropped vs kept

- **Dropped:** the expanding-window `cv_*` mean±std as a *benchmark mode* — the optimistic,
  2024-capped numbers. The clean replacement (rolling-origin, incl. 2025) is strictly more
  informative.
- **Kept:** `run_cv_pipeline` / `run_cv()` / `get_cv_runner` and their tests (ad-hoc use; freshly
  extended 2026-05-29). `cv_*` plumbing in `benchmark_utils` stays so historical JSONs render.

## Optional follow-ups (each carries a cost — not in the primary PR)

- **A — `expanding_window_folds` docstring clarification** *(separate `[docs-only]` PR)*: note it is
  a tuning-fold generator where `val == eval` is the objective; point to `rolling_origin_folds` for a
  clean forward holdout. Lives in `src/data/split.py`, which **is** a retrain trigger, so it must ship
  as its **own `[docs-only]`-tagged PR** to skip the retrain — cannot be bundled with the behavioral
  benchmark.py change.
- **B — extend clean coverage** *(`src/config.py` → 6-pos retrain)*: set
  `ROLLING_ORIGIN_TEST_SEASONS=[2022,2023,2024,2025]` (or `2021..2025`) so the clean walk-forward
  matches/exceeds the old expanding-window's 2021–2024 coverage.
- **C — "both columns" combine** *(`src/shared/pipeline.py` → 6-pos retrain + benchmark diff)*:
  surface per-model **val** MAE through `run_pipeline`/`summarize_pipeline_result` so the
  rolling-origin table shows `val` (in-loop) **vs** `test` (clean) per origin — quantifying the
  optimism gap directly. The most literal "combine," but heaviest.
- **D — full removal** *(6-pos retrain; reverses just-shipped work)*: delete `run_cv_pipeline`,
  `run_cv()`, `get_cv_runner`, `has_cv_runner`, `cv_metrics` plumbing, the CV table printer, and
  `tests/{te,dst,wr}/test_run_cv_pipeline.py`. **Not recommended now** — high churn, undoes the
  2026-05-29 TE/K/DST extension.

## Verification

1. **Dispatch behaves:** `python -m src.benchmarking.benchmark QB --cv` → prints the deprecation
   note, runs the walk-forward (prints "Rolling-Origin TEST metrics" with a per-origin table
   **including a 2025 row**), and writes a `benchmark_history/*.json` carrying `"mode":
   "rolling_origin"` + a `rolling_origin` block. `--rolling-origin` unchanged. *(Worktree: symlink
   `data/{splits,raw}` first; if local splits are stale vs main's feature whitelist the pipeline
   KeyErrors — then rely on the unit tests, which monkeypatch `_score_origin` and need no real data.)*
2. **Tests green:** `pytest -m unit` plus `tests/shared/test_benchmark_utils.py` and any
   rolling-origin dispatch tests. The `run_cv_pipeline` tests must still pass (the function is
   unchanged). Optionally add a small test asserting `--cv` selects the rolling-origin path.
3. **No retrain fires:** `git fetch origin main --quiet && git diff --name-only origin/main |
   python3 -m src.scripts.scope_positions` → **empty output**.
4. **Lint:** `ruff check . && ruff format --check .`.
