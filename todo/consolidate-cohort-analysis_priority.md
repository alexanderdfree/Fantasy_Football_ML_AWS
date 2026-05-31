# Consolidate cohort/subgroup analysis scripts into one comprehensive script

> **Status: PLANNED (not yet implemented).** Checkpointed 2026-05-31 from a planning session.
> Decisions are locked (see "Context"); a future session can implement directly from this file.
> Tracked by the `[PLANNED]` entry in [TODO.md](../TODO.md).

## Context

Cohort/subgroup error diagnostics are currently scattered across three standalone
`src/analysis/` CLIs, each with its own duplicated `MODELS`/`ACTUAL` constants,
its own data-loading, and its own per-model-error table:

- [`rb_ascension.py`](../src/analysis/rb_ascension.py) — RB backup→workhorse **ascension** cohort (+ injury attribution); lag-gap / convergence / depth-coverage deep-dives.
- [`analysis_late_week_effect.py`](../src/analysis/analysis_late_week_effect.py) — season-aware **late-week** cohort; label-anomaly (stage1), prediction-degradation (stage2), and a KEEP-vs-CUT ablation.
- [`analysis_rb_lgbm_disagreement.py`](../src/analysis/analysis_rb_lgbm_disagreement.py) — RB **sparse-history / season-opener** cohort (history-depth buckets) + an LGBM-vs-peers model-comparison deep-dive (calibration, gap decomposition).

The goal is to fold these into **one comprehensive cohort-analysis script** that also adds the
named cohorts that have no script yet. Per the clarifying answers from the planning session:

1. **Full single-file replacement** — migrate all cohort logic (incl. bespoke deep-dives) into the new module and **delete** the three old files, migrating their tests.
2. **All feasible cohorts** — unify the three existing cohorts **and** build net-new predicates for **rookies**, **one-two-punch (RB committee)**, **mid-season trade**, and **injury-return**.
3. **Suspension return** — documented as **infeasible** (no nflverse suspension data source); a registry stub prints why rather than fabricating data.

Intended outcome: a single registry-driven `src/analysis/cohort_analysis.py` giving a **uniform
per-model MAE/RMSE/bias/n-by-cohort report** across positions, with each cohort's bespoke deep-dive
preserved behind a flag, all read-only (no model/feature/target code touched → **no retrain**).

## ⚠ Reconciliation with already-merged work (added 2026-05-31, post-rebase)

The branch base lagged `main`; while this plan was being checkpointed, `main` shipped two more
scattered cohort scripts that **overlap the net-new cohorts** below. The implementation must
**absorb/reuse these, not rebuild them**:

- **`rookie`** → [`src/analysis/rookie_cohort_metrics.py`](../src/analysis/rookie_cohort_metrics.py)
  (PR #620) already computes rookie-vs-veteran MAE/bias + bias-corrected MAE (the "tracked
  rookie-subgroup metric"). Reuse its rookie-labeling + metric logic as the `rookie` cohort's
  `label_fn`/report instead of writing a new `years_exp` predicate. (Separately, the `[PRIORITY]`
  rookie-bias *fix* in TODO.md is a **model change** — out of scope here; this consolidation stays
  read-only/analysis-only.)
- **`injury_return`** → [`src/analysis/injury_subgroup_error.py`](../src/analysis/injury_subgroup_error.py)
  (PR #623) already does injury/return subgroup-error analysis (paired with the
  `src/tuning/ablate_injury_features.py` ablation). Fold its predicate/metrics into the
  `injury_return` cohort rather than re-deriving from the injuries loader.

Net effect: the consolidation now unifies **five** scattered cohort scripts (the original three +
these two), which strengthens the rationale. Re-scan `git log origin/main --grep` for
`cohort`/`subgroup`/`rookie`/`injury` at implementation time in case more has shipped, and keep the
two findings docs (`rookie_cohort_findings_priority.md`, plus the three originals) as conclusions.

## Scope

**In scope** — one new module, one consolidated test module, deletion of the three source files +
their three test files, and doc-reference updates.

**Out of scope (leave as-is):** the non-cohort analysis scripts — `analysis_feature_audit.py`,
`analysis_rb_feature_audit.py`, `analysis_k_feature_audit.py`, `audit_depth_alignment.py` (a
depth-chart data-quality audit, not a player cohort), `analysis_nflcom_baseline.py`,
`analysis_expert_comparison.py`, `analysis_dst_rare_dispersion.py`, etc. The three **findings
`.md`** write-ups (`rb_ascension_findings.md`, `late_week_effect_findings.md`,
`rb_lgbm_disagreement_findings.md`) stay — they are conclusions, not scripts; only their
"Reproduce:" command lines get updated to the new CLI.

## New module: `src/analysis/cohort_analysis.py`

A registry-driven script. Importing it must stay cheap (no torch) — the pipeline `run`/`get_runner`
is imported **lazily inside the orchestrator**, never at module top (preserve the
`if __name__ == "__main__"` gate; this is asserted by the migrated import-smoke tests).

### Cohort registry

A `CohortSpec` dataclass and a `COHORTS: dict[str, CohortSpec]` registry. Each spec carries:

- `name`, `description`, `positions` (applicable set), `cohort_value` (the "interesting" bucket label).
- `label_fn(df) -> pd.Series` — **pure**, defensive: labels each `result["test_df"]` row into the
  cohort bucket vs complement, returning all-`"unknown"` if a required column is missing (mirrors
  `label_ascension_rows`). Unit-tested on synthetic frames.
- optional `enrich` flags (`needs_rosters`, `needs_injuries`) — the orchestrator merges the aux
  columns onto `test_df` *before* `label_fn` runs, so `label_fn` stays pure/testable.
- optional `data_diagnostic()` (no-model analysis) and `deep_dive(result_test_df)` (bespoke).

### Cohort table

| Cohort (`name`) | Positions | Predicate (read-only label on `test_df`) | Source |
|---|---|---|---|
| `ascension` | RB | prior-3g opp ≤ 8 **and** week-W opp ≥ 18 | reuse `label_ascension_rows` |
| `injury_return` | QB/RB/WR/TE/K/DST | player Out/Doubtful (or absent) in W-1, active in W | net-new; reuse injuries-merge discipline from `add_injury_attribution` |
| `rookie` | QB/RB/WR/TE | `years_exp == 0` / `season == entry_year` (merged from rosters) | net-new |
| `committee` | RB | ≥2 RBs on same `(recent_team, season, week)` in a mid carry-share band | net-new |
| `trade` | QB/RB/WR/TE | `recent_team` changes within the test season for a `player_id` | net-new (re-derive `team_changed`) |
| `sparse_history` | QB/RB/WR/TE/K/DST | `n_prior_games` buckets; opener = `npg==0` | reuse `add_history_depth` |
| `late_week` | QB/RB/WR/TE/K/DST | season-aware final/penult/early week | reuse `assign_final_week_buckets` |
| `suspension_return` | — | **infeasible**: `label_fn` returns all-`"unknown"` + logs "no nflverse suspension data source" | stub |

Net-new predicates impute `NaN→0` before any feature comparison (production convention,
`feature_build.py`-style; the existing `label_ascension_rows` already does `.fillna(0)`).

**Loader-matching data path (critical):** `rookie` must read `years_exp`/`entry_year` the loader's
way — reuse `src/data/loader.py`'s roster fetch/normalize, **not** a raw `nfl_source.rosters` shim
that can lag schema. This mirrors how `rb_ascension._depth_chart_ranks` deliberately uses
`_normalize_espn_depth` instead of the raw depth shim. Same for `injury_return` via the injuries
loader path.

### Uniform per-model-error report (the core deliverable)

For each requested cohort × applicable position: run that position's pipeline **once** (lazy
`get_runner(pos)()`), label `test_df`, and emit a table of **MAE / RMSE / bias / n per cohort
bucket + ΔMAE vs overall**, per model. Build it on:

- `src/analysis/significance.py::pred_columns_from_test_df` — dynamic per-position model-column
  discovery (robust to which models a position enables), with `MODELS` labels as the fallback.
- `per_model_metrics` (migrated from the LGBM module — handles empty slices → NaN, n=0 without
  crashing) and/or `src/shared/error_analysis.py::compute_stratum_metrics`.
- optional top-k hit-rate + Spearman per bucket via the migrated `_ranking_by_bucket` /
  `_avg_weekly_ranking` (generalize the late-week ranking helper to any cohort).

Orchestrator runs the **union** of needed positions once each (≤6 pipeline runs total regardless of
how many cohorts are selected), then applies every applicable cohort label to each position's
`test_df`.

### Bespoke deep-dives (preserved behind `--deep-dive`)

Migrate verbatim, attached to their cohort so nothing is lost:

- **ascension** → lag-gap, convergence, depth-coverage, concrete examples (`prepare_weekly`,
  `find_ascension_events`, `add_injury_attribution`, `convergence_table`, `_offense_depth_ranks`,
  `_depth_chart_ranks` + their `_print_*`). The data-only nflverse-weekly path becomes the
  ascension `data_diagnostic`.
- **sparse_history** → LGBM-vs-peers disagreement, calibration table, gap decomposition, sparse+hot
  split, history-depth table, PNG plots (`available_models`, `peer_gap`, `calibration_table`,
  `gap_decomposition`, `history_depth_table`, `_make_plots`).
- **late_week** → stage1 label-anomaly (`_composition_table`, `_player_relative_drop`/`_table`,
  `_era_contrast_table`) as the late-week `data_diagnostic`; stage2 ranking; the KEEP-vs-CUT
  `run_ablation` exposed via `--ablation`.

### CLI

```
python -m src.analysis.cohort_analysis [COHORTS ...] [options]
```
- positional `COHORTS` ⊆ registry keys; default = all feasible.
- `--with-model-error` — run needed pipeline(s) and emit the uniform per-model report (default is
  **data-only**: cohort sizing + `data_diagnostic`s only — fast, no torch, matches the existing
  data-only defaults of `rb_ascension`/`late_week stage1`, avoids surprise ~6× pipeline runs).
- `--deep-dive` — also emit each cohort's bespoke sections.
- `--positions …` (restrict), `--ablation` (late_week), `--no-plots`, `--splits-dir`,
  `--established-games`, `--top-k`.

### Constant de-duplication

The merged module has **one** `MODELS` (`{Ridge,NN,Attention NN,LightGBM} → pred_*_total`), one
`ACTUAL = "fantasy_points"`, and keeps the non-colliding constants (`BACKUP_OPP`, `WORKHORSE_OPP`,
`EARLY/PENULT/FINAL`, `SKILL_POSITIONS`, `HISTORY_DEPTH_BUCKETS`, `HOT_RECENT_FP`, `PEERS`, `LGBM`,
`DISAGREEMENT_THRESHOLD`, `CALIB_BINS`, etc.). Collapsing the duplicated `MODELS`/`ACTUAL` is part of
the consolidation value.

## Files

**Create**
- `src/analysis/cohort_analysis.py` — the consolidated module (large, ~700–900 lines; expected given the full-replacement scope).
- `tests/analysis/test_cohort_analysis.py` — migrate the assertions from all three existing test modules (all use synthetic frames — no splits/network needed), re-pointed at the new module's symbols, plus an import-smoke test asserting `run`/`get_runner` is **not** bound at module scope (operator-CLI drift guard).

**Delete**
- `src/analysis/rb_ascension.py`, `src/analysis/analysis_late_week_effect.py`, `src/analysis/analysis_rb_lgbm_disagreement.py`
- `tests/analysis/test_rb_ascension.py`, `tests/analysis/test_analysis_late_week_effect.py`, `tests/analysis/test_analysis_rb_lgbm_disagreement.py`

**Modify (doc references only)**
- `TODO.md` — the `analysis_rb_lgbm_disagreement` module-path + "reproduce via the extended … CLI" mentions in the two archive entries (~lines 88 / 93) → repoint to `cohort_analysis sparse_history --with-model-error --deep-dive`.
- `src/analysis/rb_ascension_findings.md`, `src/analysis/late_week_effect_findings.md`, `src/analysis/rb_lgbm_disagreement_findings.md` — update each "Reproduce:" line to the new CLI invocation.

*(Worktree path discipline: every Edit/Write `file_path` must carry the active worktree prefix
`/.../Final-Project/.claude/worktrees/<worktree>/` — a bare `src/...` path silently writes the
parent `main` checkout. Re-prefix, then grep the new symbol in the worktree file to confirm.)*

## Conventions & stop-rules honored

- **No retrain / safe placement.** Everything lives under `src/analysis/` (+ `tests/analysis/` + docs). `src/scripts/scope_positions.py` maps `src/analysis/` → no positions (contract-tested), so this fires no GPU retrain and has zero metric impact. **Do not** edit `src/shared/error_analysis.py` or any `src/{pos}/` — import their helpers instead.
- **Rookie cohort ≠ the rejected rookie *feature*.** TODO.md's `[TESTED, REJECTED] Draft-capital / combine rookie cold-start features` rejected feeding `years_exp`/draft-capital to the **model**. This script uses rookie status only as a **read-only error-analysis label** — no model input changes. It is in fact the *"tracked rookie-subgroup metric"* that the rejection entry names as the precondition for ever revisiting rookie features. Call this out in the module docstring so a future reader doesn't think we're re-litigating it.
- **K/DST `fantasy_points` caveat.** Raw-split `fantasy_points` is skill-only (≈0 for K, absent for DST); only post-pipeline `result["test_df"]` carries correct K/DST totals. So data-only/label-anomaly diagnostics stay skill-only (as late-week already enforces via `SKILL_POSITIONS`); the per-model report covers all six via the pipeline path.
- **Lazy pipeline import**, pure helpers unit-tested, IO/presentation not — same contract as the three originals.
- **Not `[docs-only]`** — the implementation PR is behavioral (new + deleted code). (This *checkpoint* PR — plan doc + TODO note only — is docs-only.)

## Verification

1. **Unit tests (primary):** `pytest -m unit tests/analysis/test_cohort_analysis.py` — the migrated synthetic-frame assertions (ascension labeling/thresholds/injury-attribution/convergence, season-aware final week, eval-week bucketer, history-depth reset, per-model-metrics empty-slice guard, peer-gap, gap-decomposition, calibration shape) plus net-new predicate tests (rookie/committee/trade/injury_return label logic + missing-column guard) and the import-smoke test.
2. **Lint:** `ruff check src/analysis/cohort_analysis.py tests/analysis/test_cohort_analysis.py && ruff format --check`.
3. **Data-only CLI smoke:** `python -m src.analysis.cohort_analysis --data-only` (and `ascension`/`late_week` deep-dives) — exercises the no-model paths end-to-end. Run from the **miniforge3 base interpreter** (`/Users/alex/miniforge3/bin/python`) — the worktree has no `.venv` and the parent `.venv` may be stale; base has all deps and mirrors the pre-pr gate.
4. **One per-model run if splits are fresh:** `python -m src.analysis.cohort_analysis ascension committee --positions RB --with-model-error`. Note: symlinked worktree `data/splits` may lag `main`'s feature whitelist and fail-loud (`KeyError: missing cols`); if so, rely on (1)+(3) and run the full per-model path against fresh splits, rather than blocking on stale data.
5. **No-retrain confirmation:** `python -m src.scripts.scope_positions src/analysis/cohort_analysis.py` → `[]`.
6. Before PR: invoke `pre-pr-judge`; the `src/analysis/`-only diff should not trip the benchmark-freshness gate (no model/feature/target change). If it false-positives, surface options rather than running a full sweep.
