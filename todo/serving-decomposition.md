# Serving `app.py` decomposition — continuation spec

**Status: COMPLETE.** All increments shipped — `serialization.py` (#990),
`metadata.py`/`wiki.py`/`comparison.py`/`benchmark_history.py` (#998),
`core.py` (#1005), and `routes.py` + composition-root `app.py` (this PR).
**`app.py` is now ~107 LOC** (Flask app + shared `_cache`/locks + errorhandler +
explicit re-exports + bottom routes import) — down from 3085. The serving package
is 8 cohesive modules. This doc is retained as the record of how it was done.

**Key decision (revised from the original plan): there is no `state.py`.** The
shared mutable serving state — `_cache`, `_cache_lock`, `_results_write_lock`,
`_wiki_cache_lock`, and the cache-dir path constants (`_PREDICTIONS_CACHE_DIR`,
…) — **stays in `app.py`**. Extracted modules that need it reach it via a
lazy `import src.serving.app as app_pkg` (call-time attribute access:
`app_pkg._cache`), which breaks the cycle. Rationale: `_cache` is rebound by
~9 monkeypatches and read via ~136 references across fixtures (`_stub_app`,
`_mocked_app`, `degraded_mode_app` all yield the real app module), so moving it
to a `state.py` would force repointing all of them — high-risk churn — whereas
production never *reassigns* `_cache` (only mutates), so call-time attribute
access on `app.py`'s binding is correct and keeps every `_cache` patch working
unchanged. Same treatment for the cache-dir constants. So core/routes keep
`app_pkg.<state>` and their `_cache`/`_PREDICTIONS_CACHE_DIR` patches stay
`app_mod.*` (no repoint).

## Goal

Break the `src/serving/app.py` monolith (3085 LOC after #990, 18 routes, ~60 defs)
into cohesive modules with a thin composition-root `app.py`. **Pure relocation —
behavior-preserving.** Guarded by the serving test suite (`tests/test_app*.py`,
**235 tests** = 52 unit + 183 integration). No training-path files touched → no retrain.

## What's done

- **`src/serving/serialization.py`** (#990): pure JSON/scoring helpers
  (`_safe_num/_safe_str/_round_or_none`, `_validate_scoring`, `_actual_col/_pred_col`,
  `_records_to_player_rows`, `_VALID_SCORING/_MODEL_PRED_PREFIXES/_PLAYER_ROW_COLS`).
  `app.py` imports + re-exports them. Zero patch repoints (none were monkeypatched).
  This PR also moved `_MODEL_PRED_COLUMNS` here (the `(display_name, prefix)` pairs,
  shared by `comparison._best_model_arrays` and core `_compute_metrics_locked`).
- **`src/serving/metadata.py`** (this PR): `POSITION_INFO` + `_ALL_TARGETS` (static
  UI metadata; imports the 6 position config modules). Re-exported by `app.py`;
  not monkeypatched → zero repoints.
- **`src/serving/wiki.py`** (this PR): in-app markdown wiki (`WIKI_DOCS`, the ADR
  auto-register loop, `_WIKI_*` constants, `_wiki_rewrite_href`, `_render_wiki_doc`).
  Uses `app_pkg._cache`/`_wiki_cache_lock` via the lazy import. `app.py` re-exports
  `WIKI_DOCS`/`_render_wiki_doc`/`_wiki_rewrite_href`/`_WIKI_GITHUB_BLOB_BASE`; the
  `/api/wiki/*` routes stay in `app.py`. Repointed `WIKI_DOCS`/`__file__`/`markdown`
  patches (test_app_audit320.py, test_app_gaps.py) to `wiki`.
- **`src/serving/comparison.py`** (this PR): model-vs-expert helpers
  (`_load_comparison_experts`, `_load_expert_intervals`, `_best_model_arrays`,
  `_model_block_from_results`, `_model_reliability_from_results` + the two committed-JSON
  path constants). Pure over the `results` frame — no app state. `/api/comparison`
  stays in `app.py`, calling `comparison.<fn>`. Repointed 19 refs in test_app_comparison.py.
- **`src/serving/benchmark_history.py`** (this PR): History-tab JSON parsing/projection
  (`_resolve_repo_slug`, `_benchmark_*`, `_load_benchmark_history_rows`, `_TARGET_LABELS`,
  the module-local `_BENCHMARK_HISTORY_CACHE`). No app state. `/api/benchmark_history`
  stays in `app.py`, calling `benchmark_history.<fn>`. Repointed 14 refs in
  test_app_benchmark_history.py. (`app_mod.json.load` patch needs no repoint — json is a
  shared singleton and `app.py` still imports it.)

## The hard constraint (why this is not "just move code")

The serving tests do **~137 `monkeypatch.setattr` calls** against the app module,
patching **46 symbols** — including *imported* names (`joblib`, `pd`, `json`,
`markdown`, `RidgeMultiTarget`, `LightGBMMultiTarget`, `MultiHeadNet*`,
`build_position_features`, `read_scaler_meta`, `assert_scaler_matches`,
`k_data/k_features/dst_data/dst_features`, `POSITION_REGISTRY`,
`upload_predictions_cache_to_s3`, `refresh_sentinel_mtime`) and internals
(`_cache`, `_ensure_metrics`, `_apply_position_models`, `_PREDICTIONS_CACHE_DIR`, …).

Python monkeypatching only works if the **code that uses a name looks it up in the
module where the test patched it**. So when a symbol moves from `app.py` to module
`M`, every `monkeypatch.setattr(app_mod, "X", …)` for that symbol must repoint to
`monkeypatch.setattr(M, "X", …)`, and `app_mod.X` reads → `M.X`. Test aliases vary
across the 12 files (`app_mod`, `app_module`, `app`, `_stub_app`, and one direct
`joblib`), so repoints are **per-file**, not one sed.

`_cache` is the worst case: it's REBOUND by tests (`setattr(app_mod, "_cache", {})`,
8×), so re-export alone is insufficient — the code must access it as `state._cache`
(attribute), and those 8 patches must point at `state`.

## Target module layout

| Module | Holds | Notes |
|---|---|---|
| `serialization.py` | ✅ done | pure helpers (no `_cache`) |
| `metadata.py` | `POSITION_INFO`, `_ALL_TARGETS` | static dict; imports the 6 `*_cfg`; not patched → re-export only |
| `state.py` | `_cache`, `_cache_lock`, `_results_write_lock`, `_wiki_cache_lock`, `_ALL_POSITIONS`, `_APPENDED_POSITIONS`, `_MODEL_PRED_COLUMNS`, cache-dir path constants (`_PREDICTIONS_CACHE_DIR`, …) | the foundation; convert `_cache` → `state._cache` everywhere; repoint `_cache` (8) + `_PREDICTIONS_CACHE_DIR` (4) patches |
| `wiki.py` | `WIKI_DOCS`, `_wiki_*`, `_render_wiki_doc`, `_WIKI_*` | uses `markdown` (1 patch) + `state._cache`/`_wiki_cache_lock` |
| `core.py` | `_apply_position_models`, `_load_{k,dst}_splits`, `_ensure_base_data`/`_load_base_data_locked`/`_refresh_k_data_locked`, `_ensure_position_loaded`/`_ensure_all_positions_loaded`, fingerprint/snapshot/hydrate/persist, `_ensure_metrics`/`_invalidate_metrics_cache`/`_compute_metrics_locked`, `_get_data`, `_degraded_positions`, `_compute_scoring_formats` | the bulk; ~40 of the heavy patch repoints land here (model classes, `joblib`, `build_position_features`, scaler IO, `k_data`/`dst_*`, `POSITION_REGISTRY`, `upload_predictions_cache_to_s3`, `refresh_sentinel_mtime`, `pd`) |
| `comparison.py` | `_load_comparison_experts`, `_load_expert_intervals`, `_best_model_arrays`, `_model_block_from_results`, `_model_reliability_from_results` | uses `compute_metrics`; `_load_comparison_experts` patched (8) → repoint |
| `benchmark_history.py` | `_resolve_repo_slug`, `_benchmark_*`, `_load_benchmark_history_rows`, `_TARGET_LABELS` | `_benchmark_history_dir`/`_compute_models_fingerprint` patched → repoint; uses `json` |
| `routes.py` (or `routes/` pkg) | the 18 `@app.route` handlers → a `Blueprint` | imports from `core`/`serialization`/`wiki`/`comparison`/`benchmark_history` |
| `app.py` | composition root: create `Flask`, register Blueprint + `@app.errorhandler`, **re-export** the 8 `from src.serving.app import …` names | the only names tests still import directly: `POSITION_INFO`, `_ALL_TARGETS`, `_MODEL_PRED_PREFIXES`, `_WIKI_GITHUB_BLOB_BASE`, `_wiki_rewrite_href`, `_round_or_none`, `_safe_num`, `_safe_str` |

Circular-import rule: `state.py` imports nothing local; everyone imports `state`.
`core`/`wiki`/`comparison`/`benchmark_history` import `state` (+ `serialization`).
`routes` imports the feature modules. `app.py` imports all + registers — it is
imported by no one, so no cycle.

## Recommended increments (each lands green against all 235 tests)

1. ✅ **`metadata.py`** — DONE (this PR). Static, re-export only, ~0 repoints.
2. ~~`state.py`~~ — **DROPPED.** State stays in `app.py` (see Key decision above);
   no `_cache`→`state._cache` conversion, no `_cache`/`_PREDICTIONS_CACHE_DIR` repoints.
3. ✅ **`wiki.py` + `comparison.py` + `benchmark_history.py`** — DONE (this PR). Leaf
   clusters; repointed `markdown`/`WIKI_DOCS`/`__file__` (wiki), `_load_comparison_experts`
   et al. (comparison), `_benchmark_history_dir`/`_BENCHMARK_HISTORY_CACHE` (benchmark).
   `_compute_models_fingerprint` is a *core* symbol (not benchmark) — its 3 patches stay
   `app_mod.*` until increment 4.
4. ✅ **`core.py`** — DONE (this PR). Extracted ~1585 LOC (model load / data / cache /
   metrics); cache-dir consts moved with it; `_cache`+locks stay in app.py reached via a
   module-level `import src.serving.app as app_pkg` (cycle-safe both import orders). The
   ~224 test repoints used a `tokenize`-based NAME rewriter (state→`app_pkg`, core fns→`core`).
   gunicorn.conf.py's pre-warm thread repointed to `core._ensure_metrics`. Original plan note:
   work. Uses `app_pkg.<state>` for `_cache`/locks/cache-dir consts (those patches stay
   `app_mod.*`). Heavy patches to repoint to `core`: model classes, `joblib`/`torch`/`pd`
   (library singletons — repoint only the LHS module ref), `build_position_features`,
   scaler IO, `k_data`/`k_features`/`dst_data`/`dst_features`, `POSITION_REGISTRY`,
   `upload_predictions_cache_to_s3`, `refresh_sentinel_mtime`, `_apply_position_models`,
   `_ensure_metrics`, `_get_data`, `_compute_models_fingerprint`, the ensure/load/hydrate
   helpers, `_position_arch_payload`(route helper → moves with routes in inc.5).
5. ✅ **`routes.py` + composition-root `app.py`** — DONE (this PR). Moved the 19
   handlers + route helpers (`_results_for_position`, `_categorize_features`,
   `_position_arch_payload`) to `routes.py`. Used the **classic Flask pattern, not a
   Blueprint** (a Blueprint's `bp`-registration is import-order-fragile across the
   app↔routes cycle): `routes.py` does `from src.serving.app import app` for the
   `@app.route` decorators + `app_pkg` for state; `app.py` imports `routes` at the
   bottom (`# noqa: E402,F401`), which registers the handlers as a side effect —
   cycle-safe in both import orders (verified). `app.py` keeps the Flask app, shared
   `_cache`/locks, the errorhandler, and explicit `X as X` re-exports of the test-facing
   names. Only 7 test repoints needed (`_position_arch_payload`, `_categorize_features`)
   since tests exercise routes through the Flask test client, not by patching handlers. Use the comprehensive 137-patch inventory + multi-line-aware
   `monkeypatch.setattr` enumeration (a single-line grep undercounts — several patches
   span two lines).

## Repoint workflow (per increment)

1. Move symbols/code to module `M`; in `app.py` import (and, for the 8 names, re-export).
2. `grep -rn 'setattr(<alias>, "X"' tests/test_app*.py` for each moved `X` (alias ∈
   `app_mod|app_module|app|_stub_app`); change to `setattr(<M-import>, "X"`. Add the
   module import to the test file. Repoint `<alias>.X` attribute reads too.
3. Run the suite (see below) until green; the 235 tests pin behavior.

## Verification (local; the suite is xdist-heavy)

The repo's `addopts = "-n auto"` spawns ~52 workers and OOM/contention-crashes here.
**Override it** and cap threads:

```
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests/test_app*.py \
  -o addopts="--strict-markers --dist=loadfile" -n 4 --timeout=300 -q
```

(unit-only subset: add `-m unit`, ~1s; integration is ~130s at `-n 4`.) Also
`ruff check src/serving/`. Confirm no training-path file changed →
`git diff --name-only origin/main...HEAD | grep -E 'src/shared|src/(qb|rb|wr|te|k|dst)/'`
must be empty.

## Stop-rules / gotchas (don't relitigate)

- `/health` predicate is `errors AND not loaded` (cold-start 200 vs failure 503) — keep it.
- No module-level pre-warm under gunicorn `--preload` (reverted; ALB TCP-refused).
- Keep the in-flight manifest poller.
- Serving display strings (e.g. `POSITION_INFO` "formula"/"label") are **behavioral**, not docs-only.
- Work in the worktree, not the parent checkout; auto-merge is disabled repo-wide
  (watch CI then `gh pr merge --squash`, never `--admin`); `main` moves often (rebase).
