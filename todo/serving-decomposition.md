# Serving `app.py` decomposition — continuation spec

**Status:** increment 1 of N shipped (PR #990 — `serialization.py` extracted). The
heavy core/routes split below is the remaining, deliberately-deferred work. This doc
is the precise spec so the follow-up isn't a re-discovery.

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

1. **`metadata.py`** — static, re-export only, ~0 repoints (warm-up, like #990).
2. **`state.py`** — extract shared state; convert `_cache`→`state._cache`; repoint the
   `_cache`/`_PREDICTIONS_CACHE_DIR` patches. **The critical enabler; do carefully.**
3. **`wiki.py` + `comparison.py` + `benchmark_history.py`** — leaf-ish clusters that
   import `state`; repoint their handful of patches (`markdown`, `_load_comparison_experts`,
   `_benchmark_history_dir`, `_compute_models_fingerprint`).
4. **`core.py`** — the big one; repoint the ~40 heavy patches. Bulk of the work.
5. **`routes.py` Blueprint + composition-root `app.py`** — move the 18 handlers; re-export
   the 8 names; register the blueprint + errorhandler.

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
