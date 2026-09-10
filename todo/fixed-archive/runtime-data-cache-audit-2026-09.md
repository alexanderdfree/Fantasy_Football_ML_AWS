### [FIXED] Input caches omitted implementation, lookup, selector and completion identity

**File(s)**: `src/shared/feature_cache.py`, `src/shared/model_sync.py`,
`src/data/external_sources.py`, `src/data/nflcom_loader.py`, `src/k/data.py`,
`src/serving/expert_sources.py`, `src/analysis/fftoday_loader.py`.
Defects reproduced against `92be2873` during the 2026-09-10 audit.

**What**:

- Feature-selection closures with the same name shared cached columns; later
  source changes and schedule/box-score corrections also reused old features.
  Cache-directory creation errors escaped the optional-cache fallback.
- K, Sleeper and FFToday keys conflated distinct requested season/week sets.
- NFL.com persisted incomplete fetches, wrote custom-roster results into the
  default roster cache, and reused insufficient joins at a stricter threshold.
- External-source normalization could fail outside the degradation boundary;
  empty/malformed cached data could prevent recovery.
- A completely failed per-kick fetch became an empty successful K history.
- A rejected tarball could leave model files in the next fallback artifact;
  successful extraction also retained obsolete files from an existing tree.

**Fix**: Fingerprint effective ordered columns, source bytes and canonical
runtime lookup outputs. Verify dependency stability before publishing to disk
or memory, and keep computed values when optional disk writes fail. Use exact
selectors for caches; require completed fetch metadata; isolate roster
overrides and recheck precise match thresholds. Keep fetch and normalization
inside the same recovery boundary, reject failed K history acquisition, and
stage each model archive before replacing its destination.

**Validation**: Regression tests cover cold/warm and memory/disk paths,
equal-endpoint/equal-count sparse requests, source recovery, changed inputs
during computation, valid empty sources, strict thresholds, custom/default
rosters, partial extraction, and valid fallback controls. Tests live in
`tests/shared/test_feature_cache.py`, `tests/shared/test_model_sync.py`,
`tests/test_external_sources.py`, `tests/test_nflcom_loader.py`,
`tests/k/test_data_loaders.py`, `tests/analysis/test_sleeper_loader.py` and
`tests/analysis/test_fftoday_loader.py`.

The final CPU/eager production pipelines for QB/RB/WR/TE/K used frozen identical
raw/split inputs and seed 42. Prepared feature/target arrays and both neural
state dictionaries were bit-identical to `2054fe86`; loaded Ridge/LightGBM
point predictions had zero maximum difference. The paired benchmark histories
and source fingerprints are indexed by
`benchmark_history/audits/2026-09-10-runtime-comparison.json`. This establishes
healthy-path neutrality in that regime; CUDA failure recovery requires separate
hardware acceptance evidence.

**Lesson**: A cache hit requires the complete computation identity, including
data read outside the primary frame arguments. A failed or customized fetch
must not become evidence of a complete default result.

Concurrent model publication, serving cache-generation and bootstrap seeding
repairs in PR #1560 are excluded from this record and the audit branch.
