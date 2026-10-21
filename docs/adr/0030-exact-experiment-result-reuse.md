# ADR-0030: Exact experiment result reuse

**Status:** Accepted
**Date:** 2026-09-17
**Related:** ADR-0026, ADR-0027

## Decision

Analysis and A/B callers opt into automatic reuse of complete matching fitted
runs through `RunContext.reuse_results`. Production training and full training
benchmarks retain fresh execution by default. `--fresh` / `FF_FRESH=1` bypasses
reuse; callbacks, including Optuna epoch reporting and construction-capture
instrumentation, must execute and are not eligible.

Reuse binds the resolved recipe, training code and loaded numerical callables,
Python/library versions, actual input and raw side-input contents, ordered
frames, seed, and effective device/dtype/graph settings. The existing immutable
data-release marker alone is insufficient. Unsupported runtime values execute
freshly. Input or implementation changes during fitting prevent publication.

The cache stores raw per-head predictions, prepared data, and the existing
versioned fitted-model artifacts. Hits reconstruct the public training-result
contract and recompute metrics, scoring, ranking, backtests and cohorts. Each
result identifies its original run and separates original training time from
the current lookup/evaluation time. Stacked A/Bs can reuse their complete eager
non-attention baseline; their new attention training remains independent.

Artifact-based analysis has a separate identity based on the exact model
directory contents and inference inputs. It never substitutes an arbitrary
served model for a request to train current code. Incomplete model predictions
are returned with their existing errors and are never cached.

Entries are checksum-verified and atomically published; the first complete
writer wins. The default local store is `.cache/results`, overridden by
`FF_RESULT_CACHE_DIR`, with a 20 GiB LRU limit. AWS Batch uses `S3_BUCKET` under
`experiment-cache/v1/`; other callers explicitly set `FF_RESULT_CACHE_BUCKET`
to opt into S3. An empty override disables remote reuse. Entries expire after
30 days; install the matching prefix-only S3 lifecycle rule with:

```sh
python -m src.training.result_store --configure-s3-expiration --bucket ff-predictor-training
```

This preserves other bucket lifecycle rules and never expires durable run
outputs. Cache read/write failures fall back to fresh execution; actual fitting
errors propagate. Cache payloads use the project's trusted model/Python
serializers and belong only in the owner's cache and training bucket.

## Alternatives

- Reusing position/seed filenames cannot prove data, code, or model identity.
- Skipping individual stochastic stages within a new fit could change later
  RNG consumption. Reuse complete execution units instead.
- Caching final report summaries would freeze scoring and cohort semantics.
  Keep raw predictions and re-evaluate them.

## Validation

Tests cover all six position contracts, invalidation, corruption, concurrent
writers, expiry, S3 round trips, fresh overrides, callback execution and raw
prediction re-scoring. The real-data smoke uses the existing tiny E2E harness
and compares predictions, fitted artifacts, metrics and cohorts on cache hits.
These tests establish correctness; no general production speedup is claimed.

## Changelog

- 2026-09-17 · Introduce exact complete-run and pinned-artifact prediction reuse.
