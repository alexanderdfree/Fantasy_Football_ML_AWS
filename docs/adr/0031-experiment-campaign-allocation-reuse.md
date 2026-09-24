# ADR-0031: Reuse allocations across experiment campaigns

**Status:** Accepted
**Date:** 2026-09-24
**Related:** ADR-0020, ADR-0026, ADR-0027, ADR-0030

## Decision

Use a frozen campaign manifest to group existing experiment workloads by
position and CPU/GPU requirements. Support local execution and AWS Batch Spot
with the same spec and checkpoint contract. Run each step in a fresh process;
reuse the allocation and downloaded data while keeping RNG, imported overrides,
studies and model artifacts isolated.

Reuse the existing A/B grid, Optuna tuners, benchmark scorer and core pool.
Do not introduce another fitting implementation or change tuning objectives.
Keep all six positions, eager/stacked namespaces and production numerical
policies intact. Benchmarks remain fresh training evidence.

Freeze the source, effective inputs, environment and image digest before
submission. Copy local sealed inputs into campaign ownership, and use compatible
immutable data releases remotely. Results, model files and studies belong to
`campaign_runs/<id>/`; production pointers are outside the campaign lifecycle.

Resume uses verified per-step and per-cell outputs, SQLite-safe snapshots,
attempt/time budgets, and conditional S3 journals. Submission intents precede
Batch calls so ambiguous network failures cannot silently duplicate allocations.
No permanent warm capacity or idle-capacity policy is added.

## Consequences

Several experiment requests can pay one allocation and data-download cost per
position/resource group. Actual wall-time savings must be measured on Batch;
local and unit tests alone do not establish that benefit. Allocation affinity
trades finer fleet fan-out for lower startup overhead. Cross-resource steps are
independent; tuning results never implicitly alter later recipes.

The [operator guide](../experiment-campaigns.md) describes the public CLI, budget
semantics and recovery behavior.

## Changelog

- 2026-09-24 · Introduce isolated local/Batch campaigns for A/Bs, NN tuning,
  LightGBM tuning and fresh benchmarks with checkpoint-aware allocation reuse.
