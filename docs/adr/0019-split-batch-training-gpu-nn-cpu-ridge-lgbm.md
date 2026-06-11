# ADR-0019: Split Batch Training Into GPU NN And CPU Ridge/LightGBM Branches

**Status:** Accepted
**Date:** 2026-06-10
**Supersedes:** none
**Related:** ADR-0013, ADR-0017

## Context

The monolithic Batch job trains Ridge, LightGBM, base NN, and attention NN for one
position on a single `g6.xlarge`. Recent profiling showed the NN branch is
launch-bound on the L4 and uses very little VRAM, while the Ridge/LightGBM work
is CPU-bound and benefits from the existing work-conserving `core_pool` lease
path. The account now has enough Standard Spot quota to run a CPU-only pool in
parallel with the GPU pool.

The c8a probe for the WR CPU branch showed that a 4-vCPU / 8-GiB shape is safe:
single-seed CPU branch RSS was about 1.4 GiB; ten seeds with dynamic-core
sequential execution used about 1.7 GiB and closed most of the gap to fixed
4-worker packing while avoiding the 5.5-GiB memory footprint of packed workers.

## Decision

Training can opt into split mode with `src.batch.launch --split`:

- `nn` branch: submits to the existing GPU queue and trains base NN + attention
  NN only.
- `cpu` branch: submits to a new `ff-cpu-training-queue` backed by
  `c8a.xlarge` Spot and trains Ridge + LightGBM only.
- `merge` branch: submits to the CPU queue with Batch dependencies on both
  branch jobs, validates staged branch manifests/tarballs, combines artifacts
  into the normal model directory, runs the existing smoke-test/publish path,
  and only then advances the stable S3 manifest.

Branch artifacts are staged under:

```text
s3://ff-predictor-training/split-runs/{run_id}/{POS}/nn/
s3://ff-predictor-training/split-runs/{run_id}/{POS}/cpu/
```

The monolithic full path remains available and unchanged when `--split` is not
passed (CI: when the repo variable `BATCH_SPLIT_ACTIVE` is not `true`).
**Since 2026-06-11 `BATCH_SPLIT_ACTIVE=true` is the production default** — see
the changelog; unsetting the variable is the instant rollback to monolithic.

## Consequences

- The GPU pool is used only for the launch-bound NN work; Ridge/LightGBM no
  longer consume g6 capacity.
- The CPU pool uses `c8a.xlarge` as primary and `m8a.xlarge` as fallback, with
  `maxvCpus=64` so at most 16 4-vCPU CPU jobs run concurrently.
- CPU branch jobs start `src.shared.core_pool` with one active worker, set
  BLAS/OMP caps to 1, and lease all 4 cores dynamically for Ridge CV and
  LightGBM stages.
- Serving never observes partial artifacts: branch jobs write only staged
  tarballs, and the merge job is the only split path that calls the existing
  manifest publisher.
- Image builds register and stash GPU and CPU job-definition revisions
  separately, avoiding accidental CPU submissions against GPU revision pins.

## Rejected Alternatives

- **Keep monolithic GPU jobs and only raise GPU quota.** This spends scarce G/VT
  Spot quota on CPU-bound Ridge/LightGBM stages and does not improve NN
  launch-bound throughput.
- **Pack many CPU seeds as fixed worker processes.** It is fast, but duplicates
  data prep and raises memory pressure. Dynamic-core sequential nearly matches
  the timing for the measured workload with much lower RSS.
- **Publish partial branch artifacts directly.** Rejected because serving and
  smoke-test contracts expect a complete per-position artifact directory.

## Changelog

- 2026-06-11 · **Split mode enabled as the production default**
  (`BATCH_SPLIT_ACTIVE=true`) after the first full 6-position split run
  (workflow 27324790653, image `ed14915b`, seed 42): 18 jobs (6 nn → GPU queue
  rev 270, 6 cpu + 6 merge → CPU queue rev 19), all SUCCEEDED; merge
  dependencies honored; freshness gate + benchmark append green; **metrics
  byte-identical to the same-image monolithic baseline for every model × every
  position (Δ=0.0000 incl. Ridge data-identity)**; per-position branch elapsed
  nn 20–56 s / cpu 4–19 s / merge <1 s, end-to-end ~8 min. Rollback: unset the
  repo variable.
- 2026-06-10 · Initial decision: opt-in split Batch mode with GPU NN branch,
  c8a CPU Ridge/LightGBM branch, staged S3 artifacts, and merge-only manifest
  promotion.
