# Proposed ADR — Warm single-GPU MPS-packed training ("Option B")

**Status:** Proposed (pre-build). Promote to `docs/adr/0018-warm-mps-packed-training.md`
(next free number; 0017 is the current max) **only after the benchmark gate below is
green**. Lives in `todo/` (not `docs/adr/`) until then so it doesn't fire `deploy.yml`
or publish an undecided proposal to the in-app wiki. `[docs-only]`.

This is the AWS/warm cousin of Lever B in
[gpu_launch_bound_levers.md](gpu_launch_bound_levers.md) — same "pack all six positions
onto one GPU" idea, but via **real NVIDIA MPS on a warm Linux L4 box** instead of the
local single-process CUDA-streams substitute (MPS is Linux-only, so it's unavailable on
the WSL2/Windows 5080 that Lever B targets).

## Context

- **Current production (verified live 2026-06-07):** push-driven training is the Spot
  fan-out ([ADR-0013](../docs/adr/0013-spot-fan-out-via-aws-batch.md),
  [batch_design.md](../docs/batch_design.md)) — **six separate Spot hosts, one position
  each, single process per host** (`src/batch/train.py --mode=train` → `run_fn(...)`).
  **Regular training is NOT MPS-packed.** MPS exists only on the *tuner* path
  (`--mode=tune` → `src/tuning/tune_nn`, `--parallel-backend mps --n-jobs 3`), where it
  packs *concurrent Optuna trials of one position* onto that position's GPU — never the
  six positions together.
  - ⚠️ Live-state caveat found during this verification: the `ff-gpu-spot` CE is still
    `g4dn.xlarge` (T4, sm_75), **not** the `g6.xlarge` (L4, sm_89) the docs claim. On the
    T4, CUDA graphs are off (sm_80+ gate) so prod NN training is already eager. This
    proposal assumes the intended **L4** target; the gate must be measured on whatever
    hardware is actually live.
- **The lever is launch overhead, not compute** ([gpu_launch_bound_levers.md](gpu_launch_bound_levers.md)):
  the attention NN is a ~69K-param model firing hundreds of thousands of microsecond
  kernels; the GPU sits **~80% idle** between launches (measured on the 5080). That idle
  headroom is exactly what MPS co-residency fills.
- **The warm-EC2 rollback path already exists but is sequential**
  ([ec2_design.md](../docs/ec2_design.md), [ADR-0009](../docs/adr/0009-warm-training-host.md)):
  `train-ec2.yml` runs `for POS in …; do ff-train $POS; done` (~120 min) because it was
  built for the T4 ("one T4 can't host six concurrent NN jobs"). The slow part is the
  *sequential loop*, not the warmth.
- **MPS-packing N processes on one GPU is already proven on this stack** — the tuner's
  `_NvidiaMPS` (`tune_nn.py:287`) starts `nvidia-cuda-mps-control -d`, and
  `_run_mps_optimize` / `_mps_worker_entry` (`tune_nn.py:849,892`) spawn worker processes
  that share it. The missing piece is an orchestrator that packs the **six positions**
  (heterogeneous models/data) rather than trials of one.

So "Option B" = take the warm-EC2 path, swap T4→L4, and replace the sequential loop with
an MPS-packed parallel run — **started/stopped per run, not always-on.**

## Decision (proposed)

**IF** the benchmark gate shows wall-clock *and* metric parity, evolve the warm path to
MPS-pack the six positions on **one warm `g6.4xlarge` (1× L4, 16 vCPU, 64 GiB)**, started
on demand and auto-stopped on idle. **ELSE** keep the Spot fan-out as the default and put
the effort into cold-start instead (finish SOCI — see the live-verification finding).

Explicitly **not** always-on 24/7: `g6.4xlarge` on-demand is ~$1.32/hr ≈ **~$966/mo**
at 24/7 vs ~$0.35/run + scale-to-zero today. Warm ⟹ on-demand (Spot can't be kept warm
through reclaim), so the saving comes from *start/stop*, not from being perpetually hot.

## Proposed architecture (the sketch)

1. **Host:** one `g6.4xlarge` (single L4; the g6 family has no 6-GPU SKU — g6.12xlarge is
   4× L4 — so a single box inherently means GPU-*sharing* via MPS, by design). Start on
   push (or hold warm during an active dev sprint); reuse `train-ec2.yml`'s existing
   idle-auto-shutdown to stop it.
2. **Orchestrator** — a new module, **NOT under `src/batch/`** (any non-exempt file there
   trips the path-based 6-position retrain detect). Put it in `src/scripts/` (e.g.
   `run_mps_packed_train.py`). It:
   - starts one `nvidia-cuda-mps-control -d` (reuse `tune_nn._NvidiaMPS`);
   - `spawn`s six worker processes, one per position, each calling that position's
     `run_pipeline.run()` (reuse the `_run_mps_optimize` spawn+env pattern: propagate
     `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY`);
   - caps CPU per worker so six LightGBM/Ridge CV branches don't oversubscribe 16 vCPU
     (`LGBM_N_JOBS`/`OMP_NUM_THREADS` ≈ 2–3 each);
   - each worker uploads its own artifacts via the existing `src/batch/train.py`
     `upload_artifacts` path (manifest promotion is already per-position-atomic).
3. **CUDA graphs + MPS:** ON for L4 (sm_89) per the autodetect; K's nested-history trainer
   no-ops capture. This combination is **already exercised by the tuner** (Batch tune jobs
   run `FF_CUDA_GRAPH=1` under MPS), so it's low-risk — but re-confirm under the
   six-*position* mix.
4. **VRAM:** 6 × ~0.5–1 GB + six CUDA contexts on a 24 GB L4 — comfortable (the local doc
   already rates six-resident "fine at ~69K params each").

## Benchmark gate (MUST pass before any build merges)

Mirrors the project's existing A/B discipline (AGENTS.md "run the actual pipeline";
default 3 seeds, bump to 5–8 if borderline):

1. **Metric parity.** MPS-packed 6-position run vs current per-host run, same seed(s);
   diff `benchmark_history/*.json` per target. Must be within the seed band. Compare
   graphed-vs-graphed (CUDA graphs are not bit-comparable to eager — ADR-0017). MPS
   per-step math is independent of co-residency, but verify rather than assume.
2. **Wall-clock.** Total packed wall-clock on **one L4** vs (a) today's fan-out (~10 min
   incl. cold-start) and (b) a cold-start-fixed fan-out (~4–5 min). **Win condition:**
   packed-compute + warm-start latency < fan-out wall-clock by a margin worth the
   single-point-of-failure and the on-demand cost. Measure on the **real L4** — the
   ~80%-idle headroom is a 5080 figure; the weaker L4 executes each kernel longer relative
   to dispatch, so packing efficiency will be lower and six-on-one could land anywhere
   from ~1× to ~6× a single position's time.
3. **Stability.** No cross-context fault contamination under MPS (one worker crashing
   shouldn't corrupt the others); no OOM with six contexts + CUDA-graph static pools.

## Rejected / non-goals

- **Always-on 24/7** — ~$966/mo (or ~$400–580 with a 1-yr Savings Plan) to shave minutes
  off 35-cent runs. Start/stop instead.
- **Replacing Spot fan-out wholesale** — keep it the default unless the gate clears; six
  dedicated GPUs finish *compute* faster (~2 min) and scale to zero. This is an
  iteration-latency play, not a throughput one.
- **Orchestrator under `src/batch/`** — fires the 6-position retrain detect; `src/scripts/`.
- **Chasing 6 dedicated GPUs in one box** — no single g6 SKU has six L4s.

## Consequences

- **Pro:** removes Spot-acquisition + cold-start jitter → predictable, fast turnaround for
  active iteration; one host to reason about; reuses proven `_NvidiaMPS` plumbing.
- **Con:** single point of failure; on-demand cost while running; a new orchestrator to
  maintain; one L4 shared six ways (compute slower than six dedicated); MPS blast radius
  (a server fault can hit all six).
- Could share the per-position worker entry with Lever B (local CUDA-streams) so the two
  packing paths don't diverge.

## References

- [gpu_launch_bound_levers.md](gpu_launch_bound_levers.md) (Lever B; launch-bound diagnosis)
- [ADR-0013 Spot fan-out via AWS Batch](../docs/adr/0013-spot-fan-out-via-aws-batch.md)
- [ADR-0009 warm training host](../docs/adr/0009-warm-training-host.md) ·
  [ADR-0007 EC2 warm over Batch/SageMaker](../docs/adr/0007-ec2-warm-instance-over-batch-sagemaker.md)
- [ADR-0015 tuner Optuna Batch Spot fan-out](../docs/adr/0015-attention-nn-hyperparameter-tuning-via-optuna-batch-spot-fan-out.md)
  (the tuner's MPS path)
- [ADR-0017 per-arch / CUDA-graph policy](../docs/adr/0017-platform-autodetection-per-arch-optimization-policy.md)
- [batch_design.md](../docs/batch_design.md) · [ec2_design.md](../docs/ec2_design.md)
- Code: `src/tuning/tune_nn.py` (`_NvidiaMPS`, `_run_mps_optimize`, `_mps_worker_entry`),
  `src/batch/train.py` (`upload_artifacts`)

## Changelog
- 2026-06-07 · proposed · branch `claude/tender-curie-4YFnU` · drafted after confirming
  prod training is not MPS-packed (tuner-only) and the warm-EC2 path is sequential.
