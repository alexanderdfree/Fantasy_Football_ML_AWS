# CUDA graph A/B reframing + GradScaler trajectory check

> **Status:** Planned follow-up for the WSL2 / RTX 5080 box. This is analysis
> and harness work only; no production default should change unless the A/B below
> proves a fast + benchmarkable path.

## Why this is a priority

`FF_CUDA_GRAPH=1` is already a real local-iteration speed knob, but the current
writeup frames it too broadly as "not suitable for bit-comparable benchmark A/Bs."
The narrower hypothesis to test is:

- **Graph-vs-graph should be bitwise stable.** If both arms of a code-variant A/B
  run with `FF_CUDA_GRAPH=1`, replay should be deterministic and the NN side should
  compare cleanly at full FP16 speed.
- **Graph-vs-eager is the incompatible comparison.** Historical eager
  `benchmark_history/` baselines are not bit-comparable against a graphed run. The
  remedy may be a one-time graphed rebaseline, not disabling the graph knob for all
  benchmark work.
- The measured graph-vs-eager drift (~0.5% worst target in one seed) is in the
  range this project already treats as NN seed noise. Model-quality conclusions
  still need the usual >=8-seed standard for small FP-MAE deltas.

## Corrections to preserve

- **BatchNorm buffers are not the FP16-precision culprit.** BatchNorm running stats
  are held in FP32 even under autocast. The observed BN-buffer divergence is
  downstream of the weight split: divergent weights produce divergent batch
  activations, which then feed divergent running EMAs.
- **`make_graphed_callables` warmup does advance BN state.** PyTorch runs the
  warmup forward in the caller's mode; this code calls it while the model is in
  train mode. That is a real asymmetry because the eager arm does not get the same
  warmup updates.
- **The BN warmup asymmetry is probably minor.** With default BN momentum 0.1, that
  head start decays as `0.9 ** t`, so it is effectively gone after roughly 100 of
  the ~18k train steps. Still snapshot/restore BN running stats around graph
  warmup once to rule it out cleanly.

## Experiment order

1. **Graph-vs-graph reproducibility sentinel.**
   Run the same position/seed twice with `FF_CUDA_GRAPH=1` and the existing
   deterministic controls. Expected: byte/metric delta is zero. If this fails, the
   graph path has a determinism bug and the rest of this note is premature.

2. **BN warmup snapshot/restore rule-out.**
   Add an env-gated experiment that saves every BatchNorm `running_mean`,
   `running_var`, and `num_batches_tracked` before `make_graphed_callables` warmup
   and restores them immediately after. Expected: little to no movement. If it
   unexpectedly collapses the graph-vs-eager drift, keep the fix because it is cheap
   and logically cleaner.

3. **GradScaler scale-schedule diff.**
   Instrument eager vs graph runs to log, per optimizer step:
   `step`, current scale, next scale, whether the step was skipped, and the first
   overflow/scale-change step. This is the highest-information check because the
   current hypothesis is that FP16 + dynamic loss scaling amplifies a tiny
   graph/eager ordering split over many steps.

4. **Fixed-loss-scale A/B if the scale schedules diverge.**
   Try an env-gated fixed-scale mode that starts at the normal initial scale and
   prevents growth during the run. If no overflow occurs, the scale remains fixed;
   if overflow does occur, log it and treat that variant as invalid rather than
   silently changing semantics. The goal is fast FP16 + graph replay without the
   dynamic-scale amplifier.

5. **Deterministic-kernel diagnostic if scale schedules match.**
   If eager and graph have the same scale/skip trajectory but still drift, test
   `torch.use_deterministic_algorithms(True)` as a diagnostic. Do not assume this is
   shippable; use it to locate whether a nondeterministic CUDA kernel is the
   remaining splitter.

## Success criteria

- The graph-vs-graph sentinel is bitwise clean.
- Either the fixed-scale path makes graph-vs-eager close enough for a one-time
  rebaseline, or the docs are updated to state the exact remaining limitation:
  **graphed runs are comparable to graphed runs, not to eager baselines**.
- If a code fix lands, keep it env-gated first and run the actual affected pipeline
  on the 5080 before merging. This is not a unit-test-only area.

## Pickup notes for WSL Codex

- Start from [todo/gpu_launch_bound_levers.md](gpu_launch_bound_levers.md), then use
  this note as the follow-up plan.
- Use the WSL environment profile from [scripts/wsl-env.sh](../scripts/wsl-env.sh)
  before running GPU jobs.
- Prefer RB for the first pass because the existing graph measurement was on RB and
  the prior writeup already has comparison numbers.
- Do not compare the first graphed run directly against an eager
  `benchmark_history/` baseline and call the delta a regression. Establish the
  graph-vs-graph sentinel first.
