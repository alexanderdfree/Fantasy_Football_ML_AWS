# CUDA graph A/B reframing + GradScaler trajectory check

> **Status:** First-pass harness and RB run complete on the WSL2 / RTX 5080 box
> (2026-06-01 UTC). This remains analysis work only; no production default should
> change unless a follow-up proves a fast + benchmarkable path.

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

## First-pass local result (2026-06-01 UTC)

Harness:

```bash
source scripts/wsl-env.sh
.venv/bin/python -m src.analysis.cuda_graph_gradscale --position RB --include-fixed-scale
```

Default harness settings: RB attention branch only, production RB attention
architecture/data/loss, `FF_FORCE_DROPOUT_ZERO=1`, `FF_DETERMINISTIC=1`, normal
early stopping, `FF_GRADSCALER_TRACE_PATH` JSONL per variant.

Result:

| Variant | RB attn FP MAE | Steps | Scale changes | Notes |
|---|---:|---:|---:|---|
| `graph_a` | 3.990711 | 2130 | 5 | baseline graphed run |
| `graph_b` | 3.990711 | 2130 | 5 | identical metrics + scale schedule vs `graph_a` |
| `eager` | 3.986418 | 2201 | 5 | graph-eager MAE delta +0.004293 |
| `graph_restore_bn` | 3.990711 | 2130 | 5 | no movement vs stock graph |
| `graph_fixed_scale` | invalid | 1 | 1 | default fixed scale 65536 overflows/skips at step 0 |

Interpretation:

- **Graph-vs-graph is clean** for the measured RB attention run: metrics hash and
  `GradScaler` scale/skip schedule match exactly.
- **BatchNorm warmup is not the culprit** in this run: snapshot/restore around
  `make_graphed_callables` warmup produced identical graph metrics.
- **Dynamic loss scaling is the splitter:** eager and graph schedules first
  diverge at step 1. Eager skips and backs off `32768 -> 16384`; graph stays at
  `32768`.
- **Fixed normal-scale mode is invalid:** starting at the normal initial scale
  overflows immediately, so "fixed 65536" cannot be the benchmarkable path.

Local artifacts from this run were written under
`benchmark_history/ablations/2026-06-01T01-30-27_83a617b_cuda_graph_gradscale_rb.json`
and `benchmark_history/ablations/cuda_graph_gradscale/2026-06-01T01-29-20_83a617b_rb/`.

## Second-pass fixed-scale result (2026-06-01 UTC)

Follow-up harness additions:

- `eager_fixed_scale` so the fixed-scale A/B compares graph and eager under the
  same explicit `GradScaler` scale.
- `graph_fixed_scale_restore_bn` so the fixed-scale graph arm also removes the
  `make_graphed_callables` BatchNorm warmup asymmetry.
- `--fixed-scale-init` to test lower explicit scales after the default 65536
  overflowed immediately.

Command for the clean completed run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
LGBM_N_JOBS=16 \
.venv/bin/python -m src.analysis.cuda_graph_gradscale \
  --position RB \
  --variants graph_fixed_scale_restore_bn eager_fixed_scale \
  --fixed-scale-init 512 \
  --fixed-epochs 30
```

Scale search:

| Fixed init scale | Graph arm | Eager arm | Interpretation |
|---:|---|---|---|
| 65536 | invalid at step 0 | not needed | normal fixed scale overflows immediately |
| 2048 | invalid at step 104/105 | completed | too high for the graph trajectory |
| 1024 | invalid at step 488/489 | completed | still too high for the graph trajectory |
| 512 | completed | completed | clean fixed-scale A/B |

Clean 512 result (RB attention, seed 42, dropout zero, deterministic mode,
fixed 30 epochs, BN warmup restored in graph arm):

| Variant | RB attn FP MAE | Steps | Scale changes | First scale diff |
|---|---:|---:|---:|---|
| `graph_fixed_scale_restore_bn` | 4.044157 | 2130 | 0 | none |
| `eager_fixed_scale` | 4.060941 | 2130 | 0 | none |

Target deltas (graph - eager): rushing TD −0.0043, receiving TD +0.0012,
rushing yards −0.1001, receiving yards +0.0406, receptions −0.0011,
fumbles lost +0.0000.

Interpretation:

- Fixed scale 512 eliminates the dynamic `GradScaler` schedule as a confound:
  both arms use the same initial scale, take the same number of optimizer steps,
  and have identical scale/skip traces.
- The graph/eager metric still differs, so fixed loss scaling is **not** a
  bit-comparable bridge to eager baselines.
- This supports the narrower hypothesis: after removing dynamic scaling,
  early-stop count, and BN warmup asymmetry, the remaining difference is the
  expected multi-step FP16 trajectory drift from graph replay/kernel ordering.
- Fixed scale 512 is also its own training regime and is worse than the dynamic
  first-pass MAE, so it should not replace the normal dynamic FP16 path.

Artifacts:

- `benchmark_history/ablations/2026-06-01T01-51-51_83a617b_cuda_graph_gradscale_rb.json`
- `benchmark_history/ablations/cuda_graph_gradscale/2026-06-01T01-51-14_83a617b_rb/`

Decision: accept **graphed runs are comparable to graphed runs, not to eager
baselines**. Use a graphed local rebaseline for `FF_CUDA_GRAPH=1` A/Bs; do not
change the production/default path.

## Success criteria

- The graph-vs-graph sentinel is bitwise clean.
- Fixed-scale controls either make graph-vs-eager bit-comparable or prove the
  exact remaining limitation. They proved the limitation:
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
