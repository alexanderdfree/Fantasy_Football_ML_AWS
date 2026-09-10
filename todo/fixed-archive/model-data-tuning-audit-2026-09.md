### [FIXED] Model expectations, fold preprocessing and tuning boundaries diverged

**File(s)**: `src/shared/neural_net.py`, `src/shared/evaluation.py`,
`src/shared/registry.py`, `src/shared/training.py`, `src/data/preprocessing.py`,
`src/wr/features.py`, `src/te/features.py`, `src/dst/data.py`,
`src/dst/run_pipeline.py`, `src/benchmarking/benchmark.py`, and the affected
`src/tuning/` entrypoints. Reproduced against `0f0fec55` in the 2026-09-10 audit.

**What**:

- Hurdle NB/Poisson heads reported `p * mu`, treating the underlying law's mean
  as its conditional positive mean. NB `mu=1, alpha=1, p=0.5` reported 0.5
  although the fitted distribution's expectation is 1.
- WR/TE team-relative opportunity and red-zone shares crossed team stints.
  The existing neighboring team-share features already reset at each stint.
- The no-play predicate dropped nonzero signed statistics that summed to zero
  when snap data was unavailable.
- Ordinary and stacked validation averaged batch means, overweighting a short
  tail. The same four observations produced loss 25.75 or 50.5 solely from
  partitioning them into batches.
- D/ST CV/origin preparation filled missing features using global training
  years before slicing earlier folds. Perturbing only 2023 held-out scores
  changed 2013–2021 training rows and 2022 validation season openers.
- Concurrent stacked tuner trials installed process-global capture stubs and
  could steal each other's trainers, retain hooks after failure, or intercept
  unrelated training. Read-only best-study lookup selected a graph namespace
  that stacked training never wrote.

**Fix**: Convert truncated count means using their positive probability mass,
keeping raw NLL inputs and checkpoint tensor shapes unchanged. Route the loss
family through training and serving, expose the proper conditional diagnostic,
and keep sensitive half-precision arithmetic in FP32. Reuse existing team-stint
groups for the two share windows. Test every raw statistic for zero. Weight
validation totals/components by observation count while preserving weighted-MAE
checkpoint selection; version tuning objectives as `scheduler_v3`/`history_v3`.
Defer D/ST context fills to the actual fold's existing fill hook. Dispatch
temporary capture hooks by thread/nested context and unify stacked namespace
selection for training and lookup.

**Validation**: Regression suites retain ordinary Poisson, likelihood-gradient,
six-position reload, mixed precision, vmap, same-team, zero-event, empty-tail,
unrelated-thread and explicit legacy-namespace controls. Independent numerical
review passed 90 oracle/gradient/caller checks. Real D/ST preparation preserved
all 7,326 × 59 default target/feature cells, while its 2023-origin train/validation
matrices became invariant to held-out scores and retained a training-score
positive control. Full production comparisons use separately frozen source and
loader inputs, all six positions and seeds 42/123/7; a separate 2023 D/ST origin
comparison covers the affected fold path. Actual CUDA acceptance is required
for changed capture arithmetic. Final comparison and hardware evidence is added
with delivery; unit or schema checks alone do not establish those results.

**Lesson**: Keep distribution parameters distinct from reported expectations,
fit preprocessing on the population actually used for training, and preserve
trial identity through temporary instrumentation and study lookup.
