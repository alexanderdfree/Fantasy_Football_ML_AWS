### [FIXED] Model expectations, fold preprocessing and tuning boundaries diverged

**File(s)**: `src/shared/neural_net.py`, `src/shared/evaluation.py`,
`src/shared/registry.py`, `src/shared/training.py`, `src/shared/count_math.py`,
`src/shared/models.py`, `src/shared/utils.py`, `src/data/preprocessing.py`,
`src/wr/features.py`, `src/te/features.py`, `src/dst/data.py`,
`src/dst/run_pipeline.py`, `src/k/data.py`, `src/k/run_pipeline.py`,
`src/benchmarking/benchmark.py`, and the affected
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
- K CV/origin preparation similarly filled missing total/implied Vegas lines
  before slicing the fold. In a real 3,573 × 19 prepared matrix with one
  deliberately missing historical line, a later-season change altered two
  training rows. Current 2015–2025 REG schedules have no missing Vegas lines,
  so this is a supported missing-data boundary rather than observed default
  cohort incidence.
- Concurrent stacked tuner trials installed process-global capture stubs and
  could steal each other's trainers, retain hooks after failure, or intercept
  unrelated training. Read-only best-study lookup selected a graph namespace
  that stacked training never wrote.
- Explicit Apple MPS runs did not seed their accelerator RNG, and four stacked
  entrypoints discarded the captured trainer's device. Repeated seed 42 changed
  2,074 of 4,096 real MPS dropout values while explicit MPS seeding repeated them.
- At a valid low positive rate, the truncated NB likelihood assigned probability
  greater than one and reversed its rate-gradient direction. Exponentiating
  finite log-dispersion 90 also produced NaN even when the head's expected count
  was representable. These are boundary reproductions, not claims that the
  default measured cohorts reached either state.
- A training loader with fewer rows than a dropped-tail batch saved an untouched
  model after reporting zero training loss. Captured stacked/sequential tuning
  bypassed the ordinary trainer and could report a finite Optuna objective from
  untouched weights; those entrypoints require the same guard.
- All six LightGBM configurations and the tuner supplied row-sampling fractions,
  but sampling frequency remained zero. The supposedly tuned dimension never
  affected a fitted tree.

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
Seed the selected MPS accelerator and retain the captured device in stacked
callers. Reject Apple MPS stacking with actionable guidance because the actual
Torch 2.14 MPS vmap/MSE smoke fails on that backend; eager MPS remains available.
Use shared `count_math.py` tensor primitives, a stable conditional probability
and log-gamma ratio, and log-domain expectations; retain the same distributions
and parameter shapes. Fail zero-batch training clearly. Enable LightGBM sampling
for fractions below one and isolate corrected trials in `seedavg_bagging_v2`.
Existing fraction-one fits are a positive control; historical tuned fractions
do not establish the corrected recipe's accuracy.
Both ordinary and captured ensemble loops reject empty or exhausted training
epochs before validating, reporting progress, or returning a trained model.
K CV uses the same deferred-fill pattern, fitting its two context medians on
the actual filtered training frame. Ordinary K loading retains its defaults.

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
The intermediate 36-cell comparison at `e3ed317f` and its successful real L4
CUDA probes are retained under `benchmark_history/audits/2026-09-10-model-*`.
They precede subsequent numerical fixes and main merges, so they do not establish
final-branch metric neutrality. The MPS record separately distinguishes passing
RNG controls from the unsupported stacked backend smoke.
K's real prepared-matrix perturbation changes zero training rows after the
repair, while its training-value control remains active. Default loader output
is identical across 5,272 historical rows and 37 columns for both observed and
deliberately missing-context inputs; 96 focused checks passed.

**Lesson**: Keep distribution parameters distinct from reported expectations,
fit preprocessing on the population actually used for training, and preserve
trial identity through temporary instrumentation and study lookup.
