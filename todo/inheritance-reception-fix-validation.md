# Inheritance preprocessing and reception expectation — PR validation

The two requested correctness fixes are implemented. The inheritance scaler is
enabled for **WR only**. Corrected zero-truncated negative-binomial expectations
apply to the affected reception heads at **RB/WR/TE**. QB/K/DST retain their
effective production model behavior. There is no claim of metric neutrality or
of a universal accuracy improvement.

## Verified defects and resulting behavior

The frozen September test data has 92 positive WR inheritance rows containing
34 distinct float32 magnitudes. The legacy attention and base-NN input paths
collapse those magnitudes to one value, +4. With the new training-fitted
transform, **34 raw values produce 34 encoded values**, ranging from 1.686747
to 2.869163. Zero stays zero. Other features retain their existing standard
scaling/clipping. Fitted indices, scale, and bound travel in the scaler artifact;
validation/test values do not enter fitting.

The reception head's NB-2 likelihood uses an untruncated mean. Reporting
gate × that mean omitted the positive-truncation normalization. The corrected
output is `gate * mu / (1 - P_NB(0))`. A direct probability-mass check with
mu=1, alpha=1, and gate=0.75 gives 1.50; the old output was 0.75. Tests also
compare against SciPy's independent NB distribution, check small means and
autocast dtypes, and verify gradients and stacked execution.

Checkpoint expectation versions preserve old predictions on inference load.
Warm starts reuse weights but retain the new fit's requested recipe. Plain
Poisson/gated TD outputs and all raw-stat targets/loss weights are unchanged.

## Data and experiment contract

- Input release: `556115711494d5f7c10af9fe3e97b94b200e97a661078a3a1e4a874ca4b20340`.
- All 26 raw/split files were checked against the immutable release's SHA256
  entries before and after validation; zero mismatches.
- PPR comparisons use identical regular-season rows and the same projected
  components in forecasts/actuals. Expected starters use the archived
  `shared_components_v2` reference; no model chooses its own cohort.
- Existing `ab_harness`, isolated artifacts, feature cache disabled, CPU FP32
  eager, seeds 42/123/7. RB/WR/TE each ran legacy, magnitude-only,
  expectation-only, and both (36 cells). QB ran legacy and broader both (6 cells).
- **42/42 cells succeeded.** The two-cell WR smoke also succeeded and reproduced
  the full run's seed-42 frames exactly. Those repeated smoke cells are not
  additional independent seed evidence.
- Raw features/actuals and Ridge/LightGBM predictions were identical across
  paired arms. Base-NN predictions were identical for expectation-only versus
  legacy. Every cell reconstructed saved models/scalers through serving
  primitives. All per-target comparisons passed (absolute tolerance 2e-5,
  relative tolerance 1e-6), and the recorded total-forecast discrepancies were **0**.
- The old 2,768-row WR model/cache also remained reproducible within its
  two-decimal display rounding (maximum difference below 0.0051 points).

Full numeric evidence, including per-cell metrics and paired standard
deviations: [CPU A/B artifact](../benchmark_history/ablations/inheritance_reception_cpu_1b29796b.json).
Local per-row predictions and logs remain in
`analysis_output/inheritance_reception_fix/`; datasets/weights are not committed.

## Selected production recipes: measured CPU tradeoffs

The table reports **change from legacy**, mean ± paired seed standard deviation.
Positive MAE/RMSE changes are worse. These are 3-seed observations, not confidence
intervals or a comparison of CPU results with deployed GPU results.

| Position / selected arm | Attention overall MAE delta | Attention overall RMSE delta | Expected top-24 MAE delta | Expected top-24 RMSE delta |
|---|---:|---:|---:|---:|
| RB / expectation only | +0.0368 ± 0.0157 | −0.0656 ± 0.0539 | −0.0113 ± 0.0258 | −0.0860 ± 0.0930 |
| WR / both fixes | +0.0881 ± 0.0311 | +0.0092 ± 0.0194 | +0.0427 ± 0.0500 | +0.0412 ± 0.0338 |
| TE / expectation only | +0.0389 ± 0.0141 | −0.0433 ± 0.0080 | −0.0086 ± 0.0073 | −0.0354 ± 0.0052 |

For the 92 WR inheritor rows, attention MAE changes by **−0.0152 ± 0.0132**,
RMSE by **+0.0108 ± 0.0463**, and bias by **+0.1071 ± 0.0756** toward zero.
These small effects do not establish an accuracy win. The independently verified
benefit is that the model can distinguish inheritance magnitudes again.

The broader activation screen was useful for scoping: enabling the new scaler
outside WR added adverse interactions, including top-24 attention RMSE deltas
of +0.0366 QB, +0.0403 RB, and +0.0168 TE for the combined arm. Those scaler
activations are **not** enabled in the final production defaults. The generic
preprocessor remains available for a separately justified position-specific
experiment.

## Local checks and GPU status

The full local unit suite passed **4,334 tests, 2 skipped**, including four
untracked diagnostic tests retained from the preceding investigation. The
PR's own tests cover numerical expectation, gradients, legacy/new artifact
loading, warm starts, inference parity, train-only fitting, zero-only fitting,
input ordering, and all six factory/serving configurations. Ruff and format
checks pass. A current production WR benchmark is retained in
`benchmark_history/` with a content fingerprint for the final implementation.

GPU image build:
[run 34541933370](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/actions/runs/34541933370),
image commit `1b29796b9698dbdae67f956bffbf500d588b9ba6`.
The subsequent activation-only change is explicitly overridden by the A/B
recipe settings; WR `both` and RB/TE `expectation_only` match the final
production recipes. The scaler/math implementations are unchanged.

GPU WR legacy/combined smoke:
`inheritance-reception-smoke-1b29796b`, job
`b5f113f2-ea83-4f31-b578-6333836aeab6`, isolated under
`s3://ff-predictor-training/ab_runs/inheritance-reception-smoke-1b29796b/`.
It uses FP32 with production CUDA-graph autodetection and an immutable compatible
input release. **GPU results are pending**; queue/submission is not successful
pipeline evidence. No production artifacts are promoted by the A/B job.

## Reproduction

```bash
FF_FIX_AB_OUTPUT=/absolute/isolated/output \
FF_DEVICE=cpu FF_AMP_DTYPE=fp32 FF_CUDA_GRAPH=0 FF_COMPILE=0 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m src.tuning.ab_inheritance_reception --positions RB WR TE --seeds 42 123 7 -j 3

python -m src.tuning.launch_ab --spec src.tuning.ab_inheritance_reception \
  --positions WR --seeds 42 --only both \
  --collect-only --run-id inheritance-reception-smoke-1b29796b
```

The new scaler and expectation version require updated serving code before
new-format artifacts are published. Existing artifacts keep their original
behavior. This PR remains a reviewable correctness change with explicit metric
tradeoffs; model promotion/merge requires the normal production validation.

## Consolidated contracts validation — 2026-09-11

After combining #1552 with #1575 and rebasing onto #1566, the final CPU/eager
production-config comparison completed **30/30 cells**: QB/RB/WR/TE/DST,
legacy versus combined policies, seeds 42/123/7. The
[complete metrics and provenance](../benchmark_history/ablations/nn_consolidation_cpu_20260911.json)
include per-head, subgroup, mean/std and Ridge-sentinel results. All twelve
skill-position paired frames have exactly equal non-prediction values; this
direct input comparison supplements the unchanged Ridge predictions. Saved-model
inference differences for all four model families are zero across the twelve
skill-position/seed pairs. K has no affected Poisson or truncated-NB policy.

Mean attention fantasy-MAE changes (combined minus legacy) are QB **-0.0093**,
RB **+0.0262**, WR **+0.1045**, TE **+0.0377**, and DST **-0.0032**. These are mixed
accuracy effects, not a non-regression result. This run supplies no new GPU
evidence. The legacy warm-start transition has a dedicated regression test;
this comparison does not separately establish warm-start metric neutrality.

The new-fit policy reuses the legacy trunk while retaining the initialized
count-output layer when changing from a raw-rate link to a log-rate link.
Inference continues to honor checkpoint markers. The branch depends on the
contracts migration so model-policy changes remain a separate review; that
dependency also ties delivery of these focused fixes to the migration.
