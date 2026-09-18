# WR regression investigation: PR #1564

**Completed:** [findings](findings.md), [numerical report](results.md), and
[full results](results.json). All 12 fitted cells and 24 scored arms passed.

This investigation isolates depth normalization, availability semantics, and
remaining source/identity changes. It does not change a model default or propose
deploying the old data semantics. PRs #1575 and #1568 are excluded.

## Evidence contract

- Historical numerical recipe: `b9d24f9259c7fd261ab1a4e77d4212d821726420`.
  The diagnostic worker `9f4dae30ca55d79985176ab7e7f767d2c592cc43` adds only the
  input builder and experiment spec. Production model/configuration files match
  the historical commit exactly.
- The baseline availability function is extracted unchanged from
  `6406cf213cc0a5fbcd1e56e449e3d7f82804dc01`.
- All 49 required archived production objects passed their recorded hashes.
  The first corrected release is
  `556115711494d5f7c10af9fe3e97b94b200e97a661078a3a1e4a874ca4b20340`.
  Original local CPU prediction artifacts no longer exist. These are archived
  production reconstructions, not relabeled original CPU runs.
- Evaluation uses the same 2,761 regular-season 2025 WR player-weeks, corrected
  actuals, and shared projected scoring components in both forecasts and actuals.
  The seven added observations remain separate. All results are retrospective.

## Interventions

`r` selects the original or corrected raw/split generation; `a` selects the
participant-based or roster-based availability policy; `d` selects original
ESPN ordinals or corrected within-slot depth for 2025 WR inputs. Each is binary.
Availability is recomputed independently under each source generation. The
original and corrected policies reproduce their corresponding archived columns
within `3.6e-15`; their model inputs agree at FP32 precision.

The eight combinations have four distinct training/validation inputs. Within
each `r/a` pair, changing `d` preserves complete prepared training/validation
frames, selected arrays, targets, histories/masks, feature order and scaler state.
Four full pipeline fits per seed are therefore sufficient; each saved model is
scored under both depth definitions. Seeds are 42, 123 and 7. All model fitting
uses AWS Batch Spot, CUDA FP32/TF32, eager execution and disabled CUDA graphs.

Source changes include source-cache corrections, recovered observations and
their downstream feature/imputation effects. In particular, observed historical
depth values on common training rows are unchanged. Added observations change the
mean used to impute missing depth from approximately 1.702528 to 1.703777.
The legacy team-stat cache receives a schema marker only, preserving its numeric
values while passing the historical corrected loader's cache gate. These are
isolated experiment inputs; no canonical data pointer is modified.

## Reproduction

The archived object identities are in `archive/`; the verified retrieval receipt
is `recovery.json`. Recover into a private directory:

```bash
python -m src.analysis.wr_pr1564_recovery \
  --archive todo/wr-pr1564-regression/archive --output /tmp/wr-pr1564-recovered
```

Run `src.analysis.wr_pr1564_inputs` **from the pinned historical worker checkout**,
using `--recovered /tmp/wr-pr1564-recovered --output /tmp/wr-pr1564-inputs`.
It performs preparation and fingerprinting, not model fitting. The committed
input manifest records every generated file and the numerical source hashes.

The immutable prepared archive is:

```
s3://ff-predictor-training/ab_runs/wr-pr1564-20260918/inputs/4a914ff282b47472661abe063d8fd35720650cdd4b99fe4d2279c989eddaa2aa.tar.gz
```

The worker image is pinned to source `9f4dae30ca55d79985176ab7e7f767d2c592cc43`
and digest `sha256:561642ef7fea9061a5d4a776ccf62504f8ad520405fc6c6911cea19eac95344e`.
Use the current `src.tuning.launch_ab` launcher with
`--spec src.tuning.ab_wr_pr1564 --positions WR --cuda-graph false`, the image pins,
`FF_DATA_RELEASE` set to the corrected historical release, and container variables
`FF_AMP_DTYPE=fp32`, `FF_COMPILE=0`, `FF_WR1564_INPUT_URI`, and
`FF_WR1564_INPUT_SHA` bound to the archive above. Start with seed 42 and
`--only r0a0 --max-cells 1`; require its complete parity/cohort gate before
launching the three full seed jobs. Each full job contains all four variants
and uses `--max-cells 4`. Use fresh run IDs for new executions.

The verified smoke is under `ab_runs/wr-pr1564-smoke-20260918/`. Full run prefixes
are `ab_runs/wr-pr1564-full-seed42-20260918/`,
`ab_runs/wr-pr1564-full-seed123-20260918/`, and
`ab_runs/wr-pr1564-full-seed7-20260918/` in `ff-predictor-training`.
Each contains immutable `evidence/` records and content-addressed `rows/` files.
Download these two subdirectories from all three prefixes into one local root,
then run the read-only reporter:

```bash
python -m src.analysis.wr_pr1564_report \
  --root /tmp/wr-pr1564-results --output /tmp/wr-pr1564-report
```

The reporter rejects incomplete grids, mixed provenance, different paired
hardware, unequal truth/cohorts, invalid saved-inference parity, and corrupt row
artifacts. Factor contributions use exact three-factor Shapley allocation;
conditional effects and interactions are retained separately. Per-seed standard
deviations describe these runs, not uncertainty across NFL seasons.
