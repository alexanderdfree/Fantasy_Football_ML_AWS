# Why WR MAE regressed after PR #1564

**The dominant cause is the 2025 depth-rank normalization.** Replacing ESPN's
across-slot ordinals with the correct within-slot depth raises WR forecasts.
That removes an accidental downward adjustment which helped MAE on numerous
low-output player-weeks. The effect occurs in all four model families, with
their weights held fixed, and remains positive in every tested source/policy/seed
combination. Availability and other source corrections are secondary effects.

## Measured contribution

This new comparison uses archived production inputs, the original WR numerical
recipe, three seeds, and CUDA FP32/TF32 eager execution. All arms score the same
2,761 regular-season 2025 player-weeks against corrected shared-component actuals.
Positive error changes are worse; values below are fantasy points per player-week.

| Model | Full PR change: MAE | Depth contribution to MAE | Availability contribution | Other source contribution | Full PR change: RMSE |
|---|---:|---:|---:|---:|---:|
| Ridge | +0.2539 | +0.2547 | +0.0004 | -0.0012 | +0.0550 |
| Plain NN | +0.1604 | +0.1395 | +0.0096 | +0.0113 | -0.0001 |
| Attention NN | +0.1082 | +0.1468 | -0.0291 | -0.0095 | -0.0800 |
| LightGBM | +0.1131 | +0.1147 | -0.0016 | -0.0001 | +0.0270 |

The contribution columns allocate interactions by averaging all six possible
orders of applying the three changes. They sum to the full MAE change before
rounding. The report also retains the separate interactions and conditional
effects, so this conclusion does not rely solely on that allocation convention.

Depth explains approximately 87% of plain-NN's MAE increase and essentially all
of Ridge/LightGBM's increase. For attention, the other corrections offset part
of the depth-driven increase. Across all twelve conditional depth comparisons
per model, the smallest MAE increase was +0.2541 Ridge, +0.0721 plain NN,
+0.1190 attention, and +0.1065 LightGBM. Full-bundle MAE worsened in every seed
for every model.

Three-seed paired MAE standard deviations are 0.0000, 0.0227, 0.0810 and 0.0146
for Ridge, plain NN, attention and LightGBM respectively. These describe training
variation in this experiment, not uncertainty across seasons. Ridge and LightGBM
reproduce the original reported endpoint means to displayed precision. The new
neural results differ from the earlier CPU measurements; no cross-device
seed-by-seed equivalence is claimed.

## Why MAE and RMSE disagree

With the fully corrected models and preprocessing held fixed, changing only
the depth representation increases average forecasts by +0.6111 Ridge,
+0.6575 plain NN, +0.7907 attention and +0.3909 LightGBM points.

The following attention results isolate that prediction-time depth change.
These outcome bins are retrospective descriptions, not forecast-selected cohorts.

| Actual points, depth-changed rows | Player-weeks | Mean forecast increase | MAE change | Mean squared error change |
|---|---:|---:|---:|---:|
| Below 2 | 1,105 | +0.6763 | +0.6278 | +4.2891 |
| 2 to below 5 | 430 | +1.0604 | +0.4166 | +4.0341 |
| 5 to below 10 | 395 | +1.2685 | -0.1378 | -0.6444 |
| 10 to below 20 | 277 | +1.3395 | -1.1257 | -15.3970 |
| 20 or more | 65 | +1.6556 | -1.6556 | -50.8456 |

There are many more low-output observations than ceiling games. Increasing their
forecasts adds enough absolute error to worsen MAE. Reducing the larger misses
on stronger games helps squared error disproportionately, improving attention
RMSE. This is an observed forecast-distribution tradeoff, not a changed scoring
formula. Receiving-yard and reception predictions explain most of the mean
forecast increase; the JSON report includes all four raw heads.

The normalization also removes a real input-distribution defect: the share of
common test rows beyond the depth scaler's ±4-sigma boundary falls from 28.287%
to 0.036%. Correcting that defect did not guarantee a lower MAE. The older,
semantically inconsistent codes happened to suppress forecasts in a useful way
for that metric on this season.

## Cohorts and other data changes

The complete-bundle pregame-reference top-24 MAE changes are +0.0131 Ridge,
+0.0860 plain NN, -0.0331 attention and +0.0371 LightGBM, on 432 fixed rows.
Prior-season elite changes are +0.0195, +0.0975, -0.0285 and +0.0854 respectively,
on 325 fixed rows. Attention improves both metrics in those cohorts in the mean;
that does not make the overall regression a universally acceptable tradeoff.
Week-1 MAE worsens for all four families. Detailed per-seed and RMSE values remain
in the machine-readable report.

The source factor includes restored observations, refreshed dependencies and
their downstream imputation effects. Common raw training depth values themselves
are unchanged; adding training rows changes the mean used for missing-depth
imputation. The direct normalization changes 2,272 common test rows. The original
2,276-row affected-depth subset additionally includes four small imputation
changes between the full old/new endpoints.

## Validation, limits and next step

All 12 fitted cells and 24 depth replays completed on AWS Batch Spot. Every saved
raw prediction exactly matches native pipeline inference in the parity check.
The seven added test observations are reported separately and do not enter the
paired deltas. Archived pregame and prior-season importance cohorts are available
and identical across arms. Input hashes, image/source pins, paired hardware,
truth identity and attribution reconciliation passed. Focused no-fit tests pass.

Original local CPU prediction files were unavailable. The investigation recovered
49 hash-verified archived production objects and labels this reconstruction
explicitly. The complete numerical recipe is frozen at PR #1564; this does not
evaluate the separate #1575 or #1568 changes or establish prospective accuracy.

**Recommended next step:** retain the corrected depth definition and run a
separate, bounded depth-feature ablation/calibration study on corrected data.
Use earlier seasons for candidate development, then require both MAE and RMSE
improvement with protected-cohort checks before any promotion. This investigation
does not implement that repair or endorse restoring the incorrect depth values.

See [the numerical report](results.md), [full results](results.json),
[execution receipts](execution.json), and [reproduction instructions](README.md).
