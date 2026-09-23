# Independent development verification

`src.analysis.audit_development_report` reads the explicitly declared experiment
runs, verifies their immutable evidence and reconstructs the scoring truth from
hash-verified release inputs. It never fits a model/scaler, submits a job, changes
a data pointer or publishes an artifact. Downloads and reports stay in an
explicit private cache.

The plan declares every scoring year, position, seed, arm, source SHA and image
digest. Development requires the complete 2022/2023 × 42/123/7 grid for every
affected position. Smoke plans are separate and can never establish development
qualification. Duplicate or overlapping runs fail verification. Results compare
the canonical scoring components, preserving unavailable actuals and identical
player-weeks. K/DST use their native data loaders with source fetching/rebuilding
disabled; K's neural fantasy sums preserve production FP32 sign-vector arithmetic.

The verifier checks:

- SHA-256 and sizes for release inputs, cell manifests, saved rows, validation
  predictions and selected checkpoint states.
- Source/image/job identity, paired execution/hardware, prepared input and truth
  fingerprints, prior-season elite membership and complete model sample counts.
- Independent raw-head overall/elite scores and restored-validation scores,
  selected epochs/stop reasons and the pinned worker's saved-inference receipts.
- Bit-identical repeat-baseline forecasts/checkpoints and every unchanged model
  control. Stint experiments permit exactly the declared feature-input changes.
- Per-position, model and year paired three-seed MAE/RMSE differences, sample
  standard deviations and Student-t intervals for the mean seed difference.

Stint evidence additionally requires `audit-input-proof/v1`: every prepared
column in all three splits, ordered production feature lists, fitted imputation
and neural scaler fields, transformed matrices and actual resident/test neural
arguments. Only the two intended rolling columns and their corresponding
imputation/scaling transforms may change. All attention inputs/scalers, all
targets and row identities must remain identical. Missing column/split evidence
fails verification; whole-X inequality is not treated as an exemption. The
observer extension lives on the separately pinned `codex/audit-stint-input-proof`
branch, leaving the minimal TE count repair source frozen.

The intervals measure training-seed variation within each retrospective season;
they are not independent season uncertainty. Pre-2024 weekly-reference cohorts
are explicitly unavailable. A development pass requires lower mean MAE and RMSE
in both years and non-increasing prior-season elite errors. Numerical candidate
failures also block qualification. Confirmation still requires all six positions
in 2024 and 2025, both protected cohorts and any required combined experiment.
These tools cannot authorize a merge or replace that confirmation gate.

## Reproduce a report

```sh
python -m src.analysis.audit_development_report \
  --plan todo/audit-development-20260923/count-smoke-plan.json \
  --cache-dir /tmp/audit-development-verification-20260923 \
  --output /tmp/audit-development-verification-20260923/count-smoke-report.json \
  --markdown /tmp/audit-development-verification-20260923/count-smoke-report.md
```

`--offline` reproduces a previously downloaded complete run without AWS calls.
The JSON retains every source-object proof, full-precision scores, paired seed
deltas, checkpoint metadata and explicit source/image pins. A nonzero exit means
verification is incomplete or failed; a zero exit is not a model-quality pass.

## TE observer correction

The initial TE count jobs used ambiguous target-name position inference and
failed before their cell manifests could be written. WR and TE share targets.
`87213b853c4d724f6574dccddbb62dfcb6601cd1` resolves observer identity from the
production filter callable; unknown identities fail before fitting.

Version-2 plans declare each run's source/image independently. They may exclude
the initial failed TE cells only with the explicit `observer_position_identity`
supersession reason and a complete replacement TE grid. Successful cells cannot
be silently discarded. A source bridge is mandatory when combining the original
RB/WR source with the corrected TE observer. The reporter reproduces all 333
core/dependency hashes, confirms the count candidate is unchanged and verifies
that reversing only the reviewed identity replacement restores the parent AST.
It never relabels the resulting evidence as one common source SHA.

`count-initial-plan.json` preserves the original submission. `count-plan.json`
declares the valid original RB/WR cells and the two corrected TE replacement
runs, with both source/image pins and the verified bridge. The original failed
TE cells are not a forecast-quality conclusion. `bagging-plan.json` declares
the separately submitted 108-cell, six-position study on the corrected observer
source, retaining the bridge to the source used for its execution smokes.
Stint smoke plans and results remain separate from both full studies.

Submission receipts in `receipts/` preserve the exact run manifests, hashes,
job IDs and Spot queue. Root owns promotion decisions, merges and pool restore.

## Completed development evidence

- [Count precision: 54 verified cells](evidence/count-full.md). No position
  passes both development seasons. The stable numerical candidate passes its
  numerical checks but fails the required forecast/cohort gate.
- [Bagging: 108 verified cells](evidence/bagging-full.md). The global six-position
  change fails. TE LightGBM alone improves overall MAE/RMSE and preserves both
  elite metrics in both development seasons; a TE-only unit would still need
  explicit scope, a frozen candidate and confirmation.
- [WR stint execution smoke: 3 verified cells](evidence/stint-wr-smoke.md).
  Per-column containment and unchanged actual attention inputs/scaler,
  checkpoint and predictions pass. This single seed is not a development gate.

The JSON siblings retain full precision, paired seasonal results, uncertainty,
selected checkpoints, all source/image pins and immutable object proofs.
No confirmation, production default change or merge is authorized by these
results. The 2022/2023 weekly-reference cohort remains explicitly unavailable.
