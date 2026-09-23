# Completed isolated development round — 2026-09-23

All **198 decision-bearing cells** completed and passed independent evidence
verification: 54 count-precision cells, 108 LightGBM bagging cells and 36 WR/TE
stint-reset cells. Twenty-one separate execution-smoke cells are excluded from
those grids. Fitting ran on the existing AWS Batch Spot harness; local checks
were read-only reconstruction, numerical functions and no-fit tests.

The global candidates remain held. No production default, calibration, tuning
search, confirmation run or merge is approved by this report. The only frozen
development-qualified proposal is **TE-only LightGBM bagging**; canonical 2024
reference coverage still blocks its confirmation. Global PR #1606 stays held.

| Component | Complete study | Decision |
|---|---|---|
| #1613 count precision | [54 cells](count-results.md) | No position passes both years; numerical precision passes but forecast/cohort gates fail. |
| #1606 bagging | [108 cells](bagging-results.md) | Global change fails; TE LightGBM alone passes both development years and elite gates. |
| #1607 stint reset | [36 cells](stint-results.md) | Shared change fails affected NN/LightGBM gates. Tiny WR Ridge gains do not qualify the shared feature change. |

Development uses scoring years 2022/2023, seeds 42/123/7, training through T−2
and validation on T−1, with training floors 2013 and 2015 for K. Production
preprocessing/native K/DST providers, missing-data exclusions, scoring components,
FP32/TF32 and graph policies were retained. Pre-2024 weekly-reference cohorts are
explicitly unavailable. These are retrospective development results; 2024/2025
confirmation and its two protected cohorts are not established.

## Immutable evidence and pins

[publication.json](publication.json) records verified SHA-256/size/URI receipts
for compact summaries, complete reports, source-object proofs, smoke summaries
and the frozen proposal. All published objects live under the isolated
`s3://ff-predictor-training/ab_runs/audit-isolated-development-20260923/reports/`
prefix. Original per-cell rows, raw count parameters, validation predictions and
checkpoint artifacts remain at their original immutable experiment locations.
Large per-cell proof documents and fitted artifacts are not copied into Git.

- Data release: `9758de7902913140b9e1f766a63db89e558d9ebe11257c5f7946ace4ff3d8db4`.
- Baseline model source: `ecddeca88d8011cc18866b8842815bd7bc7999e5`.
- Original count RB/WR source: `90265546bb05269a697af0635633d861b3eda738`, image
  `sha256:ef2af454c8694898d03e7239c9d456532fcbecb0f48aba49a8cce008d4722ff7`.
- TE identity repair and full bagging: `87213b853c4d724f6574dccddbb62dfcb6601cd1`,
  image `sha256:1efe7ddca433f36395d500046a042e3d4854ca24cbccc99299277792aea41fbc`.
- Stint input-proof source: `b42e26a8ff79a9ca71d9fbbf5ea2067483cf8277`, image
  `sha256:654bf288cda8cf40095edd5e4d6153959c1b425a85106ca30b01a89539a7de28`.

The initial TE observer inferred WR from shared target names and failed before
writing valid cell manifests. Its replacement resolves the production filter
identity. A verified bridge establishes 333 unchanged core/dependency files and
an exact observer-only AST change. The report retains distinct per-run source
pins and excludes the superseded failures; it never relabels them as one source.

## Frozen TE proposal

[te-bagging-frozen-proposal.json](te-bagging-frozen-proposal.json) preserves
candidate ID `93d08ebb5262eb77fe4a0d7bb440f1db7fc4ce91d8b8a621b55e1a003444435b`.
Only TE LightGBM would enable `subsample_freq=1`, preserving its configured
fraction 0.7359385 and all other parameters. The recipe pins every estimator
default/seed, 108 ordered features, raw targets, non-negativity, per-target
30-round stopping, source/image/data and evidence. Its relative report path
names the original diagnostic artifact; the matching full-report hash and
immutable S3 URI are preserved in `publication.json`.

No activation is implemented. Confirmation remains blocked; no relaxed gate or
additional tuning is authorized. Independent results do not establish combined
repairs. The failed September 17 campaign remains separate historical evidence.

## Read-only reproduction

```sh
python -m src.analysis.audit_development_report \
  --plan todo/audit-development-20260923/count-plan.json \
  --cache-dir /tmp/audit-development-verification \
  --output /tmp/audit-development-verification/count.json
```

Use `bagging-plan.json` or `stint-plan.json` for the other complete studies.
`--offline` requires an already downloaded complete archive. The tool never fits,
submits jobs, changes data pointers or publishes artifacts. It verifies content
addresses, complete paired grids, native actuals and elite membership, raw-head
scores, restored checkpoints, inference receipts and unchanged controls. Stint
verification additionally requires per-column prepared/scaler proofs and actual
attention train/validation/test input equality. No missing cohort is waived.
The original experiment Git objects must remain available to verify source bridges.

This extraction includes only four read-only verifier modules, their 52 no-fit
tests, exact run plans and compact evidence. It excludes experimental trainers,
candidate math, observer installation and production configuration changes.
