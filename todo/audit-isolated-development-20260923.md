# Isolated audit development campaign, 2026-09-23

This diagnostic branch tests previously unmeasured individual fixes against
`ecddeca88d8011cc18866b8842815bd7bc7999e5`. Production defaults are unchanged.
It does not repeat the failed September 17 repair candidates. Results from this
branch cannot authorize promotion: it deliberately retains main's inference
means to isolate the actual PR semantics. In particular, count precision does
not silently include the held corrected ZTNB expectation.

| Candidate | Exact proposed source | Affected models | Controls |
|---|---|---|---|
| `count_precision` | #1613, `00873f2b9dbddfc338b53604555ccc9ae9bac97f` | RB/WR/TE attention | Ridge, LightGBM, plain NN |
| `bagging` | #1606, `d690b417e6d521d379843b924707db8ab16efff1` | All six LightGBM | Ridge, plain NN, attention |

Count functions are copied verbatim into the experiment-only
`audit_count_candidate` module. A per-cell mutator installs them only for the
count arm. The bagging mutator sets the exact PR frequency on both saved
parameters and constructed estimators before any fit. Every cell resets both
switches, including the identical `baseline_rep` noise/control arm. No training
entrypoint imports these modules outside an explicitly requested experiment.

Development origins are 2022 and 2023, each with seeds 42, 123 and 7. Training
starts in 2013 (2015 for K), ends at origin minus two, and uses the preceding
year for checkpoint validation. Native K/DST providers defer context imputation
to the existing fold-local preprocessing hook and preserve K's kick histories.
The specs reject 2024/2025; no candidate is frozen for confirmation.

All fitting, including real smokes, runs through the existing AWS Spot Batch
harness. Production nonstacked FP32/TF32 and graph policies are required. Each
worker runs its paired arms on one host. Source SHA, release, hardware, actual
execution, prepared frame/array hashes, scored player-weeks/truth, full-precision
metrics, selected checkpoint states and independent restored-weight scores are
written to content-addressed evidence. Raw count parameters, validation errors,
cross-head covariance and numerical-reference checks are included. Saved
inference must reproduce each raw target and fantasy total. Pre-2024 weekly
references are explicitly reported unavailable; prior-season elite metrics are
required. Both metrics must improve and protected cohorts must not worsen before
any further candidate work can advance. Single-season/single-seed smokes do not
establish that conclusion.

The compatible release resolved read-only from the current producer index is
`9758de7902913140b9e1f766a63db89e558d9ebe11257c5f7946ace4ff3d8db4`.
The experiment adds no data-producer changes. The baseline main image digest is
`sha256:d0313cdaf05523db92b7ba152ba84312272b2864b3ef176d6e6f3e90e43d95ba`;
it cannot run the new specs. Build this diagnostic branch and pin its own SHA
and verified digest before submission.

The first real smoke is one WR seed and all three count arms:

```sh
FF_DATA_RELEASE=9758de7902913140b9e1f766a63db89e558d9ebe11257c5f7946ace4ff3d8db4 \
FF_AUDIT_ORIGIN=2022 python -m src.tuning.launch_ab \
  --spec src.tuning.ab_audit_count_development --positions WR --seeds 42 \
  --only baseline_rep count_precision \
  --image-sha <diagnostic-source-sha> --image-digest <verified-digest> \
  --env FF_AUDIT_ORIGIN=2022 --env FF_AMP_DTYPE=fp32 --cuda-graph auto \
  --s3-prefix ab_runs/audit-isolated-development-20260923 \
  --run-id smoke-count-2022-wr-s42 --wait false
```

After every smoke cell is `ok:true` and manifests pass identity/inference checks,
the count spec has 27 cells per origin (RB/WR/TE × three arms × three seeds).
`ab_audit_bagging_development` has 36 per origin (QB/RB/WR/TE);
`ab_audit_native_bagging_development` has 18 (K/DST). First smoke the bagging
skill path on one position and each native K/DST path before those fanouts.
Pass `FF_AUDIT_ORIGIN` to both submitter and container for every launch.

Local checks are restricted to no-fit unit tests, source-equivalence checks,
lint/format and launcher dry runs. The separately held #1607 feature change is
not included in this first image.
