### [FIXED] Workflow inputs, tuning namespaces and infrastructure ownership diverged

**File(s)**: `.github/workflows/{ab-batch,ablate-rb-gate,benchmark-batch,
refresh-splits,retune-nn-batch}.yml`, `src/benchmarking/parallel_train.py`,
`src/tuning/{launch_ab,launch_tune,launch_ablate_scheduler,aggregate_results,
aggregate_scheduler,tune_nn_storage,tune_lgbm,resource_probe,ab_ensemble_seeds,
ab_opp_def}.py`, `infra/batch/teardown.sh`, `infra/aws/bootstrap.sh`.
Defects reproduced against `92be2873` during the 2026-09-10 audit.

**What**:

- Image/seed inputs reached Python or remote shell source. Valid canonical
  variant names were rejected. Configuration-only edits skipped split refresh.
- Rolling-origin workers mixed model/scaler output trees; forced concurrency
  selected CUDA on CPU hosts, and unsupported platforms received preexec hooks.
- Tuning collection chose namespaces different from submitted jobs; explicit
  eager zero was dropped and became the CUDA stacked default. Metadata-only
  launch paths imported an undeclared Torch dependency.
- Partial scheduler failures exited successfully, the tuned LightGBM comparison
  lost its configured objective, and failed environment setup leaked changes.
  Unix-only resource imports broke supported Windows startup.
- Batch teardown retained CPU resources and deleted serving's shared execution
  role. Bootstrap requested the portfolio certificate and instructed apex DNS
  changes, while deployment and clients require `fantasy.alexfree.me`.

**Fix**: Keep inputs as data, validate their actual contracts, and route
submission/collection through shared metadata. Forward explicit zeros, retain
configured objectives and restore partially applied environments. Isolate
origin output trees and honor platform/device overrides. Cover CPU teardown
resources while retaining the shared serving role. Request the application
certificate, reconcile an existing HTTPS listener, and render subdomain DNS
instructions within the portfolio's DNS zone.

**Validation**: Tests execute exact workflow/CLI fragments against local command
stubs, with harmless injection markers and valid-input controls. Native child
process checks cover device selection and unsupported affinity APIs. All-six
position controls verify namespaces, source arguments and isolated artifacts.
No live infrastructure or DNS mutation is used to establish these defects.

**Lesson**: Verify the complete producer/consumer handoff, including explicit
defaults, accepted names, interpreter boundaries and resource ownership.

Failed benchmark fanout collection belongs to PRs #1559/#1560. Manifest
seeding, bootstrap model preflight and serving-role policy repairs belong to
PR #1560 and were removed from this audit branch. The hostname/listener repair
is separate and absent from that PR's reviewed diff.
