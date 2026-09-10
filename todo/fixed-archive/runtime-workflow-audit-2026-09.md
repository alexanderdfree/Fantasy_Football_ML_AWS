### [FIXED] Workflow inputs, tuning namespaces and infrastructure ownership diverged

**File(s)**: `.github/workflows/{ab-batch,ablate-rb-gate,benchmark-batch,
refresh-splits,retune-nn-batch}.yml`, `src/benchmarking/parallel_train.py`,
`src/tuning/{launch_ab,launch_tune,launch_ablate_scheduler,aggregate_results,
aggregate_scheduler,tune_nn_storage,tune_lgbm,resource_probe,ab_ensemble_seeds,
ab_opp_def,feature_selection,attn_knob_experiments}.py`, lightweight A/B specs,
`infra/batch/teardown.sh`, `infra/aws/bootstrap.sh`, provider hooks and their
shared/generated helpers, and `gemini-scheduled-triage.yml`.
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
- Stacked Stage1 plans advertised 24 seeds but generated three-seed commands.
  Advertised A/B specs imported undeclared heavy dependencies at dispatch time.
- Claude/Gemini and generated WSL guards compared unnormalized paths; Claude
  promoted local splits after queued or mismatched merge events.
- Claude/Gemini configured hook commands split repository paths containing
  spaces, preventing every hook from running even though the scripts handled
  those paths correctly.
- Scheduled issue triage required both unlabeled and needs-triage status,
  selecting neither intended population.

**Fix**: Keep inputs as data, validate their actual contracts, and route
submission/collection through shared metadata. Forward explicit zeros, retain
configured objectives and restore partially applied environments. Isolate
origin output trees and honor platform/device overrides. Cover CPU teardown
resources while retaining the shared serving role. Request the application
certificate, reconcile an existing HTTPS listener, and render subdomain DNS
instructions within the portfolio's DNS zone.
Emit canonical stacked seeds explicitly and import heavy dependencies only in
the functions that need them. Share canonical path/current-PR validation across
providers and require the matching completed merge before promoting splits.
Quote configured hook executable paths while preserving every other setting.
Select the union of the two intended triage populations.

**Validation**: Tests execute exact workflow/CLI fragments against local command
stubs, with harmless injection markers and valid-input controls. Native child
process checks cover device selection and unsupported affinity APIs. All-six
position controls verify namespaces, source arguments and isolated artifacts.
No live infrastructure or DNS mutation is used to establish these defects.
Provider regressions use isolated temporary Git fixtures; no real merges or
parent-checkout writes are exercised. Twenty advertised A/B specs resolve in a
clean environment containing only the workflow's lightweight dependencies.
Triage controls preserve repository/open-issue scope and perform no live
label or comment writes.
All twelve configured Claude/Gemini commands now execute from space-containing
paths; ordinary-path controls remain valid. Ninety-seven related checks pass.

**Lesson**: Verify the complete producer/consumer handoff, including explicit
defaults, accepted names, interpreter boundaries and resource ownership.

Failed benchmark fanout collection belongs to PRs #1559/#1560. Manifest
seeding, bootstrap model preflight and serving-role policy repairs belong to
PR #1560 and were excluded from this audit's changes. The hostname/listener repair
is separate and absent from that PR's reviewed diff.
