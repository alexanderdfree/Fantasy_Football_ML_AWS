# Consolidated PR correctness review — paused checkpoint

Status: paused at the owner's request on 2026-09-11. **No merge, deployment,
production migration, model promotion, or AWS workload was performed.** This is
a review checkpoint, not final correctness approval. The owner plans to continue
in a fresh session.

The original request was: exhaustively review recent consolidated PRs, especially
#1566 and #1577; the prior independent judge checked scope only; validate #1566's
first production migration; clear #1534's three unresolved threads. Do not merge.

## Resume locations and checkout state

- Review worktree: `/Users/alex/.codex/worktrees/992b/Final-Project`.
- Local checkpoint branch: `codex/review-consolidated-20260911`.
- Evidence directory: `/Users/alex/.codex/worktrees/992b/Final-Project/logs/consolidated-review-20260911`.
- The evidence directory is ignored by Git. It includes complete logs, captured
  production manifests, all 30 retained model archives, the 26-file production
  data release, native test results, review probes, PR diffs, and the exact
  inputs/weights required by the successful fitted replay. Do not commit the
  data, weights, generated native build, or large logs.
- The temporary integration assembly used #1577 as its base, then applied only
  `src`, `tests`, and `ios` changes from #1575, #1534, #1576, and #1479 in that
  order. No conflict occurred. It intentionally did not assemble their prose
  and benchmark-history changes. Those full diffs are separately saved.
- Before pausing, the temporary source assembly is reversed. The local branch
  retains the #1577 source plus this checkpoint; it is not a replacement PR.
  No reviewed PR branch was pushed or changed.
- Reconstruct the tested assembly with the saved `assemble-1575.patch`,
  `assemble-1534.patch`, `assemble-1576.patch`, and `assemble-1479.patch`, in that
  order. Check each patch separately before applying. First refresh the remote
  graph and determine whether the reviewed heads have moved.
- Pinned Python used:
  `/private/var/folders/lb/5gkmj44x5z1c5w60kmrjk1mm0000gn/T/pr-consolidation-0n9objn5/venv/bin/python`.
  Python 3.12.12, NumPy 2.5.3, pandas 3.0.5, Torch 2.14.0, sklearn 1.9.0.
  The parent checkout's venv and miniforge environment have older dependencies;
  do not silently substitute them for production-parity claims.

## Reviewed revisions and scope

`origin/main` was refreshed to `ccab867901e3c3e89177460032953860addc42cf`.
The PRs below were open. GitHub checks for #1566 and #1577 were green at these
heads. Refresh their current state in the next session.

| PR | Bundle | Head | Base |
| --- | --- | --- | --- |
| #1557 | Tooling / remote bootstrap | `aef915d95379274dafde85e4148593a78be88dd3` | main |
| #1574 | Expert scoring / Timeline | `f4c6edc84df9eab9274d09d40b62f0eb10e702bb` | #1557 |
| #1566 | Training / data / prediction / serving contracts | `2335b132a5f8e0a2da94672c7e343d9051305afb` | #1574 |
| #1575 | NN correctness / inheritance | `b07403ef28d2ac1cd9616014fc82c681b535dcfe` | #1566 |
| #1576 | QB historical role proxy / pregame depth | `a67513964a13d6a2c805856740d84e0c1b69889d` | #1566 |
| #1577 | Scheduled maintenance | `a90ff165ddbdf16b33ade393a03084ffcdf42dd9` | #1566 |
| #1534 | Optional bounded flag scaling | `65228ec8a8017bcf62dfb5efb0014eace5840594` | #1575 |
| #1479 | Research screens / dated evidence | `a002e00b3ac22c0fd3e8809dd39120a08979cb6e` | #1557 |

The separate open #1565, #1568, #1578 and old tested-rejected #1326 were inventoried
but not included in the eight-bundle assembly. Scope can be expanded if intended.
#1566 alone has 331 changed files, not the first 100 returned by `gh pr view`.
Use saved complete Git diffs/file inventories, not the truncated API file list.

## Findings confirmed so far — not fixed

### F1 — P1: #1577 rejects the artifact paths emitted by #1566

`src/maintenance/storage.py:82` validates a stable artifact with
`startswith(f"models/{pos}/")`. Its manifest reader correctly looks under
`models/releases/v3/<POS>/manifest.json`, while the canonical publisher writes
artifact keys under `models/releases/v3/<POS>/history/`. Every valid new stable
artifact therefore fails `model_pins()` with `No verified stable manifest for QB`.
Both shadow and active workflows call this during `begin`, so neither can start
after the required migration.

- Exact source: [storage.py at reviewed head](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/blob/a90ff165ddbdf16b33ade393a03084ffcdf42dd9/src/maintenance/storage.py#L82).
- Existing `tests/maintenance/test_contracts.py::models` puts legacy artifact
  paths into new-format manifests, hiding the integration error.
- Independent failing probe:
  `test_review_maintenance.py::test_maintenance_accepts_the_model_keys_its_base_publishes`.
  It creates entries with the actual `new_history_key`, `build_manifest`, and
  `manifest_key` helpers. This fails on the reviewed code.
- Repair direction: validate against the canonical `history_prefix`, and update
  maintenance fixtures to use real publisher-produced keys throughout.

### F2 — P2: #1577 leaves Batch/EC2 snapshot publication outside its lease

The new lease steps are in the later `ecs_rollout` jobs. The preceding `train`
jobs still run `python -m src.prediction.build_snapshot` without the shared
`serving-publication` lease (`train-batch.yml:513-525`, `train-ec2.yml:430-440`).
Setting `FF_MAINTENANCE_LOCK_TABLE` does not serialize those snapshot writes with
weekly activation. The canonical snapshot publisher has CAS checks but does not
read DynamoDB itself. A CI builder starting during activation can capture and
advance its pointer while maintenance owns the lease; maintenance recovery then
refuses to overwrite that later publication. This contradicts the runbook's
claim that the opted-in writers share publication coordination.

- Sources: [Batch publisher](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/blob/a90ff165ddbdf16b33ade393a03084ffcdf42dd9/.github/workflows/train-batch.yml#L513),
  [EC2 publisher](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/blob/a90ff165ddbdf16b33ade393a03084ffcdf42dd9/.github/workflows/train-ec2.yml#L430).
- Independent workflow probes fail for Batch and EC2, and pass for
  `refresh-splits.yml`, which really surrounds publication with the lease.
- `test_snapshot_can_change_while_the_maintenance_lease_is_held` demonstrates a
  real canonical snapshot write succeeding while an independently held lease
  remains owned. This isolates the missing coordination from F1.
- Repair direction: acquire/release around the actual training-triggered cache
  publication as well as the rollout, preserving cleanup on failures. Review
  the full state transition after that change; do not remove CAS checks.

### F3 — P2: #1479 fails #1566's package boundary in the combined tree

Four new research specs import `available_models` / `per_model_metrics` from
`src.analysis.cohort_analysis` instead of their canonical
`src.evaluation.metrics` owner:

- `src/tuning/ab_games_gap.py:170`
- `src/tuning/ab_oline_continuity.py:236`
- `src/tuning/ab_proe_pace.py:241`
- `src/tuning/ab_qb_context_receivers.py:180`

Their compatibility re-exports prevent an immediate missing-symbol exception,
but pull the analysis entrypoint back into experiment metrics and deterministically
fail the newly introduced
`tests/shared/test_package_boundaries.py::test_experiment_metrics_do_not_reintroduce_analysis_entrypoint_dependency`.
Independent PR CI does not exercise this sibling combination. Update all four
imports to the canonical metrics module, preserving any separately needed `MODELS`
constant, and rerun the combined boundary/spec tests.

All three findings are saved locally with executable probes and complete logs.
They have not been posted as new GitHub review comments or fixed in PR branches.

## #1534 review threads — completed

All three threads were already addressed in the consolidated source. Verified
the actual call sites, documentation and tests; ran the focused tests in the
pinned environment; replied with the exact reviewed SHA and evidence; resolved
the threads. A fresh connector read confirmed all three are resolved.

| Thread | Concern | Evidence |
| --- | --- | --- |
| `PRRT_kwDOSCHeqs6gE4ot` | Optional feature columns / diagnostic drift | All direct `_train_nn` callers pass prepared `feature_cols`; `_scale_xs` rejects missing or unmatched columns when enabled |
| `PRRT_kwDOSCHeqs6gE4o6` | Overstated scaler integrity hash | Manifest prose explicitly distinguishes name/count hashes from persisted fitted statistics |
| `PRRT_kwDOSCHeqs6gE4pA` | Untested A/B spec | Real-spec tests cover ranges, activation failures, cohorts/bias, Ridge key, invalid status and coexistence with magnitude scaling |

Reply comment IDs: `3986358960`, `3986359133`, `3986359324`.
No new model training or bounded-flag enablement was performed. The flag remains
default-off. The focused pinned-environment run passed 54 tests.

## First-production-migration evidence for #1566

Production was inspected using read-only AWS ECS/ELB/S3 calls. At capture time:

- Service `fantasy-cluster/fantasy-service` was healthy at task revision 589,
  image `fantasy-predictor:ccab867901e3c3e89177460032953860addc42cf`, 2/2 tasks.
- Container and ALB checks still use `/health`; the ALB matcher is 200.
- Data release:
  `556115711494d5f7c10af9fe3e97b94b200e97a661078a3a1e4a874ca4b20340`.
- Producer fingerprint:
  `bdae10360ba1037a6a78808243bff84549c1a1f9faa082cb7b9d0bc0f8e59425`.
- No `models/predictions_cache/current.json` and none of the six
  `models/releases/v3/<POS>/manifest.json` keys existed. Production still uses
  the predecessor `models/<POS>/releases/manifest.json` protocol.
- All six predecessor source frontiers are main commit
  `b9d24f9259c7fd261ab1a4e77d4212d821726420`, order 1508.

Captured all 30 retained archives (17,188,229 bytes) using their enumerated ETags.
Every archive identifies a verifiable full main-history Git SHA. Executed the
actual `protect_legacy(..., dry_run=True)` at #1566's exact head against captured
manifests and bytes, with a replay transport that forbids writes. All six
positions passed, five archives each, preserving the source frontier. The
lineage basis was current main; no branch source was registered in production.

Captured the actual sealed production data release: 26 files, 56,207,111 bytes,
all checksum verified. Its 14 maintenance archive checks pass. An earlier probe
of an old local validation directory found duplicate obsolete cache versions;
that was **not** a production finding. The exact live manifest has only the
current caches. Do not report the discarded local-directory ambiguity as a bug.

The six actual stable production model sets also passed the canonical loader's
load/predict smoke in the combined integration tree, including the newer NN
compatibility paths. Existing first-cutover, rollback, source/intent, data and
snapshot tests passed at the exact #1566 head (262 targeted tests).

This validates retained-artifact compatibility and the offline migration
preconditions tested above. **A real first production cutover remains unperformed.**
Before claiming rollout validation, the new source needs its matching sealed data
release, six approved v3 model heads and a complete schema-11 serving generation.
Then the exact new image/task and `/ready` ALB transition and rollback must be
verified in an authorized rehearsal/cutover. The current absence of the new
objects is a prerequisite gap, not evidence that the migration code failed.

## Test evidence and limits

| Check | Result | Evidence file |
| --- | --- | --- |
| #1534 focused tests, exact head, pinned deps | 54 passed | `1534-pinned-tests.log` |
| #1566 contract/publication/data/prediction/deployment tests, exact head | 262 passed | `1566-contract-tests.log` |
| #1577 maintenance tests plus independent canonical-path probe | 67 passed, 1 failed (F1) | `1577-maintenance-tests.log` |
| Combined eight-bundle unit suite | 5,146 passed, 6 failed, 2 skipped initially | `integrated-unit-tests.log` |
| Rerun socket/ESPN-dependent failures outside sandbox | 14 passed; all five environment failures cleared | `retest-network-tests.log` |
| Remaining combined-suite code failure | F3 package boundary | `integrated-failures.txt` |
| Independent finding probes plus package boundary | 4 failures, 2 passing controls: F1, two F2 cases, F3 | `confirmed-findings.log` |
| Browser contracts against committed bundle | 9 passed | `browser-tests.log` |
| Native app compilation and hosted XCTest | 26 passed | `native-tests.log`, `ios-derived/Logs/Test/` |
| Actual production stable model loader smoke | 6 passed | `production-model-smoke.log` |
| Actual production archive readiness | 14 source checks ready | `production-archive-readiness.json` |
| Six previously fitted bundles through combined Predictor/frame adapter | 191,844 values bit-identical, max difference 0 | `fitted-replay/summary.json`, `fitted-replay.log` |

The five initial environment failures were three AF_UNIX socket tests, a local
HTTP server test, and a new reference test that unexpectedly calls real ESPN
without an injected ESPN loader. They passed with the necessary network/socket
access. Consider making that unit test hermetic, but do not misreport its sandbox
failure as a production scoring defect.

The fitted replay used saved production-configuration inputs/weights, all six
positions, and `FF_NN_NORM=layer` while saved bundles specify batch norm. Saved
constructors correctly won. This was **inference replay, not new training or GPU
validation**. It does not replace the merge gates for the NN/QB metric changes.

## Work remaining for the fresh session

1. Refresh `origin/main`, current PR heads/bases, overlapping PRs and review state;
   compare with the table above before reusing this checkpoint's evidence.
2. Reconstruct the source integration tree from the saved patches if the heads
   remain unchanged. Keep each real PR's base-specific diff separate from the
   combined working-tree proof.
3. Finish the exhaustive correctness review. Broad boundary tracing, full unit
   coverage, browser/native checks, real-artifact migration checks and fitted
   replay are done, but **this session did not complete a line-by-line correctness
   review of every changed file**. Do not label it exhaustive approval.
4. Decide/deliver scoped repairs for F1–F3 under the owner's review-only/no-merge
   instruction. Do not silently edit/push all PR branches merely because the
   probes fail. Clear F3 in the combined tree and retest the real publisher keys
   after F1. Exercise an actual activation/publication interleaving after F2.
5. Complete #1566's first migration rehearsal: sealed data and generation creation,
   actual image/task readiness, rollback, IAM, and any remaining production
   dependency checks. No production write was made by this review.
6. #1577 still needs the runbook's real AWS shadow rehearsals, full writer
   coordination, notification verification and active-publication checks before
   enablement. CloudFormation lint/template validation from the PR body was read,
   but this session did not independently call the state-machine validation API.
7. Review/reconcile current GPU and affected-position metric evidence for #1575
   and #1576 before any future merge authorization. No fresh GPU A/B was run here.
8. Prepare the final self-contained correctness report with ranked findings and
   evidence. Leave all PRs unmerged unless the owner later explicitly changes
   that instruction.

No background automation or delegated agent was created. All tests running when
the owner asked to pause completed before this checkpoint was saved.
