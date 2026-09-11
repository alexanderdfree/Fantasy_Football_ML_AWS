# ADR-0011: Smoke-test gate + always-stable artifact (manifest v3)

**Status:** Accepted

**Decision.** Publishing training jobs upload a versioned artifact and validate its structure and load/predict behavior. Manifest v3 keeps `current`/`previous` for candidate visibility, and `stable`/`previous_stable` exclusively for approved artifacts. An eligible passing output advances `stable` and retains the prior distinct approved artifact as `previous_stable`; failed candidates preserve both approved pointers. Each entry records `smoke_passed`. Model consumers try only the approved pointers, with isolated extraction directories so partial corrupt extraction cannot contaminate fallback. V2 `stable` remains trusted during migration; candidate-only v1/v2 manifests fail closed. Historical production HTTP serving follows the complete serving-generation pointer defined in [ADR-0027](0027-versioned-prediction-and-execution-contracts.md), rather than rebuilding whenever an individual model head advances.

Manifests and history live under `<prefix>/releases/v3/<POSITION>/`, outside legacy writers' mutation and collection paths. When protected state is absent, migration verifies legacy archive provenance against registered source ancestry and copies retained bytes into protected storage before publishing a pointer. Protected read failures never downgrade to legacy state. Legacy runtime readers record the object actually installed, so the first protected manifest replaces legacy bytes even on the poller's first observation.

Training images bake their actual full Git SHA. Registered first-parent ancestry and a retained source frontier reject older-image completions; a pre-training intent orders runs on the same source and binds their position, dataset and run identity. Split jobs reuse the immutable plan's intent. A successful output claims its canonical receipt **before** pointer mutation, and concurrent/retried attempts use that same output. Failed smoke results cannot poison the accepted-output slot. Superseded successes retain their own verified output without changing the active model.

Publication reads manifest content and ETag together and uses `IfMatch` or create-only `IfNoneMatch`. After a conflict it rereads and revalidates source, intent and operator rollback eligibility before retrying. Manual rollback rotates a dedicated `rollback_epoch`; intents retain the epoch present before computation (the serialized `publication_revision` field). Generic CAS revisions also rotate during collection, but cannot masquerade as operator rollback. Retries preserve their original epoch.

Automatic destructive retention is suspended. Independent Batch publishers cannot safely delete from a local manifest snapshot, and re-reading immediately before deletion leaves a time-of-check/time-of-use race. `artifact_gc.prune` retains its callable signature but warns and returns no deletions without contacting S3; producers no longer call it. Explicit operator collection now lives in `src.artifacts.gc`: dry-run by default, 24-hour grace, a non-expiring manifest CAS lock, and protection for manifest references, retained receipts, and plan-owned objects. Every manifest write carries a fresh revision nonce to prevent ABA after lock release. Producers and operator promotion reject a held lock; interrupted collectors require exact-token recovery only after confirming the original process stopped. Retained-plan artifacts and dataset objects are deliberately outside this collection policy. The manifest's recent `history[]` list remains bounded; object count is not bounded by it.

The `promote.py --to` interface downloads/extracts and smoke-tests the selected history artifact before advancing both `stable` and `current`, preserving `previous_stable` and the strongest source/intent frontiers. Legacy cutover copies approved bytes before pointer mutation. `--dry-run` validates and previews without writing objects or manifests. A history entry or operator selection alone is not serving approval.

**Context.** D10's CI gate catches code regressions before they merge; it does not catch artifact regressions — a model that trains successfully but predicts NaN, or whose feature-column hash drifted past the scaler's, will pass pytest and then silently degrade the live dashboard. The weather/Vegas-missing-at-inference incident (TODO archive) is the canonical example: training-pipeline and serving-pipeline drift shipped past every test and only surfaced when users saw zeros in the dashboard. Prior to D11, any successful S3 upload silently became "live" to the next ECS task that booted. Production safety is a core claim of the project; this decision is what backs it.

**Options considered.**

| Option | Safety | Operator cost | CI/CD fit |
|---|---|---|---|
| Always advance — newest upload is live | Low | None | Matches D10's trunk-based ratchet |
| Manual promote step (operator approval before live) | High | Every push needs a human | Defeats D10 |
| **Smoke-test gate + stable/current/previous/history (chosen)** | High | None on success; rollback is a `promote.py` invocation | Compatible with D10 — gate runs in the same CI job |
| ECS canary deploy (split live traffic) | Very high | Needs traffic-splitting infra | Over-engineered for single-task ECS service |

**Chosen rationale.** The smoke test is *non-fatal*: `current` advances on successful conditional publication (so smoke failures are visible in the manifest for post-mortem), but `stable` only moves on success. The frontend resolves the artifact pointer at boot from `stable`, so a failed gate means new ECS tasks keep loading the prior good model while a human investigates. PR #179 (`8c42e88`) closed the operational loop: after a successful train, the workflow now issues `aws ecs update-service --force-new-deployment` so the newly-promoted `stable` is actually consumed by a fresh task instead of staying invisible to the long-running one. The smoke test itself ([src/shared/smoke_test.py](../../src/shared/smoke_test.py)) catches the four failure modes that matter at promotion time: pickle/torch.load deserialization errors after class-path drift, state-dict shape mismatches (feature-count drift between training and the runtime registry), scaler `feature_cols_hash` drift caught by `assert_scaler_matches`, and NaN/Inf predictions on benign input from a collapsed head.

**Rejected.** Manual-promote breaks the trunk-based CI/CD ratchet from D10 — every push would need a human. Canary deploys need real traffic-splitting infrastructure absent on a single-task ECS service. Doing nothing and relying on tests proved insufficient (see Context).

**Consequence.** Training completion, smoke approval and current selection are
separate states. Model builders consume approved pointers; forensics and the
operator listing also expose candidate/history entries. Failed candidates retain
the prior approved fallbacks. Smoke validation exercises its recorded inputs;
production-path and cohort tests remain necessary for prediction correctness.
`promote.py --list` shows retained history, and `--to <key>` validates a target
before conditional publication. The legacy `model.tar.gz` mirror remains removed
as established by D13 layer C.

**References.** [src/shared/smoke_test.py](../../src/shared/smoke_test.py) (smoke test entrypoint + `SmokeTestFailed`), [src/shared/model_sync.py](../../src/shared/model_sync.py) (manifest v3 schema, `build_manifest`/`load_manifest`/`write_manifest`, `_sync_one` consumer with approved-only fallback), [src/batch/train.py](../../src/batch/train.py) (upload → smoke → promote sequence), [src/scripts/promote.py](../../src/scripts/promote.py) (operator rollback CLI), [src/shared/artifact_gc.py](../../src/shared/artifact_gc.py) (suspended retention compatibility entrypoint), [.github/workflows/train-ec2.yml](../../.github/workflows/train-ec2.yml) (ECS force-new-deployment after train). Commit arc: `1b20e9e` (versioned history / PR #104) → `e8bf2a7` (promote CLI / PR #122) → `c7fa2d7` (smoke-test gate + bucket versioning / PR #130) → `8c42e88` (ECS force-rollover / PR #179).

## Dataset and run identity

Batch pins immutable dataset/build-plan IDs before submission and claims
per-plan artifact receipts before conditional model publication. Normal Batch
benchmark/download consumers verify those receipts, including artifact checksum
and serialized code/dataset/plan provenance, instead of following later global
promotions. Dataset capture/replay includes K/DST provider calls and distinguishes
observed empty responses from unavailable sources. The deployment prerequisites,
source scope, legacy EC2 boundary, and deferred collector requirements are in
[Training build plans](../training-build-plans.md). Transport implementation now
lives in `src/artifacts` with `src/shared` module aliases for compatibility.
The EC2 path retains explicit mutable input selection, but its image is digest
pinned and its collector follows exact source/run/position receipts. Serving
cache construction verifies the selected plan/run's output keys before building.

## Changelog

- **2026-09-10** — Isolated v3 storage from queued legacy writers/GC; added baked source identity, registered ancestry, pre-training intents, canonical first-success receipts and exact legacy-run collection. Dedicated rollback epochs survive GC and retries. Legacy dry-run migration is read-only, and first protected refresh replaces installed legacy bytes.

- **2026-09-10** — Added an explicit dry-run-first operator collector with manifest CAS ownership, revision nonces, grace-period and retained-plan protection, plus stopped-process lock recovery. Legacy publisher-snapshot pruning remains non-destructive.

- **2026-09-10** — Added immutable dataset/build plans, exact publication receipts and output verification, provider snapshot capture/replay, serving-cache-before-rollout workflow gates, and an explicit artifact transport package. First deployment requires an initial snapshot refresh; numerical package moves remain covered by retraining and version-2 benchmark fingerprints.

- **2026-09-10** — Manifest v3 retains a previous approved artifact and fails closed for unapproved fallback. Producer and manual rollback publication use ETag preconditions; operator rollback revalidates its target. Automatic destructive retention is suspended because publisher-local GC can delete concurrent active/in-flight artifacts. Regression tests cover conflicting first/existing writes, failed/missing approved artifacts, candidate-only legacy state, read failures, and contaminated partial extraction. The later source/intent contract addresses completion ordering; retained-plan/dataset expiration remains separate policy.

- **2026-05-19** — D11 (smoke-test gate + always-stable manifest v2) and D12 (training-step perf composition; `torch.compile` measured and rejected on T4) added. D2 extended to note the user-facing PPR/Half-PPR/Standard scoring switch (PR #153). D4/D6 reconciled with K's `ATTN_L1_FEATURES` removal (PR #199) — K's attention static branch now matches the documented "no rolling features in the attention static channel" convention across all six positions.
