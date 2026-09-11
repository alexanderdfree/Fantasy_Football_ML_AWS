### [FIXED] Scheduled maintenance used the superseded cache and rollout protocols

**File(s):** `src/maintenance/`, `src/prediction/build_snapshot.py`,
`src/artifacts/{deployment,serving_snapshot}.py`,
`src/scripts/advance_data_release.py`, and maintenance/CI infrastructure (PR #1577).

**What:** The maintenance branch read predecessor model manifests, staged a cache
tarball, and called the former ECS-only data-advance interface. The contracts
branch instead consumes v3 manifests and immutable snapshot generations and
requires an ECS/ALB readiness transaction. Combining them without adaptation
would leave prepared output unused or bypass the rollout contract. The original
90-second Lambda and 240-second lease also could not contain that transaction.

**Fix:** Stage the canonical builder's four cache files and model/data/pointer
token; validate the pinned request before conditional snapshot publication.
Preserve the deployed image and advance its data and snapshot pins through the
canonical readiness transaction. Persist snapshot intent and rollout state before
mutation, preserve later publishers on rollback, and recover a lost completion
receipt from verified live state. Use explicit Lambda, state-machine, readiness
and lease budgets. Keep schedules disabled and shadow mode as the defaults.

**Validation:** Local integration tests exercise staging without publication,
file corruption, unchanged-data/new-snapshot rollout, failure before/after ECS
acceptance, missing completion receipts, stale preparation, durable intent,
later-publication protection, standalone packaging and CI lease wiring. The
original recorded AWS evidence is historical; no new AWS stack was installed or
active/shadow execution launched during this consolidation.

**Lesson:** A textually resolved merge is insufficient when artifact ownership
and activation protocols change. Reuse the publication and readiness authorities,
including their recovery state, and verify each caller's execution budget.
