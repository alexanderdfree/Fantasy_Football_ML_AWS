### [FIXED] Incomplete cache/model identity and uncoordinated artifact publication

**File(s):** `src/prediction/`, `src/artifacts/`, `src/training/`,
`src/shared/feature_cache.py`, `src/serving/state.py`, `src/contracts/`, and the
corresponding tests. Foundation commit `911223e1`; complete integration is
included in PR #1566.

**Readiness migration follow-up (2026-09-11, PR #1577):** The deployed legacy image
`13a5d4f7570677f2a7bbe27444caa22ea2d3107e` returned 200 from `/health`, but
its missing `/ready` route became 500 when the non-API exception handler
rethrew `NotFound`. The planned ALB `/ready` matcher `200,404` would remove
healthy legacy targets before replacement tasks were hydrated. The rollout
helper now uses `/health?readiness=1` with matcher 200: old images ignore the
query; new `routes.health` delegates it to strict artifact readiness. Only
after exact revision/image/container readiness does ALB switch to `/ready`.
Tests cover the observed legacy 500, new missing/corrupt/revoked generations,
new 404/500/503 failures, successful hydration, and prior-settings restoration.

**What:** A repository-wide design audit reproduced four gaps: a changed feature
projection reused the old cache entry; reordered history inputs passed shape/hash
checks; concurrent publisher cleanup deleted another publisher's selected object;
and fallback selected a candidate that failed smoke validation. Integration review
also demonstrated verified generation A being replaced by B before model loading,
while the consumer still reported A's identity. Native comparison notes described
superseded scoring/sample semantics. A failed final cache write could publish an
earlier generation without its assembled comparison data.

**Fix:** Complete computation identities drive prepared-data caches and memos.
Model bundles carry ordered schemas, fitted state, constructors and file identities;
loading pins verified bytes before deserialization. Publication uses conditional
updates, approved fallback, immutable receipts and a coordinated collector. Explicit
execution contexts separate training from output effects. Historical serving uses
verified complete generations and app-owned request snapshots. Shared wire definitions
and generated fixtures exercise current browser/native behavior. New plans,
standalone receipts and serving snapshots share the canonical data-release ID.
Offline publication requires the final complete write and publishes that exact
generation even if the local pointer subsequently changes.

The PR follow-up removes ML libraries from the serving dependency subset and
selects an artifact reader before importing any builder. Historical/live
prediction construction and provider adapters now have owners outside serving;
HTTP-independent state and compatibility imports retain existing callers.
The image's runtime probe serves all six positions and checks unavailable,
corrupt and stale artifacts with ML/build imports forbidden. Offline publication
is exercised in a separate interpreter with Flask and serving imports forbidden.

**Evidence:** `tests/shared/test_feature_cache.py`,
`tests/prediction/test_prediction_consumer_parity.py`, `tests/artifacts/test_gc.py`,
`tests/artifacts/test_serving_snapshot.py`, `tests/serving/test_state.py`, browser
contract tests, and native XCTest tests. The PR also records actual six-position
multi-seed baseline/candidate evidence; synthetic serialization tests are not
presented as model-quality validation.

**Lesson:** Names, dimensions and a green build do not establish computational
identity. Capture complete inputs and fitted state, make approval distinct from
candidate existence, and protect the bytes actually consumed. The design and
migration boundaries are recorded once in ADR-0027 and the build-plan runbook.
