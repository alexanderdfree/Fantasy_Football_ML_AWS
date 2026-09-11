### [FIXED] Consolidated data isolation, truth provenance and client boundaries

**File(s):** `src/data/release.py`, `src/data/dst_scoring.py`,
`src/evaluation/records.py`, `src/shared/comparison_truth.py`, the six target
builders, `src/prediction/historical.py`, `src/contracts/serialization.py`,
`ios/Sources/Stores/SnapshotStore.swift`, and associated contract/E2E tests
(implementation `28002bb8`, PR #1566 follow-up).

**What:** Nested provider symlinks escaped isolated hydration; DST threads lost
capture context; evaluation identities omitted scored observations; old E2E
fixtures/assertions no longer matched strict recipes and partial results.
Adjacent audit fixes also needed adaptation to the new data/HTTP/client owners.

**Fix:** Validate nested destinations before downloads/quarantine/install, copy
provider context into workers, explicitly bind actual columns and availability
into evaluation identity, and migrate E2E fixtures without weakening contracts.
Keep pre-fill missingness in non-feature reporting fields through target/feature
construction. Serialize matching shared actuals/forecasts separately from native
display values. Preserve HTTP headers, wiki source timestamps and native request
generation/scoring ownership. Recover mutable external caches without fetching
or removing verified empty pinned data. Retain Ridge selection sidecars in
fitted bundles and repair the identified recorded-activity filter.

**Validation:** Focused isolation/capture/evaluation/cache tests, all-six truth
and whitelist checks, enabled WR/TE/split E2Es, native delayed-response tests,
and 18 production-input target comparisons. The target callbacks preserve every
original column and add only reporting fields. Activity-row retention is a
separate behavior change. Final combined checks and input/source fingerprints
are recorded in the PR validation update.

**Lesson:** File-disjoint PRs can still disagree at producer/consumer boundaries.
Preserve and test observation truth, exact source ownership and request identity
when moving code; passing mocks must use the actual producer's contracts.
