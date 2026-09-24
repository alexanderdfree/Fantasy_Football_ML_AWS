### [FIXED] CI image-cache exports were discarded on exact hits

**Files:** Training/serving image workflows and image-build contract tests.

**What:** Image workflows exported a fresh multi-GB layer archive on every warm
build, then actions/cache discarded it because the immutable primary key had
already matched. They also shared a broad Linux restore prefix and declared
GHA flags that the inline CLI did not actually export.

**Fix:** Exact primary-key hits skip only cache export and replacement. Misses
and fallback restores still export/save a new primary-key archive. Each image
has a local archive with image/architecture-specific keys and restore prefixes;
inactive GHA flags are removed. Image builds, publishing, source-SHA validation,
runtime requirements and smoke checks are unchanged.

The former post-training fingerprint-change wait was already removed in
PR #1566, with readiness recovery refined in #1577. Tooling scoping and its
conservative fallback, plus readiness regression coverage, landed separately in
PR #1626. Those improvements remain on main and are not duplicated here.

**Validation:** 6,261 unit tests passed (3 skipped), including cache hit/miss,
fallback, source identity and branch publication contracts. Native comparisons use isolated
caches, fresh workers, local OCI exports, one cold sample and three warm
source-invalidated samples. No ECR publication, Batch registration/training,
artifact promotion or deployment is involved.

The [first measured trial](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/actions/runs/35303372669)
rejected the proposed per-layer GHA backend: serving cold/warm-median build and
cache time went from 24/15 seconds to 43/17; training went from 288/136 to
420/128, with a 236-second warm outlier. Training's cold GHA export alone took
177 seconds (158 for its largest layer). Dependency inventories matched across
all samples. A [local-cache linked-layer follow-up](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/actions/runs/35305366835)
also missed the training warm-time gate: median 142 seconds versus 136 (cold
276 versus 288). That Dockerfile prototype was reverted. The final experiment
isolates discarded-export elimination on current main; measurements are pending.

The first attempt also exposed a real cache-budget lock: even a tiny write was
rejected as read-only. Removing 7,466,398,711 bytes of redundant/superseded
archives reduced listed storage from 11,635,645,311 to 4,169,246,600 bytes and
restored writes, retaining the current main caches and existing spending limits.
Measurement stages clean up their own cache entries to stay within the allowance.

**Lesson:** Scope known paths explicitly without allowing them to hide unknown
paths. Measure cache transfer and materialization costs, preserve provenance,
and distinguish already-landed improvements from the current change.
