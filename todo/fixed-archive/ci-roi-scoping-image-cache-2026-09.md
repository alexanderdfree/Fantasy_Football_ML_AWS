### [FIXED] CI tooling scope and image-cache work inflated turnaround

**Files:** Test-scope selector and tests; training/serving image workflows;
training Dockerfile; rollout and image-provenance contract tests.

**What:** Agent configuration changes fell back to all eight test shards.
Conversely, an unknown path mixed with a recognized path could be overlooked.
Image workflows shared a broad Linux cache restore prefix and declared GHA
flags that the inline CLI did not actually export. They restored and exported
large local layer caches. Writing the training source-SHA file
after copying source required materializing the heavy dependency filesystem.

**Fix:** Explicit provider configuration/hook routes use the shared shard;
every unclassified non-documentation path forces the full matrix. Each image
has one local layer archive with image/architecture-specific keys and restore
prefixes; inactive GHA flags are removed. A small metadata stage
validates the immutable full source SHA, then linked source/metadata layers
reuse the dependency image. Runtime requirements and smoke checks are intact.

The former post-training fingerprint-change wait was already removed in
PR #1566, with readiness recovery refined in #1577. Regression coverage now
checks immediate success for an already-ready revision, unchanged readiness
responses, and propagation of failed readiness requests. This PR does not
claim the earlier removal as a new implementation or remove rollout checks.

**Validation:** 5,405 unit tests passed (2 skipped), including selector, hook,
provenance and rollout contracts. Native AMD64/ARM64 comparisons use isolated
caches, fresh workers, local OCI exports, one cold sample and three warm
source-invalidated samples. No ECR publication, Batch registration/training,
artifact promotion or deployment is involved.

The [first measured trial](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/actions/runs/35303372669)
rejected the proposed per-layer GHA backend: serving cold/warm-median build and
cache time went from 24/15 seconds to 43/17; training went from 288/136 to
420/128, with a 236-second warm outlier. Training's cold GHA export alone took
177 seconds (158 for its largest layer). Dependency inventories matched across
all samples. The follow-up retains local archives and isolates the linked-layer
layout improvement; final measurements are pending.

The first attempt also exposed a real cache-budget lock: even a tiny write was
rejected as read-only. Removing 7,466,398,711 bytes of redundant/superseded
archives reduced listed storage from 11,635,645,311 to 4,169,246,600 bytes and
restored writes, retaining the current main caches and existing spending limits.
Measurement stages clean up their own cache entries to stay within the allowance.

**Lesson:** Scope known paths explicitly without allowing them to hide unknown
paths. Measure cache transfer and materialization costs, preserve provenance,
and distinguish already-landed improvements from the current change.
