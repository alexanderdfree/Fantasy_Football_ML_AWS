### [FIXED] CI tooling scope and image-cache work inflated turnaround

**Files:** Test-scope selector and tests; training/serving image workflows;
training Dockerfile; rollout and image-provenance contract tests.

**What:** Agent configuration changes fell back to all eight test shards.
Conversely, an unknown path mixed with a recognized path could be overlooked.
Image workflows shared the default GHA cache scope and additionally restored
and exported large local layer caches. Writing the training source-SHA file
after copying source required materializing the heavy dependency filesystem.

**Fix:** Explicit provider configuration/hook routes use the shared shard;
every unclassified non-documentation path forces the full matrix. Each image
has one architecture-specific GHA v2 layer-cache scope. A small metadata stage
validates the immutable full source SHA, then linked source/metadata layers
reuse the dependency image. Runtime requirements and smoke checks are intact.

The former post-training fingerprint-change wait was already removed in
PR #1566, with readiness recovery refined in #1577. Regression coverage now
checks immediate success for an already-ready revision, unchanged readiness
responses, and propagation of failed readiness requests. This PR does not
claim the earlier removal as a new implementation or remove rollout checks.

**Validation:** Focused selector, hook, provenance and rollout tests cover the
changed contracts. A temporary draft-only workflow compares cold and warm
build/cache behavior on existing native GitHub runner types, with isolated
caches and local OCI exports. It never publishes to ECR, registers Batch jobs,
promotes artifacts, or deploys. Performance results will be recorded here after
the comparison; projections from the initial investigation are not measured
speedups. The temporary workflow/helper are removed after preserving evidence.

**Lesson:** Scope known paths explicitly without allowing them to hide unknown
paths. Measure cache transfer and materialization costs, preserve provenance,
and distinguish already-landed improvements from the current change.
