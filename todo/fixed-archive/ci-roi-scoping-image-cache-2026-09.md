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
fallback, source identity and branch publication contracts. The
[final native run](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/actions/runs/35968269140)
used current-main inputs at 66514437 and candidate d20ad3ab. One cold dependency
seed per image was shared only after verifying identical Docker inputs; three
warm samples per variant ran interleaved on fresh native workers. Every exported
image passed the original smoke checks, source-SHA checks and matching
source/dependency inventory checks. The miss path is unchanged and covered by
workflow execution tests. Local OCI exports excluded production ECR push and
deployment; no Batch training, artifact promotion or deployment was performed.

| Build/cache time | Baseline warm samples | Candidate warm samples | Median change |
|---|---|---|---|
| Training/AMD64 | 167, 128, 193 s | 131, 119, 130 s | 167 → 130 s (-22.2%) |
| Serving/ARM64 | 18, 15, 15 s | 16, 17, 14 s | 15 → 16 s (+6.7%; effectively flat) |

The shared cold controls were 542 s for training and 21 s for serving; these are
single controls, not separate before/after cold measurements. Full job queues,
ECR push and live rollout are outside this timing claim. The
[retained measurements](../validation/ci-cache-export-2026-09.json) include
per-step durations, source revisions, dependency digests and job links. The
temporary workflows/helper and their experiment caches were removed after
collecting results, including the review-flagged cleanup/probe code.

The [first measured trial](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/actions/runs/35303372669)
rejected the proposed per-layer GHA backend: serving cold/warm-median build and
cache time went from 24/15 seconds to 43/17; training went from 288/136 to
420/128, with a 236-second warm outlier. Training's cold GHA export alone took
177 seconds (158 for its largest layer). Dependency inventories matched across
all samples. A [local-cache linked-layer follow-up](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/actions/runs/35305366835)
also missed the training warm-time gate: median 142 seconds versus 136 (cold
276 versus 288). That Dockerfile prototype was reverted. The final change
isolates discarded-export elimination and leaves the Dockerfiles unchanged.

The first attempt also exposed a real cache-budget lock: even a tiny write was
rejected as read-only. Removing 7,466,398,711 bytes of redundant/superseded
archives reduced listed storage from 11,635,645,311 to 4,169,246,600 bytes and
restored writes, retaining the current main caches and existing spending limits.
Measurement stages clean up their own cache entries to stay within the allowance.
On September 24, two obsolete disposable-integration image caches totaling
7,215,776,487 bytes were also removed before validation; production caches and
spending settings were retained.

**Lesson:** Scope known paths explicitly without allowing them to hide unknown
paths. Measure cache transfer and materialization costs, preserve provenance,
and distinguish already-landed improvements from the current change.
