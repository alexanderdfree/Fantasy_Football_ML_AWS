### [FIXED] Serving image cache shared restore prefixes across image families

**Files:** `.github/workflows/deploy.yml`,
`tests/artifacts/test_serving_image_cache_workflow.py`. Serving-only extraction
from #1587 at `994d36ade28328ed600a468383a3d318c29a37fc`.

**Status (2026-09-23): HELD.** The serving-cache transport change has no paired
native acceptance result. Its local source patch is preserved for follow-up;
it must not be merged based only on the successful candidate arm.

**What:** The serving workflow restored a broad Linux cache namespace also used
by the training image, and declared both GHA and local cache backends.

**Fix:** Scope its local archive key and restore prefix to serving, OS and
architecture, and select one local layer-cache backend. Platform, context,
publication tags, push behavior, Dockerfile and runtime dependencies remain
unchanged. The training-image layout and temporary benchmark workflows are not
part of this extraction.

**Evidence:** The [retained native ARM64 record](evidence/serving-cache-native-arm64-35305366835.json)
preserves four cold/warm pairs from workflow run 35305366835. Each pair has
identical source-file hashes, installed dependency inventory and build
materials, with successful Dockerfile import/API smoke receipts. Both serving
arms used local cache exports in that follow-up harness: it does not directly
validate removal of the old GHA flags or establish a speedup. OCI image/config
digests differ and the complete archives were not retained, so this is not
whole-image bitwise-equivalence evidence. A fixed-source native cache-backend
comparison remains the explicit acceptance gate for that transport change.

**Native attempts:** [Run 35942815000](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/actions/runs/35942815000)
used reviewed diagnostic commit `92fc113ce23e8a87ad60b9d7fab88a14bcab5c8f`
and fixed application source `fe368b1ae6cadafd94bb67f7ffd92499266188bb`.
The legacy local+GHA arm failed on both attempt 1 and the single authorized
failed-job retry with `error writing layer blob: failed to reserve cache`.
The Dockerfile build completed before the cache-export failure, but neither
attempt produced a legacy inspection receipt; both comparison jobs were skipped.

The candidate local-only arm passed on attempt 1. Its receipt preserves 198
source-file hashes, 6,304 dependency-file hashes, 27 installed packages and
runtime OCI configuration. Independent checks matched its source files to the
fixed Git blobs, verified all nine direct dependency pins and checked the
runtime command, artifact-only environment and readiness contract. Attempt 2
reused that earlier success and artifact; it was not a second candidate result.
The [small outcome receipt](evidence/serving-cache-native-35942815000-held.json)
records attempt/job/artifact identities and the candidate receipt checksum.

This remains candidate-only evidence: no paired pass, whole-image equality or
performance improvement is claimed. No further retry, source/cache-policy
change, cache deletion, model fitting, image publication or deployment followed
the repeated failure. The observed cache-usage snapshot does not establish the
reservation failure's cause. Training-image optimizations remain separately
unvalidated.

**Lesson:** Keep cache namespaces specific to image and architecture. Preserve
source/dependency/runtime evidence separately from performance claims, and do
not turn historical inventory parity into a stronger output-equality claim.
