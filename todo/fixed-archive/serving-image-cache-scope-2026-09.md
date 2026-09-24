### [FIXED] Serving image cache shared restore prefixes across image families

**Files:** `.github/workflows/deploy.yml`,
`tests/artifacts/test_serving_image_cache_workflow.py`. Serving-only extraction
from #1587 at `994d36ade28328ed600a468383a3d318c29a37fc`.

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

**Lesson:** Keep cache namespaces specific to image and architecture. Preserve
source/dependency/runtime evidence separately from performance claims, and do
not turn historical inventory parity into a stronger output-equality claim.
