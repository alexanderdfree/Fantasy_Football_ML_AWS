# Training dataset and publication identities

Training selects ADR-0026's canonical immutable data release before submitting
jobs. Its SHA256-addressed manifest covers raw inputs, train/validation/test
splits, and any captured provider responses. Each file has a digest and size.
The producer fingerprint comes from the selected image commit's data-building
sources, not the workflow runner's newer checkout. `DATA_PRODUCER_PATHS` in
`src/data/release.py` is the shared producer/trigger authority.

`data/by-producer/<fingerprint>/manifest.json` selects the latest complete release
for that recipe; `data/manifest.json` is the global current pointer. Selection
captures one release ID, so later refreshes cannot change a plan's inputs.
Missing releases, incomplete publication, corruption, permission errors and
timeouts block submission. Schema-2 plans use `data-release-v1`. Retained
schema-1 plans can explicitly read `dataset-v1`; new work never publishes a second
mutable `datasets/sources` authority.

## First deployment and operator use

Before a gated train, run **Refresh data splits in S3** for the image's producer
revision and wait for its sealed data upload. The refresh regenerates numerical
splits, prewarms native K/DST inputs and the archived evaluation reference, and
captures provider responses before sealing. No cloud refresh is performed by
opening this PR.

For an operator-controlled local refresh, use a dedicated unpinned checkout with
the normal dependencies and AWS credentials configured:

```bash
python -m src.data.build
python -m src.orchestration.datasets publish --bucket BUCKET --revision COMMIT --output dataset.json
```

The build completes and seals inputs; upload refuses an absent or changed seal.
Do not label a working tree with another revision's producer identity. Historical
images without compatible releases require an explicit refresh at that revision.

`python -m src.orchestration.build_plan` registers the actual image SHA's ancestry
from a full checkout, selects the matching canonical release, and writes a
content-addressed plan with code, data, producer, seed, positions, run ID,
publication intents/rollback epochs, and resolved Batch revision/image references.
Manual dispatch can infer the full SHA from the selected GPU image tag. Selection
pins that definition before waiting for data; untagged images cannot establish
code identity. Full, NN, CPU and merge jobs receive the same `FF_BUILD_PLAN_ID`,
`FF_DATA_RELEASE`, `FF_DATASET_ID`, `FF_DATA_FORMAT`, `FF_TRAIN_GIT_SHA` and
`FF_TRAIN_IMAGE_ID`. The two data ID variables must agree. Workers validate these
before computation, including the executable's baked `.training-source-sha`.
Batch uses full-SHA tags; registry tag immutability remains infrastructure policy.
EC2 resolves its selected tag to an immutable digest.

Standalone EC2/local publication also pins canonical data. Its explicit
`FF_LEGACY_RUN_ID` identifies the run without a Batch plan; the name does not imply
mutable data. Register the executable source before direct operator publication:

```bash
python -m src.scripts.register_training_source --sha FULL_IMAGE_SHA --bucket BUCKET
```

The canonical `models` prefix accepts main's first-parent history. Branch
experiments use a separate model prefix. Nonpublishing ablations do not reserve
publication intents. Strict Batch CI requires a plan; EC2 instead passes its run
ID, source SHA, image digest and canonical data ID through training, collection
and cache construction. Warm hosts receive the same validated `ff-train` wrapper
as new bootstrap installations. Mutable historical inputs are only available
through explicit `FF_DATA_RELEASE=legacy`, never after a failed release lookup.
Generic launch/benchmark publication requires either a plan or an identified
standalone run and follows exact output receipts. Download-only use can retrieve
current approved models without attributing them to a newly submitted run.

## Provider response boundary

Provider captures record provider/version, effective request, retrieval time,
content digest, row count and observed/empty/unavailable status. These files are
part of the sealed raw inventory. Replay never contacts a provider on a cache
miss; missing/corrupt responses fail the release boundary. Older canonical
releases without captures can replay their sealed derived caches with historical
network fallback forbidden. The archived evaluation reference is taken from the
same release, not refreshed while building historical serving results.

Installation verifies every object before replacing raw/split inputs, rejects
symlinked destination directories, and removes unrelated older cache files.
Hydration completes before consumers open either mounted directory. Live builders
use an isolated overlay and explicit source context for current observations;
they cannot modify or relabel the frozen historical release.

## Exact outputs and serving publication

Successful smoke validation claims a per-plan, per-position receipt under
`build-plans/<id>/artifacts/<position>.json`. The receipt records the exact object
key, SHA256, size, smoke approval, intent and provenance. The first accepted
output wins before model-pointer mutation; retries and competing attempts use
that canonical output. Failed smoke candidates never claim the successful slot.
Mutable promotion status belongs to the model pointer, not receipt identity.
Legacy publishing runs use exact source/position/run receipts under
`<prefix>/releases/v3/run-outputs/`. Artifact downloads and
benchmark aggregation in identified runs follow these receipts rather than
the mutable global model manifest. Failed smoke validation, missing receipts,
wrong provenance, or object corruption fail verification; another run's model
cannot substitute for the missing output.

The Batch workflow verifies receipts and publishes the serving cache before the
ECS rollout. Refresh and EC2 compatibility workflows also build the cache before
rolling the service; cache-build failure blocks rollout. Production request
handlers can therefore consume published generations rather than rebuilding
the model pipeline.
Legacy-run cache builds pass `FF_BUILD_POSITIONS` for the trained subset and the
same source/image/run identity; a newer global model cannot stand in for a
missing or superseded output from that run.

Initial infrastructure bootstrap verifies both approved model artifacts and a
complete published serving generation before creating or changing AWS resources.
The initialize-only seeder imports weights; it does not build a serving cache.
After seeding, run `FF_MODEL_S3_BUCKET=BUCKET python -m src.scripts.build_serving_cache`
before bootstrap. The predecessor cache archive and loose files are not a valid
substitute for the new generation.

## Retention and migration boundaries

Artifact transport lives in `src/artifacts`; the old `src.shared.model_sync` and
`src.shared.artifact_gc` imports alias the same module objects for compatibility.
Transport/control-plane changes run shared/serving tests as appropriate.
Numerical changes in `prediction`, `training`, `evaluation`, or shared/data code
retain six-position retraining and benchmark requirements; HTTP contracts do not
trigger numerical retraining. Benchmark fingerprints are version 2 because their
path domain now includes the numerical packages.

Automatic producer-side retention remains suspended. An explicit coordinated
collector is available for unowned model-history objects:

```bash
# Inventory only; default grace is 24 hours. Review the generated report.
python -m src.artifacts.gc --bucket BUCKET --position QB --output gc-plan.json
# Recomputes the eligible set under an exclusive manifest lock, then deletes.
python -m src.artifacts.gc --bucket BUCKET --position QB --execute --output gc-result.json
```

Execution acquires a CAS lock without expiration. Publishers and manual promotion
reject a locked manifest; publications captured before lock acquisition fail their
ETag precondition even after release because every manifest write has a unique
revision nonce. Eligible publishers must reread and revalidate before retrying.
The collector protects current/previous and approved pointers,
recent history, all retained receipts, and any artifact stamped with a build-plan
or publication-intent owner (including a delayed or interrupted receipt writer). Recent objects remain
within the grace period. Receipts, plans, datasets and plan-owned artifacts are
retained intentionally; expiring retained plans is a separate operator policy.

While holding the collection lock, it conditionally removes eligible deletion
keys from manifest history before deleting their objects. A failed deletion may
leave retryable unreferenced bytes; rollback listings retain only referenced
objects protected by that policy.

A normal exception releases only the collector's own unchanged lock. A process
crash or uncertain write can leave an auditable lock in the manifest. There is
no timeout-based lock stealing. After independently confirming the original
collector process has stopped, recover using its exact owner token:

```bash
python -m src.artifacts.gc --bucket BUCKET --position QB \
  --recover-lock OWNER_TOKEN --confirm-collector-stopped
```

Recovery during an active collector would violate exclusive ownership. The
command requires explicit confirmation and a matching token and conditionally
releases the lock; a conflicting manifest update is never overwritten.

Source ancestry rejects older-code publication, and the pre-training intent
sequence handles newer data/runs on the same code. Operator rollback has its own
epoch, separate from CAS/GC revisions; retries retain their original rollback
barrier. Receipts identify each run's exact output independently of later global
promotions. Manifests/history use `<prefix>/releases/v3/<POSITION>/`, so queued
legacy writers cannot overwrite or collect their objects. Initial migration
copies retained legacy bytes after validating their source provenance. The
serving snapshot pointer remains a separate complete-release boundary.
