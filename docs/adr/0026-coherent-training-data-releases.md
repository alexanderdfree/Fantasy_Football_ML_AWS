# ADR-0026: Coherent training-data releases

**Status:** Accepted

## Context

Refreshing only the three split files left Batch and serving with independently
aged raw dependencies. In the September 2026 audit, a current-code rebuild and
production preparation differed on thousands of rows because red-zone caches,
team statistics, identity bridges, and splits came from different generations.
Column presence did not detect changed source values or changed aggregation
semantics. Kickers also fetched historical 2025 PBP on every load.

## Decision

Publish raw dependencies and train/validation/test splits as one immutable,
content-addressed data release. The producer completes the full data build,
prewarms derived K/DST inputs and the shared evaluation reference, and seals its
file hashes, source-code hashes, runtime versions, and source-season coverage.
Uploading is a separate step and rejects missing or changed seals. Every file is
uploaded and verified before the single current-release pointer advances.
The producer verifies that historical loader dependencies can be replayed;
an optional-source fetch failure cannot silently create an incomplete release.

Consumers resolve a release once, verify its manifest and files, finish local
hydration, and only then prepare models. Batch split branches, EC2, tuning, and
A/B fleets receive one pinned release ID. Model and branch metrics retain it;
combining branches from different releases fails. Local hydration installs files
after all downloads pass validation; it is not an atomic transaction across two
mounted directories, so consumers must bootstrap before opening data files.

An aggregate training-history run also binds its data release, alongside the
source revision and seed. Results with another or unknown release cannot enter a
pinned run. Workflow retry attempts use separate split staging identifiers so a
late branch from an older attempt cannot overwrite the current attempt's input.

Historical source caches in a pinned release are read-only inputs. Missing or
incompatible caches raise `DataReleaseError`; network fallback must not convert
them to a mixed generation or zero-filled optional feature. Semantic cache
versions invalidate old weekly, depth-rank, red-zone, and opportunity data.
Historical kicker backfill PBP and player identity/name metadata are cached with
the release. `FF_DATA_RELEASE=legacy` is an explicit migration/rollback mode,
not an automatic fallback when a manifest is missing.

The live builder needs current-season data in addition to the frozen training
history. Its CLI starts a fresh child with a copied mutable raw-data overlay and
an explicit original-history directory. This initializes existing cache-path
imports without process-global patching and retains the normal model/split and
artifact paths. Historical subsets are filtered from the full sealed archive;
only uncovered seasons can fetch into a separate, explicitly allowed live
cache. Schedule/team rollups modify the overlay, never the original release.

Deployment and training gates verify that a published release matches the
data-producing source files in the image revision. Initial migration therefore
waits for the first successful publication before replacing the running service
or submitting new training jobs. A failed rebuild or timeout fails the gate;
absence of a pending marker is not evidence that data is ready. Documentation or
unrelated changes can reuse a matching release.

The execution and serving integration in [ADR-0027](0027-versioned-prediction-and-execution-contracts.md)
uses this release ID as the only new dataset authority. Build plans, standalone
run receipts, model metadata and cache-schema-10 snapshots preserve it. Serving
data advancement requires a fully verified matching snapshot and the readiness
transaction; it cannot independently replace raw data beneath a running model.
Captured provider response files are included in the sealed inventory. Retained
releases without captures can only replay their existing historical derived
caches with network fallback forbidden.

## Completeness and identity

An absent upstream season is recorded in coverage metadata. Both upstream 2012
snap-count formats were empty during the 2026-09-10 audit; those counts cannot
be reconstructed by assuming zero participation. The configured 2012 context
year remains available for other statistics, with unknown snap coverage.

Invalid player IDs are excluded before roster merges and feature engineering.
Participation first uses the fantasy ID crosswalk, then unambiguous roster
PFR/ESPN identities and exact normalized team-season names. NFL-documented name
variants provide a final conservative bridge; ambiguous names stay unresolved.
The one additional Nathan/Nate Carter alias is tied to his GSIS identity and
roster context, with the official Falcons source recorded in the helper.
Identity sources are part of the release rather than a per-run network lookup.

## Consequences

- Changing data definitions requires rebuilding a release and rebaselining
  affected positions; equal schemas do not establish equal inputs.
- A model result can be traced to exact data files and producer code.
- A failed publication leaves the previous current release available.
- The old unversioned objects remain available for explicit rollback, but new
  default readers require a compatible release.
- Publication and model deployment remain separate authorized operations.

## References

- [Data release producer/consumer](../../src/data/release.py)
- [Rollout gate](../../src/scripts/wait_data_release.py)
- [Source identities](../../src/data/identity.py)
- [Live builder overlay](../../src/serving/live_build.py)
- [Training-data audit incident](../../todo/fixed-archive/training-data-audit-remediation-2026-09.md)

## Changelog

- **2026-09-10** — Add a staged-publication option for scheduled maintenance: seal and verify immutable candidates before a separate compatible cache/data activation (ADR-0028). Default CI publication behavior is retained.

- **2026-09-10** — Seal, verify, publish, pin, and gate coherent raw/split releases;
  isolate live-data overlays and record source coverage. (PR #1564)

- **2026-09-10** — Bind typed execution plans, standalone receipts and serving
  snapshot rollback to this canonical release authority. (PR #1566)
