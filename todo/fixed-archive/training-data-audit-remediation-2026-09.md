### [FIXED] Training labels, role semantics, identity joins and raw/split generations diverged

- **File(s):** `src/data/{loader,identity,dst_scoring,release}.py`,
  `src/features/{engineer,roster_availability}.py`, `src/k/{config,data}.py`,
  Batch/EC2/serving/tuning consumers and rollout workflows; ADR-0026.
- **What:** The 2026-09-10 production audit found 292 omitted D/ST touchdowns,
  162 omitted punt blocks and one spurious cached touchdown across 441 of
  6,814 modeled games; ESPN receiver-slot ordinals changed the depth feature's
  meaning in 2025; availability used the realized participant pool; 298
  skill-player snap records lacked an ID bridge; cached versus current
  source generations produced different model inputs. K was filtered at four
  games and again at six. The upstream 2012 snap files contained no rows.
- **Fix:** Reconcile D/ST events against complete PBP while preserving the
  owner's points-allowed scoring contract. Normalize multi-slot ranks, use
  source-era-correct pregame roster eligibility in both paths, resolve all
  audited skill snap identities conservatively, and discard invalid IDs before
  joins. Align K's filters and cache its historical backfill. Publish verified
  raw/split releases, pin fleets and record provenance, gate incompatible
  rollouts, and build live overlays without mutating sealed history. Record
  unavailable 2012 snap coverage instead of fabricating observations.
- **Review corrections:** Apply the same slot-relative receiver depths to
  archived and live ESPN payloads, including per-athlete slot identifiers.
  Reject incomplete historical loader caches before publication, and include
  D/ST scoring and kicker backfill inputs in serving-cache invalidation.
- **Evidence:** Full-source D/ST reconciliation corrected exactly 441 games
  (+2,070 aggregate points; maximum 12 points/game), retaining other raw
  targets including points_allowed and yards_allowed. Identity replay left
  zero unmatched positive-snap QB/RB/WR/TE records in the audited source.
  Availability replay preserved 88,517 earlier rows under future-stat
  perturbation and reproduced Dak Prescott's live/training flag consistently.
  Focused source, event, overlay, release and rollout regressions accompany the
  changes. Pipeline comparison evidence belongs to the PR/benchmark records;
  these corrections do not imply metric neutrality.
- **Lesson:** Finite matrices and matching schemas do not prove semantic
  correctness or completeness. Reconcile events/populations, preserve
  pregame input definitions, and identify the exact data generation.
