# ADR-0024 — Shared comparison truth and pregame evaluation cohorts

**Status:** Accepted (2026-09-10)

## Context

The Comparison tab combined live model metrics on full fantasy actuals with
static expert metrics on position-specific target sums. WR/TE rushing and QB
receiving therefore counted against only one side. The static season-leader
selector also included postseason results, and the columns covered different
player-weeks. Separately, Batch artifacts omitted the local benchmark's cohort
reporting, leaving the `elite_top24` follow-up gate (#1354/#1537) without evidence.

## Decision

All displayed comparison sources are evaluated from the same cached prediction
table, against regular-season actuals scored on the **same projected components**
as the forecasts, on a common player-week intersection. The owner clarified this
contract on 2026-09-10: compare only quantities both the models and experts
calculate. This replaces the initial full-fantasy actual basis without restoring
the old asymmetry between model and expert labels. Components are declared in
`src/shared/comparison_scoring.py`: QB passing/rushing/turnovers; RB
rushing/receiving/lost fumbles; WR/TE receiving/lost fumbles; K made-yardage, PATs
and misses; DST the common defensive stats and PA/YA tiers. Missing actual
components make a row unavailable; they never fall back to full fantasy points.
Source metadata may come from the historical JSON; its accuracy
cells and stored player IDs are never a fallback for a missing live table.
The UI reports common sample sizes and missing data. Quartile bias uses the same
common sample. Ranking metrics evaluate each source's own selections on that
shared slate.

Four cohort definitions remain separate:

| Name | Selection | Purpose |
|---|---|---|
| `weekly_reference_top24` | Top 24 per position/week by a fixed archived expert reference | Primary expected-starter accuracy and bias |
| `elite_top24` | Top 24 distinct players by prior-season mean shared-component points | Historical continuity; pre-season importance |
| `seasonal_actual_top24` | Top 24 by current-season regular-season shared-component actual total | Retrospective season-leader accuracy |
| `weekly_actual_top24` | Actual weekly top 24 versus each source's predicted selection | Hit rate, points captured, and lineup regret |

The Comparison tab also retains explicitly retrospective season-leader top-12
and top-30 tables. Cohorts are selected before forecast coverage is applied;
missing rows never promote the next-ranked player. Ties use player ID, and
seasons are ranked independently. Actual-week winners' negative bias is not a
calibration target: selection on realized outcomes creates that pattern.

## Timeline records

The Timeline applies the same component truth to its `all` regular-season cohort.
It separates offense (QB/RB/WR/TE, NFL.com and RotoWire), K (ESPN), and DST
(RotoWire and ESPN). All four models and the group's required experts share one
finite player-week intersection. The source set is fixed, including when a whole
source or week is missing; an unavailable source never relaxes the comparison.
The selected season is explicit. Missing position-weeks remain unavailable entries
when that week exists elsewhere in the season's cached slate.

Every model retains its own weekly errors, edges, and season record. A model's
edge is the minimum of expert MAE minus its own MAE on the common rows. A positive
edge requires beating every expert in the group. Unevaluable weeks are excluded
from the win denominator, and ties are not wins. These decisions use unrounded
errors; only display formatting rounds them. Season MAE pools player-week errors
rather than averaging differently sized weekly means. No weekly winner or
season-selected champion supplies an aggregate performance claim.

The API reports the actual basis, components, source set, position scope, common
sample size, pre-intersection coverage, and unavailable/excluded-source reasons.
The web UI renders all four fixed model series, leaves gaps for unavailable weeks,
and keys requests by scoring, group, and season. This is retrospective evaluation
of the current cached forecasts, separate from the dated release changelog.
Timeline schema v2 replaces the old winner/edge summary with per-model records;
its web consumer and committed bundle ship together.

Expert comparison totals must continue to use the declared component contract.
Preserving additional raw expert stats or full-fantasy forecast totals for another
view does not authorize grading those totals against restricted comparison truth.

## Reference artifact

`data/raw/weekly_evaluation_reference_v1.parquet` contains only player/week keys,
position, pregame reference score/rank, source recipe, and generation metadata.
The versioned recipe is the mean of archived NFL.com and RotoWire forecasts for
QB/RB/WR/TE, ESPN for K, and RotoWire for DST. The current recipe is
`shared_components_v2`; old recipe rows are preserved in the versioned parquet.
NFL.com K is excluded from matched comparisons because its native bucket total
cannot represent our made-yardage and miss targets. ESPN supplies those targets. Both required offense sources
must exist for a candidate; it never becomes a mean of whichever happens to be
available. This is a two-provider reference, not a claim of industry consensus.

Ranks are computed from the full published forecast pool, independently of
actual outcomes and model forecasts. NFL.com offense before 2024 is excluded
because the hvpkod archive backfilled actuals; RotoWire before 2018 is excluded.
Unavailable seasons/weeks are explicit and never replaced with model rankings.

`python -m src.scripts.build_evaluation_reference --seasons 2025 --upload`
builds/publishes the artifact from the existing source loaders. The existing
off-container serving-cache builder refreshes it too. The normal S3 raw-data
hydration already delivers this parquet to Batch and serving. No requests or
training occur in the metric helper; the artifact is never a model feature.
Reference publication failure retains the old archive and does not discard a
valid serving prediction cache.
Refreshes are restricted to the exact requested, provider-supported seasons;
replacements that lose archived player-week coverage are rejected before writing.
Other seasons and recipe versions remain intact.

## Serialization and validation

`src/shared/evaluation_cohorts.py` computes compact JSON-safe summaries while
held-out rows still exist, including normal, split-branch, and CV pipeline
returns. Local/parallel/rolling-origin, Batch, and EC2 summaries preserve them.
Every origin retains its own cohort report. A missing row set or prior/reference/component
data produces an explicit unavailable entry with `n: null`, not an omitted key
or a zero score. Valid empty cohorts are distinct.

K/DST prior importance is computed from their position-native post-target
training/validation totals; the offensive-only generic split is not a valid
substitute. Reports include per-model sample counts, MAE, RMSE, signed bias,
cohort identity, reference identity, `actual_basis`, and the component list.
Prior-season importance is recomputed from raw components; a precomputed full-score
prior mean is not a substitute for missing component history. Split merges combine disjoint model
blocks and reject differing cohorts/truth/reference vintages.

Regression tests cover equal forecasts receiving equal scores, invariance to
unprojected actual stats, missing components, exclusion of NFL.com K, ESPN K
reference selection, cross-position scoring components, missing forecast weeks, postseason exclusion, reference
selection independent of actuals/model forecasts, no rank-25 promotion when an
actual is missing, and serialized Batch/rolling-origin output.

## Rejected alternatives

- Updating only static expert numbers: leaves coverage and future data drift.
- Combining full-fantasy model actuals with restricted expert actuals: different
  labels cannot be compared. Full-fantasy scoring remains a distinct benchmark;
  this comparison intentionally measures shared projected components.
- Each model's own top-24 pool for MAE comparison: changes the grading population.
- Actual weekly winners as the primary bias target: confuses hindsight selection
  with forecast miscalibration.
- Live expert API calls during training/reporting or web requests: adds latency
  and makes comparison membership vary with network availability.
- Rewriting historical benchmark rows using freshly trained predictions: does
  not describe those historical runs.

## Consequences

Comparison values intentionally change because truth, sample, and retrospective
cohorts are corrected. Existing model weights and predictions do not change.
Historical static tables remain dated research snapshots and are not comparable
to the corrected primary metric without rerunning their evaluation.

## Changelog

- 2026-09-10 — Apply matched component scoring and compatible position groups to
  Timeline; replace hindsight-selected winners with per-model records (PR #1573).

- 2026-09-10 — Establish matched full-score comparison and versioned pregame
  top-24 reporting across all benchmark paths (PR pending).

- 2026-09-10 — Owner clarification: score only common projected components on
  both sides; version reference as shared_components_v2 with ESPN K, retain
  paired coverage and explicit scoring metadata (PR pending).
