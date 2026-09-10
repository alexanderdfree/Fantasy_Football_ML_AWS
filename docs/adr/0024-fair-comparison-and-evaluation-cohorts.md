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
table, against full regular-season PPR actuals, on a common player-week
intersection. Source metadata may come from the historical JSON; its accuracy
cells and stored player IDs are never a fallback for a missing live table.
The UI reports common sample sizes and missing data. Quartile bias uses the same
common sample. Ranking metrics evaluate each source's own selections on that
shared slate.

Four cohort definitions remain separate:

| Name | Selection | Purpose |
|---|---|---|
| `weekly_reference_top24` | Top 24 per position/week by a fixed archived expert reference | Primary expected-starter accuracy and bias |
| `elite_top24` | Top 24 distinct players by prior-season mean full fantasy points | Historical continuity; pre-season importance |
| `seasonal_actual_top24` | Top 24 by current-season regular-season actual total | Retrospective season-leader accuracy |
| `weekly_actual_top24` | Actual weekly top 24 versus each source's predicted selection | Hit rate, points captured, and lineup regret |

The Comparison tab also retains explicitly retrospective season-leader top-12
and top-30 tables. Cohorts are selected before forecast coverage is applied;
missing rows never promote the next-ranked player. Ties use player ID, and
seasons are ranked independently. Actual-week winners' negative bias is not a
calibration target: selection on realized outcomes creates that pattern.

## Reference artifact

`data/raw/weekly_evaluation_reference_v1.parquet` contains only player/week keys,
position, pregame reference score/rank, source recipe, and generation metadata.
The versioned recipe is the mean of archived NFL.com and RotoWire forecasts for
QB/RB/WR/TE, NFL.com for K, and RotoWire for DST. Both required offense sources
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

## Serialization and validation

`src/shared/evaluation_cohorts.py` computes compact JSON-safe summaries while
held-out rows still exist, including normal, split-branch, and CV pipeline
returns. Local/parallel/rolling-origin, Batch, and EC2 summaries preserve them.
Every origin retains its own cohort report. A missing row set or prior/reference
data produces an explicit unavailable entry with `n: null`, not an omitted key
or a zero score. Valid empty cohorts are distinct.

K/DST prior importance is computed from their position-native post-target
training/validation totals; the offensive-only generic split is not a valid
substitute. Reports include per-model sample counts, MAE, RMSE, signed bias,
cohort identity, and reference identity. Split merges combine disjoint model
blocks and reject differing cohorts/truth/reference vintages.

Regression tests cover equal forecasts receiving equal scores, cross-position
scoring components, missing forecast weeks, postseason exclusion, reference
selection independent of actuals/model forecasts, no rank-25 promotion when an
actual is missing, and serialized Batch/rolling-origin output.

## Rejected alternatives

- Updating only static expert numbers: leaves coverage and future data drift.
- Scoring everyone only on modeled targets: excludes genuine fantasy points.
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

- 2026-09-10 — Establish matched full-score comparison and versioned pregame
  top-24 reporting across all benchmark paths (PR pending).
