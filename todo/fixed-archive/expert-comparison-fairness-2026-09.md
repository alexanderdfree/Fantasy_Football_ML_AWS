### [FIXED] Model-vs-expert comparison graded a stale duplicate expert, used an expert-selected headline cohort and declared winners from noise

- **File(s):**
  - `src/shared/{comparison_scoring,comparison_uncertainty,evaluation_cohorts}.py`
  - `src/data/expert_sources.py`
  - `src/prediction/{historical,comparison,comparison_snapshot}.py`
  - `src/scripts/build_evaluation_reference.py`
  - `src/serving/timeline.py`
  - `src/serving/frontend/src/views/{Comparison,Timeline}.jsx`
  - `ios/Sources/{Models/Comparison,Views/Comparison/ComparisonView}.swift`

  PR #1595 (audits of 2026-09-18 and 2026-09-25); follow-up PR pending (served-model
  verdicts, no verdict on hindsight cohorts, information-set disclosure; see the
  end of this record).
- **What:** ADR-0024 made truth and components symmetric, but the served comparison
  was still unfair in five ways.
  1. **Placeholder zeros were graded.** The hvpkod NFL.com archive lists every
     rostered player, and 168 all-zero placeholder rows were graded as 0.0
     forecasts. NFL.com QB MAE was 6.516 with them and 6.199 without.
  2. **"NFL.com" is not an independent expert.**
     - Its components reproduce RotoWire's series: 94% of passing-yard forecasts
       match within 0.01.
     - `hvpkod/NFL-Data` commit times show weeks 2–6 and 10–14 were captured
       Tuesday–Thursday, before the final injury report. Those weeks still project
       85 players ruled Out.
     - Its whole deficit against RotoWire (+0.052 MAE) is those stale weeks.
     - The Timeline graded "NFL.com + RotoWire", which is one provider twice, and
       left out ESPN.
  3. **The expert reference selected the headline cohort.** `weekly_reference_top24`
     was chosen by the graded experts' own forecasts, so their winner's curse
     inflated the models' RB/WR lead to about two to three times its size on
     forecast-free cohorts.
  4. **Winners were highlighted from noise.** The best of seven cells was
     highlighted at 1e-9 tolerance, with four model draws against the experts. No
     headline gap's 95% paired interval excluded zero.
  5. **The default metric masked a flip.** MAE rewards median-like forecasts on
     right-skewed points, and winners flipped under RMSE (TE, RB).
- **Fix:**
  - All-zero provider rows are missing forecasts (`score_forecast_components`).
  - NFL.com offense joins `EXCLUDED_SOURCES` and leaves the reference recipe
    (`shared_components_v4` is RotoWire-only).
  - The headline cohorts are `weekly_depth_starters` (pregame depth chart) and
    `elite_top24` (prior-season importance). Neither is selected by any forecast.
  - Every cell carries paired, player-clustered bootstrap intervals, with group
    minima taken inside each replicate. A row names a winner only when MAE and
    RMSE agree and exclude zero. Cells also carry signed bias.
  - The Timeline grades RotoWire+ESPN with season edge intervals.
  - 2025 is labelled a development-season backtest.
  - Cache schema 12.
- **Lesson:**
  - Identical truth is not an identical population or an independent source.
    Check provider provenance, meaning capture time (git history or fetch
    timestamps) and near-duplication with other providers, before counting it as
    an expert.
  - Select cohorts with something no graded source produced.
  - Never declare a winner without a paired interval that accounts for picking
    the best of several models.
  - Replicate the served numbers pandas-only from the current cache generation
    (`current.json`) before and after any comparison change.
- **Follow-up (2026-09-25, PR pending):** the row verdict still graded the best of
  four models against the best expert. The bootstrap took the minimum inside each
  draw, so the interval was honest, but the framing gave the model family four
  draws and hid that the served model (Attention NN at QB/TE/DST, LightGBM at
  RB/WR, Ridge at K; the Next Week board's ranking chain) loses K on both metrics
  and trails on RMSE at RB/WR/TE in 2025. The first out-of-sample record (2026
  weeks 1–2, last archived `upcoming_week.json` version before each kickoff,
  models and experts from the same fetch) has the served Attention NN behind
  RotoWire by +0.26 MAE [+0.10, +0.43] pooled and +0.52 at WR. Season-leader and
  expert-reference cohorts also carried verdict lines although their rows are
  selected on outcomes or on a graded expert's own forecasts. Now: the verdict
  grades `SERVED_MODEL` (shared with the Next Week board through the API
  contract), best-of-four is a context line, hindsight cohorts carry
  `not_applicable`, and both surfaces disclose the backtest information set.
  Lesson: a selection-aware interval does not fix a selection-biased headline;
  name the pre-specified model, and never attach a verdict to a hindsight cohort.
