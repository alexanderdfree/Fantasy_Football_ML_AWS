> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Expert comparison mixed scoring labels, coverage, and postseason-selected leaders
- **File(s):** `src/serving/comparison.py`, `src/serving/routes.py`, `src/serving/frontend/src/views/Comparison.jsx`, `src/analysis/analysis_nflcom_baseline.py`; ADR-0024.
- **What:** Models were graded against full fantasy actuals while committed expert metrics used only modeled targets. WR rushing and QB receiving counted against only one side. Static top-12/top-30 IDs were ranked with postseason totals, and per-source coverage differed.
- **Fix:** Compute all columns from cached predictions on full regular-season actuals and a shared player-week intersection. Rebuild seasonal membership from that truth, select a new primary weekly top-24 cohort from a separate archived pregame reference, and distinguish actual-week leader capture from accuracy/bias. Retire the static accuracy snapshot as the live authority and label historical research tables accordingly.
- **Lesson:** Same player IDs do not establish comparable evaluation: truth, season type, row coverage, and cohort-selection time must all agree. Actual-week winners' negative bias is not a training target.
