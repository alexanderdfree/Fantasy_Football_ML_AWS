### [FIXED] Consolidated offline comparison and provider-cache boundaries

**File(s):** `src/analysis/analysis_expert_comparison.py`,
`src/analysis/topn_expert_gap.py`, `src/shared/comparison_scoring.py`,
`src/shared/evaluation_cohorts.py`, `src/data/nflcom_loader.py`,
`src/serving/expert_sources.py`, `src/serving/core.py` (PR #1574 follow-up;
reviewed starting revision `f4c6edc84df9`).

**What:** Explicitly unavailable expert components reached numeric metrics;
partial source fetches could become reusable caches; adjacent audit repairs
needed reconciliation with shared-component scoring and separate display totals.

**Fix:** Use one finite paired population, report unavailable comparisons,
retain certified pre-fill truth when supplied, rescore fresh raw forecasts in
the requested format, preserve valid nonpositive predictions and zero-hit F1,
and prevent failed/partial provider data from becoming a complete prediction
generation. Raw provider caches retain exact season identity and completion.

**Validation:** The affected evaluation/provider checks passed 260 tests and
the full unit suite passed 4,460 tests (2 skipped) before restacking. Regression
fixtures include missing components, finite intersections and partial sources.

**Lesson:** A projector's unavailable value is an interface contract. Its metric
consumer and cache publisher must preserve that meaning through the whole path.
