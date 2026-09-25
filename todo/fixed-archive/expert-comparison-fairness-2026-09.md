### [FIXED] Expert placeholders graded as zeros, expert-selected headline cohort, and split surface populations

- **File(s):** `src/shared/comparison_scoring.py`, `src/data/expert_sources.py`,
  `src/prediction/{historical,comparison,comparison_snapshot}.py`,
  `src/shared/evaluation_cohorts.py`, `src/serving/timeline.py`,
  `src/artifacts/serving_snapshot.py`, `src/serving/frontend/src/views/{Comparison,Timeline}.jsx`
  (PR pending; audit branch `claude/expert-comparison-fairness-6eab0c`, 2026-09-18).
- **What:** After ADR-0024 the truth and components were symmetric, but three population
  defects remained. (1) The hvpkod NFL.com archive lists every rostered player, so 64–77% of
  its 2025 offense rows are all-zero placeholders; 168 of them reached the served common slate
  as confident 0.0 forecasts while ESPN dropped its zero rows at ingestion. Unprojected backups
  who started (a 0.0 against a 25.7-point game) cost NFL.com 0.32 QB MAE, and deep-bench zeros
  handed it small free wins elsewhere. (2) The headline `weekly_reference_top24` cohort was
  selected by the graded experts' own forecasts (NFL.com/RotoWire mean; ESPN alone for K), so
  those sources' errors were conditioned on their own selection: own-selection bias measured
  +0.40 for NFL.com against a neutral selector. (3) The Comparison tab intersected all seven
  sources while the Timeline offense group omitted ESPN, so the same model had two MAEs on two
  tabs (RB attention 4.74 vs 4.35).
- **Fix:** Score provider rows through `score_forecast_components`: a row whose shared components
  are all zero is a missing forecast for every source and surface (reference recipe
  `shared_components_v4`, cache schema 12). Add `weekly_consensus_top24`, selected by the
  equal-weight mean of every displayed source on the common slate, as the headline cohort; keep
  the archived reference as a labelled secondary view. Give the Timeline offense group the tab's
  displayed sources (ESPN included) and pin the parity with a test.
- **Lesson:** Identical truth is not identical population. Provider archives encode "not
  projected" differently, a selector drawn from graded sources conditions their errors, and two
  surfaces that intersect different source sets are two different benchmarks. Replicate the served
  numbers pandas-only from the current cache generation (`current.json`, not the top-level
  tarball) before and after any comparison change.
