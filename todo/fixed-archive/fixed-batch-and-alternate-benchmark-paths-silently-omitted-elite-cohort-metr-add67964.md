> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Batch and alternate benchmark paths silently omitted elite cohort metrics (#1537)
- **File(s):** `src/shared/evaluation_cohorts.py`, `src/shared/pipeline.py`, `src/shared/benchmark_utils.py`, `src/batch/train.py`, `src/benchmarking/benchmark.py`, `src/scripts/build_evaluation_reference.py`; ADR-0024.
- **What:** The local benchmark appended cohort metrics after the common summary step. Batch serialization and split merges lost them, and rolling-origin output omitted them, so the elite-bias follow-up gate (#1354) lacked production evidence.
- **Fix:** Calculate cohorts while held-out rows exist and preserve them in every benchmark summary and each rolling origin. Split merges require matching cohort/truth/reference identities. Preserve `elite_top24` as prior-season importance, add `weekly_reference_top24` as pregame importance, and report unavailable data explicitly rather than silently dropping the field or using zero metrics. The reference artifact contains forecasts/ranks only and is hydrated by existing raw-data sync.
- **Lesson:** Assert the final serialized artifact, not just the helper. Retrospective leader cohorts, pregame expectation cohorts, and ranking metrics answer different questions and need distinct names.
