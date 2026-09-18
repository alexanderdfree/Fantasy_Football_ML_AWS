### [FIXED] Reporting discarded optional model-selection provenance

**Files:** `src/batch/train.py`, `src/shared/benchmark_utils.py`,
`src/tuning/aggregate_results.py`. Extracted from the reporting hunks of
PR #1568 at `1a15bdc1281c5dfe310dbfd6daabe8adb589bbe2`.

**What:** Optional selection metadata present in a pipeline result or NN
history was omitted from Batch metrics, split-job merges, and history summaries.
The tuning summary also labeled every recorded objective as combined loss.

**Fix:** Carry existing `*_selection` or `checkpoint_selection` data through
reporting and display the recorded objective name/value. Legacy results without
selection metadata retain their existing metrics and do not acquire a policy.
These reporting functions do not create checkpoints, select models or trials,
change tuning namespaces, or update production configuration. The corresponding
selection changes in #1568 remain separately subject to the dual-metric gate.

**Lesson:** Preserve provenance as optional reporting data without making its
availability a reason to activate a model policy. No-fit tests cover legacy
results, all six position summaries, split merging, and duplicate-key rejection.
