> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Pipeline evaluation added adjustment to predictions but not to the total target
- **Files:** `src/shared/pipeline.py:305, 349` (and equivalent in `run_cv_pipeline`)
- **What:** After fixing the total aux loss target to `sum(targets)`, evaluation still added `adj_test.values` to Ridge and NN total predictions. This compared `sum(preds) + adj` against `sum(targets)` — the adjustment inflated the evaluation error.
- **Fix:** Removed `+ adj_test.values` from evaluation totals. Adjustment is only applied at inference in `src/serving/app.py`.
- **Lesson:** When changing a training target, trace all downstream consumers — evaluation metrics, ensemble computation, and plotting all need to stay consistent. This was a cascading side-effect of Fix 1 that we caught.
