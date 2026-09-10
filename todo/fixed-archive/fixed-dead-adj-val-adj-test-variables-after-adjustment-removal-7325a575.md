> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Dead `adj_val`/`adj_test` variables after adjustment removal
- **File:** `src/shared/pipeline.py:778, 921-922`
- **What:** After removing `+ adj_val.values` from CV and holdout totals (Fix 6), the `adj_val` and `adj_test` variables were still computed but never used.
- **Fix:** Deleted the dead lines.
