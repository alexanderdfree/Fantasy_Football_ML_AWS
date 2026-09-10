> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] No feature clipping after StandardScaler
- **Files:** `src/shared/pipeline.py:311-313`, `app.py:320`
- **What:** Test features could produce z-scores up to 19.5 — far outside the training distribution. NN predictions were unpredictable for these inputs.
- **Fix:** Added `np.clip(..., -4, 4)` after all `StandardScaler.transform()` calls.
- **Lesson:** Always clip scaled features. StandardScaler assumes train/test distributions are similar, but outliers in test data can produce extreme z-scores. Clip at +/-4 (catches 0.3-0.4% of values, prevents catastrophic extrapolation).
