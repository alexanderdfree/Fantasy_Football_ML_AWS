> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] `run_cv_pipeline` missing `non_negative_targets` on MultiHeadNet
- **File:** `src/shared/pipeline.py:804`
- **What:** `_train_nn` and `_train_attention_nn` both pass `non_negative_targets` to `MultiHeadNet`, but `run_cv_pipeline` constructed its own `MultiHeadNet` without it. DST's `pts_allowed_bonus` (range [-4, +10]) was incorrectly clamped to >= 0 during CV.
- **Fix:** Added `non_negative_targets=cfg.get("nn_non_negative_targets")` to the CV pipeline's `MultiHeadNet` call.
- **Lesson:** When the same model is constructed in multiple code paths, all paths must pass the same kwargs. `_train_nn` is the reference — any manual `MultiHeadNet(...)` call elsewhere must mirror it.
