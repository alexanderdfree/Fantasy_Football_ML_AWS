> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] K attention scaler metadata was missing the full `attn_static_features` list
- **Files:** `src/k/run_pipeline.py` and scaler save path (PR #145, `e01507b`).
- **What:** K's attention NN scaler was being saved with a truncated `feature_cols` meta — only a subset of the actual `attn_static_features` it had been fit on. At inference, `assert_scaler_matches` (the canonical training/inference skew check) compared the truncated meta against the runtime feature list and either filtered columns or raised, depending on the runtime path. Either way the served K model was operating with a different feature set than the trained one.
- **Fix:** Write the full `attn_static_features` list into the scaler meta so `assert_scaler_matches` sees the same set the scaler was actually fit on.
- **Lesson:** Scaler metadata must always be written with the full feature list it was fit on. After a rename or schema change, rebuild the artifact rather than letting a stale meta file silently filter columns at inference. This is exactly the failure mode D11's smoke test now catches (`assert_scaler_matches` check inside `src/shared/smoke_test.py`).
