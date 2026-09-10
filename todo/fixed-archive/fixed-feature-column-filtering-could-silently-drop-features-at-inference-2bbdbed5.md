> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Feature column filtering could silently drop features at inference
- **File:** `app.py:308-311`
- **What:** `feature_cols = [c for c in feature_cols if c in pos_train.columns]` filters to available columns. If a feature from training is missing, the model gets fewer features than expected → dimension mismatch → crash. The crash would happen, but the error message wouldn't identify which feature is missing.
- **Fix:** Added count comparison and warning log when columns are dropped.
