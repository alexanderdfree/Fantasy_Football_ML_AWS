> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] `fill_nans_safe` dead code (and its self-pinning `TestFillNansSafe`) removed
- **File(s):** [src/features/engineer.py](../../src/features/engineer.py) + `tests/test_feature_leakage.py` (both removed in PR [#323](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/323), commit `f0a8421`).
- **What:** `fill_nans_safe` had zero callers in `src/` (production uses `src/shared/feature_build.py::fill_nans_with_train_means`); its only exerciser was `tests/test_feature_leakage.py::TestFillNansSafe`, a test that existed solely to pin the dead-code contract.
- **Fix:** PR #323 (Tier A docs + dead-symbol cleanup) deleted the function and the entire `tests/test_feature_leakage.py` atomically; `fill_nans_with_train_means` remains the production helper.
- **Lesson:** A test that exists only to pin a dead symbol should be deleted with the symbol, not kept as false coverage.
