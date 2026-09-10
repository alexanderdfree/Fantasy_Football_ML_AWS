> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Weather/Vegas features missing at inference in `src/serving/app.py`
- **File:** `app.py:310-311`
- **What:** Training pipeline (`_prepare_position_data`) merged schedule features, but `src/serving/app.py`'s inference path (`_apply_position_models`) did not. Models trained with 12 weather/Vegas features received zeros at serving time.
- **Fix:** Added `merge_schedule_features(_df)` calls in `_apply_position_models` before feature computation.
- **Lesson:** Any feature engineering done in the training pipeline must also be done in the inference/serving path. Diff the two code paths when adding new features.
