> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Predictions are always PPR but API serves multiple scoring formats
- **File:** `app.py:505-561`
- **What:** `/api/predictions` accepts a `scoring` param (standard, half_ppr, ppr). It selects the correct *actual* column, but `ridge_pred` and `nn_pred` are always trained on PPR targets. When a user selects "standard" scoring, actuals change but predictions don't — the comparison is apples-to-oranges.
- **Fix:** Added `scoring_note` field to API response when scoring != ppr. UI already displays a "PPR Scoring" badge and has no scoring selector, so users aren't misled. Training separate models per format is out of scope.
- **Impact:** API consumers now get a clear warning.
