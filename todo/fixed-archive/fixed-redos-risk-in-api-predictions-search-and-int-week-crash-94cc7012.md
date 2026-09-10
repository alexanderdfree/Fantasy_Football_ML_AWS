> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] ReDoS risk in `/api/predictions` search and `int(week)` crash
- **File:** `app.py:496-498`
- **What:** User search input was passed directly to `str.contains()` as a regex pattern (ReDoS risk). Also, `int(week)` could crash with `ValueError` on invalid input.
- **Fix:** Added `regex=False` to `str.contains()` and wrapped `int(week)` in try/except with 400 response.
