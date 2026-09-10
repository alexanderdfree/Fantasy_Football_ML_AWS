> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] No API error handling
- **Files:** `app.py:85-90`
- **What:** All API routes lacked try/except. If `_get_data()` or model loading failed, the user saw a generic 500 with no useful message.
- **Fix:** Added Flask `@app.errorhandler(Exception)` that returns JSON `{"error": ...}` for `/api/` routes. Logs full traceback to console.
