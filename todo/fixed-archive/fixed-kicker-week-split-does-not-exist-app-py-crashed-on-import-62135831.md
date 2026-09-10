> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] `kicker_week_split` does not exist — app.py crashed on import
- **File:** `app.py:62, 262`
- **What:** Imported `kicker_week_split` from `K.k_data`, but the function was renamed to `kicker_season_split`. App crashed immediately with `ImportError`.
- **Fix:** Changed import and call site to `kicker_season_split`.
- **Lesson:** When renaming functions, grep for all call sites across the project — not just the file where the function is defined.
