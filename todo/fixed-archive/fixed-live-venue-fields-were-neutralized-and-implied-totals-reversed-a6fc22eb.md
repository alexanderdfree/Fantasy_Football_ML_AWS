> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Live venue fields were neutralized and implied totals reversed
- **File(s):** `src/serving/live_schedule.py`, `src/serving/espn_live.py`, `src/shared/weather_features.py`, `src/k/data.py`, `tests/test_live_schedule.py`, `tests/shared/test_weather_features.py` (issues #1519/#1529; PR pending).
- **What:** ESPN-only schedule rows dropped venue/rest/weather fields, making all live inputs outdoor, 65F, windless, and seven days rested. Shared/K formulas assigned the favored team the underdog's implied points.
- **Fix:** Join live lines to the current calendar, verify neutral-site surfaces/roof, and supply kickoff forecasts with source coverage. Use home=(total+spread)/2 for nflverse's positive-home-favorite spread and the inverse for away. Validate the actual downstream feature merge and rebaseline the changed model inputs.
- **Review follow-up (#1545):** Neutral games require an authoritative venue ID plus boolean grass/indoor details; successful but incomplete source responses cannot preserve a nominal home-stadium default. Unknown retractable-roof states retain imputation and JSON-safe source disclosure. Missing ESPN lines keep non-null current-calendar odds.
- **Lesson:** Reusing feature-building code does not establish source parity; verify event identity, source sign conventions, and the values after the last merge.
