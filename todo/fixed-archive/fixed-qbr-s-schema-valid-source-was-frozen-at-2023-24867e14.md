> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] QBR's schema-valid source was frozen at 2023
- **File(s):** `src/data/nfl_source.py`, `src/data/external_sources.py`, `tests/test_qbr_source_freshness.py` (PR pending).
- **What:** The old espnscrapeR-data CSV lacked 2024–25 observations, zeroing both QBR prior features for every live QB. The maintained replacement also published 540 2025 records under a 2026 label.
- **Fix:** Use the maintained release, validate game ID/season/week against completed schedules, and invalidate derived caches with a source-version bump. Rebuild data and rebaseline the affected pipeline.
- **Lesson:** Check observed season/game coverage as well as file timestamps and schema; a newer file can still mislabel an entire prior season.
