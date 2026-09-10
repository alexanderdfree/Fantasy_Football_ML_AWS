> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Partial practice coverage silently became full participation
- **File(s):** `src/serving/practice_reports.py`, `src/serving/live_sources.py`, `src/serving/upcoming_week.py`, `src/serving/core.py`, `tests/test_practice_reports.py` (PR pending).
- **What:** Four teams in the primary report suppressed the all-or-nothing fallback; Sleeper's practice field mapped no players. Known limited participants such as Flowers and Odunze consequently received full-practice values.
- **Fix:** Read official current-week NFL tables, join by canonical team/name/position, and combine coverage per team. Official reports supersede older fallback values; unpublished/unknown reports use the fitted training mean and expose coverage metadata.
- **Review follow-up (#1545):** An unmatched official name can be a roster alias (Andrew/Drew Ogletree). Preserve an ID-matched fallback or unknown for that unresolved team/position group; do not overwrite it with synthetic full participation.
- **Lesson:** A nonempty feed is not league-wide coverage, and an unknown report is not a healthy player. Normalize historical team-directory aliases before matching current nicknames.
