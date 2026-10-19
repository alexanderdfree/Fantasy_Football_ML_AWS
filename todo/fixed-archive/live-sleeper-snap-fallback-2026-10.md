### [FIXED] Live snap history lacked a verified actual-observation fallback

**File(s):** `src/data/live_snap_counts.py`, `src/prediction/upcoming.py`,
`tests/test_app_live_snap_counts.py`, and `tests/test_app_live_history.py`.
PR pending.

**What:** The investigated 2026 Week 6 artifact reported snap coverage of
2,054/2,266 completed skill-player games. Its primary nflverse/PFR feed lagged
Sunday games. Sleeper exposes actual player and team offensive snap counts through
a free public endpoint, but its Sunday records also lacked those fields when
checked. A populated box score or an absent snap key cannot establish zero snaps.

**Fix:** Add an optional live-history fallback that admits only explicit actual
counts after unique primary identity and completed-game validation. Preserve
native values and row membership, round the observed ratio with exact decimal
half-even arithmetic to the existing fractional precision, and retain retrieval
provenance and recovery/error counts.
Count coverage after recovery so every remaining missing observation still warns
and follows the existing model encoding. Archived training/evaluation inputs and
the serving container remain outside this retrieval path.

**Lesson:** Validate actual observations and their publication scope before
claiming an alternate API fills a gap. An open endpoint, populated player record,
or provider label does not establish snap availability or independent collection.
An optional fallback should improve coverage only when matching source values
exist; missing fields and source failures must remain visible.
