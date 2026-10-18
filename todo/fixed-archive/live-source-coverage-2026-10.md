### [FIXED] Upcoming source coverage lost practice identities and London's weather

**File(s):** `src/data/{practice_reports,roster_identity,live_schedule}.py`,
`src/prediction/upcoming.py`, and their regression tests (PR pending;
baseline `0125b68b`).

**What:** The October 18, 2026 live Week 6 artifact reported unknown practice
participation for 84 players despite covering all 26 scheduled teams. The
official practice report also lists players already excluded from projections
as Out or reserve. Matching only projected players made those known identities
appear unresolved, which withheld published-absence participation values from
their active teammates. Separately, ESPN supplied Wembley Stadium's country as
`England`; the weather adapter accepted `United Kingdom` but rejected `England`
before requesting a forecast.

**Fix:** Retain the validated roster before Out exclusions for practice identity
joins, while counting coverage and returning values only for projected players.
Recognize same-week, same-team, same-position reserve identities without making
them active roster members; keep ambiguous, conflicting and unknown identities
unresolved. Ignore reports for teams outside the upcoming slate. Normalize UK
constituent-country names to Open-Meteo's `GB` country code.

**Validation:** Regression tests cover excluded teammates, scoped reserve
identities, competing aliases, missing/active reference status, wrong-week
references, off-slate reports and the real upcoming builder's pre-Out roster
handoff. Weather tests exercise geocoding and kickoff-hour forecasts with each UK
country name. A live probe of the corrected adapter returned Wembley weather
in the required Fahrenheit/mph units. Replaying the actual Week 6 source inputs
reduced unknown practice participation from 84 to 3 of 439 projected players,
with all 26 teams covered. Full unit validation passed 6,309 tests (3 skipped).
The remaining practice values, two conflicting roster eligibility records and
two unconfirmed retractable roofs remain explicitly uncertain.

**Lesson:** Match source identities before narrowing the prediction population,
and measure coverage against the population actually being projected. A
successful source response can still lose observations through local identity
or location normalization. Genuine eligibility disagreements and unconfirmed
retractable-roof states must remain explicit; complete coverage requires evidence.
