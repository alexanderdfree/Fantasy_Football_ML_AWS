### [FIXED] Upcoming source coverage lost practice identities and London's weather

**File(s):** `src/data/{practice_reports,roster_identity,live_schedule}.py`,
`src/prediction/upcoming.py`, and their regression tests (PR #1649, `39bb46d6`;
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
unresolved. An inactive reference cannot retarget an existing validated player's
identity across teams or positions. Ignore reports for teams outside the upcoming
slate. Normalize UK constituent-country names to Open-Meteo's `GB` country code.

**Validation:** Regression tests cover excluded teammates, scoped reserve
identities, competing aliases, missing/active reference status, wrong-week
references, off-slate reports and the real upcoming builder's pre-Out roster
handoff. Weather tests exercise geocoding and kickoff-hour forecasts with each UK
country name. A live probe of the corrected adapter returned Wembley weather
in the required Fahrenheit/mph units. Replaying the actual Week 6 source inputs
reduced unknown practice participation from 84 to 3 of 439 projected players,
with all 26 teams covered. Full unit validation passed 6,311 tests (3 skipped).
At that rollout, the remaining practice values, two conflicting roster eligibility
records and two unconfirmed retractable roofs remained explicitly uncertain.

**Lesson:** Match source identities before narrowing the prediction population,
and measure coverage against the population actually being projected. A
successful source response can still lose observations through local identity
or location normalization. Genuine eligibility disagreements and unconfirmed
retractable-roof states must remain explicit; complete coverage requires evidence.

### Verified remaining source gaps

**File(s):** `src/data/{official_rosters,club_practice,roof_reports}.py`, their
existing roster/practice/schedule integrations, and focused regression tests
(follow-up to PR #1649).

**What:** The remaining Week 6 gaps required additional primary sources. Official
NFL rosters and club records identify Miller Moss and Patrick Herbert as practice
squad (`DEV`), and the Chargers' October 17 release announcement and current NFL
roster identify Marquez Valdes-Scantling as `CUT`, superseding the stale weekly
`ACT` reference. His practice entry had blocked two active WR teammates. The
Steelers explicitly list Pat Freiermuth as absent from all three practice reports
before his Saturday Questionable addition. Official game pages announce the
Atlanta roof closed and Indianapolis roof open for the affected matchups.

**Fix:** Verify inactive identities against unique current-week GSIS/name/DOB and
official season-roster/profile context; retain these as attributed exclusions or
report identities. Accept the existing published-absence `2.0` encoding only with
complete club table, legend and dated daily-report evidence, recording observed
participation as null and retaining the independent Questionable game status.
Bind official roof decisions to the exact teams, venue and kickoff before weather
handling. Each resolution records its official URLs, retrieval/effective identity
and source hashes. Missing or conflicting evidence still leaves a disclosed gap.

**Evidence:** Primary sources include the [NFL Bears roster](https://www.nfl.com/teams/chicago-bears/roster),
[NFL Chargers roster](https://www.nfl.com/teams/los-angeles-chargers/roster),
[Chargers release announcement](https://www.chargers.com/news/activate-keandre-lambert-smith-sign-caleb-murphy-active-roster-2026),
[Steelers injury report](https://www.steelers.com/team/injury-report/), its
[dated Week 6 update](https://www.steelers.com/news/week-6-injury-report-buccaneers-x2319),
and the official [Atlanta](https://www.mercedesbenzstadium.com/falcons-gameday/chicago-bears)
and [Indianapolis](https://www.colts.com/game-day/regular-season-game-3) game pages.
These captures were retrieved after kickoff on October 18 Eastern and establish
retrospective source resolution, not pregame source availability or accuracy.
The follow-up full unit suite passed 6,468 tests (3 skipped), including source
identity, unavailable/malformed evidence, game-date and redirect boundaries.

**Lesson:** Provider disagreement needs an identity-bound primary source. An
explicit published absence can establish a derived encoding without establishing
measured participation; a building's indoor flag cannot establish a game's roof
decision. Retain current franchise names before constructing official matchup
keys from a directory that also contains historical relocations. Source event
time and retrieval time must remain distinct.
