### [FIXED] Upcoming refresh mixed legacy hydration with historical provider replay

**File(s):** `.github/workflows/refresh-upcoming-week.yml`,
`src/prediction/upcoming.py`, and their regression tests
(implementation `f0c695a1`).

**What:** After the production contract migration, upcoming refresh
[run 34662692415](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/actions/runs/34662692415)
pinned the new immutable data release and loaded all six approved model
generations, but its old sync helper downloaded only 17 mutable-prefix files
instead of the release's 160 files. The current-season schedule fetch then ran
under historical replay and failed. The historical serving snapshot was healthy
while the separate upcoming artifact remained on its previous generation.

**Fix:** Materialize the selected release through the existing checksum-verified
dataset API before loading models, including provider captures and the archived
evaluation reference. Fetch current schedules within the existing live-source
scope in the CLI's isolated mutable cache. Historical replay remains strict, and
the data producer contract and training policies are unchanged.

**Validation:** Execute the workflow's hydration block against a sealed release
containing provider metadata, response bytes and evaluation-reference data;
reject a corrupt captured response before model loading. Exercise the real
schedule-fetch call under both captured and derived-only historical modes,
confirming live access works and the historical guard is restored afterward.
The unpublished full CLI build produced 475 players across all six positions
and three scoring formats with no blocked live-source checks; all 160 sealed
input files and 254 model files remained byte-identical. Existing roster,
practice and weather coverage warnings stayed explicit. The isolated production
QB benchmark (`2026-09-12T01-01-15_f0c695a1.json`, no S3 sync) records the matching
code fingerprint and completed all four model families.

**Lesson:** An environment pin does not materialize its dataset. Verify the
consumer's actual file source, and give live requests an explicit scope instead
of weakening historical replay when an upcoming builder hits a missing input.
