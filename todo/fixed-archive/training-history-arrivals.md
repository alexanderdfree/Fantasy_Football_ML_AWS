### [FIXED] Out-of-order and late Batch runs lost or mislabeled History rows

**File(s):** `src/batch/run_history.py`, `src/batch/launch.py`,
`src/batch/train.py`, `src/batch/benchmark.py`, `.github/workflows/train-batch.yml`,
`src/serving/benchmark_history.py`, and `src/serving/frontend/src/views/History.jsx`.
Baseline: `586a08da`; fix: PR pending.

**What:** The History API sorted and synchronized stored rows correctly, but
the collector read mutable serving manifests. An overlapping run could place
another run's metrics under the requested commit/PR. The SHA check only warned.
Completions beyond the workflow's five-hour wait and ten-minute recovery did not
publish history. The browser also cached its first response until a full reload.

**Fix:** Register expected positions under an immutable run id before submitting
jobs. Full/merge jobs save their own metrics and the last completing job publishes
the complete summary with a conditional S3 write. The CLI retrieves that exact
row for its local/git copy; known legacy SHA mismatches fail before writing.
History refreshes on visits, focus, and a visible-page timer, retaining good data
on a transient error and preserving expanded rows by stable run identity.

**Evidence:** Tests interleave all six positions across two runs, race completion
and retries, complete a run days after registration without a waiter, reject
foreign/missing SHAs, and verify both monolithic and split submission wiring.
The API cache regression covers both newer and older timestamps arriving late.
The original reproduction returned run B's MAE for three positions in run A's row;
the repaired collector retrieves A's immutable six-position summary.

**Lesson:** A serving pointer identifies the artifact served now, not the result
of a particular training run. Result publication must survive the coordinator's
lifetime. Existing jobs using older images do not gain completion-side publication
retroactively; collect/backfill those separately using verified artifact provenance.
