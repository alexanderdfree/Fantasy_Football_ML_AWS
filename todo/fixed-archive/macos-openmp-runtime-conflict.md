### [FIXED] macOS tests and benchmarks crashed with multiple OpenMP runtimes

**File(s):** `scripts/fix_macos_openmp.py`, `tests/scripts/test_fix_macos_openmp.py`, `SETUP.md` ([PR #1554](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/1554), initial repair `f894c5d0`).

**What:** On 2026-09-10, 41 local Python crash reports pointed to the same
PyTorch-bundled OpenMP binary. Each process also loaded scikit-learn's bundled
copy and Homebrew's copy through LightGBM. Crashes occurred with Python 3.12 and
3.13, including `pthread_mutex_init` error 179/22 and a pytest worker dying during
LightGBM prediction. Environment thread caps were already present in a failing
run. A process-only loader override reduced the loaded runtimes to one and
passed 57 model/training tests, establishing a repair candidate.

**Fix:** The setup script repairs only the selected interpreter's owned library
entries, preserving originals without modifying hardlinked package-cache
binaries. A fresh-process check requires one canonical runtime and rolls back
changes on failure. Package reinstalls can be repaired again with new backups.

**Validation:** Seven local environments each loaded one runtime after repair.
The current pinned Python 3.12 environment passed `pytest -m unit -n 4` without
a loader override: 3,672 passed, 2 skipped in 66.70 seconds after rebasing onto
current main and addressing review, including the previously crashing
`test_api_rows_expose_age_and_rookie` endpoint test. Repair
tests cover preservation of hardlinked package-cache files, rollback of multiple
libraries and cancellation during replacement/verification, repeated application,
package reinstalls, environment ownership, and
Linux/Windows no-op behavior. No new Python crash reports appeared during these
checks. The older Python 3.12 and 3.13 environments each also passed the 57-test
model/training subset with one GPU-only test skipped.

**Lesson:** Inspect native crash frames and actual loaded libraries before
changing Python versions, device defaults, or model code. A green small test
subset is preliminary evidence; validate the repaired environment on the full
unit suite and the affected native workload.
