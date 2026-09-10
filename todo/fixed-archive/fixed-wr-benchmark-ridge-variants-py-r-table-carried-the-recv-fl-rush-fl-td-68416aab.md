> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] WR `benchmark_ridge_variants.py` R² table carried the `recv_fl`/`rush_fl`/`td_pts` mislabel M18 left unfixed
- **File(s):** [src/wr/benchmark_ridge_variants.py](../../src/wr/benchmark_ridge_variants.py) (commit `7c67469`, PR [#356](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/356), finding F6).
- **What:** M18 (PR #312) migrated the MAE-table headers to iterate `TARGETS` but scoped out the parallel R² table, which kept hardcoded `recv_td`/`recv_yd`/`recs` columns and dropped `fumbles_lost`.
- **Fix:** The R² table now builds `r2_target_header` from `TARGETS` (`benchmark_ridge_variants.py:274`) and prints `r2_cells` over `TARGETS` (`:284`), so all four raw stats render.
- **Lesson:** When migrating one table's headers to iterate a target list, grep the file for sibling tables sharing the same hardcoded literals and fix them in the same pass.
