> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] DST `build_data` fabricated team-week rows for unplayed fixtures
- **File(s):** [src/dst/data.py](../../src/dst/data.py) (`build_data(include_unplayed=False)`), [src/prediction/upcoming_special_teams.py](../../src/prediction/upcoming_special_teams.py) (`build_defense_frame` opts in), [tests/dst/test_data_build.py](../../tests/dst/test_data_build.py). audit-1499 Tier B PR; issue #1520.
- **What:** `build_data` seeded the D/ST frame from every REG schedule row and nflverse publishes the full fixture list up front, so an in-season cache's not-yet-played games (NaN `home_score`/`away_score`) became team-week rows: `yards_allowed` fell back to 350, `compute_targets` filled `points_allowed=21` and zero counts, and each phantom row scored `fantasy_points = -1.0`. Reproduced with one injected unplayed fixture → two fabricated rows. Latent on today's fully-played 2012–2025 cache (0 unplayed REG rows), so the training path is Δ=0; it fires the moment `SEASONS` includes an in-progress season. K is immune (its current-season arm comes from the played-games weekly parquet).
- **Fix:** Drop rows with NaN scores right after the REG filter unless `include_unplayed=True`. The live upcoming-week defense builder sets it, because it deliberately NaNs the target week's scores and needs that fixture's spread/total/home/opponent/rest/roof context (it then NaNs the targets on those rows). Tests pin: dropped by default, kept when opted in with `points_allowed` still NaN, and `assert_frame_equal` between both modes on an all-played schedule (the Δ=0 guard).
- **Lesson:** A frame seeded from a *schedule* is seeded from the future; any "known-before-kickoff" source must be filtered to played games before targets are derived, and the filter must be a keyword the live path can opt out of rather than a blanket drop — the live builder legitimately consumes the same code with fabricated NaN outcomes. Verify the activation precondition (0 unplayed rows today) so the fix ships as Δ=0 rather than as a mover.

**Frozen-release verification (2026-09-23):** The [exact-value receipt](evidence/audit-1499-tier-b-9758de79-20260923.json)
compares main `0db5cfb496036e6f524ae54182353c30f2ad8df4` with candidate
`2e9e1516b7545bb32a0460d368d9cdfff2548d92` on release
`9758de7902913140b9e1f766a63db89e558d9ebe11257c5f7946ace4ff3d8db4`.
The manifest and all four native DST input files passed SHA-256 and size checks.
All 3,663 regular-season schedule rows were played: the added filter removes
zero rows. The complete built frame (7,326 rows), native and prepared splits
(5,726 train / 544 validation / 544 test), and all 38 ordered feature columns
match exactly between revisions. Feature/target arrays are byte-equal.

[The verifier](../../src/analysis/verify_kdst_inert.py) uses the native data
provider and unscaled preparation, including deterministic pandas fills. It
blocks estimator/scaler/trainer fitting and optimizer steps, disables the
feature cache, and stubs only display-only team-logo retrieval identically.
This is a release-specific equivalence proof on the recorded local runtime,
not a new MAE/RMSE benchmark or GPU result. The older Batch row records rounded
historical scores and does not establish bitwise prediction equality.

Positive controls retain the intended effect: an injected unplayed fixture
produces two phantom rows with the legacy-equivalent opt-in, while the new
default removes them; upcoming replay preserves its feature context and hides
target-week outcomes. The focused K/registry and DST/upcoming suites passed
84 no-fit tests. Re-run the proof or require affected-path Batch validation
when the input release, effective configuration, or production-code delta
changes; a future in-progress season activates the filter.
