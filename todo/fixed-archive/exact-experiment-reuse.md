### [FIXED] Exact experiment reuse and prediction-cache identity

**Files:** `src/training/result_store.py`, `src/training/reuse.py`,
`src/analysis/prediction_reuse.py`, analysis adapters and the A/B harness.

**What:** Repeated analysis and unchanged A/B baselines refit identical models;
the weekly-accuracy cache could reuse a position/seed parquet after inputs or
implementation changed.

**Fix:** Use verified complete-fit and artifact-prediction identities. Preserve
fresh training gates, exact model source, raw predictions and original run
provenance. Recompute evaluation outputs, and treat uncertain identities as
misses. ADR-0030 contains the storage and execution contract.

**Lesson:** Saved predictions save work only when their full computation is
identified; a filename, seed, or matching headline score is not that identity.
