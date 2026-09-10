> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] DST `pts_allowed_bonus` clamped to >= 0, but target ranges from -4 to +10
- **Files:** `src/shared/neural_net.py`, `src/dst/config.py`, `src/dst/run_pipeline.py`, `src/serving/app.py`
- **What:** The softplus-to-clamp fix applied `clamp(min=0)` globally to all heads. But DST's `pts_allowed_bonus` ranges from -4 (35+ points allowed) to +10 (shutout). The model couldn't predict negative tiers.
- **Fix:** Added `non_negative_targets` parameter to `MultiHeadNet.__init__` (defaults to all targets). DST config specifies `{"defensive_scoring", "td_points"}`, leaving `pts_allowed_bonus` unconstrained.
- **Lesson:** Output constraints must be per-head when targets have different valid ranges. A global clamp works for most positions but breaks any target that can legitimately be negative.
