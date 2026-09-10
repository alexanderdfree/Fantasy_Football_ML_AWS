> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Softplus floor inflated low-scoring predictions (twice)
- **Files:** `src/shared/neural_net.py:106, :407, :589` (all three `MultiHeadNet*` variants)
- **What:** `softplus(0) ~ 0.693` per head creates a floor that compounds across heads: ~2.08 pts for a 3-head position, ~2.8 pts for K's 4-head sum, etc. No player could be predicted below the floor. Also created a scale mismatch with Ridge (`np.maximum(..., 0)` allows exact zeros), biasing the ensemble.
- **First fix:** Replaced with `torch.clamp(val, min=0.0)` so heads can emit exact zeros.
- **Regression:** Commit `845d93b` reverted the clamp back to `F.softplus` at lines 106 and 407 and extended the softplus floor into the new `MultiHeadNetWithNestedHistory` at line 589 (so K inherited the bug). Also dropped a separate `torch.clamp(td_pred, min=0.0)` around GatedTDHead's output — left unchanged this time because GatedTDHead's `sigmoid × softplus(value)` is already in `[0, +∞)` and doesn't need an outer clamp.
- **Re-fix:** Restored `torch.clamp(val, min=0.0)` at all three sites.
- **Lesson:** Non-negativity constraints on output layers must allow exact zeros. Softplus is good in hidden layers (smooth gradient), but on outputs its floor compounds across heads. When porting a new model variant, re-check which output transforms the existing variants use — don't copy the pre-fix pattern. Consider a shared `_apply_non_negative` helper so a future reviewer can't silently diverge one variant.
