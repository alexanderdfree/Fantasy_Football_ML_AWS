> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Total aux loss double-counted adjustments
- **Files:** `src/shared/pipeline.py:208-211`
- **What:** Training total target was `fantasy_points` (includes INT/fumble penalties), but the model predicts `sum(heads)` (clean targets only). The total aux loss trained heads to absorb penalties. Then at inference, adjustments were added *again* via `adj.values`. Net effect: ~1.9 pts/game double-counted penalty for QB.
- **Fix:** Changed total target to `sum(pos_train[t].values for t in targets)`.
- **Lesson:** When a loss term compares a derived quantity (sum of heads) to a label, the label must match the derivation exactly. Any mismatch between what the model produces and what it's trained against will leak into predictions.
