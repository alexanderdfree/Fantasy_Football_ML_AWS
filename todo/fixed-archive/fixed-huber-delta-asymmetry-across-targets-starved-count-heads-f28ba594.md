> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Huber delta asymmetry across targets starved count heads
- **Files:** Position config files (`*_config.py`)
- **What:** Pre-rebalance loss weights were roughly equal across heads, so yards targets (δ ≈ 15–30) dominated count-head gradients (δ ≈ 0.25–0.5) by ~20–2500× per sample, collapsing the count heads toward their mean. The DST `pts_allowed_bonus` head also had a too-forgiving delta relative to its range, and the old QB `td_points` delta was too small relative to its point scale.
- **Fix:** (1) Rebalanced NN loss weights to ≈ `2.0 / huber_delta` per head across RB (`d229830`), QB (`4ac478f`), WR (`35e611b`), and TE (`a03f795`). (2) DST targets were migrated to 10 raw stats (`cc0c627`), retiring `pts_allowed_bonus` entirely; QB's `td_points` was likewise replaced by split `passing_tds`/`rushing_tds` heads with δ = 0.5 and matching w = 4.0.
- **Lesson:** Huber δ and loss weight are coupled — changing one without the other either starves or drowns a head. Encode the pairing in the config (`2.0/δ`) and re-derive the weight whenever δ moves. See CLAUDE.md "Loss weights are tuned inverse-to-Huber-delta".
