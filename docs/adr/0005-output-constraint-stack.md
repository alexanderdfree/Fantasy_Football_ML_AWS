# ADR-0005: Output-constraint stack

**Status:** Accepted

**Decision.** Combine four constraints on NN outputs: (a) Huber loss with per-target deltas, (b) per-head `clamp(min=0)` controlled by a `non_negative_targets` set, (c) a gated TD head that models P(TD>0) and E[TD|TD>0] separately, (d) ±4σ feature clipping after StandardScaler.

**Context.** Fantasy targets have three nasty properties: they're zero-inflated (most players don't score a TD on a given week), non-negative (with one exception — DST `pts_allowed_bonus`, which runs −4 to +10), and have outliers (40+ point games do happen). Vanilla MSE regression with no output bound produces nonsense.

**Options considered.** Rather than a single option table, each constraint has its own rationale, and several replaced earlier bugs:

- **Huber over MSE.** Outlier games dominate MSE gradients. Huber with per-target delta (≈1.5–3.0) caps the penalty.
- **Clamp instead of Softplus.** An earlier version used Softplus on head outputs, which has a floor of `softplus(0) ≈ 0.693`. Across three heads that's a ~2-point floor no player could drop below, and it created a scale mismatch with Ridge's `np.maximum(·, 0)`. Clamp allows exact zeros. (Fixed in commit `fe507e0`.)
- **`non_negative_targets` parameter, not a global clamp.** DST's `pts_allowed_bonus` is legitimately negative when the defense gives up a lot of points. A global clamp broke DST; making the set configurable per-position fixed it.
- **Gated TD head.** TDs are discrete and mostly zero. Binary gate + value head reflects the actual data-generating process. (Added in commit `18170a6`.)
- **Hurdle loss families.** Two zero-inflated value losses available alongside the gate: `hurdle_negbin` (zero-truncated NB-2, fits overdispersed counts like receptions where var/mean ≈ 2) and `hurdle_poisson` (zero-truncated Poisson, fits dispersion-≈1 counts like RB TDs and fumbles_lost). Both train the value head on positives only, scaling by fraction-positive so loss magnitude stays comparable to neighbouring Huber/Poisson heads. `hurdle_poisson` was added 2026-05-20 specifically to mirror Ridge's `gated_ordinal` decomposition for sparse Poisson-shaped count heads.
- **±4σ feature clip.** Test-set outliers were producing z-scores up to ~19, sending NN outputs off a cliff. Clipping after scale catches 0.3% of values and prevents catastrophic extrapolation.

**Chosen rationale.** Each constraint was added in response to a specific observed failure, not as a precaution. This ADR captures them together because they form a *coherent* stack — remove any one and a specific failure mode returns. Choosing *which* hurdle family to use on which head is a per-position config call (see RB ablation in TODO.md archive).

**References.** [src/shared/neural_net.py:274-305](../../src/shared/neural_net.py) (`non_negative_targets` set + per-head clamp), [src/shared/training.py](../../src/shared/training.py) (`MultiTargetLoss` with Huber; `hurdle_negbin_value_loss` / `hurdle_poisson_value_loss` + their ZTNB/ZTP log-pmfs), [src/dst/config.py:174](../../src/dst/config.py) (`nn_non_negative_targets=set(_TARGETS)` — after the commit `cc0c627` migration all 10 raw DST heads are non-negative, so the set is simply the full target list; the `pts_allowed_bonus` head that used to warrant DST opting out of the global clamp is no longer a head — its negative values are produced downstream by the tier-lookup in `src/shared/aggregate_targets.py`), feature clipping in [src/shared/pipeline.py](../../src/shared/pipeline.py). The `GatedHead` is now parameterized over a list of gated targets (`RB` has three: `receptions`, `rushing_tds`, `receiving_tds`; `WR`/`TE` have two: `receptions`, `receiving_tds`; `QB`, `K`, and `DST` have none — see D2). See also the "Fixed" section of [TODO.md](../../TODO.md) for each bug history.

---
