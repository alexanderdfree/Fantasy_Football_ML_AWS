# ADR-0005: Output-constraint stack

**Status:** Accepted

**Decision.** Combine position-specific raw-stat losses, per-head non-negativity, gated count heads, and bounded NN inputs. Ordinary static inputs use StandardScaler followed by ±4 clipping; explicitly selected sparse continuous inputs preserve their magnitudes through a training-fitted bounded transform.

**2026-09-10 amendment — ungated Poisson heads.** New training uses log-rate
outputs for ungated heads whose loss family is `poisson_nll`. The predicted
raw count is `exp(log_rate)` and the loss consumes the log-rate directly with
`log_input=True`. A clamped negative output previously received zero gradient
even when its label was positive; the stable RB/WR/TE attention fumble heads,
WR base-NN fumble head, and DST attention safety head were zero throughout
their 2025 holdouts. Log-space loss also avoids the vanishing gradient of
`log(rate + epsilon)` when a rate is very small.

Initialize each such head's final weights to zero and its bias to the log of
the TRAIN-only event mean (minimum initial rate `1e-6` for an all-zero training
target). Other heads, their losses/weights, and fantasy-point aggregation keep
their existing behavior. In particular this does not restore global Softplus
outputs or introduce the previously rejected hurdle-Poisson loss. K has no
Poisson heads and is unchanged; the generic path also covers its nested model.

The head persists `_log_rate_version` in its state dict. Missing/false markers
retain the legacy raw-rate/clamp interpretation on load, including after a
legacy artifact is re-saved. This lets serving deploy before retraining without
reinterpreting old weights. Training factories and the inference registry use
the same target resolver. `nn_poisson_log_rate=False` retains the baseline for
paired validation via `src.tuning.ab_poisson_log_rate`; production defaults on.

The log-rate amendment supersedes the clamp requirement only for new ungated
Poisson heads. The original clamp-based behavior remains for legacy artifacts.

**Context.** Fantasy targets have three nasty properties: they're zero-inflated (most players don't score a TD on a given week), non-negative (with one exception — DST `pts_allowed_bonus`, which runs −4 to +10), and have outliers (40+ point games do happen). Vanilla MSE regression with no output bound produces nonsense.

**Options considered.** Rather than a single option table, each constraint has its own rationale, and several replaced earlier bugs:

- **Huber over MSE.** Outlier games dominate MSE gradients. Huber with per-target delta (≈1.5–3.0) caps the penalty.
- **Clamp instead of Softplus.** An earlier version used Softplus on head outputs, which has a floor of `softplus(0) ≈ 0.693`. Across three heads that's a ~2-point floor no player could drop below, and it created a scale mismatch with Ridge's `np.maximum(·, 0)`. Clamp allows exact zeros. (Fixed in commit `fe507e0`.)
- **`non_negative_targets` parameter, not a global clamp.** DST's `pts_allowed_bonus` is legitimately negative when the defense gives up a lot of points. A global clamp broke DST; making the set configurable per-position fixed it.
- **Gated TD head.** TDs are discrete and mostly zero. Binary gate + value head reflects the actual data-generating process. (Added in commit `18170a6`.)
- **Hurdle loss families.** Two zero-inflated value losses available alongside the gate: `hurdle_negbin` (zero-truncated NB-2, fits overdispersed counts like receptions where var/mean ≈ 2) and `hurdle_poisson` (zero-truncated Poisson, fits dispersion-≈1 counts like RB TDs and fumbles_lost). Both train the value head on positives only, scaling by fraction-positive so loss magnitude stays comparable to neighbouring Huber/Poisson heads. `hurdle_poisson` was added 2026-05-20 specifically to mirror Ridge's `gated_ordinal` decomposition for sparse Poisson-shaped count heads.
- **±4σ feature clip.** Test-set outliers were producing z-scores up to ~19, sending NN outputs off a cliff. Clipping after scale catches 0.3% of values and prevents catastrophic extrapolation.

**Chosen rationale.** Each constraint was added in response to a specific observed failure, not as a precaution. This ADR captures them together because they form a *coherent* stack — remove any one and a specific failure mode returns. Choosing *which* hurdle family to use on which head is a per-position config call (see RB ablation in [todo/fixed-archive.md](../../todo/fixed-archive.md)).

**References.** [src/shared/neural_net.py:274-305](../../src/shared/neural_net.py) (`non_negative_targets` set + per-head clamp), [src/shared/training.py](../../src/shared/training.py) (`MultiTargetLoss` with Huber; `hurdle_negbin_value_loss` / `hurdle_poisson_value_loss` + their ZTNB/ZTP log-pmfs), [src/dst/config.py:174](../../src/dst/config.py) (`nn_non_negative_targets=set(_TARGETS)` — after the commit `cc0c627` migration all 10 raw DST heads are non-negative, so the set is simply the full target list; the `pts_allowed_bonus` head that used to warrant DST opting out of the global clamp is no longer a head — its negative values are produced downstream by the tier-lookup in `src/shared/aggregate_targets.py`), feature clipping in [src/shared/pipeline.py](../../src/shared/pipeline.py). The `GatedHead` is now parameterized over a list of gated targets (`RB` has three: `receptions`, `rushing_tds`, `receiving_tds`; `WR`/`TE` have two: `receptions`, `receiving_tds`; `QB`, `K`, and `DST` have none — see D2). See also [todo/fixed-archive.md](../../todo/fixed-archive.md) for each bug history.

## Changelog

- **2026-09-11 — Consolidated checkpoint policies.** A new-fit warm start keeps
  both the requested reception-expectation mode and Poisson output link. When
  loading a legacy raw-rate count head into a log-rate fit, retain the new fit's
  train-only final-layer initialization rather than interpreting old raw-rate
  weights as logarithms. Ordinary inference loading continues to honor saved
  legacy markers. The policies are validated together; accuracy tradeoffs remain
  separate from the correctness of the probability and gradient contracts.

- **2026-09-10 — Inheritance magnitude and reception expectation.** The 92 positive
  WR `inherited_opportunity` rows contained 34 distinct float32 values, all of
  which mapped to +4 under ordinary z-score clipping. New NN scalers map this
  selected feature to `4*x/(s+abs(x))`, where `s` is the median nonzero absolute
  training value (1 when training has no nonzero values). Zero remains zero;
  validation/test values do not influence the scale. Activation is explicit in
  WR's production configuration; the other positions retain their existing
  scaler after the broader A/B showed adverse interactions. The shared
  transform applies to base/attention NNs, including CV and captured
  tuning paths; it does not alter the raw feature or Ridge/LightGBM inputs.
  The fitted scaler carries the policy and parameters into serving. Old
  StandardScaler artifacts retain their original transform.

  For `hurdle_negbin` reception heads, the loss fits an **untruncated** NB-2 mean
  `mu`, so reported expectation is `sigmoid(gate) * mu / (1-P_NB(0))`. The former
  `sigmoid(gate)*mu` omitted the truncation normalization. Probability arithmetic
  uses log1p/expm1 in at least FP32 and stays in torch. Each gated head stores a
  tensor expectation version in its checkpoint: absent/zero preserves legacy
  behavior, one enables the corrected NB mean. Loading respects the saved mode
  rather than reinterpreting old weights using today's defaults. Other loss
  families, gated TD outputs, and target/loss-weight definitions are unchanged.
  New archives require serving code that understands the new scaler and head
  version; existing archives remain readable. These are correctness changes,
  not a claim of metric neutrality. See the
  [focused validation record](../../todo/fixed-archive/inheritance-reception-contracts-2026-09.md).

- **2026-09-10** — Train ungated Poisson heads as log-rates with training-mean initialization, consistent eager/graphed losses, and checkpointed legacy compatibility. Validate sparse-head calibration alongside full fantasy metrics with `ab_poisson_log_rate`. (PR pending)

- **2026-05-20** — D5 extended with `hurdle_poisson` loss family (zero-truncated Poisson on positives + BCE gate) as an available primitive alongside `hurdle_negbin`. RB sparse-count ablation (Variants D/E/Bf added to `src/tuning/ablate_rb_gate.py`) showed Variant E (hurdle_poisson on rushing_tds, receiving_tds, fumbles_lost) wins per-target MAE — count_sum 0.353 vs Ridge 0.369 — but regresses aggregate FP MAE +0.163 vs current Variant C. **Rejected for shipping**; primitive kept available for future use, current RB config unchanged.
