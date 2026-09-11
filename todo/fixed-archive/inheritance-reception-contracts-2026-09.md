### [FIXED] Inheritance magnitude collapsed and reception expectations omitted truncation

Consolidation follow-up (2026-09-11): legacy warm starts must preserve both the
requested reception-mean and Poisson log-rate policies. Reuse legacy trunks but
retain the new fit's train-only final-layer initialization when the count link
changes. The combined checkpoint regression test distinguishes this new-fit rule
from inference's requirement to preserve saved legacy behavior.

**File(s):** `src/shared/feature_build.py`, `src/shared/pipeline.py`,
`src/shared/neural_net.py`, configuration/factory/registry wiring and
`src/tuning/ab_inheritance_reception.py` (PR pending).

**What:** On the sealed September baseline, all 92 positive WR inheritance rows
were mapped to +4 by z-score clipping, collapsing 34 distinct float32 input
values into one. The attention reception head separately fit a zero-truncated
NB-2 likelihood but reported gate × the untruncated mean, understating the
expectation of its fitted distribution.

**Fix:** Fit a bounded magnitude-preserving scaler on training-only nonzero
inheritance values and carry it with the saved model. Enable it explicitly for
WR; the measured broader activation adds regressions and is not a production
default. Ordinary features and
legacy StandardScaler artifacts retain their prior transformation. Correct
the new reception output to gate × mu / (1 − P_NB(0)), using torch log1p/expm1
arithmetic in at least FP32. A per-head tensor version preserves interpretation
across serialization, including legacy checkpoints with no version field.
Base-NN and attention preprocessing, CV, captured tuning, and serving use the
same fitted scaler contract. Existing gated TD and non-NB loss behavior stays
unchanged. K/DST have no inheritance feature or affected NB head.

**Validation:** A real WR legacy/combined smoke preserved all 34 inheritance
magnitudes in both NNs and reproduced saved-model total predictions exactly,
with per-target parity checks passing through the serving primitives. Old WR artifacts also reproduce the
captured 2,768-row production cache within its 0.005-point display rounding.
The full isolated A/B separates legacy, magnitude-only, expectation-only, and
combined arms on the same raw inputs and seeds: 42/42 cells succeeded with
exact saved-model inference parity. Scaler activation is limited to WR after
the broader screen showed regressions. The
[validation report](../inheritance-reception-fix-validation.md) records paired
metrics, tradeoffs, provenance, and current GPU status.

**Lesson:** A rare continuous feature can become a flag after generic z-score
clipping. Check observed input distinguishability, not only presence in an
allowlist. A distribution's latent rate is not necessarily its reported mean;
verify output expectations against probability mass and preserve semantics in
the model artifact. Correctness and metric improvement are separate claims.
