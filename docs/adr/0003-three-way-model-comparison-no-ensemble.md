# ADR-0003: Three-way model comparison, no ensemble

**Status:** Accepted

**Decision.** Train and report Ridge (L2 linear), multi-head NN, and LightGBM independently per position. Do not ensemble or stack.

**Context.** A core goal is comparing multiple model architectures quantitatively. Ensembling would dominate any single model's MAE, but it would also muddle the question the project is trying to answer.

**Options considered.**

| Option | What it answers | What it costs |
|---|---|---|
| Ensemble (weighted average) | "Lowest MAE possible" | Loses the per-architecture comparison |
| Stacking (meta-model) | Same as ensemble + meta-risk | Extra CV pass, minimal gain at this sample size |
| Independent comparison (chosen) | "Which architecture wins, and why" | Leaves some accuracy on the table |

**Chosen: independent comparison.** Ridge is the honest baseline (same feature matrix, same scaling, just L2 regression). The NN is the headline "custom architecture" deliverable. LightGBM is the "what could a boosted tree do with the same inputs" reference point. Reporting all three per-position surfaces real trade-offs: Ridge wins on stability and interpretability; NN pulls ahead on WR/TE where interactions matter; LightGBM is competitive where there's enough data and falls apart on K/DST.

**Rejected.** Ensembling was considered and rejected because it would obscure exactly the finding the project is trying to produce. (A future production system would of course blend these — that's a follow-up, not this ADR.)

**References.** [src/shared/models.py](../../src/shared/models.py) (`RidgeMultiTarget`, `LightGBMMultiTarget`), [src/shared/neural_net.py](../../src/shared/neural_net.py), [src/shared/pipeline.py](../../src/shared/pipeline.py). LightGBM added in commit `f343c20`.
