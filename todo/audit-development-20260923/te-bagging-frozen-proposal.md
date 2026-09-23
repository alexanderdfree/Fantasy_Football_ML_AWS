# Frozen TE-only bagging proposal

Candidate ID: `93d08ebb5262eb77fe4a0d7bb440f1db7fc4ce91d8b8a621b55e1a003444435b`.
The [machine-readable recipe](te-bagging-frozen-proposal.json) freezes the exact
parameters, 108-feature order, targets, source/image/data pins and evidence.

This is a development-qualified proposal only. It changes TE LightGBM's
`subsample_freq` from 0 to 1, preserving its configured `subsample=0.7359385`.
No other position or model family is part of the proposed activation. Production
activation is not implemented, global PR #1606 remains held, and confirmation is
not authorized while canonical 2024 reference coverage is blocked.

The resolved learner uses 1,900 trees maximum, learning rate 0.08219987,
15 leaves, depth 8, column fraction 0.5750816, L2 regularization 1.1011751,
L1 regularization 1.228984, minimum child samples 51, minimum split gain
0.16878265, and the regression objective. The complete estimator defaults and
seed-specific constructor parameters are in JSON. Per-target validation and
30-round early stopping are unchanged. The recorded Batch A/B route resolves
LightGBM to one CPU thread; its job environments and pinned resolver establish
that setting. Parameter resolution instantiated estimators only; no fitting
ran locally.

The six paired TE seed comparisons (18 cells including repeat controls) improve
both overall errors and preserve elite errors in both development seasons:

| Scoring year | MAE delta | RMSE delta | Elite MAE delta | Elite RMSE delta |
|---|---:|---:|---:|---:|
| 2022 | −0.008870 | −0.008050 | −0.007371 | −0.017069 |
| 2023 | −0.018394 | −0.002072 | −0.003712 | −0.004966 |

Deltas are candidate minus baseline, averaged over paired seeds 42, 123 and 7.
Full precision and seed uncertainty are preserved in the frozen JSON and
[complete bagging report](evidence/bagging-full.json). This is retrospective
development evidence, not confirmation or a claim of a production improvement.

Validated source: `87213b853c4d724f6574dccddbb62dfcb6601cd1`.
Image: `sha256:1efe7ddca433f36395d500046a042e3d4854ca24cbccc99299277792aea41fbc`.
Data: `9758de7902913140b9e1f766a63db89e558d9ebe11257c5f7946ace4ff3d8db4`.
Any future confirmation must use a concrete TE-only implementation with this
frozen recipe, retain all six positions and both required protected cohorts,
and preserve unchanged controls. No new tuning or relaxed gate is authorized.
