> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [TESTED, REJECTED] RB hurdle_poisson on sparse-count heads regressed FP MAE
- **Files:** `src/shared/training.py` (`ztp_log_prob`, `hurdle_poisson_value_loss`, `SUPPORTED_HEAD_LOSSES` expansion, dispatch in `MultiTargetLoss.forward`), `src/shared/pipeline.py` (hurdle-family downgrade path extended to cover `hurdle_poisson` for non-`GatedHead` models), `src/tuning/ablate_rb_gate.py` (Variants D / E / Bf + new per-target-sum decision rule). Benchmark JSON at `benchmark_history/ablations/2026-05-20T03-42-20_fdcba72_rb_td_gate.json`.
- **What:** RB attention NN was losing to Ridge `gated_ordinal` on all three sparse-count heads (rushing_tds NN 0.305 vs Ridge 0.234, receiving_tds 0.108 vs 0.064, fumbles_lost 0.074 vs 0.071 in the same-seed Variant C run). Ridge's gated_ordinal wins because it trains Stage-2 (ordinal regressor) *only on positives*, while the NN's Poisson NLL trains the value head on all samples — dragging mu toward the marginal (mostly-zero) mean. Added `hurdle_poisson` (zero-truncated Poisson on positives + BCE gate) as the NN-side architectural mirror.
- **Ablation result (seed=42, same-run Ridge baseline):**
  ```
  Variant            FP MAE   Rush TD   Rec TD   Fum lost   CntSum
  A (huber+gate)      4.418    0.271    0.081    0.037      0.389
  B (Poisson/no gt)   4.330    0.316    0.109    0.076      0.501
  C (Poisson+gate)    4.316    0.305    0.108    0.074      0.487  ← current ship
  D (hurdle_poisson)  4.522    0.254    0.067    0.074      0.395
  E (D + fumbles)     4.479    0.249    0.066    0.039      0.353
  Bf (C + fum gate)   4.305    0.314    0.099    0.072      0.485
  Ridge (gated_ord.)    -      0.234    0.064    0.071      0.369
  ```
  Variant E wins per-target MAE on all three count heads — beating Ridge on `fumbles_lost` (-0.032) and matching on both TD heads — but regresses aggregate FP MAE +0.163 vs Variant C. The yards heads also drift slightly (cross-target gradient interaction from the loss-family swap).
- **Decision:** Held back from shipping. The per-target wins are real but the +0.163 FP MAE regression is well above the project's de-facto sensitivity threshold (~0.05 pt/game). Empirical dispersion diagnostic confirmed ZTP was the right family — rushing_tds 1.16, receiving_tds 1.06, fumbles_lost 1.01 — so the experiment was correctly designed, the trade-off is just unfavorable. Kept the `hurdle_poisson` primitive available in `src/shared/training.py` and the D/E/Bf variants in `src/tuning/ablate_rb_gate.py` so future work can re-test if the aggregate-FP balance shifts.
- **Lesson:** Optimizing on per-target MAE for sparse heads can directly trade against aggregate FP MAE through cross-target gradient interaction in a shared-backbone model. A per-head architectural win doesn't compose linearly to a FP-MAE win — the head whose calibration improves may absorb gradient capacity that previously supported a different head's calibration. When evaluating a new loss family, *always* measure FP MAE alongside per-target MAE; the constrained criterion (per-target ≤ Ridge + tolerance, lowest FP MAE among qualifying) is the safer default than either metric alone.
