# RB LightGBM "underprediction" vs Ridge/NN — findings

**Date:** 2026-05-30 · **Verdict: EXPECTED behaviour, not a bug.**
**Reproduce:** `python -m src.analysis.analysis_rb_lgbm_disagreement`
(2025 test set, 1643 RB player-weeks; local CPU retrain reproduces the serving-UI numbers).

## The question

On the serving UI some RB player-weeks show LightGBM far below the other models.
Canonical case **Derrick Henry 2025 Week 2** — LGBM ≈ 12.6, NN ≈ 16.3,
Attn-NN ≈ 17.2, **Ridge ≈ 26.3**. Is this a bug, or just model differences?

## Decisive fact: LGBM was the *most accurate*, not buggy

Henry's Week 2 **actual was 2.3 FP** (11 carries, 23 rush yds, 0 TD, 0 rec) — he
cratered after a 29.2-pt Week 1. Errors vs the 2.3 actual:

| Model | Pred | Error |
|---|---|---|
| Ridge | 26.3 | +24.0 (worst) |
| NN | 16.3 | +14.0 |
| Attn-NN | 17.2 | +14.9 |
| **LightGBM** | **12.6** | **+10.3 (best)** |

"LGBM predicts lower than its peers" ≠ "LGBM is wrong." Henry's gap of **7.30 FP**
is the **single largest LGBM-vs-peers disagreement in the entire RB test set** —
the flagged example is the literal extremum, and on it LGBM is the closest model.

## Class-level confirmation (all 1643 RB rows)

**Global — LGBM wins, no anomalous bias.** All four models share a similar small
negative bias (right-skewed FP), so there is no LGBM-specific low-bias.

| | MAE | bias | RMSE |
|---|---|---|---|
| Ridge | 4.514 | −0.78 | 6.465 |
| NN | 4.268 | −0.71 | 6.123 |
| Attn-NN | 4.255 | −0.96 | 6.205 |
| **LightGBM** | **4.155** | −0.99 | 6.114 |
| *(Season-avg baseline)* | *4.310* | | |

Note Ridge (4.514) is **worse than the naive season-average baseline (4.310)** —
its extrapolation is a net liability.

**The flagged class "LGBM ≪ peers" (peer-mean − LGBM ≥ 4 FP, n=12, incl. Henry W2).**
LGBM is the most accurate and nearly unbiased; Ridge overpredicts by +8.9.

| | MAE | bias |
|---|---|---|
| Ridge | 14.258 | +8.90 |
| NN | 9.796 | +3.57 |
| Attn-NN | 7.788 | +1.28 |
| **LightGBM** | **7.258** | **−0.16** |

The top-decile-gap cut (n=165) shows the same ordering (LGBM 5.858 < Ridge 6.904).
Extreme disagreements are **12-to-1 one-directional** (LGBM lower): LGBM is the
conservative anchor, not an erratic one.

**Calibration — every model regresses to the mean; Ridge runs hottest where it hurts.**

| actual bin | n | avg actual | Ridge | NN | Attn | LGBM |
|---|---|---|---|---|---|---|
| [0,5) | 826 | 1.4 | **4.1** | 3.8 | 3.8 | **3.6** |
| [10,15) | 191 | 12.1 | 9.1 | 9.5 | 8.9 | 9.1 |
| [25,∞) | 76 | 31.2 | 12.8 | 13.4 | 12.8 | 12.7 |

On the 826 low-actual weeks (50% of rows) Ridge predicts highest; on true ceiling
weeks every model tops out near ~13. **Ridge's willingness to predict high does
not buy ceiling accuracy** — on genuinely high weeks (actual ≥ 20, n=129) Ridge's
MAE (15.23) ≈ LGBM's (15.27). It just lands high on the wrong, mean-reverting weeks.

## Mechanism

At Week 2 the only "recent" history is Week 1, so Henry's lagged inputs were
`rolling_mean_fantasy_points_L3 = ewma = 29.2`, `rolling_mean_rushing_yards_L3 = 169`.
Every model "sees" a red-hot elite back.

- **Ridge (linear)** extrapolates that signal straight up → 26.
- **LightGBM (trees + Huber + heavy reg `reg_lambda=8.25`/`reg_alpha=4.92`,
  `num_leaves=16`)** cannot extrapolate beyond training-leaf values and shrinks
  toward typical outcomes → 12.6.

Per-target decomposition of the +9.06 FP Ridge−LGBM gap on the disagreement class:

| target | FP contribution |
|---|---|
| **rushing_tds** | **+6.31** |
| rushing_yards | +2.20 |
| receiving_yards | +0.87 |
| receptions | +0.50 |
| receiving_tds | −0.79 |

Two-thirds of the gap is **extrapolated rushing TDs** — a sparse, high-variance
count where the linear model chases the recent rate and the tree model holds near
the base rate. On the mean-reverting week the back scores 0 TDs and Ridge eats the
miss.

## Residual cost (honest note)

All models — LGBM included — badly underpredict true ceiling games (actual ≥ 20:
bias ≈ −15). That is a universal RB-predictability ceiling from the features, **not**
specific to LightGBM, and Ridge is no better there. A Ridge/LGBM blend would mostly
re-import Ridge's overprediction variance on the common low weeks without ceiling
benefit, so it is unlikely to lower MAE — left as a separately-benchmarkable
question, not pursued here.

## Conclusion

The disagreement is textbook tree-vs-linear behaviour on a volatile, mean-reverting
target, and it is precisely why LightGBM is the production-best (lowest-MAE) model
for RB. **No model change recommended.**
