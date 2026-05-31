# RB model comparison — LGBM "underprediction" + NN/attention by history depth

**Date:** 2026-05-30 · **Verdict: EXPECTED behaviour, not a bug** (both parts).
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

## NN & attention NN by in-season history depth

Follow-up question: the attention NN is *history-aware*, so for Henry W2 — where
the within-season sequence holds only **1 game** — shouldn't it have regressed
toward LGBM's ~12 rather than printing 17.2?

**The structure is as assumed, and the masking is correct.**
`build_game_history_arrays` ([src/features/engineer.py](src/features/engineer.py))
groups by `(player_id, season)`, so the sequence is within-season: Henry W2 = 1
real game + 16 zero-padded slots, and `AttentionPool` masks the padding with
`-inf` before softmax — the model attends *only* to the one real game.

**But a 1-item sequence does not imply "regress."** Attention over a length-1
sequence *extracts* that game — there is no small-sample → hedge reflex — so the
lone explosive Week 1 (plus the static prior-season branch, ≈19.8) yields a high
prediction. Define `n_prior_games` = in-season games played before the row (= the
attention sequence's real length). Per-model MAE by depth (2025 test); **bold =
best**:

| n_prior_games | n | avg actual | Ridge | NN | Attn | LGBM |
|---|---|---|---|---|---|---|
| 0 (opener) | 159 | 4.5 | 4.51 | **3.26** | 3.81 | 3.33 |
| 1 | 148 | 5.7 | 4.01 | 3.71 | 3.68 | **3.60** |
| 2–3 | 273 | 6.3 | 4.02 | 3.71 | 3.66 | **3.65** |
| 4–7 | 440 | 7.2 | 4.19 | 4.20 | 4.14 | **4.07** |
| 8+ | 623 | 9.0 | 5.09 | 4.95 | 4.85 | **4.78** |

1. **At `n_prior_games==1` (Henry's bucket) the attention NN (3.68) is on par with
   LGBM (3.60), and the average bias is *negative*** — so Henry W2 is a within-bucket
   outlier, not a systematic over-trust of sparse history. On the 22 sparse-history
   rows whose recent game was hot (≥18 FP), players **actually averaged 16.6 FP next
   week** — hot starts usually persist, so ~17 was the right expected value. LGBM
   under-predicts these the most (bias −3.31); it only looked "right" on Henry
   because Henry is the 1-of-1 crater. Pushing the attention NN toward ~12 would
   *degrade* the typical case.
2. **The one genuine weak spot is the season opener (`n_prior_games==0`).** With no
   real games the attention history branch is inert (all-padding → 0), so the model
   leans on its static prior-season branch and over-predicts (bias **+1.13**, MAE
   3.81 — the worst non-Ridge model; the plain NN handles no-history best at 3.26).
   Deferred to a tracked follow-up.

**The attention NN never has the lowest MAE in any history-depth bucket** — LGBM
(or, at the opener, the plain NN) wins everywhere. For RB the attention architecture
is not earning its complexity over the tree model.

## Conclusion

Part 1 — the LGBM "underprediction" is textbook tree-vs-linear behaviour on a
volatile, mean-reverting target, precisely why LightGBM is the production-best
(lowest-MAE) RB model. Part 2 — the attention NN's history-awareness does **not**
make it regress on a short sequence, and its ~17 for Henry was well calibrated; its
only real gap is the empty-history season opener (`n_prior_games==0`), tracked as a
follow-up. **No model change recommended in this analysis** (the season-opener
follow-up was implemented separately — see the Update below).

## Update (2026-05-30): season-opener fix shipped

The `n_prior_games==0` weak spot (the attention NN over-predicting on empty in-season
history — the "deferred follow-up" in Part 2) was addressed with a **learned no-history
embedding** in `MultiHeadNetWithHistory` ([src/shared/neural_net.py](../shared/neural_net.py)):
under `attn_no_history_embedding`, an empty-history row (all-padding mask) gets a
per-target learned vector in place of the dead pooled-history vector (which the
per-target LayerNorm otherwise maps to its constant bias, so the head leaned entirely on
the hot static prior-season branch). The parameter is zero-initialised and built last, so
it draws no RNG and is bit-identical to the baseline at init — the flag OFF/ON A/B is
exactly ceteris-paribus (Ridge/NN/LGBM MAE identical to 6 d.p., confirming clean
attribution).

**RB result (two seeds, same splits).** The fix reduces season-opener over-prediction in
both: opener bias `+1.13 → +0.52` (seed 42) and `+0.72 → +0.59` (seed 123), opener MAE
`3.81 → 3.39` and `3.42 → 3.37`, with overall attention-NN MAE flat within seed noise
(`4.25 → 4.15` seed 42; `4.19 → 4.20` seed 123). The **baseline opener problem is itself
seed-sensitive** (bias +1.13 vs +0.72), so the gain's magnitude varies while its direction
(toward 0) is robust. Re-running this script now prints post-fix Part 2 numbers, which
will differ from the pre-fix table above.

**Rejected variant — mirroring the embedding onto the opp-defense branch.** A team's
opponent-defense history is empty iff it is **week 1**, and every player in week 1 is also
playing their own opener — so `opp-empty ⊆ player-empty` (confirmed: 99 week-1 rows ⊂ 159
`npg==0` rows). A separate opp-branch embedding therefore reaches no row the player
embedding doesn't already cover; it double-counts the opener signal, competes for the same
week-1 gradient (diluting the player correction — opener bias only reached +0.86, overall
regressed to 4.25), and adds overfit-prone capacity. Reverted: the player branch is the
correct and sufficient home for the opener signal.
