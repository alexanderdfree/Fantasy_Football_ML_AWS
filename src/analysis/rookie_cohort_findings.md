# Rookie cohort metric — findings

Output of `python -m src.analysis.cohort_analysis rookie --with-model-error` (seed 42, 2025 test
season, four skill positions). This is the *tracked rookie-subgroup metric* whose
absence was the stated precondition for ever revisiting draft-capital features
(see `TODO.md` `[TESTED, REJECTED] Draft-capital / combine rookie cold-start
features`, PR #519).

**Cohort definition.** A test row is a rookie iff the player's earliest season
anywhere in the splits equals its season (data spans 2012–2025, so any player
first appearing in 2025 is an unambiguous rookie — exactly the rows whose
`prior_season_*` were NaN before imputation). Rookies split into the first 3
games played (`rookie_early`) and the rest (`rookie_rest`).

## Headline: rookies are NOT a high-MAE cohort — the defect is *bias*, not MAE

| Pos | Best model (overall MAE) | Rookie MAE | Veteran MAE | **Δ MAE** | rookie_early MAE |
|-----|--------------------------|-----------:|------------:|----------:|-----------------:|
| QB  | Ridge (6.552)            | 6.417 | 6.575 | **−0.159** | 6.222 |
| RB  | Attention NN (4.151)     | 4.057 | 4.176 | **−0.119** | 3.505 |
| WR  | Multi-Head NN (4.198)    | 3.396 | 4.384 | **−0.988** | 2.865 |
| TE  | Attention NN (3.476)     | 3.492 | 3.473 | **+0.019** | 2.685 |

At three of four positions the rookie MAE is **lower** than the veteran MAE, and
TE is flat. Rookies simply score fewer fantasy points (realized rookie vs veteran
FP/game: QB 9.9 vs 13.6, RB 6.1 vs 7.6, WR 4.3 vs 7.2, TE 5.5 vs 5.6), so their
absolute errors are smaller. **A MAE-based benchmark can never show a rookie
feature helping** — the cohort that needs help isn't a high-error cohort on MAE.
This is the mechanism behind #519's flat benchmark, now made visible.

## Where the rookies actually break — the bias (`mean(pred − actual)`)

The error is *directional*. It barely registers in MAE — rookies score little, so
each game's `|error|` is small regardless of sign — but it shows up as systematic
over/under-prediction in the **signed bias**, and pooling `rookie_early` (over)
with `rookie_rest` (under) cancels it further (nothing cancels *inside* MAE; the
cancellation is in the signed mean). Signed bias on `rookie_early` vs
`rookie_rest` (every model, total FP):

| Pos | bucket | Ridge | NN | Attn NN | LightGBM |
|-----|--------|------:|----:|--------:|---------:|
| QB  | rookie_early | **+3.62** | **+3.76** | **+2.95** | **+4.37** |
| QB  | rookie_rest  | −0.30 | −0.47 | −2.23 | −0.67 |
| WR  | rookie_early | **+2.74** | +0.43 | +1.73 | +1.16 |
| WR  | rookie_rest  | +1.17 | −0.92 | −0.39 | −0.06 |
| TE  | rookie_early | **+1.83** | +0.61 | +0.69 | +1.22 |
| TE  | rookie_rest  | −0.35 | −0.91 | −1.10 | −0.90 |
| RB  | rookie_early | +0.77 | −0.15 | −0.52 | −0.25 |
| RB  | rookie_rest  | −0.68 | −0.08 | **−1.59** | −1.05 |

Two systematic patterns:

1. **QB / WR / TE: massive *over-prediction* of rookies in their first ~3 games**
   (QB especially: +3 to +4.4 FP/game across every model). With no NFL history,
   every history feature is imputed to 0 and the model defaults a rookie toward a
   league-average starter — but rookie pass-catchers and QBs start slow. Once a
   role is established (`rookie_rest`) the sign flips to *under*-prediction.
2. **RB: under-prediction throughout**, worst on `rookie_rest` (−1.0 to −1.6) —
   the same lagged-input structural under-prediction the RB-ascension diagnostic
   found (`src/analysis/rb_ascension.py`): a rookie back who seizes a workhorse
  role is scored off backup-level lagged inputs.

So "handling rookies" is a **calibration/bias** problem, position-specific in
sign, not an overall-MAE problem — and it is invisible to the MAE benchmark the
project optimizes.

## How much of that error is recoverable? (bias-corrected MAE)

`MAEbc` = MAE after removing each cohort's mean bias (`mean(|e − mean(e)|)`), so
`MAE − MAEbc` is the systematic, calibration-recoverable part and the remainder is
irreducible spread. (`RMSE² = bias² + Var` decomposes the same way and is why RMSE
surfaces bias that MAE folds into spread; MAE has no such decomposition.) Best
model per position on `rookie_early`, plus the largest recoverable across models:

| Pos | best model | MAE | MAEbc | recoverable | max recoverable (any model) |
|-----|-----------|----:|------:|------------:|----------------------------:|
| QB  | Ridge   | 6.222 | 5.172 | **+1.05** | +1.60 (LGBM 6.68→5.08) |
| WR  | NN      | 2.865 | 2.711 | +0.15 | +1.25 (Ridge 4.08→2.83) |
| TE  | Attn NN | 2.685 | 2.396 | +0.29 | +0.80 (Ridge 3.29→2.49) |
| RB  | Attn NN | 3.505 | 3.671 | −0.17 | none for best model (Ridge +0.29) |

- **QB rookies carry ~1 FP/game of recoverable error in *every* model** — the
  strongest calibration target. The best model (Ridge) over-predicts rookie QBs by
  +3.6 in their first 3 games; correcting that removes 1.05 of its 6.22 MAE.
- **WR / TE: the recoverable bias lives in the *weaker* models (Ridge / LightGBM).**
  The best NN/attn models already largely handle rookies (low bias), so a rookie
  calibration buys *them* little — a sharper restatement of #519's "the gain must
  reach the best model."
- **RB: nothing recoverable for the best model** — its `rookie_early` bias is ≈0;
  the error is genuine spread (some rookie RBs are immediate workhorses, some
  aren't). The negative "recoverable" is the diagnostic saying *there is no
  systematic bias to remove* (MAD is minimized at the median, not the mean, so
  `MAEbc ≥ MAE` when bias ≈ 0 — read it as zero, not a defect).

Net: a rookie **calibration** (not a draft-capital feature) is worth ~1 FP/game at
QB and helps the weaker WR/TE models, but does nothing for the best RB model — and
none of it moves overall MAE. Judge any such fix on MAEbc / bias + ranking.

## Implication for draft capital (#519)

The precondition is now satisfied: a tracked rookie metric exists. But it argues
*against* re-introducing draft capital as an MAE play. The rookie cohort isn't
high-MAE, so an MAE-judged feature will read flat (exactly #519's result). If
rookies are to be addressed, the candidate levers and how to judge them:

- **Target the early-game over-prediction bias** (QB/WR/TE), e.g. a rookie-aware
  shrink-to-low-prior or an `is_rookie × early_game` interaction — and **judge it
  on the `rookie_early` bias and on ranking (top-12 hit rate), not overall MAE.**
- If draft capital is revisited at all, the #519 stop-rule still holds: scope to
  RB / LightGBM-only, and watch this metric's rookie bias, not the benchmark MAE.

## Caveats

- **Single seed (42).** Best-model identities for RB/WR/TE are within ~0.05 MAE
  of the runner-up (NN/Attn/LGBM are neck-and-neck; cf. the seed-sensitivity
  note in auto-memory). The bias *direction* is the robust signal here, not the
  4th-decimal MAE ordering.
- **Rookie fraction ≈ 18%** of skill rows (QB 14%, RB 21%, WR 19%, TE 15%) —
  slightly above #519's ~14% because "first season in the 2012–2025 data" is a
  touch broader than `years_exp == 0` (it includes late UDFA/practice-squad
  debuts). Both pick out the same all-`prior_season_*`-NaN cold-start rows.
- **The MAE Δ is confounded by scoring magnitude** (rookies score less ⇒ lower
  absolute error). The bias column is the de-confounded read.

Reproduce: `python -m src.analysis.cohort_analysis rookie --with-model-error`
(add `--no-model` for instant cohort sizes, `--positions RB` to scope).
