# RB models vs backups ascending into a workhorse role

**Date:** 2026-05-30 · **Verdict: structural blind spot — the model reacts, it cannot anticipate.**
**Reproduce:** `python -m src.analysis.cohort_analysis ascension --deep-dive` (data-only; no splits/torch).
Literal per-model MAE/bias on the cohort: `python -m src.analysis.cohort_analysis ascension --positions RB --with-model-error`
(runs the RB pipeline).

## The question

When a starter goes down, a low-usage back can jump from ~5 to ~22 touches in a
single week. How do the RB models handle that transition? This is the *role-change*
regime — orthogonal to the history-depth lens in
[rb_lgbm_disagreement_findings.md](rb_lgbm_disagreement_findings.md), which buckets
by `n_prior_games` and would average an ascending back (often 8+ prior games) in
with stable veterans, hiding the transition.

## Cohort

Backup → workhorse event = a back averaging **≤ 8 opportunities/game** (carries +
targets) over the prior 3 games who posts **≥ 18 opportunities** in week W.
**205 events over 2012–2025 (~15/season, 14 in the 2025 test season); 60% coincide
with the team's prior lead RB being Out/Doubtful or inactive** (the rest: committee
shake-ups, ejections, game-script). ~1% of RB-weeks (≈3% of startable) — small
enough that any fix is invisible in aggregate MAE (see Honest note).

## Literal model undershoot on the 2025 test cohort

Measured 2026-06-01 with the production RB pipeline and the corrected ascension
label (`min_prior_games=2`, so season openers are not counted as backup-to-workhorse
events). The test cohort is **14 RB games**. Every model under-predicts every
ascension row, so MAE and absolute signed bias are identical:

| Model | Ascension N | Ascension MAE | Ascension bias | Established MAE | ΔMAE vs global |
|---|---:|---:|---:|---:|---:|
| Ridge | 14 | 11.846 | −11.846 | 4.181 | +7.600 |
| NN | 14 | 12.096 | −12.096 | 3.992 | +8.035 |
| Attention NN | 14 | 12.188 | −12.188 | 3.906 | +8.212 |
| LightGBM | 14 | 12.462 | −12.462 | 3.895 | +8.494 |

Takeaway: the trained models do **not** claw back the role-change information
bound; the production-best LightGBM is still about **12.5 FP low** on the breakout
week. Treat NN/Attention numbers as directional only (single seed, n=14), but the
Ridge/LightGBM undershoot is deterministic and large.

## Why the model can't see it coming

Every RB volume/role feature is a `shift=1` rolling aggregate
([src/shared/feature_build.py](src/shared/feature_build.py)::`rolling_agg`) or the
prior-game attention sequence ([src/rb/config.py](src/rb/config.py)
`attn_history_stats`, "prior in-season games only"). At week W they all encode
*backup* usage. `practice_status`/`game_status` are the player's **own** injury
status, not the teammate's. There is **no vacated-volume / teammate-availability
feature**. The only forward-looking signal is `depth_chart_rank` (current-week
charts publish pre-game) — and it is the weak link (see below).

## The lag gap (what inputs encode vs realized, mean over 205 events)

| Quantity | Model input (lagged) | Realized @ W |
|---|---|---|
| Opportunities (car+tgt) | **5.40** | **22.05** |
| Game carry share | 0.20 (`carry_share_L3`) | 0.70 |
| Fantasy pts (std) | 3.41 (L3-mean) / 4.76 (last wk) | **14.18** |
| Fantasy pts (PPR) | 4.37 (L3-mean) | **17.12** |

**Structural under-prediction: 10.78 FP std (76% of realized), 12.75 PPR.** The
inputs capture ~¼ of the output. This is an upper bound on any model anchored on
these features — it isolates the information content of the inputs, not the literal
NN error (use `--with-model-error` for that).

## Convergence — the miss is one week, then it overshoots

| offset | realized FP | L3-input FP | `carry_share_L3` | realized share |
|---|---|---|---|---|
| W+0 | 14.18 | 3.41 | 0.20 | 0.70 |
| W+1 | 8.02 | 7.40 | 0.41 | 0.55 |
| W+2 | 8.19 | 8.98 | 0.51 | 0.52 |
| W+3 | 8.72 | 10.54 | 0.60 | 0.51 |

By W+1 the lagged input (7.40) ≈ realized (8.02) — the model self-corrects within
one game. By W+2–W+3 it slightly **overshoots**, because the L3 window still carries
the one spike while usage settles to a normal lead-back load. The damage is
concentrated almost entirely on the ascension week itself.

## depth_chart_rank — the only forward-looking signal, and it's the weak link

- **2025 is covered, not missing.** The 2025 depth feed is the ESPN-format schema
  (no legacy `formation`/`depth_team`); the loader normalizes it via
  `_normalize_espn_depth` ([src/data/loader.py](src/data/loader.py), PR #370). Loaded
  the production way, 2025 has 24,596 offensive player-weeks and **100% of the 14 2025
  ascension events carry a depth_chart_rank**. (An earlier cut of this doc said "2025
  dead" — that was a bug in *this diagnostic*, which pulled the raw
  `nfl_source.depth_charts` shim and filtered on `formation`, silently dropping the
  un-normalized 2025 ESPN rows. Fixed; it now loads depth the way the loader does.)
- **The real weakness is rank quality, not coverage.** Across all 205 events 97%
  carry a depth_chart_rank that week, but only **70% are listed rank ≤ 2 (median rank
  2)** — the official chart frequently *still lists the ascending back as the backup*
  the week he explodes. It's present, but a weak, single, low-weight ordinal slot; it
  does not encode the vacated *volume* (which is the lever in "Where a fix could help").

## Concrete examples (largest input→output gaps)

Thomas Rawls 2015 W11 (4.7 prior opp/g → 33, scored 37.5 vs ~2.3 implied), DeAngelo
Williams 2015 (Le'Veon Bell injury, 36.5), Zach Charbonnet 2024 (Kenneth Walker out,
31.3), Alexander Mattison 2020 (Dalvin Cook out, 26.5), Mike Davis 2018 — most are
lead-back-injury events.

## Honest note (why this won't move aggregate MAE)

The cohort is ~1% of RB-weeks, so even a perfect ascension fix is invisible in
overall MAE — the exact trap that killed the draft-capital feature
([TODO.md](../../TODO.md) `[TESTED, REJECTED]`). Any remediation needs a *tracked
ascension-subgroup metric* (this module is that metric) or it reads as "no effect."

## Where a fix could help (not pursued here)

The only lever that touches the ascension week itself is a **forward-looking
teammate-availability / vacated-volume feature** (teammate RB ruled Out on the
current-week injury report → projected available carry share). The injury data
supports it leakage-safely (reports publish pre-game). Scope narrowly to RB + the
subgroup metric above; the attention NN is the architecture most able to exploit it
(it can weight the freed-up role rather than diluting it in a rolling mean). The
plain rolling features already converge by W+1, so the prize is purely the W-week
breakout.
