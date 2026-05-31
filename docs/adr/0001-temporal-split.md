# ADR-0001: Temporal split (2012–23 train / 2024 val / 2025 test)

**Status:** Accepted

**Decision.** Split by season, not by random row. Train on 2012–2023, validate on 2024, hold out 2025 for test.

**Context.** Weekly fantasy data is a time series with heavy week-over-week autocorrelation within a player (rolling features by construction). A random train/test split would leak week W+1's rolling stats into week W's label.

**Options considered.**

| Option | Complexity | Leakage risk | Data-efficiency |
|---|---|---|---|
| Random row split | Low | **High** (rolling features leak) | High |
| K-fold time-series CV | Medium | Low | Medium |
| Single season-based holdout (chosen) | Low | Low | Medium |

**Chosen: single season holdout.** Matches how the model will actually be used (next season is unknown; last season is the fairest holdout). Ridge hyperparameter tuning still uses expanding-window CV *inside* the 2012–2023 train window to avoid a single fold's noise — so we get some of the statistical benefit of K-fold without contaminating the test year.

**Rejected.** K-fold over seasons was considered but over-spent our limited post-2012 history and broke the "deployment mirror" intuition — at serve time we're always predicting a future we've never trained on, and a single holdout faithfully simulates that.

**Extension (2026-05-29) — rolling-origin reporting.** The single holdout makes every headline metric an `n=1`-season point estimate (a 0.02 MAE gap between two models is indistinguishable from season noise). `rolling_origin_folds` ([src/data/split.py](../../src/data/split.py)) + `benchmark.py --rolling-origin` optionally score N forward holdouts (`ROLLING_ORIGIN_TEST_SEASONS=[2023,2024,2025]`; per origin train `[..T-2]` / val `T-1` / test `T`) and report per-model MAE/R²/top-12 as **mean±std**. This is the walk-forward extension of the same expanding-window principle already used for Ridge tuning — **not** the K-fold rejected above: every origin still trains strictly on the past, and the final origin (test 2025) reproduces this split season-for-season. It is complemented by a *within-season* paired bootstrap ([src/analysis/significance.py](../../src/analysis/significance.py)) that answers the orthogonal question — sampling noise within one season vs. variance across seasons. Headline/operator-invoked only (~N× cost); the per-PR benchmark stays single-split.

**References.** [src/data/split.py:6-37](../../src/data/split.py), season constants in [src/config.py](../../src/config.py). Knowledge cutoff to 2012 landed in commit `f400a5c`.

## Changelog

- **2026-05-31** — **D1 extended:** the train-only games filter became a per-position `min_games_per_season` knob ([src/shared/position_config.py](../../src/shared/position_config.py) field → `build_pipeline_config` → [pipeline.py](../../src/shared/pipeline.py), keyed in [feature_cache.py](../../src/shared/feature_cache.py)), replacing the bare global `MIN_GAMES_PER_SEASON=6`. The filter was applied to TRAIN only (val/test kept every row), so the model trained on ≥6-game players but was scored on everyone — a self-inflicted train→test covariate shift on the cold-start subgroup, surfaced by the [src/tuning/ablate_min_games.py](../../src/tuning/ablate_min_games.py) harness (finding shipped retrain-free in PR [#638](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/638)). **RB/WR/TE relaxed to 1**, validated full-model 8-seed on each position's *served* model (cold-start `would_filter` MAE: RB LGBM −0.175±0.021, WR LGBM −0.209±0.027, TE attn −0.214±0.122; established `kept` flat; overall slightly better — adding low-volume player-seasons back to TRAIN is informative, not noise). **QB excluded** — its served Ridge benefits but the attention NN regresses established QBs (`kept` +0.404±0.158 MAE) and threshold 3 is unstable, so it's deferred to a per-target follow-up ([TODO.md](../../TODO.md) Open). K/DST inert (no team/kicker has <6 games; K already floors its own `min_games` in `k/data.py`), left at the global default. **Triggers a 6-position retrain.** Relates to **D1** (the temporal split) and the eval-methodology covariate-shift guard; distinct from the K-only `min_games` pre-filter. Tests: [tests/tuning/test_ablate_min_games.py](../../tests/tuning/test_ablate_min_games.py).
