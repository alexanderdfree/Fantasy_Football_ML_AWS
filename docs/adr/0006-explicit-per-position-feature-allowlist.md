# ADR-0006: Explicit per-position feature allowlist

**Status:** Accepted

**Decision.** Every position has an explicit `{POS}_INCLUDE_FEATURES` list. The feature engineer computes ~155 features; the trainer uses *only* what's on the allowlist. Adding a feature to training requires changing a config.

**Context.** Feature leakage is the single most common source of "my model works in training and collapses in production" in time-series ML. Opt-out allowlists (compute everything, exclude the bad ones) are easy to get wrong silently — one new feature that accidentally peeks into the current week breaks everything, and nobody notices until deployment.

**Options considered.**

| Option | Leakage resilience | Convenience | Auditability |
|---|---|---|---|
| All features in → trust the builder | Low | High | Low |
| Opt-out blocklist | Medium | Medium | Medium |
| Opt-in allowlist (chosen) | **High** | Lower | High |

**Chosen: opt-in allowlist.** A reviewer can diff a PR and see exactly what features the model sees. Adding a feature is a deliberate act. The inconvenience (a config edit per experiment) is the point — it forces intentionality.

**Extension: per-position attention-static allowlist.** When D4 grew to cover all six positions, the Ridge/base-NN allowlist turned out not to be enough. The attention branch already consumes temporal signal through its game-history channel, so feeding it the *same* rolling/EWMA/trend features again is both redundant and a leakage risk (the rolling features reach back farther than the attention window, so they silently smuggle older-season information past what's on the visible sequence). Landed in commit `2500ecc`: every position now defines a `{POS}_ATTN_STATIC_FEATURES` list that the pipeline consults when building the attention NN's static channel, distinct from the Ridge/base-NN `{POS}_INCLUDE_FEATURES`. Rolling, EWMA, and trend columns are explicitly kept out. Example: DST's attention static branch sees only 9 columns (`is_home`, `week`, `spread_line`, `total_line`, `rest_days`, `div_game`, `is_dome`, `prior_season_dst_pts_avg`, `prior_season_pts_allowed_avg`) — see [src/dst/config.py:218-228](../../src/dst/config.py); K swaps its rolling features out for contextual / shift-1 "last-game" features (`_CONTEXTUAL_FEATURES`, [src/k/config.py:39-47](../../src/k/config.py), wired into the attention static branch at L207) because everything further back is in the inner kick sequence anyway.

K was the lone exception until PR #199 (`dff43fb`): when the convention landed in `2500ecc`, K still carried a separate `ATTN_L1_FEATURES` block that pushed L1-rolling columns into the attention static channel. The nested per-kick attention (D4) was already learning the same per-game aggregates the L1 rollups encoded, so the columns were redundant signal — and they violated the rule the other five positions followed. Removing them made the convention uniform across all six positions; this is the kind of residue a cross-cutting refactor leaves behind, and the lesson is to audit every position's config for it rather than trusting the touched-files list (TODO archive entry on the K `ATTN_L1_FEATURES` violation).

**Extension: context-augmented history tokens.** The static-vs-history split D6 establishes (rolling/EWMA/trend live in static, raw per-game stats live in history) doesn't pin down where *per-game raw context* belongs — Vegas implied totals, home/away, days rest, team-level box score for that game. Under the original RB schema these reached the model only through the static branch (carrying the *target week's* values), which meant the attention encoder had to assume every historical game shared the prediction week's environment. For RB that's a meaningful gap: usage is driven by game script, and a 12-carry game in a -7 road dog spot is different signal than the same line in a +10 home favorite spot. RB's `ATTN_HISTORY_STATS` now includes four game-script columns and eight team box-score columns alongside the player's own per-game stats; D6's "no double-counting" rule is satisfied because these are per-game raw values, not rolling/EWMA/trend. The columns are materialised by `src/shared/team_box_score.py::merge_team_box_score_features` (Vegas/home/rest already plumbed per-row by [merge_schedule_features](../../src/shared/weather_features.py)) and consumed automatically by `build_game_history_arrays` once they're listed in the position's `ATTN_HISTORY_STATS`. QB/WR/TE/K/DST configs deliberately do not opt in — the pattern is RB-first, with other positions to follow if benchmarks warrant.

**Rejected.** Opt-out was the earlier pattern and was exactly how the feature-clipping bug and the schedule-features-at-inference bug slipped in. Allowlist refactor landed in commit `18170a6` alongside the gated TD change.

**References.** [src/qb/config.py::POSITION_CONFIG](../../src/qb/config.py) (`include_features` + `attn_static_features` fields on the position-config object), [src/te/config.py](../../src/te/config.py), [src/dst/config.py:218-228](../../src/dst/config.py), [src/k/config.py:140-152](../../src/k/config.py), [src/shared/pipeline.py](../../src/shared/pipeline.py). Weather/Vegas features (from [docs/archive/design_weather_and_odds.md](../archive/design_weather_and_odds.md)) are opted in per-position through the same mechanism.

---
