# Rolling-origin multi-season eval (2023/2024/2025) — report + deferred UI follow-up

**Date:** 2026-06-22 · **Base:** origin/main `1e67be7` · **Data:** `benchmark_history/2026-06-22T21-10-13_1e67be7.json` (`mode: rolling_origin`, `--no-sync`)

## Why
The site's reported eval is a **single most-recent season** (TEST_SEASONS=[2025]). The ask was to evaluate on more than one season and see **per-season** results, not just one point. Widening the *production* holdout was rejected because it would shorten the served model's training window (staleness). The leakage-free way to get a multi-season read **without touching the served model** is the **rolling-origin walk-forward** (`src/data/split.py::rolling_origin_folds`, `benchmark.py --rolling-origin`): for each test season T, train a clean model on `[..T-2]`, val `T-1`, test `T`. The T=2025 origin ≈ today's served model, so reporting 2023/2024/2025 needs no model change.

## Decisions taken (this work)
- **Served model + splits unchanged** (origin/main: train 2013-2023, val 2024, test 2025). No retrain, no staleness, no serving change.
- Report the multi-season eval via rolling-origin (already-built machinery), `ROLLING_ORIGIN_TEST_SEASONS=[2023,2024,2025]` (gives 2024+2025 plus 2023 for a 3-point mean±std).
- **UI deferred** (owner choice): generate + record the data + this report now; the History-tab display is a follow-up (see below).

## Result — Attention NN (served model family), per-season MAE + 3-season mean±std

| Pos | 2023 | 2024 | 2025 | mean±std | 2025 vs mean |
|----|----|----|----|----|----|
| QB  | 5.64 | 6.04 | 6.01 | 5.90 ± 0.22 | +0.12 (2025 pessimistic) |
| RB  | 4.11 | 4.10 | 4.05 | 4.09 ± 0.03 | −0.03 (flat) |
| WR  | 4.29 | 4.34 | 3.94 | 4.19 ± 0.21 | −0.25 (2025 flattered WR) |
| TE  | 3.36 | 3.43 | 3.40 | 3.40 ± 0.04 | +0.00 (flat) |
| K   | 3.84 | 4.17 | 4.15 | 4.06 ± 0.19 | +0.10 (2025 pessimistic) |
| DST | 5.32 | 5.04 | 5.02 | 5.13 ± 0.17 | −0.11 (2025 optimistic) |

Best model by 3-season mean (served Attn NN vs others): **RB/WR/DST → Attn NN best**; **QB → LightGBM 5.891 ≈ Attn 5.895 (tied)**; **TE → PlainNN 3.36** (Attn 3.40 close); **K → Ridge 3.95** (Attn worst — known K/Attn-noise). All four models' per-season + mean±std live in the JSON's per-result `rolling_origin` block.

## Finding
The single-season (2025) headline is **per-position biased in both directions**: ~0.25 too good for WR and ~0.1 too good for DST, but ~0.1 too pessimistic for QB and K (RB/TE flat). The 3-season mean±std is the honest central estimate; the std (±0.03 RB → ±0.22 QB/WR) shows how much a single season can swing. Rankings are **unchanged** vs production — this corrects the *reported number's* season bias, it doesn't change any model decision.

## Deferred follow-up — surface this on the site
The `rolling_origin` block is generated but **not displayed**: `src/serving/benchmark_history.py::_benchmark_row()` reads only the flat single-split headline keys; no serving code or `app.js` references `rolling_origin`. Production also never runs `--rolling-origin` (`train-batch.yml` uses `benchmark --download-only`, single-split). To make the multi-season report the on-site eval:
1. **Backend** — `_benchmark_row()` folds the `rolling_origin` block into the row payload (additive; old readers ignore it).
2. **Frontend** — `app.js` History tab renders per-season + mean±std (~20-40 LOC).
3. **(optional) CI** — add `--rolling-origin` to `train-batch.yml` so it stays current each retrain (~3× benchmark cost), and sync the rolling-origin JSON to S3 so serving (which reads S3, not the repo) picks it up.

Until then, this committed JSON + report is the record.
