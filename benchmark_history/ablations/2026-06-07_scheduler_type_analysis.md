# LR-scheduler-type A/B — all six positions (2026-06-07)

Raw per-position results: `s3://ff-predictor-training/ablate_scheduler/{POS}/result.json`
Merged: `2026-06-07_scheduler_type_results.json` (this dir).

## Run scope
- Positions: QB, RB, WR, TE, K, DST (one Spot **g4dn.xlarge / T4 (sm_75)** per position —
  the live AWS Batch compute environment as of 2026-06-07, not the g6/L4 in the design docs)
- Seeds: 42, 43, 44 (3) — reported mean ± sample-std
- Scheduler types: `onecycle`, `cosine_warm_restarts`, `plateau`
- Image: `ff-training:13891f45…` (branch `claude/scheduler-type-ablation`, **not merged**)
- **Eager** (FP16): on the T4 (sm_75) CUDA graphs are gated off regardless (graphs need
  sm_80+), so this run **is** the production-Batch training regime; `FF_CUDA_GRAPH=0` was
  belt-and-suspenders. The A/B is both internally bit-comparable and representative of Batch
  production — the only regime it does NOT match is a local sm_80+ (5080/L4) graphed run.
- LightGBM disabled (scheduler-free); NN configs at production values (no proxy shrink).
- Headline metric: production **attention-NN** fantasy-point MAE.

## Method (asymmetric by design)
Only `scheduler_type` (+ its hyperparameters) differs between a position's three variants.
The variant matching the position's **production** type keeps that position's **tuned**
params; the two alternatives use **canonical** params (cosine T0=40/Tmult=2/eta_min=1e-5;
onecycle max_lr=4·nn_lr/pct_start=0.3; plateau factor=0.5/patience=8). So: an **untuned
alternative beating tuned production is the strong signal**; production winning is weak
(home-field advantage). Ridge FP MAE is the **data-identity sentinel** — it must be
identical across the three variants of a position. **All six sentinels passed.**

## Results — attention-NN FP MAE (mean ± std, 3 seeds)

| pos | prod | onecycle | cosine | plateau | winner | Δ vs prod | pooled σ | verdict |
|---|---|---|---|---|---|---|---|---|
| QB | cosine | 6.0499±0.031 | **6.0405**±0.052 | 6.0140±0.077 | plateau | +0.0265 | 0.093 | flat (noise) |
| RB | cosine | 4.0124±0.007 | **4.0367**±0.047 | 4.0576±0.038 | onecycle | +0.0243 | 0.047 | flat (noise) |
| WR | cosine | 3.9716±0.030 | **4.0172**±0.012 | 4.0172±0.017 | onecycle | +0.0456 | 0.032 | **candidate** |
| TE | onecycle | **3.3851**±0.027 | 3.3572±0.006 | 3.3682±0.012 | cosine | +0.0279 | 0.028 | borderline (≈σ) |
| K  | onecycle | **4.2239**±0.028 | 4.2002±0.025 | 4.1885±0.020 | plateau | +0.0354 | 0.034 | candidate (marginal) |
| DST| cosine | 5.0280±0.009 | **5.0108**±0.034 | 5.0152±0.037 | cosine | +0.0000 | 0.049 | production best |

(bold = production type. Δ vs prod = MAE(prod) − MAE(winner); + means winner better.)

Per-target attention MAE for the two flagged positions (where the gap lives):

| WR target | onecycle | cosine | plateau |   | K target | onecycle | cosine | plateau |
|---|---|---|---|---|---|---|---|---|
| receiving_yards | **19.225** | 19.629 | 19.706 |   | fg_yard_points | 4.120 | 4.105 | **4.100** |
| receptions | 1.316 | 1.301 | **1.294** |   | pat_points | 1.096 | 1.089 | **1.085** |
| receiving_tds | **0.265** | 0.272 | 0.275 |   | fg_misses | **0.399** | 0.400 | 0.405 |
| fumbles_lost | 0.011 | 0.011 | 0.011 |   | xp_misses | **0.161** | 0.163 | 0.167 |

## Interpretation

**1. Scheduler type is a second-order knob for this model.** Four of six positions
(QB, RB, TE, DST) are within 3-seed noise of their best variant — the production choice is
not distinguishable from the alternatives, let alone beaten. The nominal "winners" split
evenly across types (onecycle: RB, WR; cosine: TE, DST; plateau: QB, K), which is what you
expect when the effect is mostly noise.

**2. Two positions show a real (beyond-noise), if modest, alternative — both untuned:**
- **WR → onecycle** is the firmest result: 3.9716 vs production cosine 4.0172, Δ=+0.046
  ≈ 1.4× pooled σ (~1.1% of WR's FP baseline). The entire gap is in **receiving_yards**
  (19.23 vs 19.63, ~0.40 raw-yard MAE) — WR's dominant high-magnitude head. onecycle's
  higher peak LR (4·nn_lr = 4e-3) appears to help that head; the small count heads are a
  wash. This is the one I'd actually pursue.
- **K → plateau** is real but marginal: 4.1885 vs production onecycle 4.2239, Δ=+0.035
  ≈ 1.0× pooled σ — right at the noise edge. The gain is on the dense heads
  (fg_yard_points, pat_points); plateau is slightly *worse* on the sparse miss heads.

**3. Production choices are largely vindicated.** DST's production cosine is literally the
winner. QB/RB/TE alternatives don't clear noise. So 4/6 production scheduler-type choices
stand; the experiment found no reason to touch them.

## Caveats
- **3 seeds.** Per project rule, a delta landing inside the seed band needs 5–8 seeds.
  WR (1.4σ) is the only result I'd call firm at 3 seeds; **K and TE are borderline (≈1σ)**
  and could evaporate at 5–8 seeds (cf. backbone-norm: 3-seed −0.022 → 8-seed +0.007).
- **Alternatives are untuned (canonical params).** A tuned onecycle (peak LR sweep) could
  widen the WR gap — or a tuned cosine could close it. The win is a *candidate*, not a ship.
- **`plateau` is unreachable from `PositionConfig`** (no `plateau_factor`/`plateau_patience`
  fields). Shipping K→plateau would require adding those fields first — not worth it for a
  1σ result.
- Eager FP16 matches the live T4 Batch path (graphs gated off on sm_75); the only regime
  not covered is a local sm_80+ (5080/L4) graphed run, which a candidate should re-confirm on.

## Recommended follow-up
1. **WR**: a focused 5–8-seed confirmation of onecycle vs cosine, then a small `onecycle_max_lr`
   sweep (e.g. 2e-3 / 3e-3 / 4e-3) — the receiving_yards head is the lever. Only then consider
   flipping `src/wr/config.py` to onecycle.
2. **K**: bump to 5–8 seeds before believing the plateau edge; if it holds, it argues for
   *adding* plateau support to `PositionConfig` (currently absent) rather than a quick switch.
3. Leave QB / RB / TE / DST on their production schedulers — no evidence to change.
