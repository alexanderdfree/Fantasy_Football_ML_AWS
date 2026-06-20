# Methodical per-position feature selection

Tooling to determine the optimal combination of the ~100+ `include_features` per
position, judged on the **real pipeline** via the shared A/B harness + stacked
seeds. Read this with [ab_harness_priority.md](ab_harness_priority.md).

**Why staged, not brute-force:** 2^N subset search is infeasible, and per the
0.02 FP-MAE noise floor a single-feature leave-one-out is mostly noise — which is
why the screens work at the family / sub-family level and zoom to columns only
where a group shows signal above noise.

## The method (coarse → fine)

1. **Family screen (Stage 1)** — Plackett-Burman 12-run main-effects over feature
   *families*; every family's effect is estimated from all 12 runs.
   - Skill (QB/RB/WR/TE): [ab_feature_screen.py](../src/tuning/ab_feature_screen.py)
     (validated core-8) + [ab_feature_screen_extended.py](../src/tuning/ab_feature_screen_extended.py)
     (adds the position-defining `specific`; `ewma` is QB-only → sub-screen it).
   - K / DST: [ab_feature_screen_k.py](../src/tuning/ab_feature_screen_k.py) /
     [ab_feature_screen_dst.py](../src/tuning/ab_feature_screen_dst.py) — their flat
     `all_features` lists are partitioned into named groups (validated exhaustive),
     cfg-mutator-only, eager seeds (K/DST can't stack).
2. **Sub-family zoom (Stage 2)** — [ab_feature_subscreen.py](../src/tuning/ab_feature_subscreen.py),
   parametrized by `(position, family)` via env, on the families Stage 1 flags
   neutral/borderline. `rolling`/`prior_season` group by stat-root; smaller
   families resolve per column (PB if ≤11 sub-groups, else leave-one-out).
3. **Confirm (Stage 3)** — re-run the chosen drop-set *together* at a high seed
   count (PB assumes additivity; this catches interactions).

Shared grouping + the **metric-agnostic** (MAE *and* RMSE) DoE main-effects
estimator live in [feature_groups.py](../src/tuning/feature_groups.py). Every drop
variant declares `expect_ridge_identical=False` — a real drop MUST move Ridge, so
a Δ=0 ("drop didn't take", the #1172 bug) fails the run loudly.

## Run recipe ([feature_selection.py](../src/tuning/feature_selection.py) driver)

```bash
# 1. plan: cell counts + exact launch_ab commands (fires real Spot jobs; --dry-run to preview)
python -m src.tuning.feature_selection plan --positions RB WR K DST

# 2. launch a stage on the GPU Batch fleet (local training SIGSEGVs on macOS libomp)
python -m src.tuning.launch_ab --spec src.tuning.ab_feature_screen_extended --positions RB --stacked-seeds --max-cells 350
python -m src.tuning.launch_ab --spec src.tuning.ab_feature_screen_k --positions K          # eager
python -m src.tuning.launch_ab --spec src.tuning.ab_feature_subscreen --positions RB --env FF_SUBSCREEN_FAMILY=rolling --stacked-seeds

# 3. report: per-model (Ridge/LGBM/NN/Attn) MAE+RMSE effects -> todo/feature_selection/{pos}.md + .json
python -m src.tuning.feature_selection report --spec src.tuning.ab_feature_screen_extended --run-id <id> --positions RB

# 4. apply the cut YOU chose (draft PR; CI + benchmark gated; never auto-merge)
python -m src.tuning.feature_selection apply --position RB --drop trend_targets target_share_L3 --pr
```

## Stage-2 / Stage-3 orchestration ([feature_selection_stage2.py](../src/tuning/feature_selection_stage2.py))

Four subcommands turn Stages 2-3 into a smooth workflow. They **only print the exact
`launch_ab` commands** (and write a `plan.json`); they never submit a Batch job
(`--exec` opts in). Always **smoke one real cell first** — the printed flow leads with
the riskiest arm at one seed (`--max-cells 2`), because degenerate arms only crash live
(`--list` / unit tests validate grid construction, not the pipeline; #1187 → #1212).

```bash
# Stage 2a — auto-select + print the sub-family screen commands from the Stage-1 reports.
#   Comprehensive: zooms every decomposable family that is a drop-candidate/borderline OR
#   large/heterogeneous (rolling/prior_season/specific + trend); skips atomic + clean KEEP.
#   Skill stacks 24 seeds; K/DST eager 8 (their 3-seed Stage-1 mid-tier is noisy). Trim with
#   --only-family / --skip-family / --max-families; cost preview is a rough ±2x gut-check.
python -m src.tuning.feature_selection substage --positions RB WR TE QB K DST
#   -> writes todo/feature_selection/stage2/plan.json (the report's source of truth)

# Stage 2b — after the runs finish, consolidate them into one per-position report.
python -m src.tuning.feature_selection substage-report --positions RB

# Stage 3 — confirm the chosen drop-set TOGETHER on the production config (PCA-Ridge ON;
#   the screen is skip-PCA). Skill stacked-24 by default (--eager for faithful attention);
#   K/DST eager-8. PB assumes additivity — this catches interactions.
python -m src.tuning.feature_selection confirm --position RB --from-stage2
python -m src.tuning.feature_selection confirm-report --position RB
#   -> todo/feature_selection/stage2/rb.confirm.{md,json}; then `apply` the cut you choose.
```

## Decisions baked in

- **Per-model analysis, no auto-drop rule.** The report shows each model
  separately (MAE + RMSE) and flags a clearly-labelled *suggested* conservative
  cut (neutral-or-helpful for every model); the final whitelist is the operator's.
- **`apply` is human-in-the-loop.** It edits `include_features` / `all_features`
  (+ mirrors into `attn_static_features`) via a self-contained, idempotent
  marked block, verifies the drop took in a fresh subprocess, and opens a **draft**
  PR. Editing these fires a 6-position retrain — review the benchmark delta first.
- **Stacked vs eager** results never mix: skill screens stack (vmap, FP32/LN/
  fixed-epochs, ~24 seeds); K/DST eager (~3). Compare stacked only to stacked.

## Caveats (in every report)

- **Neutral overall ≠ useless** — judge borderline groups by subgroup *bias*
  (rookies/RB, returners), not overall MAE (the draft-capital lesson).
- **Only dropping** from the whitelist and mirroring into the attention static
  branch — never *promoting* rolling/ewma/trend into the non-temporal static branch.
- Static feature audits (VIF / |r|>0.95 under `analysis_output/`) corroborate drop
  candidates when present (best-effort).
