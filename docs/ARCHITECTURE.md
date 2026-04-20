# ADR-001: Fantasy Football Predictor — Consolidated Architecture

| Status | Date | Author |
|---|---|---|
| Accepted | 2026-04-16 | Alex Free (CS 372) |
| Updated | 2026-04-19 | D7/D9 and §2 diagram reconciled with the EC2 training switch; the Batch path is preserved as standby (see [docs/batch_design.md](batch_design.md)). |

## Table of Contents

1. [Context](#1-context)
2. [System Overview](#2-system-overview)
3. [Decision Log](#3-decision-log)
   - [D1: Temporal split](#d1-temporal-split-201223-train--2024-val--2025-test)
   - [D2: Multi-target decomposition with shared NN backbone](#d2-multi-target-decomposition-with-shared-nn-backbone)
   - [D3: Three-way model comparison, no ensemble](#d3-three-way-model-comparison-no-ensemble)
   - [D4: Attention over game history (skill positions only)](#d4-attention-over-game-history-skill-positions-only)
   - [D5: Output-constraint stack for zero-inflated, non-negative targets](#d5-output-constraint-stack)
   - [D6: Explicit per-position feature allowlist](#d6-explicit-per-position-feature-allowlist)
   - [D7: EC2 warm instance over Batch/SageMaker for parallel training](#d7-ec2-warm-instance-over-batchsagemaker)
   - [D8: Two Docker images (slim Flask, heavy training)](#d8-two-docker-images)
   - [D9: Warm training host (replaces Batch cold-start stack)](#d9-warm-training-host)
   - [D10: Trunk-based CI/CD with test-gated deploys](#d10-trunk-based-cicd-with-test-gated-deploys)
4. [Cross-Cutting Consequences](#4-cross-cutting-consequences)
5. [Open Issues / Follow-Ups](#5-open-issues--follow-ups)
6. [References](#6-references)

---

## 1. Context

**Problem.** Predict weekly fantasy football points for individual NFL players across six positions (QB, RB, WR, TE, K, DST) for the 2025 season, using 2012–2024 as training history. Primary output is a per-player point projection (regression); ranking metrics (top-12 hit rate, Spearman correlation) are derived from projections post-hoc.

**Constraints.**
- Solo CS 372 final project, ~2 weeks of execution time.
- Small-sample ML regime: after position filtering and ≥8-games-per-season minimum, roughly 200–600 player-seasons per position — orders of magnitude smaller than datasets most modern NN architectures assume.
- Public data only — `nfl_data_py` ([nflverse](https://github.com/nflverse)) weekly stats, rosters, schedules, snap counts. Snap count coverage starts 2012, which bounds the training window.
- Deliverables must hit 15 of 16 CS 372 ML rubric items for ~73 pts (see [instructions/DESIGN_DOC.md](../instructions/DESIGN_DOC.md)). Rubric item "Documented design decision with technical tradeoffs" is what this ADR satisfies.

**Scope.**
- In: per-player weekly projections, three scoring formats (Standard / Half-PPR / Full-PPR), a Flask dashboard for lookup, automated training on AWS Batch.
- Out: lineup optimization, DFS salary-aware construction, betting odds beyond what's embedded as features.

**Forces driving the architecture.**
- The positions are not the same sport — QB and DST share maybe 10% of meaningful features. A one-model-fits-all approach under-fits all of them.
- Fantasy points are a *sum* of component stats (rushing yards, TDs, receptions). Modeling the sum directly wastes structure that modeling the components preserves.
- Training is cheap (~2 min on a GPU per position), but inference must run in a standard ECS container with no CUDA. This asymmetry shapes the whole deployment story.

---

## 2. System Overview

```
                    ┌─────────────────────────┐
  nflverse API ───▶ │ Data ingest + features  │ ─┐
  (2012–2025)       │  src/data, src/features │  │
                    └─────────────────────────┘  │
                                                 ▼
                    ┌──────────────────────────────────────────┐
                    │   Three model families (per position)    │
                    │  ┌────────┐ ┌─────────┐ ┌──────────────┐ │
                    │  │ Ridge  │ │ Multi-  │ │ Attention NN │ │
                    │  │ (base) │ │ HeadNet │ │ (QB/RB/WR/TE)│ │
                    │  └────────┘ └─────────┘ └──────────────┘ │
                    │  + LightGBM (selective positions)        │
                    └──────────────────────────────────────────┘
                                         │
                       Compared, not     │
                       ensembled ──────▶ │
                                         ▼
         ┌─────────────────────┐   ┌────────────────────┐
         │ EC2 g4dn (train)    │   │ Flask app (serve)  │
         │ Dockerfile.train    │   │ Dockerfile (slim)  │
         │ warm, 6 parallel    │   │ ECS, CPU-only      │
         │ positions via SSM   │   │                    │
         └─────────────────────┘   └────────────────────┘
                    │                          ▲
                    │   model artifacts → S3 → │
                    └──────────────────────────┘
```

> **Standby:** the AWS Batch + Spot path ([docs/batch_design.md](batch_design.md)) is fully kept and is reactivated by flipping the `BATCH_ACTIVE` repo variable. D7 explains the trade-off; D9 covers the warm-host implementation.

A training run is triggered by a push to `main`, which invokes [`.github/workflows/train-ec2.yml`](../.github/workflows/train-ec2.yml): the workflow starts the warm g4dn.xlarge (if auto-shutdown stopped it), runs SSM commands to launch six per-position training containers, streams CloudWatch logs to the Actions job, and verifies fresh `model.tar.gz` artifacts landed in S3 per position. The Flask service is built separately and deployed to ECS on every push to `main`; it reads pre-baked models from S3 and serves projections through a dashboard.

---

## 3. Decision Log

Each decision below follows the same structure: what was decided, the forces at play, options considered, the chosen option's trade-offs, why the rejected alternatives were rejected, and references to code.

---

### D1: Temporal split (2012–23 train / 2024 val / 2025 test)

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

**References.** [src/data/split.py:6-37](../src/data/split.py), season constants in [src/config.py](../src/config.py). Knowledge cutoff to 2012 landed in commit `f400a5c`.

---

### D2: Multi-target decomposition with shared NN backbone

**Decision.** Decompose each position's prediction into a small set of *raw NFL stat* targets (yards, TD counts, receptions, interceptions, fumbles_lost) rather than pre-scored fantasy-point buckets. Train one neural network per position with a shared backbone and one head per target. Convert per-target predictions to fantasy points *after* the model runs via a deterministic aggregator ([shared/aggregate_targets.py](../shared/aggregate_targets.py)) that multiplies each raw-stat prediction by the corresponding coefficient in `src/config.py:SCORING_PPR` (or `SCORING_HALF_PPR` / `SCORING_STANDARD`). The final totals (`preds["total"]`) and the `w_total · Huber(total)` auxiliary loss are produced through that aggregator rather than a naive `sum(heads)`.

Current target sets (6 for QB/RB, 4 for WR/TE):

| Pos | Targets |
|---|---|
| QB | `passing_yards`, `rushing_yards`, `passing_tds`, `rushing_tds`, `interceptions`, `fumbles_lost` |
| RB | `rushing_tds`, `receiving_tds`, `rushing_yards`, `receiving_yards`, `receptions`, `fumbles_lost` |
| WR | `receiving_tds`, `receiving_yards`, `receptions`, `fumbles_lost` |
| TE | `receiving_tds`, `receiving_yards`, `receptions`, `fumbles_lost` |

**Context.** Fantasy points collapse heterogeneous events (a receiving TD is structurally very different from a passing yard) into a single scalar. Decomposing lets each head specialize, lets us apply different loss deltas per component, and — critically — keeps MAE reporting interpretable in native stat units ("the model is off by ±18 passing yards, ±0.4 passing TDs per game") rather than in ambiguous point buckets. An earlier iteration of this ADR had targets like `passing_floor = passing_yards × 0.04` and `td_points = pass_TD × 4 + rush_TD × 6` baked in; the migration to raw stats moved all scoring coefficients to one place and decoupled model error from scoring-format choice.

**Options considered.**

| Option | Complexity | MAE interpretability | Scoring-format flexibility |
|---|---|---|---|
| Single-target NN predicting total fantasy points | Low | Low | None (retrain per format) |
| Fantasy-point-component heads (`passing_floor`, `td_points`, …) | Medium | Medium (points, ambiguous) | None (formats baked in) |
| Shared backbone + raw-stat heads + post-aggregation (chosen) | Medium | High (yards / TDs / receptions) | Full (aggregator takes `scoring_format` arg) |

**Chosen: raw-stat heads with post-aggregation.** The backbone learns position-general features ("is this player healthy? on the field? getting opportunity?"). Heads specialize on individual countable events. The aggregator is the single source of truth for turning those events into fantasy points — swap `SCORING_PPR` for `SCORING_STANDARD` without retraining. Auxiliary total-loss (`w_total · Huber(total_pred, total_actual)`) routes through the same aggregator, so the sum the loss tracks is always actual fantasy points.

Zero-inflated TD targets get a `GatedTDHead` (BCE gate on `tds > 0` plus a value head for conditional mean). RB has two gates (`rushing_tds`, `receiving_tds`); WR and TE each have one (`receiving_tds`). QB has none — QBs score too often for the zero-inflation argument to hold.

**Rejected.** Single-target models under-fit the structure — every head would implicitly have to learn "what is a TD" separately from "what is a rushing yard." Fantasy-point-component heads (the previous iteration) made MAE hard to reason about — a `td_points` MAE of 4.25 could mean either "off by ~0.7 TDs/game" or "off by ~1 TD/game and a PAT." Raw-stat targets are unambiguous.

**Consequence.** Aux total-loss and inference-time totals both flow through `shared/aggregate_targets.py:predictions_to_fantasy_points` — changing scoring coefficients is a single-file edit. The per-position adjustment functions (`compute_qb_adjustment`, `compute_fumble_adjustment`, etc.) are retired for QB/RB/WR/TE; their effects (interception penalty, fumble penalty) are now direct targets (`interceptions`, `fumbles_lost`) that the aggregator prices in. K and DST retain `compute_adjustment_fn` (out of scope for this migration).

**References.** [shared/aggregate_targets.py](../shared/aggregate_targets.py) (aggregator + `TARGET_UNITS` + `POINT_EQUIVALENT_MULTIPLIER`), [src/config.py](../src/config.py) (`SCORING_PPR`, `SCORING_HALF_PPR`, `SCORING_STANDARD`), [shared/neural_net.py:38-145](../shared/neural_net.py) (`MultiHeadNet`, `aggregate_fn` plumbing), [shared/training.py](../shared/training.py) (`MultiTargetLoss`), per-position `compute_{pos}_targets` in `QB/qb_targets.py`, `RB/rb_targets.py`, `WR/wr_targets.py`, `TE/te_targets.py`, and `{POS}/{pos}_config.py` (target lists + loss weights + Huber deltas in raw-stat units). Consolidated in commit `99d7086`; raw-stat migration follows.

---

### D3: Three-way model comparison, no ensemble

**Decision.** Train and report Ridge (L2 linear), multi-head NN, and LightGBM independently per position. Do not ensemble or stack.

**Context.** The CS 372 rubric explicitly asks for "compared multiple model architectures quantitatively." Ensembling would dominate any single model's MAE, but it would also muddle the question the project is trying to answer.

**Options considered.**

| Option | What it answers | What it costs |
|---|---|---|
| Ensemble (weighted average) | "Lowest MAE possible" | Loses the per-architecture comparison |
| Stacking (meta-model) | Same as ensemble + meta-risk | Extra CV pass, minimal gain at this sample size |
| Independent comparison (chosen) | "Which architecture wins, and why" | Leaves some accuracy on the table |

**Chosen: independent comparison.** Ridge is the honest baseline (same feature matrix, same scaling, just L2 regression). The NN is the headline "custom architecture" deliverable. LightGBM is the "what could a boosted tree do with the same inputs" reference point. Reporting all three per-position surfaces real trade-offs: Ridge wins on stability and interpretability; NN pulls ahead on WR/TE where interactions matter; LightGBM is competitive where there's enough data and falls apart on K/DST.

**Rejected.** Ensembling was considered and rejected because it would obscure exactly the finding the project is trying to produce. (A future production system would of course blend these — that's a follow-up, not this ADR.)

**References.** [shared/models.py](../shared/models.py) (`RidgeMultiTarget`, `LightGBMMultiTarget`), [shared/neural_net.py](../shared/neural_net.py), [shared/pipeline.py](../shared/pipeline.py). LightGBM added in commit `f343c20`.

---

### D4: Attention over game history (skill positions only)

**Decision.** For QB, RB, WR, TE, replace pure rolling features with a variable-length game-history branch processed by learned-query attention, fused with the standard static feature vector. Skip this for K and DST.

**Context.** Rolling means lose order — "three good games then a bad one" looks identical to "one bad then three good." Attention over the last N games lets the model weight recent games higher, attend more to games pre-injury, and in principle learn role-change signals (backup becomes starter) that a fixed window can't capture.

**Options considered.**

| Option | Signal captured | Sample-efficient? |
|---|---|---|
| Pure rolling features | Mean/variance only | Yes |
| LSTM / Transformer over history | Order + interactions | No (overfits on ~300 rows) |
| Rolling + attention-pool over history (chosen for skill positions) | Both | Marginal |
| Rolling only for K/DST | (N/A) | Yes |

**Chosen: learned-query attention pool for skill positions.** A small attention head (`d_model=32`, 2 heads) is cheap enough not to overfit, with positional encoding for recency. For K/DST, the per-game signal is so noisy and the feature count so small that the attention branch added variance without meaningfully reducing MAE — dropped.

**Rejected.** A full LSTM or Transformer was tried conceptually (see [docs/design_lstm_multihead.md](design_lstm_multihead.md)) but rejected as over-parameterized for this regime. Kept the design doc as an artifact of the consideration.

**References.** [shared/neural_net.py:209-393](../shared/neural_net.py) (`AttentionPool`, `MultiHeadNetWithHistory`, `GatedTDHead`), [docs/design_lstm_multihead.md](design_lstm_multihead.md). Evolved across commits `b31bdf7` → `c399c12` → `99d7086`.

---

### D5: Output-constraint stack

**Decision.** Combine four constraints on NN outputs: (a) Huber loss with per-target deltas, (b) per-head `clamp(min=0)` controlled by a `non_negative_targets` set, (c) a gated TD head that models P(TD>0) and E[TD|TD>0] separately, (d) ±4σ feature clipping after StandardScaler.

**Context.** Fantasy targets have three nasty properties: they're zero-inflated (most players don't score a TD on a given week), non-negative (with one exception — DST `pts_allowed_bonus`, which runs −4 to +10), and have outliers (40+ point games do happen). Vanilla MSE regression with no output bound produces nonsense.

**Options considered.** Rather than a single option table, each constraint has its own rationale, and several replaced earlier bugs:

- **Huber over MSE.** Outlier games dominate MSE gradients. Huber with per-target delta (≈1.5–3.0) caps the penalty.
- **Clamp instead of Softplus.** An earlier version used Softplus on head outputs, which has a floor of `softplus(0) ≈ 0.693`. Across three heads that's a ~2-point floor no player could drop below, and it created a scale mismatch with Ridge's `np.maximum(·, 0)`. Clamp allows exact zeros. (Fixed in commit `fe507e0`.)
- **`non_negative_targets` parameter, not a global clamp.** DST's `pts_allowed_bonus` is legitimately negative when the defense gives up a lot of points. A global clamp broke DST; making the set configurable per-position fixed it.
- **Gated TD head.** TDs are discrete and mostly zero. Binary gate + value head reflects the actual data-generating process. (Added in commit `18170a6`.)
- **±4σ feature clip.** Test-set outliers were producing z-scores up to ~19, sending NN outputs off a cliff. Clipping after scale catches 0.3% of values and prevents catastrophic extrapolation.

**Chosen rationale.** Each constraint was added in response to a specific observed failure, not as a precaution. This ADR captures them together because they form a *coherent* stack — remove any one and a specific failure mode returns.

**References.** [shared/neural_net.py:61-102](../shared/neural_net.py) (clamp + non-negative set), [shared/training.py](../shared/training.py) (`MultiTargetLoss` with Huber), [DST/dst_config.py:76](../DST/dst_config.py) (`DST_NN_NON_NEGATIVE_TARGETS = {"defensive_scoring", "td_points"}` — DST is out of scope for the raw-stat migration and keeps its prior target set), feature clipping in [shared/pipeline.py](../shared/pipeline.py). The `GatedTDHead` is now parameterized over a list of gated targets (`RB` has two: `rushing_tds` + `receiving_tds`; `WR`/`TE` have one: `receiving_tds`; `QB` has none — see D2). See also the "Fixed" section of [TODO.md](../TODO.md) for each bug history.

---

### D6: Explicit per-position feature allowlist

**Decision.** Every position has an explicit `{POS}_INCLUDE_FEATURES` list. The feature engineer computes ~155 features; the trainer uses *only* what's on the allowlist. Adding a feature to training requires changing a config.

**Context.** Feature leakage is the single most common source of "my model works in training and collapses in production" in time-series ML. Opt-out allowlists (compute everything, exclude the bad ones) are easy to get wrong silently — one new feature that accidentally peeks into the current week breaks everything, and nobody notices until deployment.

**Options considered.**

| Option | Leakage resilience | Convenience | Auditability |
|---|---|---|---|
| All features in → trust the builder | Low | High | Low |
| Opt-out blocklist | Medium | Medium | Medium |
| Opt-in allowlist (chosen) | **High** | Lower | High |

**Chosen: opt-in allowlist.** A reviewer can diff a PR and see exactly what features the model sees. Adding a feature is a deliberate act. The inconvenience (a config edit per experiment) is the point — it forces intentionality.

**Rejected.** Opt-out was the earlier pattern and was exactly how the feature-clipping bug and the schedule-features-at-inference bug slipped in. Allowlist refactor landed in commit `18170a6` alongside the gated TD change.

**References.** [QB/qb_config.py](../QB/qb_config.py) (`QB_INCLUDE_FEATURES`), [TE/te_config.py](../TE/te_config.py), [shared/pipeline.py](../shared/pipeline.py). Weather/Vegas features (from [docs/design_weather_and_odds.md](design_weather_and_odds.md)) are opted in per-position through the same mechanism.

---

### D7: EC2 warm instance over Batch/SageMaker

**Decision.** Train on a single warm EC2 g4dn.xlarge driven by CI. Six per-position training containers run in parallel on the instance via SSM commands, invoked by [.github/workflows/train-ec2.yml](../.github/workflows/train-ec2.yml). AWS Batch with Spot is kept as a standby path ([docs/batch_design.md](batch_design.md)), reactivated by setting `BATCH_ACTIVE=true`.

**Context.** Per-position training takes ~2 minutes on a GPU. We went through three iterations: SageMaker first (commit `eedacfc`), then Batch + Spot (`57d52f9` → `ffb3119`), then the current warm-EC2 design ([docs/ec2_design.md](ec2_design.md), landed 2026-04-19). Each pivot was driven by the same realization: a 2-minute training job amplifies cold-start overhead, so eliminating it is worth more than the per-hour savings.

**Options considered.**

| Option | Cold-start | Cost pattern | Operational overhead |
|---|---|---|---|
| Train locally | 0 s | $0 | Blocks laptop ~12 min per full run; no audit trail |
| SageMaker Training Jobs | 3–5 min | $0.53/hr × 6 | Managed, but full cold-start every run |
| AWS Batch + Spot | 30–90 s (with pull-through + SOCI) | $0.16/hr × 6 ≈ $0.03/run | Scales to zero; own the IAM/ECR/queue |
| **EC2 warm instance (chosen)** | ~0 s (already running) | ~$0.53/hr while active, $0 while stopped via idle auto-shutdown | Single host to babysit; SSM is the only control plane |

**Chosen: EC2 warm instance.** The container is pre-pulled; the CUDA drivers are already loaded. `train-ec2.yml` just `aws ec2 start-instances` (no-op if already running) then fans six SSM commands out to the host. Per-run cost is effectively the 2 min of training plus the Actions runtime. An idle auto-shutdown timer ([infra/ec2/auto-shutdown.timer](../infra/ec2/auto-shutdown.timer)) stops the instance after 4 h quiet, bringing the idle cost to zero; the next push pays the start-up tax once and reuses the warm host for the rest of the day.

The commit↔model relationship is now one-to-one: every merge to `main` produces a measured, logged training run. Under Batch, cold-starts dominated observability — the Actions log was mostly "waiting for compute environment."

**Why Batch remains the standby path.** Batch is strictly better when training *dominates* wall time (long jobs) or when we genuinely want $0-idle with no manual stop semantics. For constant fine-tuning on a 2-minute job, the always-on-but-auto-stopped EC2 pattern dominates. We keep the Batch image pipeline live ([.github/workflows/batch-image.yml](../.github/workflows/batch-image.yml)) so switching back is one `BATCH_ACTIVE=true` away.

**Rejected.** SageMaker (`eedacfc` → `57d52f9`): managed overhead without training-time dominance. Kubernetes (GKE/EKS): too much machinery for a single GPU job. Long-lived instance without auto-shutdown: leaves an expensive GPU running unused.

**References.** Active path: [docs/ec2_design.md](ec2_design.md), [infra/ec2/README.md](../infra/ec2/README.md), [.github/workflows/train-ec2.yml](../.github/workflows/train-ec2.yml), [batch/train.py](../batch/train.py) (reused as the in-container entrypoint). Standby path: [docs/batch_design.md](batch_design.md), [batch/launch.py](../batch/launch.py). Commit arc: `eedacfc` (SageMaker) → `57d52f9` (pivot to Batch) → `ffb3119` (final Batch) → `4b96c41` / `deb3cc7` (EC2 wiring) → `ec5ab17` (SSM polling fix).

---

### D8: Two Docker images

**Decision.** Build and deploy two separate Docker images: a slim `python:3.12-slim` image for the Flask inference service (~150 MB) and a `pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime` image for GPU training (~5–6 GB). The heavy image is consumed by the EC2 warm host today (D7) and by AWS Batch on the standby path.

**Context.** Inference runs CPU-only on ECS and does not need CUDA, `torch.cuda.*`, or the pytorch wheel's CUDA libs. Training needs all of them plus `nfl_data_py`, `lightgbm`, and the training scripts. A single image would either bloat inference (slow ECS deploys, higher cold-start) or strip training capability.

**Options considered.**

| Option | Inference image size | Training setup | Ops |
|---|---|---|---|
| One shared image | ~5–6 GB | Easy | Slow ECS deploys |
| Two images (chosen) | 150 MB + 5–6 GB | Explicit split | Two pipelines |
| Multi-stage build | Smaller, but fragile | Complex | Debug-hostile |

**Chosen: two images.** They have different requirements, different deploy cadences, and different failure modes. Keeping them separate means the Flask app can deploy without rebuilding torch, and a training dep bump doesn't ship to prod inference.

The training Dockerfile ([batch/Dockerfile.train](../batch/Dockerfile.train)) uses *explicit* COPYs rather than `COPY . .` to drop the Flask UI, scratch scripts, and analysis notebooks out of the image — see the comments on lines 25–38 of that file.

**Rejected.** Multi-stage builds that share a base were considered but rejected as debug-hostile: when a training run fails on Batch, the fastest debug path is `docker run` the exact training image locally. A multi-stage build obscures that.

**References.** [Dockerfile](../Dockerfile) (Flask), [batch/Dockerfile.train](../batch/Dockerfile.train) (Batch), [.dockerignore](../.dockerignore). Landed in commit `0e814a1`.

---

### D9: Warm training host

**Decision.** Keep a single g4dn.xlarge EC2 instance warm with the training image already pulled and CUDA drivers loaded. Trigger per-push training from CI via SSM `RunCommand`, stream CloudWatch logs back to Actions, and stop the instance after 4 h of inactivity via a systemd timer. The earlier Batch cold-start stack (ECR pull-through + SOCI + aggressive `.dockerignore`) is kept in-repo for the standby path.

**Context.** D7 picked the warm-host pattern; this decision is the implementation. The old Batch design had to fight cold-start (image pull, instance provisioning, Docker startup) because Batch intentionally scales to zero. For a 2-minute training job, every second spent warming up is overhead we pay on every run. Leaving a GPU idle at $0.53/hr is also unacceptable — so the design has to stop the instance when it's genuinely unused.

**Chosen (composed).**

| Component | Mechanism | Why it matters |
|---|---|---|
| Deep Learning AMI (Ubuntu 22.04, PyTorch) | NVIDIA drivers, Docker, SSM agent, ECR credsStore pre-installed | First boot is ~90 s; subsequent starts are ~25 s |
| `ff-training:latest` pre-pulled, cached on root EBS | `docker pull` at user-data time, then again on every `systemctl start` | Container-start from the CI command is ~2 s (image is already present) |
| SSM `RunCommand` as the only control plane | No SSH, no open ingress, IAM-scoped per command | Security: instance has egress-only SG; auditability: every run is a logged SSM invocation |
| Per-position `ff-train` helper on PATH | `train-ec2.yml` fires 6 parallel SSM commands (one per position) | Re-uses the 6-parallel-position pattern from the Batch design unchanged |
| [`auto-shutdown.timer`](../infra/ec2/auto-shutdown.timer) | systemd timer fires every 15 min, stops instance if idle > 4 h | Brings idle cost to zero; next push pays one start-up (~25 s), subsequent pushes are warm |
| [`cloudwatch-agent.json`](../infra/ec2/cloudwatch-agent.json) | Ships `/var/log/ff-train/*.log` to `/ff/training` | Logs survive the instance stop/start cycle |

Net effect on a typical push: if the instance is already warm, training starts within seconds; if it was idle and stopped, the first push eats ~25 s of start-up and every subsequent push that day is warm. The total wall-clock time from `git push` to six `model.tar.gz` in S3 is ~3–5 min, of which ~2 min is actual training.

**Standby path — Batch cold-start stack.** The Batch design used three optimizations to minimize cold-start: ECR pull-through cache (~120 s → ~30 s pull), SOCI lazy loading (container starts before image fully pulled), and aggressive `.dockerignore` + explicit `COPY` in the training Dockerfile (~8 GB → ~5–6 GB image). Full tables in [docs/batch_design.md](batch_design.md). These stay in force on the Batch path and are independent of the EC2 choice — they also help the first EC2 image pull during user-data.

**Rejected.** Dedicated-instance reserved-pricing: commits to 24/7 usage we don't need. Spot on EC2 with no auto-shutdown: interrupts mid-training. On-demand with no auto-shutdown: burns $0.53/hr through idle weekends. Lambda-backed GPU (not generally available at this size): no GPU, and would add a cold-start problem back.

**References.** [docs/ec2_design.md](ec2_design.md), [infra/ec2/launch-instance.sh](../infra/ec2/launch-instance.sh), [infra/ec2/user-data.sh](../infra/ec2/user-data.sh), [infra/ec2/auto-shutdown.sh](../infra/ec2/auto-shutdown.sh), [.github/workflows/train-ec2.yml](../.github/workflows/train-ec2.yml). Standby assets: [batch/build_and_push.sh](../batch/build_and_push.sh), [batch/Dockerfile.train](../batch/Dockerfile.train). Arc: `4145257` → `8a50eec` → `ffb3119` (Batch cold-start stack) → `4b96c41` → `deb3cc7` (EC2 warm-host implementation) → `ec5ab17` (SSM polling fix).

---

### D10: Trunk-based CI/CD with test-gated deploys

**Decision.** All deployments happen from `main`. Three GitHub Actions workflows: [tests.yml](../.github/workflows/tests.yml) runs on every push and PR; [batch-image.yml](../.github/workflows/batch-image.yml) builds and registers a new Batch job definition revision when training code changes; [deploy.yml](../.github/workflows/deploy.yml) builds and pushes the Flask image to ECS. Both deploy workflows gate on `tests.yml` passing.

**Context.** Solo project, short timeline. Branching models designed for teams add ceremony without benefit. What's actually needed is a ratchet: broken code can't reach production, every push is traceable to a green test run, every image is tagged by SHA for rollback.

**Options considered.**

| Option | Ceremony | Rollback | Fit for solo |
|---|---|---|---|
| Trunk-based + test-gated (chosen) | Low | SHA-tagged images | Excellent |
| Env branches (dev/staging/prod) | High | Revert + redeploy | Overkill |
| Manual deploy | None | Manual | Easy to skip tests |

**Chosen: trunk-based + test-gated.** Images are tagged by `${{ github.sha }}`; all historical tags stay in ECR for rollback. Batch job definitions are registered as new *revisions* (never deregistered), so rolling back is "submit a job with definition-name:revision-N-1."

**Rejected.** Environment branches would add a staging deploy with nothing behind it — in a solo project the "prod monitoring" is the dashboard on my laptop. Manual deploys were the original state; replacing them was the point.

**References.** [.github/workflows/tests.yml](../.github/workflows/tests.yml), [batch-image.yml](../.github/workflows/batch-image.yml), [deploy.yml](../.github/workflows/deploy.yml). Landed in commit `ffb3119`.

---

## 4. Cross-Cutting Consequences

**What becomes easier.**
- *Parallel iteration per position.* A change to RB features affects only RB's training job, only RB's models, only RB's tests. D2 (multi-head), D6 (allowlist), and D7 (parallel Batch jobs) compose into position-independent evolution.
- *Reproducible serving.* The Flask image is immutable and SHA-tagged (D10); models are baked in, not pulled at runtime. No "it worked yesterday" class of bugs.
- *Audit trail for leakage.* The allowlist (D6) plus the temporal split (D1) plus the ±4σ clip (D5) means any new feature has to survive three independent checks before it affects training.

**What becomes harder.**
- *Six configuration surfaces instead of one.* Each position has its own config, targets, loss weights. A framework-level change (e.g., a new regularizer) needs propagation to six places. This is a deliberate trade (D2/D6) but real.
- *Training-inference skew.* The Flask app must run the same preprocessing as the training pipeline, or models get zeros for features they were trained on. This happened once already (weather/Vegas features missing at inference, fixed after the fact — see [TODO.md](../TODO.md)).
- *Two images to maintain.* D8 doubles the Dockerfile surface; a requirements bump in one does not automatically propagate to the other.

**What we'll need to revisit.**
- K-position features use cross-season rolling windows (see [TODO.md](../TODO.md) "Open"). Technically a leakage source; currently justified by the specialist-role-stability argument, but worth re-measuring once the 2025 season completes.
- Single-format (PPR) training models. The scoring-format flexibility is only at the fantasy-points *computation* layer; the models themselves are trained on PPR. Retraining per format is straightforward but not automated.
- No lineup-construction layer. Per-player projections alone aren't a DFS product.

---

## 5. Open Issues / Follow-Ups

From [TODO.md](../TODO.md) "Open" section, mapped to decisions:

1. **K cross-season rolling leakage** — related to D1 (temporal split) and D5 (per-position non-negative targets). Requires either collecting more K games or accepting the bias.
2. **PPR-only training** — related to D2 (multi-target). Needs a training-matrix flag for scoring format and re-running six position pipelines.
3. **No lineup optimizer** — out of scope for this ADR; tracked as a follow-up project, not a revision of D1–D10.

---

## 6. References

### Source files by subsystem

- **Data & features:** [src/data/loader.py](../src/data/loader.py), [src/data/split.py](../src/data/split.py), [src/data/preprocessing.py](../src/data/preprocessing.py), [src/features/engineer.py](../src/features/engineer.py), [shared/weather_features.py](../shared/weather_features.py).
- **Models:** [shared/models.py](../shared/models.py) (Ridge, LightGBM, ordinal, two-stage), [shared/neural_net.py](../shared/neural_net.py) (MultiHeadNet, attention, gated TD head), [src/models/baseline.py](../src/models/baseline.py).
- **Training:** [shared/training.py](../shared/training.py) (MultiTargetLoss, trainer, schedulers), [shared/pipeline.py](../shared/pipeline.py) (pipeline orchestrator).
- **Per-position configs:** `QB/qb_config.py`, `RB/rb_config.py`, `WR/wr_config.py`, `TE/te_config.py`, `K/k_config.py`, `DST/dst_config.py`.
- **Serving:** [app.py](../app.py), [Dockerfile](../Dockerfile).
- **Training infra (active, EC2):** [infra/ec2/](../infra/ec2/), [batch/train.py](../batch/train.py), [batch/Dockerfile.train](../batch/Dockerfile.train), [batch/build_and_push.sh](../batch/build_and_push.sh), [.github/workflows/train-ec2.yml](../.github/workflows/train-ec2.yml).
- **Training infra (standby, Batch):** [batch/launch.py](../batch/launch.py), [.github/workflows/batch-image.yml](../.github/workflows/batch-image.yml).
- **CI:** [.github/workflows/tests.yml](../.github/workflows/tests.yml), [train-ec2.yml](../.github/workflows/train-ec2.yml), [batch-image.yml](../.github/workflows/batch-image.yml), [deploy.yml](../.github/workflows/deploy.yml).

### Related design docs

- [instructions/DESIGN_DOC.md](../instructions/DESIGN_DOC.md) — rubric mapping + full repo walkthrough (authoritative for rubric-claimed items).
- [docs/ec2_design.md](ec2_design.md) — warm-host training design (authoritative for D7 active path + D9).
- [docs/batch_design.md](batch_design.md) — Batch cold-start analysis, cost breakdown (authoritative for the D7 standby path).
- [infra/aws/README.md](../infra/aws/README.md) — ECS + ALB + domain runbook (authoritative for D8 serving ops).
- [infra/ec2/README.md](../infra/ec2/README.md) — EC2 warm-host runbook (authoritative for D9 ops).
- [docs/design_weather_and_odds.md](design_weather_and_odds.md) — weather/Vegas feature rationale (folded into D6).
- [docs/design_lstm_multihead.md](design_lstm_multihead.md) — LSTM exploration, kept as artifact of the rejection under D4.
- [docs/design_xgboost_ensemble.md](design_xgboost_ensemble.md) — ensembling consideration, rejected under D3.
- [docs/expert_comparison.md](expert_comparison.md) — benchmark against published projections (evaluation evidence).
- [TODO.md](../TODO.md) — issue log (fixed + open).

### Commit timeline of inflection points

| Commit | Phase | What changed |
|---|---|---|
| `974f00d` | Prototype | Monolithic local pipeline |
| `f400a5c` | Data | Knowledge cutoff set to 2012 (D1) |
| `99d7086` | Modeling | Attention V4 + model consolidation (D2, D4) |
| `fe507e0` | Modeling | Softplus → clamp output constraints (D5) |
| `18170a6` | Modeling | Gated TD head + allowlist refactor (D5, D6) |
| `f343c20` | Modeling | LightGBM added (D3) |
| `eedacfc` | Infra | SageMaker attempt |
| `57d52f9` | Infra | Pivot to AWS Batch (D7) |
| `0e814a1` | Infra | Docker optimization — two images (D8) |
| `4145257`, `8a50eec` | Infra | Batch cold-start stack (D9) |
| `ffb3119` | Infra | CI/CD + test gating + final batch infra (D10) |
