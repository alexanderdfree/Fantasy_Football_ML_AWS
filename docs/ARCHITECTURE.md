# ADR-001: Fantasy Football Predictor — Consolidated Architecture

| Status | Date | Author |
|---|---|---|
| Accepted | 2026-04-16 | Alex Free (CS 372) |

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
   - [D7: AWS Batch over SageMaker for parallel training](#d7-aws-batch-over-sagemaker)
   - [D8: Two Docker images (slim Flask, heavy Batch)](#d8-two-docker-images)
   - [D9: Batch cold-start stack](#d9-batch-cold-start-stack)
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
         │ AWS Batch (train)   │   │ Flask app (serve)  │
         │ Dockerfile.train    │   │ Dockerfile (slim)  │
         │ g4dn Spot, parallel │   │ ECS, CPU-only      │
         └─────────────────────┘   └────────────────────┘
                    │                          ▲
                    │   model artifacts → S3 → │
                    └──────────────────────────┘
```

A training run is invoked locally (`python batch/launch.py`): the launcher uploads the split parquets to S3, submits six parallel Batch jobs (one per position), polls to completion, and downloads model tarballs. The Flask service is built separately and deployed to ECS on every push to `main`; it reads pre-baked models and serves projections through a dashboard.

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

**Decision.** Decompose each position's fantasy points into 2–3 *component* targets (e.g., RB: `rushing_floor`, `receiving_floor`, `td_points`). Train one neural network per position with a shared backbone and one head per target; the total is the sum of head outputs.

**Context.** Fantasy points collapse heterogeneous events (a receiving TD is structurally very different from a passing yard) into a single scalar. Decomposing lets each head specialize and lets us apply different loss deltas, regularization, and output constraints per component.

**Options considered.**

| Option | Complexity | Expected MAE | Interpretability |
|---|---|---|---|
| Single-target NN predicting total | Low | Medium | Low |
| Separate NN per target | High | Low (best) but overfits | Medium |
| Shared backbone + per-target heads (chosen) | Medium | Low | Medium |

**Chosen: shared backbone + per-target heads.** The backbone learns position-general features ("is this player healthy? on the field? getting opportunity?"). Heads specialize. A small auxiliary loss on `sum(heads) ≈ total_target` keeps the decomposition honest. With only a few hundred training rows per position, parameter sharing is a meaningful regularizer.

**Rejected.** Single-target models under-fit the structure — every head implicitly has to learn "what is a TD" separately from "what is a rushing yard." Fully separate per-target models would triple parameter count with no offsetting sample-size gain.

**References.** [shared/neural_net.py:38-145](../shared/neural_net.py) (`MultiHeadNet`), [shared/training.py](../shared/training.py) (`MultiTargetLoss`), per-position `{POS}/{pos}_targets.py` (target decomposition formulas), `{POS}/{pos}_config.py` (loss weights). Consolidated in commit `99d7086`.

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

**References.** [shared/neural_net.py:61-102](../shared/neural_net.py) (clamp + non-negative set), [shared/training.py](../shared/training.py) (`MultiTargetLoss` with Huber), [DST/dst_config.py:76](../DST/dst_config.py) (`DST_NN_NON_NEGATIVE_TARGETS = {"defensive_scoring", "td_points"}`), feature clipping in [shared/pipeline.py](../shared/pipeline.py). See also the "Fixed" section of [TODO.md](../TODO.md) for each bug history.

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

### D7: AWS Batch over SageMaker

**Decision.** Train on AWS Batch with g4dn.xlarge Spot instances, six parallel jobs (one per position). Orchestration is a single `boto3`-based Python script.

**Context.** Per-position training takes ~2 minutes on a GPU. A 3–5 minute cold-start from a managed service more than doubles wall time, and we pay for it whether training succeeds or fails.

**Options considered.**

| Option | Cold-start | Cost (6 runs) | Operational overhead |
|---|---|---|---|
| Train locally | 0s | $0 | Blocks laptop ~12 min |
| SageMaker Training Jobs | 3–5 min | $0.53/hr × 6 | Managed, but overkill |
| AWS Batch + Spot (chosen) | 30–90s | $0.16/hr × 6 ≈ $0.03/run | Own the IAM/ECR/queue |
| Kubernetes (GKE/EKS) | Low | Low | Too much for a solo project |

**Chosen: AWS Batch + Spot.** ~70% cost savings vs on-demand, scales to zero when idle, no standing cluster to babysit. The orchestration overhead (IAM roles, ECR repo, job definition, job queue) is front-loaded but stable.

**Rejected.** SageMaker was actually *tried* (commit `eedacfc`) and reverted (`57d52f9`) once it became clear the managed-service overhead wasn't buying anything for a 2-minute job. The reverse lesson: "managed" is the right answer when training dominates, not when startup does.

**References.** [batch/launch.py](../batch/launch.py), [batch/train.py](../batch/train.py), [docs/batch_design.md](batch_design.md), [docs/AWS_DEPLOYMENT_GUIDE.md](AWS_DEPLOYMENT_GUIDE.md). Commit arc: `eedacfc` (SageMaker) → `57d52f9` (pivot to Batch) → `ffb3119` (final batch).

---

### D8: Two Docker images

**Decision.** Build and deploy two separate Docker images: a slim `python:3.12-slim` image for the Flask inference service (~150 MB) and a `pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime` image for Batch training (~5–6 GB).

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

### D9: Batch cold-start stack

**Decision.** Combine three cold-start optimizations: (a) ECR pull-through cache for the pytorch base image, (b) SOCI (Seekable OCI) lazy loading, (c) aggressive `.dockerignore` + explicit COPYs in the training Dockerfile.

**Context.** After picking Batch over SageMaker (D7), the dominant bottleneck became the container pull: the pytorch base is ~7 GB, and Docker Hub throttles multi-region pulls. Left alone, a cold Batch instance spent ~120 s just pulling the image.

**Chosen (all three, composed).**

| Optimization | Before | After | Mechanism |
|---|---|---|---|
| ECR pull-through cache (2c) | 120 s pull | 20–40 s | First pull seeds ECR in-region; subsequent pulls stay inside AWS network |
| SOCI lazy loading (2a) | wait for full pull | ~30–60 s earlier | Container starts before image fully pulled; essential files stream first |
| `.dockerignore` + explicit COPY (2b) | ~8 GB image | ~5–6 GB | Drops Flask UI, tests, docs, benchmarks, scratch scripts from training image |

Net effect: cold start goes from ~180 s to ~60–90 s, which is the difference between "this is slow" and "this is fine."

**Rejected.** Pre-warming a long-lived EC2 instance was considered and rejected — it defeats the "scale to zero when idle" property that made Batch + Spot attractive in the first place (D7).

**References.** [batch/build_and_push.sh](../batch/build_and_push.sh), [batch/Dockerfile.train](../batch/Dockerfile.train), [docs/batch_design.md](batch_design.md) (full cost & timing tables in §"Strategy 2"). Arc: `4145257` → `8a50eec` → `ffb3119`.

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
- **Training infra:** [batch/launch.py](../batch/launch.py), [batch/train.py](../batch/train.py), [batch/Dockerfile.train](../batch/Dockerfile.train), [batch/build_and_push.sh](../batch/build_and_push.sh).
- **CI:** [.github/workflows/tests.yml](../.github/workflows/tests.yml), [batch-image.yml](../.github/workflows/batch-image.yml), [deploy.yml](../.github/workflows/deploy.yml).

### Related design docs

- [instructions/DESIGN_DOC.md](../instructions/DESIGN_DOC.md) — rubric mapping + full repo walkthrough (authoritative for rubric-claimed items).
- [docs/batch_design.md](batch_design.md) — full Batch cold-start analysis, cost breakdown (authoritative for D7 & D9).
- [docs/AWS_DEPLOYMENT_GUIDE.md](AWS_DEPLOYMENT_GUIDE.md) — deployment runbook (authoritative for D7/D8 ops).
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
