# Attribution

_Last verified: 2026-04-21._

Sources, libraries, and AI-tool usage for this project.

## Data

- **[nfl_data_py](https://github.com/nflverse/nfl_data_py)** 0.3.3 (MIT, wrapper for [nflverse](https://github.com/nflverse)) — weekly player stats, rosters, schedules, snap counts, injuries, depth charts for seasons 2012–2025. Used for every target and nearly every feature. Cached under `data/raw/` via [src/data/loader.py](src/data/loader.py). 2025+ weekly stats come from nflverse `stats_player` directly (nfl_data_py lags a season).
- **Vegas odds + weather features** — joined in [shared/weather_features.py](shared/weather_features.py). Source and feature rationale are documented in [docs/design_weather_and_odds.md](docs/design_weather_and_odds.md).

No data was manually labelled or scraped.

## Third-party libraries

Canonical version pins live in [requirements.txt](requirements.txt) (serving), [batch/requirements.txt](batch/requirements.txt) (training), [requirements-dev.txt](requirements-dev.txt) (torch + tooling), plus image-level pins in [Dockerfile](Dockerfile) and [batch/Dockerfile.train](batch/Dockerfile.train). Key choices:

| Library | Version | Used for | Why this one |
|---|---|---|---|
| PyTorch | 2.11.0 | `MultiHeadNet`, attention branch, training loop | Needed a framework flexible enough for per-head output constraints and a learned-query attention pool on variable-length history ([shared/neural_net.py](shared/neural_net.py)). CUDA 12.6 runtime base image on EC2 training ([batch/Dockerfile.train](batch/Dockerfile.train)); CPU-only wheel for local dev and CI. |
| scikit-learn | 1.8.0 | Ridge baseline, `StandardScaler`, metrics | Standard, well-tested; the Ridge baseline's interpretability is a rubric deliverable. |
| LightGBM | 4.6.0 | Gradient-boosted baseline | Fast to train on the ~300-row-per-position regime, and provides a third architecture class for the comparison ([shared/models.py](shared/models.py)). |
| Flask + gunicorn | 3.1.3 / 25.3.0 | Serving dashboard | Matches the CPU-only ECS deploy target; gunicorn for multi-worker WSGI ([app.py](app.py), [Dockerfile](Dockerfile)). |
| boto3 | 1.42.89 | S3 + SSM calls in training orchestration | Required by AWS SDK; used in [batch/launch.py](batch/launch.py) and [batch/train.py](batch/train.py). |
| pandas / numpy / pyarrow | 3.0.2 / 2.4.4 / 23.0.1 | Data manipulation, parquet I/O | Standard; parquet because every split is cache-friendly and ~10× smaller than CSV. |
| pytest, ruff | 9.0.3 / 0.15.0 | Test runner, lint | Only in [requirements-dev.txt](requirements-dev.txt). Config in [pyproject.toml](pyproject.toml). |

## AI development tools

Claude Code (Anthropic) was used throughout this project. A substantive account:

### What was AI-generated

_TODO — Alex to fill in with specifics. Suggested framing: list which concrete files or sections were initial-drafted by Claude, e.g. "boilerplate pytest fixtures in `tests/`", "scaffolding for each `{POS}/run_{pos}_pipeline.py`", "the initial `docs/ec2_design.md` cloud-init script", "AWS CLI snippets in `docs/` and `infra/ec2/`", etc. Be honest — graders specifically want to know what was generated vs modified vs written from scratch._

### What was substantially modified after generation

_TODO — Alex to fill in. Suggested framing: list places where AI-generated code was the starting point but required meaningful rework to match the project's constraints. Candidates from the commit history: the `non_negative_targets` mechanism in `MultiHeadNet` (generated code assumed a global clamp), the weather/Vegas feature join (required reasoning about how to handle bye weeks), the EC2 `user-data.sh` (required debugging the credsStore path), the benchmark harness (had to be reshaped around per-position configs)._

### What Alex had to debug, fix, or substantially rework

Concrete examples documented in [TODO.md](TODO.md) under the "Fixed" archive — each entry captures the lesson Alex took from that debug cycle. Notable ones:

- **Softplus near-zero bias inflating low-scoring predictions** — AI-generated NN applied `softplus` to every head output unconditionally. `softplus` approaches 0 only for very negative inputs; at input 0 it's ≈ 0.69, so heads initialized or operating near zero were biased upward and low-end predictions were systematically too high. Alex identified this by noticing the prediction distribution was off, traced it to the output activation, and reworked the head-output gating so non-negativity is applied only where intended (see the `non_negative_targets` entry below). Current head-output logic in [shared/neural_net.py](shared/neural_net.py) (around line 100); see also the first entry in the TODO archive.
- **DST `pts_allowed_bonus` range bug** — the initial non-negativity constraint was applied too broadly and incorrectly constrained DST's `pts_allowed_bonus` head (subsequently retired in the 10-raw-stat DST migration), which legitimately ranged from −4 to +10. Alex added a per-position `non_negative_targets` parameter so only the heads that *should* be non-negative get `softplus`; everything else is linear. The CV path was later caught missing the same kwarg and fixed ([shared/pipeline.py](shared/pipeline.py)).
- **Total-aux-loss / adjustment double-counting** — the aux loss compared `sum(heads)` against `fantasy_points` (which already folded in INT/fumble penalties), and then inference added the adjustment *again*. Net: QB predictions were biased by ~1.9 pts/game. Alex traced the full cascade — finding and fixing four dependent sites across the pipeline.
- **Weather/Vegas features missing at inference** — training merged schedule features but serving did not. Models trained on 12 weather/Vegas features received zeros at serving time. Fix in [app.py](app.py) (`merge_schedule_features` call inside `_apply_position_models`, around line 351).
- **No feature clipping after StandardScaler** — test features could produce z-scores up to ~19.5, far outside the training distribution, so NN predictions went wild on a few outliers. The fix (`np.clip(..., -4, 4)` after every `StandardScaler.transform()`) had to be mirrored in both the training pipeline and the serving path — a canonical instance of the training-vs-inference drift CLAUDE.md warns about.

### What Alex kept ownership of

- **Design decisions**: target decomposition per position, feature allowlist per position, the EC2-vs-Batch trade-off, when to add attention, when to *not* ensemble. Rationale lives in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Rubric mapping and claim choices** in [instructions/DESIGN_DOC.md](instructions/DESIGN_DOC.md).
- **Error analysis and evaluation framing** — deciding what the comparison is actually trying to answer, and how to report it.
- **All code review, debug-to-root-cause work, and the final say on any change that lands.**

### How this is tracked

- Git history is unedited — every commit is Alex's commit, many representing Claude-assisted drafting followed by Alex's review and rework. Recent commit messages like `refactor(attn): per-position whitelist for attention static features` and `perf(training): sub-phase timing + parallel Ridge + pin_memory` show the kind of verification-driven iteration that defined the collaboration.
- The [TODO.md](TODO.md) "Fixed" archive is itself an artifact of this workflow: every "Lesson" entry captures something Alex learned by debugging an AI-proposed implementation down to root cause.
