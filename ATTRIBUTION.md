# Attribution

Sources, libraries, and AI-tool usage for this project.

## Data

- **[nfl_data_py](https://github.com/nflverse/nfl_data_py)** (MIT, wrapper for [nflverse](https://github.com/nflverse)) — weekly player stats, rosters, schedules, snap counts, injuries, depth charts for seasons 2012–2025. Used for every target and nearly every feature. Cached under `data/raw/` via [src/data/loader.py](src/data/loader.py).
- **Vegas odds + weather features** — joined in [shared/weather_features.py](shared/weather_features.py). Source and feature rationale are documented in [docs/design_weather_and_odds.md](docs/design_weather_and_odds.md).

No data was manually labelled or scraped.

## Third-party libraries

Canonical list: [requirements.txt](requirements.txt) (serving) and [batch/requirements.txt](batch/requirements.txt) (training). Key choices:

| Library | Used for | Why this one |
|---|---|---|
| PyTorch 2.11 | `MultiHeadNet`, attention branch, training loop | Needed a framework flexible enough for per-head output constraints and a learned-query attention pool on variable-length history ([shared/neural_net.py](shared/neural_net.py)). |
| scikit-learn | Ridge baseline, `StandardScaler`, metrics | Standard, well-tested; the Ridge baseline's interpretability is a rubric deliverable. |
| LightGBM 4.6 | Gradient-boosted baseline | Fast to train on the ~300-row-per-position regime, and provides a third architecture class for the comparison ([shared/models.py](shared/models.py)). |
| Flask + gunicorn | Serving dashboard | Matches the CPU-only ECS deploy target; gunicorn for multi-worker WSGI ([app.py](app.py), [Dockerfile](Dockerfile)). |
| boto3 | S3 + SSM calls in training orchestration | Required by AWS SDK; used in [batch/launch.py](batch/launch.py) and [batch/train.py](batch/train.py). |
| pandas 3.0, numpy 2.4, pyarrow 23 | Data manipulation, parquet I/O | Standard; parquet because every split is cache-friendly and ~10× smaller than CSV. |
| pytest, ruff | Test runner, lint | Only in [requirements-dev.txt](requirements-dev.txt). Config in [pyproject.toml](pyproject.toml). |

## AI development tools

Claude Code (Anthropic) was used throughout this project. A substantive account:

### What was AI-generated

_TODO — Alex to fill in with specifics. Suggested framing: list which concrete files or sections were initial-drafted by Claude, e.g. "boilerplate pytest fixtures in `tests/`", "scaffolding for each `{POS}/run_{pos}_pipeline.py`", "the initial `docs/ec2_design.md` cloud-init script", "AWS CLI snippets in `docs/` and `infra/ec2/`", etc. Be honest — graders specifically want to know what was generated vs modified vs written from scratch._

### What was substantially modified after generation

_TODO — Alex to fill in. Suggested framing: list places where AI-generated code was the starting point but required meaningful rework to match the project's constraints. Candidates from the commit history: the `non_negative_targets` mechanism in `MultiHeadNet` (generated code assumed a global clamp), the weather/Vegas feature join (required reasoning about how to handle bye weeks), the EC2 `user-data.sh` (required debugging the credsStore path), the benchmark harness (had to be reshaped around per-position configs)._

### What Alex had to debug, fix, or substantially rework

Concrete examples documented in [TODO.md](TODO.md) under the "Fixed" archive — each entry captures the lesson Alex took from that debug cycle. Notable ones:

- **Softplus floor inflating low-scoring predictions** — AI-generated NN used `softplus` on head outputs, which has a floor of ~0.69. Across three heads this imposed a ~2-point minimum no player could drop below. Alex identified this by noticing the prediction distribution was off, traced it to the activation, and swapped to `torch.clamp(min=0)`. See [shared/neural_net.py:62-70](shared/neural_net.py) and the first entry in the TODO archive.
- **DST `pts_allowed_bonus` range bug** — the clamp fix above then incorrectly clamped DST's `pts_allowed_bonus` head, which legitimately ranges from −4 to +10. Alex added a per-position `non_negative_targets` parameter so only the heads that *should* be non-negative are constrained.
- **Total-aux-loss / adjustment double-counting** — the aux loss compared `sum(heads)` against `fantasy_points` (which already folded in INT/fumble penalties), and then inference added the adjustment *again*. Net: QB predictions were biased by ~1.9 pts/game. Alex traced the full cascade — finding and fixing four dependent sites across the pipeline.
- **Weather/Vegas features missing at inference** — training merged schedule features but serving did not. Models trained on 12 weather/Vegas features received zeros at serving time. Fix in [app.py:310-311](app.py).

### What Alex kept ownership of

- **Design decisions**: target decomposition per position, feature allowlist per position, the EC2-vs-Batch trade-off, when to add attention, when to *not* ensemble. Rationale lives in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Rubric mapping and claim choices** in [instructions/DESIGN_DOC.md](instructions/DESIGN_DOC.md).
- **Error analysis and evaluation framing** — deciding what the comparison is actually trying to answer, and how to report it.
- **All code review, debug-to-root-cause work, and the final say on any change that lands.**

### How this is tracked

- Git history is unedited — every commit is Alex's commit, many representing Claude-assisted drafting followed by Alex's review and rework. Recent commit messages like `fix(batch/tests): mock sync_raw_data and populate model_dir in main() tests` show the kind of verification-driven iteration that defined the collaboration.
- The [TODO.md](TODO.md) "Fixed" archive is itself an artifact of this workflow: every "Lesson" entry captures something Alex learned by debugging an AI-proposed implementation down to root cause.
