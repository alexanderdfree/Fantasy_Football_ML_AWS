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

Claude produced first drafts of the following before I reviewed and reworked them. None of these landed as-is — each had to be edited to fit the project's constraints, but the initial scaffold saved typing time:

- **Per-position pipeline scaffolds** — the six `{POS}/run_{pos}_pipeline.py` entrypoints share a near-identical shape; Claude wrote the first one (RB) and I had it generate the others as templates that I then specialized.
- **Pytest fixtures and boilerplate** — `tests/conftest.py`, the per-position `{POS}/tests/conftest.py` files, and the synthetic-DataFrame builders (`_make_player_games()`, etc.) used across feature tests. I authored the assertions; Claude authored the fixture skeletons.
- **AWS infra glue** — initial drafts of [infra/ec2/launch-instance.sh](infra/ec2/launch-instance.sh), [infra/ec2/user-data.sh](infra/ec2/user-data.sh), the systemd unit files in [infra/ec2/](infra/ec2/), [infra/ec2/cloudwatch-agent.json](infra/ec2/cloudwatch-agent.json), and [infra/aws/bootstrap.sh](infra/aws/bootstrap.sh). I then debugged each against the real AWS environment (see TODO.md "Fixed" archive — multiple ECS deploy bugs traced to drift between these files and live AWS state).
- **Markdown structure** — first-draft skeletons of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (the ADR template, the per-decision "Options considered" table layout) and [docs/expert_comparison.md](docs/expert_comparison.md). I wrote the actual decision content, options, and rejected-alternative reasoning; Claude only laid down the structural scaffold.
- **GitHub Actions YAML** — the initial shapes of [.github/workflows/tests.yml](.github/workflows/tests.yml), [.github/workflows/train-ec2.yml](.github/workflows/train-ec2.yml), [.github/workflows/batch-image.yml](.github/workflows/batch-image.yml), and [.github/workflows/deploy.yml](.github/workflows/deploy.yml) — pytest-shard matrix, SSM-RunCommand polling, ECR push steps. The matrix configurations and the deploy gating logic I tuned by hand after watching CI runs fail in specific ways.
- **Front-end stub** — the initial [templates/index.html](templates/index.html) and [static/css/style.css](static/css/style.css). Visual layout and the side-by-side prediction-comparison card are mine; the surrounding HTML/CSS scaffolding was AI-drafted.

### What was substantially modified after generation

These are places where Claude's first draft was a usable starting point but I had to do meaningful re-design to match the project's actual constraints:

- **`non_negative_targets` mechanism in `MultiHeadNet`** — generated code applied a global `softplus`/`clamp` across every head. That broke DST's `pts_allowed_bonus` (range −4 to +10) and created the softplus floor that biased low-scoring predictions across all positions. I redesigned the constructor to take a per-position `non_negative_targets` set, and switched the activation to `torch.clamp(min=0)` so heads can emit exact zeros (TODO.md "Fixed" archive entries on softplus and DST `pts_allowed_bonus`). The CV pipeline missed the same kwarg later — also caught and fixed (TODO.md entry on `run_cv_pipeline`).
- **Weather/Vegas feature join** in [shared/weather_features.py](shared/weather_features.py) — the AI-generated draft handled the simple case (one row per game) but didn't address bye weeks, mid-season trades, or the training-vs-inference path skew. I reworked the join to be index-preserving, added the `merge_schedule_features()` calls in [app.py](app.py) that the inference path was missing (TODO.md "Fixed" archive entry), and added a leakage test for the result.
- **Benchmark harness** ([benchmark.py](benchmark.py) + [shared/benchmark_utils.py](shared/benchmark_utils.py)) — Claude's first version was a single linear flow; I reshaped it to the per-position registry pattern that lets the EC2 path fan six SSM commands out in parallel and that lets `python benchmark.py {POS}` run a single position locally. The `benchmark_history.json` append-with-config-snapshot pattern was a deliberate addition — every run should be reproducible from the JSON entry alone.
- **Loss-rebalance refactor** — I designed the `w ≈ 2.0/δ` convention (TODO.md:146-150 has the rationale); Claude helped propagate the change across all six position configs and update the corresponding tests. The pairing convention is now documented in [CLAUDE.md](CLAUDE.md) so a future contributor can't accidentally tune one without the other.
- **EC2 `user-data.sh` credsStore path** — Claude's draft assumed the standard Docker credsStore configuration; the Deep Learning AMI ships with a different layout. I debugged the resulting `docker pull` failures and reworked the cloud-init to handle both (commit `ec5ab17` fixed an SSM polling regression that surfaced during the same pass).
- **TODO.md "Fixed" archive structure** — the What → Files → Fix → Lesson template is mine, designed to make each debugging cycle teach future-me something. Claude helped fill in subsequent entries once the template was in place, but every "Lesson" line is mine — those summarize what *I* learned from each cycle, not what Claude inferred.

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
