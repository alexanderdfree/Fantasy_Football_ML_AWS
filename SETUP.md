# Setup

First-time installation and local-run instructions. Once the environment is up, see the Quick Start section of [README.md](README.md) for everyday commands.

## Prerequisites

- Python 3.12 (the training Dockerfile targets 3.12; local venvs should match)
- `git`
- ~2 GB free disk space for the cached NFL dataset
- (Optional) AWS credentials, only needed for EC2 training or deploying to ECS

## Install

```bash
git clone https://github.com/alexanderdfree/Fantasy_Football_ML_AWS.git
cd Fantasy_Football_ML_AWS

python3.12 -m venv .venv
source .venv/bin/activate

# Core deps (numpy, pandas, sklearn, flask, lightgbm, boto3, …)
pip install -r requirements.txt

# PyTorch (CPU wheel — swap in the CUDA wheel if training locally)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

# Dev/test tooling (pytest, ruff) — only needed for running tests or lint
pip install -r requirements-dev.txt
```

## Apple Silicon (macOS) — optional MPS

The install above uses the **CPU** wheel; on an Apple Silicon Mac the pipeline runs on the CPU
and is byte-identical to CI. To try the Mac's GPU for the attention-NN phase, **opt in** with
`--device mps` (equivalently `FF_DEVICE=mps`) — it is *not* auto-selected: `auto` stays on the CPU
on macOS so benchmark numbers stay comparable to CI/the GPU hosts. MPS only accelerates the NN
(Ridge / ElasticNet / LightGBM stay on the CPU), and for this small model the speedup is unproven,
so measure before relying on it:

```bash
# A/B one position: default (CPU) vs MPS
time python -m src.wr.run_pipeline
time python -m src.wr.run_pipeline --device mps
```

If MPS is meaningfully faster on your Mac, the default can be flipped later in the `auto` branch of
[src/shared/utils.py](src/shared/utils.py). See [AGENTS.md](AGENTS.md)'s *Platform & hardware targets* for the
full per-platform matrix and rationale.

## Windows 11 + NVIDIA GPU install (e.g. RTX 5080)

The block above installs the **CPU** PyTorch wheel. To train/tune locally on an NVIDIA GPU —
Windows 11 with an RTX 5080 (or any Blackwell `sm_120` card), or Linux + NVIDIA — use the
**CUDA 12.8** build instead. It's the same `torch==2.11.0`, just the `cu128` wheel (the `cu126`
wheel the AWS Tesla-T4 path uses tops out at `sm_90`, so Blackwell needs `cu128` specifically).

**Prerequisites**

- Python **3.12** — matches the project; the `cu128` cp312 wheel exists. Install from python.org and tick *"Add python.exe to PATH"*.
- A recent NVIDIA driver (R570+, the GeForce 50-series launch driver or newer). You do **not** need a separate CUDA Toolkit — the pip wheel bundles its own CUDA 12.8 runtime via `nvidia-*` packages.
- `git`.

**Create and activate a venv** (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked by execution policy, run once in this shell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

(`cmd` users: `.venv\Scripts\activate.bat`.)

**Install** the GPU dependency set ([requirements-gpu.txt](requirements-gpu.txt) — the GPU analog of `requirements-dev.txt`, with the `cu128` torch build):

```powershell
pip install -r requirements-gpu.txt
```

To match CI's `uv` path instead, set `UV_INDEX_STRATEGY` first — `$env:UV_INDEX_STRATEGY="unsafe-best-match"` in PowerShell (or `set UV_INDEX_STRATEGY=unsafe-best-match` in `cmd`) — then `uv pip install -r requirements-gpu.txt`. To swap an existing CPU env in place without a full reinstall: `pip install --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`.

**Verify the GPU is visible:**

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → 2.11.0+cu128 12.8 True NVIDIA GeForce RTX 5080
```

**First-time data pull** — the heredoc in the next section is bash-only; on Windows use this cross-shell one-liner instead:

```powershell
python -c "from src.data.loader import load_raw_data; from src.data.preprocessing import preprocess; from src.features.engineer import build_features; from src.data.split import temporal_split; temporal_split(build_features(preprocess(load_raw_data())))"
```

**Run training / tuning** — identical commands to macOS/Linux, run from the repo root.
**Set `$env:OPENBLAS_NUM_THREADS='1'` first — REQUIRED on Windows, not a perf knob:** without it,
local pipeline / benchmark / tuning runs crash partway through a position with a `0xC0000005` access
violation (process exit `-1073741819`, no Python traceback), because concurrent threaded LAPACK `eigh`
calls (the Ridge-PCA alpha CV in [src/shared/pipeline.py](src/shared/pipeline.py)) segfault Windows'
bundled OpenBLAS. Linux OpenBLAS tolerates it, which is why AWS Batch / CI never hit it (the Batch
job-def and `conftest.py` pin the same thread caps — this is the local-dev equivalent). Capping it to
one thread fixes it and only throttles numpy's BLAS — LightGBM uses a separate OpenMP runtime
(`LGBM_N_JOBS`, below), so tree-building still uses every core. Set it once per PowerShell session; it
covers every command below, including the tuners.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"   # REQUIRED on Windows (cmd: set OPENBLAS_NUM_THREADS=1)

python -m src.qb.run_pipeline                       # one position, full pipeline
python -m src.benchmarking.benchmark RB --no-sync   # benchmark one position
python -m src.tuning.tune_nn RB --n-trials 30       # Optuna NN tuning
python -m src.tuning.tune_lgbm RB                   # Optuna LightGBM tuning
```

Each `tune_nn` / `tune_lgbm` run also appends a git-tracked history entry under
`benchmark_history/tuning/` (one JSON per run, `{timestamp}_{git_hash}_{tune_nn,tune_lgbm}.json`) —
commit it to version-control that run's best params, the same way `benchmark_history/` tracks
benchmark runs.

Run from the repo root — the tuners look for a relative `data/raw/` (same as on macOS, not a
Windows-specific quirk). The RTX 5080 is picked up automatically by the `torch.cuda.is_available()`
device check ([src/shared/pipeline.py](src/shared/pipeline.py)); the attention NN trains in FP16,
which is native (and full-throughput) on Blackwell. The remaining sections below (*Run benchmarks*,
*Run tests*, *Lint*) work as written with the same `python -m …` / `pytest` / `ruff` commands. To
get full use of the 9950X3D's cores, see the next subsection.

### Use all 16 cores (performance)

Most parallelism is already wired: Ridge/ElasticNet CV fans its alpha grid across cores via `joblib`
(one thread per alpha — each doing a single-threaded BLAS solve under the REQUIRED
`OPENBLAS_NUM_THREADS=1` above, so the cores stay busy via the alpha fan-out, not multi-threaded
BLAS), and a single position overlaps its CPU classical-ML models with the GPU attention NN
([src/shared/pipeline.py](src/shared/pipeline.py)). Two things to know to fully use the 9950X3D:

**NN training is GPU-bound.** During the attention-NN phase the CPU will look mostly idle — that's
expected; the 5080 is doing the work. The CPU-heavy parts are the classical-ML models and tuning.

**LightGBM is single-threaded by default** — the one lever that matters on a 16-core box.
`_lgbm_n_jobs()` ([src/shared/models.py](src/shared/models.py)) reads the `LGBM_N_JOBS` env var,
defaulting to `1` (a guard for macOS's nested-OpenMP segfault and small Linux-CI boxes — neither
applies to Windows). Set it to your **physical** core count for multithreaded tree-building:

```powershell
$env:LGBM_N_JOBS = "16"     # cmd: set LGBM_N_JOBS=16
```

Use `16` (physical cores), not `32` (logical) — LightGBM tree-building usually regresses under
hyperthreading, and `-1` means "all 32 logical". Quick A/B on your box: time
`python -m src.rb.run_pipeline` with `LGBM_N_JOBS=1` vs `16`.

**Parallel tuning — match the knob to the bottleneck:**

```powershell
# LightGBM tuning is CPU-bound: parallelize TRIALS, keep each trial single-threaded.
$env:LGBM_N_JOBS = "1"
python -m src.tuning.tune_lgbm RB --n-jobs 16

# NN tuning is GPU-bound: trials share the one GPU. --n-jobs default is 2 (fits 16 GB);
# try 3 if VRAM allows — diminishing past that.
python -m src.tuning.tune_nn RB --n-jobs 2
```

Don't stack the two — many parallel LightGBM-tuning trials *and* `LGBM_N_JOBS=16` is 16×16 thread
oversubscription (the `--n-jobs` help flags this). And unset `LGBM_N_JOBS` (or use a separate shell)
when running the full `pytest` suite — with `-n auto` xdist workers a high per-process thread count
oversubscribes the runner.

## WSL2 (Linux on Windows) + RTX 5080

Running the project from WSL2 (Ubuntu on Windows 11) instead of native Windows: the GPU still works
(WSL passes the 5080 through to CUDA), and you get the Linux `bash` toolchain. Two differences from
the native-Windows section above.

**Install — `uv` handles the Python 3.12 + `cu128` wheel.** WSL distros usually ship a newer system
Python (e.g. 3.14), and the project needs 3.12. `uv` fetches the right interpreter for you, so you
don't need a system `python3.12`:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
# Blackwell sm_120 needs the cu128 wheel (the AWS cu126 path tops out at sm_90):
UV_INDEX_STRATEGY=unsafe-best-match uv pip install -r requirements-gpu.txt
uv pip install -r requirements-dev.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → 2.11.0+cu128 12.8 True NVIDIA GeForce RTX 5080
```

**The Windows `OPENBLAS_NUM_THREADS=1` crash does NOT apply here — but still cap BLAS, for speed.**
WSL2 is Linux, so the native-Windows `0xC0000005` LAPACK segfault can't happen. The caps are still
worth setting for a *throughput* reason (not safety): the Ridge/ElasticNet alpha CV fans alphas over
`joblib.Parallel(n_jobs=-1, prefer="threads")`, and if each joblib thread also spawns a BLAS
thread-pool you oversubscribe the 16 cores — the same pathology the AWS Batch job-definition caps fix
addresses (ADR D13). Capping BLAS to 1 lets the outer joblib axis own a clean 16-way fan-out. Source
the helper instead of exporting by hand:

```bash
source scripts/wsl-env.sh                 # OMP/MKL/OPENBLAS/NUMEXPR=1, LGBM_N_JOBS=16
python -m src.qb.run_pipeline             # single position, full pipeline
python -m src.benchmarking.benchmark RB   # benchmark one position

# Tuning — the script prints the reminder, but: unset LGBM_N_JOBS first so the
# Optuna trials (not each LightGBM model) get the cores.
unset LGBM_N_JOBS
python -m src.tuning.tune_lgbm RB --n-jobs 16   # CPU-bound: parallel trials
python -m src.tuning.tune_nn   RB --n-jobs 2    # GPU-bound: 2 trials share the 5080
```

Everything else in the *Use all 16 cores* subsection above applies verbatim — only the shell syntax
differs (`export FOO=bar` / `source` instead of `$env:FOO`).

### Parallel local training — all six positions at once

Train every position concurrently instead of the sequential `benchmark` loop (the runner partitions
the physical cores across positions and rebalances as each finishes). Each position trains once,
writing its `src/{pos}/outputs/` artifacts and one consolidated `benchmark_history/` entry; with AWS
creds present it mirrors that to the website History tab (metrics only). Per-position logs go to
`logs/local-train-<POS>.log`.

```bash
source scripts/wsl-env.sh
scripts/train-local-parallel.sh            # all 6, concurrency autodetected
scripts/train-local-parallel.sh QB RB WR   # a subset
scripts/train-local-parallel.sh -j 4       # cap concurrency
scripts/train-local-parallel.sh --rolling-origin  # walk-forward report
scripts/train-local-parallel.sh --dry-run  # print the core plan, launch nothing
scripts/train-local-parallel.sh --no-sync  # don't mirror to the website
```

## First-time data pull and split

`src.data.loader.load_raw_data()` caches the nflverse pulls to `data/raw/`. `src.features.engineer.build_features()` materialises the ~150 engineered columns (rolling_*, ewma_*, trend_*, prior_season_*, opp_*, contextual, position one-hots) every position's `include_features` whitelist references — without this step every engineered column ends up constant-zero via the silent backfill that used to live in `src/shared/feature_build.py` (now raises `KeyError`). `src.data.split.temporal_split()` writes `train.parquet`, `val.parquet`, `test.parquet` under `data/splits/`. The app and benchmark both read from `data/splits/`, so these must exist before anything else runs.

```bash
python - <<'PY'
from src.data.loader import load_raw_data
from src.data.preprocessing import preprocess
from src.features.engineer import build_features
from src.data.split import temporal_split

df = build_features(preprocess(load_raw_data()))
temporal_split(df)           # writes data/splits/{train,val,test}.parquet
PY
```

First run takes several minutes (downloads ~14 seasons of weekly stats, rosters, schedules, snap counts, injuries, depth charts). Subsequent runs use the parquet cache in `data/raw/` and are near-instant.

## Run the Flask app locally

```bash
python -m src.serving.app
# → http://localhost:5050
```

The dashboard loads pre-trained model artifacts from each position's `src/{pos}/outputs/models/` directory. If a position's models are missing, run the benchmark for that position first (see below) to populate them.

## Run benchmarks

```bash
python -m src.benchmarking.benchmark              # all positions, full comparison
python -m src.benchmarking.benchmark RB           # one position
python -m src.benchmarking.benchmark QB RB WR     # several positions
python -m src.benchmarking.benchmark RB --no-sync # skip the S3 mirror (throwaway run)
```

Each run writes a `{run_id}.json` file under [benchmark_history/](benchmark_history/) with the git SHA, timestamp, and per-position config snapshot (used by CI to track benchmark drift) and refreshes the model artifacts under `src/{pos}/outputs/models/`. Headline results are summarized in the Evaluation section of [README.md](README.md).

**Cloud sync (so local runs appear on the website's History tab).** When `FF_MODEL_S3_BUCKET` is set (with AWS credentials available — e.g. `export FF_MODEL_S3_BUCKET=ff-predictor-training`), each run also mirrors its JSON to `s3://{bucket}/{FF_MODEL_S3_PREFIX:-models}/benchmark_history/`, the same key the serving container syncs at boot — so the run shows up on the History tab at the next container restart. Without the env var the upload is skipped (you'll see a one-line notice and the local JSON is still written); pass `--no-sync` to skip the upload even when a bucket is configured.

## Run tests

```bash
pytest                       # full suite — unit, integration, e2e
pytest -m unit               # fast subset (<1 s per test)
pytest tests/rb/             # just one position's tests
```

The e2e tests read `data/splits/*.parquet`, so the first-time data pull must have been done. Individual markers are defined in [pyproject.toml](pyproject.toml).

## Lint

```bash
ruff check .
ruff format --check .
```

Config is in `[tool.ruff]` in [pyproject.toml](pyproject.toml).

## Train in the cloud (owner only)

Pushes to `main` trigger one of two training workflows, selected by the `BATCH_ACTIVE` repo variable. The default since 2026-05-20 is `true` — Spot fan-out via AWS Batch.

- **`BATCH_ACTIVE=true`** (default) → [.github/workflows/train-batch.yml](.github/workflows/train-batch.yml): fans out six g6.xlarge Spot instances via AWS Batch (one position per host, parallel; ~15 min wall-clock, ~$0.35/run GPU-compute estimate on g6 at the measured ~10 min Batch wait — migrated from g4dn 2026-05-31). One-time provisioning via [infra/batch/setup.sh](infra/batch/setup.sh); operator runbook in [infra/batch/README.md](infra/batch/README.md); full design in [docs/batch_design.md](docs/batch_design.md).
- **`BATCH_ACTIVE=false`** (rollback) → [.github/workflows/train-ec2.yml](.github/workflows/train-ec2.yml): starts the warm g4dn.xlarge OD instance, runs the six position pipelines sequentially via SSM (~120 min wall-clock; one T4 can't fit concurrent NN runs). See [infra/ec2/README.md](infra/ec2/README.md) and [docs/ec2_design.md](docs/ec2_design.md).

Flip with `gh variable set BATCH_ACTIVE --body "true"` or `--body "false"`; the new value takes effect on the next push. `workflow_dispatch` on either workflow bypasses the gate for smoke-testing the inactive path. See D7 + D13 in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the trade-off.

## Sync agent auto-memory across machines (owner only)

Agent memories live outside this public repo and are deliberately **not** committed. [scripts/agent-memory-sync.sh](scripts/agent-memory-sync.sh) syncs the markdown memory trees through the private `ff-predictor-training` S3 bucket, using separate prefixes for each agent:

- Claude Code: `~/.claude/projects/<slug>/memory/` -> `s3://ff-predictor-training/claude-memory/<repo>/memory/`
- Codex: `${CODEX_HOME:-~/.codex}/memories/` -> `s3://ff-predictor-training/codex-memory/<repo>/memories/`

Claude session transcripts/subagent metadata and Codex SQLite runtime files are never synced.

```bash
scripts/agent-memory-sync.sh all status       # dry-run both agents; writes nothing
scripts/claude-memory-sync.sh pull            # S3 -> local Claude memory
scripts/codex-memory-sync.sh pull             # S3 -> local Codex memories
scripts/agent-memory-sync.sh all push         # local -> S3 for both local trees
scripts/agent-memory-sync.sh all push --prune # mirror-delete, opt-in only
```

- **Discipline:** pull at session start, push at session stop. You work one machine at a time, so this keeps the conflict hotspots (`MEMORY.md` / `memory_summary.md`) clean.
- **Separate remotes:** Claude and Codex never share an S3 folder. Override with `FF_CLAUDE_MEMORY_S3_PREFIX` or `FF_CODEX_MEMORY_S3_PREFIX`; the legacy `FF_MEMORY_S3_PREFIX` remains a Claude-only alias. `FF_MEMORY_S3_BUCKET` overrides the bucket for both.
- **Additive by default:** a memory created on the other box is never silently dropped; `--prune` opts into mirror-delete (the bucket has versioning enabled as the recovery net).
- **Credentials:** needs AWS creds (env or `~/.aws/credentials`); it cleanly no-ops when the `aws` CLI or creds are absent, so it is safe in a hook.
- **Full-auto hooks:** tracked Claude and Codex `SessionStart` hooks pull the respective agent memory. Tracked `Stop` hooks push both local memory trees, so a cross-agent memory update made on a machine reaches both S3 prefixes when both local stores exist.
- **Fresh-machine seed:** `bash scripts/bootstrap-claude-wsl.sh --with-memory-sync` does an initial Claude pull while installing Claude global conveniences; `bash scripts/bootstrap-codex-local.sh --with-memory-sync` installs Codex prompt templates and does an initial Codex pull.
- **Durable vs incidental:** this syncs incidental, machine-local recall across your boxes. **Durable, share-worthy project knowledge belongs in version-controlled [AGENTS.md](AGENTS.md)**, the cross-agent source of truth both Claude Code and Codex read.

## Bootstrap Codex local prompts (owner only)

Codex project hooks are version-controlled under [`.codex/`](.codex/) and load when the repo is trusted. After hook changes, open `/hooks` in Codex, review the changed commands, and trust them.

Start local Codex sessions through the fresh-worktree launcher:

```bash
scripts/codex-fresh-worktree.sh
```

It reuses the current directory only when it is already a clean Codex-owned worktree under `${CODEX_HOME:-~/.codex}/worktrees/*/Final-Project`. Otherwise it fetches `origin/main`, creates a fresh `codex/session-<id>` worktree under `${CODEX_HOME:-~/.codex}/worktrees/<id>/Final-Project`, best-effort links ignored `data/raw` and `data/splits` from the main checkout, and starts Codex with `--cd` pointed at the fresh worktree. Useful launcher options: `--force-new`, `--base <ref>`, `--branch <name>`, `--no-fetch`, and `--print-path`.

Codex custom slash prompts are local-user files, not repo-scoped files. The repo keeps templates in [`.codex/prompts/`](.codex/prompts/); install or refresh the actual `~/.codex/prompts/*.md` copies with:

```bash
scripts/bootstrap-codex-local.sh
```

Restart Codex after installing prompts. They appear as `/prompts:pre-pr-judge`, `/prompts:pre-pr-gate`, `/prompts:post-pr-followup`, `/prompts:post-session-critique`, and `/prompts:solve-issues`.

Known difference from Claude Code: Codex `SessionStart` hooks can add model-visible context, but they cannot persist environment exports like Claude's `CLAUDE_ENV_FILE` flow or move the active session into a new worktree. Use the normal setup instructions above for the Python environment; the Codex pre-PR hook will probe the main worktree's `.venv` when the current worktree lacks one.
