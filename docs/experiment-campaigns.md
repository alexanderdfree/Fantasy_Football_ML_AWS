# Experiment campaigns

A campaign runs A/B grids, NN searches, LightGBM searches and fresh training
benchmarks through the existing workload implementations. Copy
`examples/campaigns/smoke.json`, choose a unique `id`, then change its steps.
The example deliberately uses one trial and two NN epochs to validate execution;
remove those limits for production comparisons.

```sh
# Validate and inspect the resolved work without allocating anything.
python -m src.tuning.campaign --file my-campaign.json --backend batch --dry-run

# Local: copy a completed, sealed producer build into campaign-owned inputs.
python -m src.tuning.campaign --file my-campaign.json --backend local --data-dir data

# Batch: first build the selected branch using batch-image.yml.
python -m src.tuning.campaign --file my-campaign.json --backend batch \
  --image-sha FULL_COMMIT_SHA --wait

# Use the same file, backend and --fresh setting when resuming.
python -m src.tuning.campaign --file my-campaign.json --backend batch --resume --wait
python -m src.tuning.campaign --file my-campaign.json --backend batch --status
```

`experiment-campaign.yml` exposes the same Batch runner as a dispatch-only
workflow. It writes results to `campaign_runs/<id>/` and never promotes a model,
changes a production configuration, or starts an always-on pool. The workflow
does not build images: dispatch `batch-image.yml` on the experiment branch first.

The manifest freezes resolved A/B variants/seeds, source bytes, execution
overrides, the image digest and one compatible immutable data release. Local
data must have been produced and sealed by the current data builder. An explicit
`dataset_id` can select a privately staged release; producer compatibility is
still required. Resume rejects changed source, data or campaign settings.

Batch groups steps into one job per position and resource type. A/B, NN tuning
and benchmark steps normally share a GPU allocation; LightGBM uses the CPU
queue. `options.device: "cpu"` routes another workload to CPU. Steps execute in
their listed order within each allocation, in fresh subprocesses. Different
positions and CPU/GPU allocations run independently. Tune results are reported,
not automatically applied to later steps. Local execution retains the existing
parallel grid and core-pool decisions inside each step.

Every step has `id`, `kind`, `positions`, optional `options`, and optional `env`.
A/B steps also require a `src.tuning.*` spec. Supported options are:

| Kind | Options |
| --- | --- |
| `ab` | `seeds`, `only`, `jobs`, `stacked_seeds` (boolean), `stacked_epochs`, `feature_cache`, `device`, `max_cells` |
| `nn_tune` | `seed`, `n_trials`, `timeout`, `n_jobs`, `parallel_backend`, `scope`, `stacked_seeds` (0 or at least 2), `stacked_epochs`, `device` |
| `lgbm_tune` | `seeds`, `n_trials`, `timeout`, `n_jobs` |
| `benchmark` | `seed`, `jobs`, `rolling_origin`, `significance`, `device` |

Omitted options retain workload defaults. Batch NN tuning retains its GPU
profile, with hardware-gated stacking and separate eager/stacked study names.
Tuner objectives remain the current validation PPR RMSE objectives. History-scope
tuning remains limited to QB/RB/WR/TE. A/B Batch grids default to a 120-cell cap;
larger intentional grids must specify `max_cells`.

Allowed `env` entries are numerical `FF_*` overrides. Credentials, dispatch
switches and managed output/data/identity variables are rejected. The parent
process cannot inject new numerical flags into a resumed step. Explicit CUDA or
MPS requests fail if the worker cannot use that device.

Completed work is verified before skipping it. A/B cells or whole stacked groups
and benchmark folds checkpoint as they complete. SQLite studies use SQLite's
backup API, including committed WAL state, plus periodic and SIGTERM snapshots
on Batch. Trial budgets count attempts, including failed/interrupted trials;
timeouts charge active search time across retries, excluding queue time. An
exhausted search with no successful trial remains failed and needs a new,
deliberately budgeted campaign.

`--fresh` bypasses the exact fit cache for newly executed A/B work. It does not
discard completed campaign checkpoints. Benchmarks always perform fresh fits
for missing cells. Use a new campaign ID to repeat an entire experiment.

A concurrent controller cannot submit the same attempt twice. Transport failures
with an uncertain submission outcome require reconciliation by the recorded job
name; they never trigger a blind duplicate. `--resume` reattaches active jobs and
retries failed allocations. Ordinary failed steps do not discard independent
later work. Reports and logs remain under the campaign's S3 prefix; the local
controller keeps its manifest and job receipts under `.cache/campaigns/<id>/`.
