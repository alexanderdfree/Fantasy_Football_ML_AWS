# ADR-0029: Synthetic player-history diagnostics

**Status:** Proposed (initial infrastructure; draft PR)

## Context

Aggregate accuracy does not describe how a trained model responds to a sustained
elite history, a change in role, or an extreme player archetype. Reproducible
synthetic cohorts can expose these behaviors, provided the generator preserves
the relationships among raw stats, opportunities, team context and external
signals. Arbitrary edits to one feature can create contradictory inputs.

## Decision

Build an offline donor-based diagnostic layer under `src/analysis/`. Keep its
recipes and artifacts separate from training splits, evaluation cohorts and
`benchmark_history/`. Synthetic responses have no observed future outcome and
must not be reported as forecast accuracy or used as synthetic training labels.

The first implementation supports QB **attention history inputs only**:

- A versioned JSON recipe specifies the seed, case count, history length, allowed
  donor seasons and optional bounds on the donor window's historical points.
  Donors are restricted to `TRAIN_SEASONS`; validation/test seasons are excluded.
- `replay` samples real windows with replacement, retaining their game order.
- `block_bootstrap` samples contiguous blocks with replacement from each selected
  window. All production history fields move together, including QBR, modeled
  opportunity and team game context. Block boundaries are marked. This preserves
  within-game values and within-block order, **not** a complete consistent NFL
  schedule or transitions between blocks. Bye/rest metadata is the donor game's
  context, not recomputed synthetic elapsed time.
- Each candidate has N observed games preceding a real forecast key, within one
  player-season. The forecast game's outcomes are excluded from selection and
  tensors. Sampling is uniform over eligible windows, not over players; long
  seasons contribute more windows. Overlapping/repeated windows are dependent,
  so the case count is not an independent statistical sample size.
- PPG means the existing QB projected scoring components, excluding receiving
  and two-point conversions. Bounds select the **original donor window**; a
  bootstrap realization can have a different mean, and both means are recorded.
- Input columns must include the current `POSITION_CONFIG.attn_history_stats`.
  Missing columns fail; missing external values survive in parquet and are
  counted in the manifest. The production history builder applies its usual
  NaN-to-zero behavior to tensors. Missing raw outcomes cannot define a cohort.
- Output includes only raw history signals and provenance. Cached rolling,
  prior-season and static features are excluded. The production
  `build_game_history_arrays` builds unscaled tensors and masks in newest-first
  order. Internal ordinal slots are never presented as NFL calendar weeks.

Artifacts explicitly carry `full_model_input_ready: false`. These tensors alone
cannot be passed to a full model: checkpoint-specific history preprocessing,
static features, prior-season summaries and any other input branches must also
be supplied coherently. Identity replay is the first positive control.

## Operator workflow

Use the project's configured Python environment. Start from a pinned data release
and its enriched, **unscaled** production QB frame, after target computation and
schedule/team-box-score merges. A bare raw weekly cache or stale split is not a
valid source. For example, the current shared preparation function can export
the training frame without training a model:

```python
from pathlib import Path
import pandas as pd
from src.qb.run_pipeline import CONFIG
from src.shared.pipeline import _prepare_position_data

train = pd.read_parquet("data/splits/train.parquet")
val = pd.read_parquet("data/splits/val.parquet")
prepared = _prepare_position_data("QB", CONFIG, train, val)
Path("analysis_output/synthetic_sources").mkdir(parents=True, exist_ok=True)
prepared[6].to_parquet("analysis_output/synthetic_sources/qb.parquet", index=False)
```

Preparation uses the normal local raw dependencies; hydrate and verify the same
release first (ADR-0026). Generation itself only reads the supplied parquet and
recipe; it does not fetch data, train, invoke serving, or write production paths.

```bash
python -m src.analysis.synthetic_history \
  --source analysis_output/synthetic_sources/qb.parquet \
  --recipe src/analysis/synthetic_history_recipes/qb_replay.json \
  --output analysis_output/synthetic/qb-replay-001

python -m src.analysis.synthetic_history \
  --source analysis_output/synthetic_sources/qb.parquet \
  --recipe src/analysis/synthetic_history_recipes/qb_block_bootstrap.json \
  --output analysis_output/synthetic/qb-bootstrap-001
```

Each new output directory contains:

| File | Contents |
|---|---|
| `games.parquet` | Case/step/block IDs, original donor player/season/week, teams, raw history signals, historical projected-component points |
| `cases.parquet` | Forecast key, original/generated history averages, unique donor-game counts |
| `history.npz` | Unscaled `history` and Boolean `mask`; load with `allow_pickle=False` |
| `manifest.json` | Recipe, consumed-value/source-file hashes, implementation hashes, runtime versions, signal order, coverage and artifact hashes |

Existing output directories are never overwritten. Source row order does not
affect sampling; the recipe, consumed values, code and runtime identify a replay.
Generated datasets stay outside Git. Manifests retain content identity, not a
claim that the supplied source came from the currently deployed data release.

## Alternatives and next slices

Whole-window replay supplies the most faithful control. Block resampling adds
controlled variation while keeping opaque signals paired with their game.
Independent column noise was rejected because it breaks those relationships.
A learned generator introduces another model whose realism must be validated;
it is unnecessary for this initial infrastructure.

Follow-ups, not enabled by this version:

1. Full-model replay against saved checkpoints, with static/context branches,
   fixed fitted scalers, production feature reconstruction and recorded responses.
2. Explicit transformations for usage, efficiency and role changes. Every
   transformation must enumerate dependent fields, team accounting and a policy
   for QBR/EPA/opportunity signals; none may silently leave stale derived values.
3. Deliberate extreme fixtures such as a sustained 100-point QB. The donor
   sampler fails clearly when no qualifying window exists; it never relaxes a
   recipe or invents a correct future score.
4. Position-specific schemas and checks for RB/WR/TE/K/DST, including their
   distinct loading and scoring paths, plus cohort response reporting.

## Changelog

- 2026-09-10: Start versioned QB donor-history generation, provenance, consistency
  checks and production attention-history replay in a draft PR.
