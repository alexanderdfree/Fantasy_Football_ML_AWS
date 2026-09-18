# ADR-0029: Synthetic player-history diagnostics

**Status:** Accepted (schema 2: QB forecast context and full-model replay; transforms and other positions tracked below)

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

Schema version 2 supports QB and exports, per case, the attention history and
the forecast game's static context:

- A versioned JSON recipe specifies the seed, case count, history length, allowed
  donor seasons and optional bounds on the donor window's historical points.
  Donors are restricted to `TRAIN_SEASONS`; validation/test seasons are excluded.
  Per-position column groups, validity relations and provenance paths live in
  `src/analysis/synthetic_history_schema.py`, derived from `POSITION_CONFIG`.
- `replay` samples real windows with replacement, retaining their game order.
- `block_bootstrap` samples contiguous blocks with replacement from each selected
  window. All production history fields move together, including QBR, modeled
  opportunity and team game context. Block boundaries are marked. This preserves
  within-game values and within-block order, **not** a complete consistent NFL
  schedule or transitions between blocks. Bye/rest metadata is the donor game's
  context, not recomputed synthetic elapsed time.
- Each candidate has at least N observed games preceding a real forecast key,
  within one player-season; `window: exact` restricts candidates to forecasts
  with exactly N prior games, so every case replays the real window in full.
  Each case records `real_prior_games` and `exact_window`. The forecast game's
  outcomes are excluded from selection and tensors. Sampling is uniform over
  eligible windows, not over players; long seasons contribute more windows.
  Overlapping/repeated windows are dependent, so the case count is not an
  independent statistical sample size.
- PPG means the existing QB projected scoring components, excluding receiving
  and two-point conversions. Bounds select the **original donor window**; a
  bootstrap realization can have a different mean, and both means are recorded.
- Input columns must include the current `POSITION_CONFIG.attn_history_stats`
  and every whitelisted production feature column. Missing columns fail; missing
  external history values survive in parquet and are counted in the manifest,
  while feature columns must be finite (a raw or stale frame is rejected). The
  production history builder applies its usual NaN-to-zero behavior to tensors.
  Missing raw outcomes cannot define a cohort.
- Generated histories include only raw history signals and provenance; cached
  rolling, prior-season and static features never enter a resampled history.
  The production `build_game_history_arrays` builds unscaled tensors and masks
  in newest-first order. Internal ordinal slots are never presented as NFL
  calendar weeks.
- The real forecast game's unscaled production feature row is exported as
  `context.parquet` and held fixed. Its windowed columns and the sequence-coupled
  columns the schema names (`week`, `days_rest`, `season_starts_to_date`,
  `is_returning_from_absence`, `rookie_early`) describe the real prior games,
  not the synthetic history; the manifest records both the list and this policy.
- The consumed-values hash covers the history projection, the recomputed history
  points and every feature column of the selected donor rows, labelled by dtype
  kind so string spellings do not change it; a scoring or whitelist change does
  change it, and the manifest says so.

The manifest's `model_input_readiness` block states, per saved-model family,
whether the artifact can feed it coherently, and the replay re-derives it from
the recipe and the cases rather than trusting the field. The attention NN is
ready in every mode: its static branch is the non-temporal forecast context and
its history branch is the synthetic tensor, which the production model consumes
unscaled. Ridge, the base NN and LightGBM read the forecast row's windowed
rolling/ewma/trend/share features, which describe the real history; they are
ready only for `replay` cohorts whose every case is an exact window, and are
otherwise recorded with the reason (resampled history, or the count of cases
that truncate the real history).

`src/analysis/synthetic_replay.py` replays a cohort against a saved checkpoint
directory (served or producer path, optionally synced from S3) through the
production prediction adapter: it hand-builds the checkpoint's ordered inputs
from `context.parquet` and `history.npz`, applies the checkpoint's fitted scaler
and clip, refuses bundled families from different training generations (the
same provenance, target and fitted-preprocessing gate serving applies), and
records raw per-target responses plus totals for every scoring format in a new
directory. It never re-featurizes synthetic rows and never writes an outcome
column. With the consumed source, an identity control rebuilds the forecast
rows' inputs with the production tensor builder and requires equal static values
for every case in every mode; for `replay` cohorts it also requires the equal
newest-first history prefix and mask, and predictions on exact windows must
match production's whole-frame predictions within a float tolerance (replay
shares production's code path, not its batch, so the largest difference is
recorded rather than assumed zero). The control reports `passed`,
`inputs_only` (no exact window to compare) or `context_only` (bootstrap); a
failed control publishes nothing.

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
| `cases.parquet` | Case index, forecast key, real prior games and exact-window flag, original/generated history averages, unique donor-game counts |
| `context.parquet` | Per case, the forecast game's teams and every unscaled production feature column |
| `history.npz` | Unscaled `history` and Boolean `mask`; load with `allow_pickle=False` |
| `manifest.json` | Recipe, consumed-value/source-file hashes and their scope, implementation hashes, runtime versions, signal order, coverage, per-family readiness and artifact hashes |

Replay a cohort against the served QB checkpoint (`--sync` pulls it from S3 via
`FF_MODEL_S3_BUCKET`; `--model-dir` overrides the resolved directory). The
default family is the attention NN; `all` expands to every bundled family but
skips LightGBM on macOS (libomp), where it must be named explicitly and run on
Linux/Batch instead:

```bash
python -m src.analysis.synthetic_replay \
  --cohort analysis_output/synthetic/qb-replay-001 \
  --output analysis_output/synthetic_replays/qb-replay-001 \
  --families attn_nn ridge nn --sync \
  --source analysis_output/synthetic_sources/qb.parquet
```

The replay directory contains `predictions.parquet` (one row per case: the case
metadata, `pred_{family}_{target}`, `pred_{family}_total` in PPR and
`pred_{family}_total_{half_ppr,standard}`) and `replay_manifest.json` (cohort
manifest and file hashes, recipe, model directory and sync summary, requested
families, per-family bundle ids or file hashes with provenance and
feature-column hash, families excluded with reasons, the identity-control result
with its tolerance and largest prediction difference, response semantics,
runtime versions and code hashes).

Existing output directories are never overwritten. Source row order does not
affect sampling; the recipe, consumed values, code and runtime identify a replay.
Generated datasets stay outside Git. Manifests retain content identity, not a
claim that the supplied source came from the currently deployed data release.

## Alternatives and next slices

Whole-window replay supplies the most faithful control. Block resampling adds
controlled variation while keeping opaque signals paired with their game.
Independent column noise was rejected because it breaks those relationships.
A learned generator introduces another model whose realism must be validated;
it is unnecessary for this initial infrastructure. Reconstructing the windowed
features for resampled histories would require reusable windowed builders in
`src/features/engineer.py` (a retrain-gated shared path); the flat families
therefore stay identity-only until that is worth its cost.

Follow-up sequence (one PR each; the per-position schema registry and the
shared validator are the extension points):

1. **Delivered (schema 2):** full-model replay against saved checkpoints with
   the forecast context, fixed fitted scalers and recorded responses. Production
   feature reconstruction for resampled histories is deliberately out of scope.
2. Explicit transformations for usage, efficiency and role changes. Every
   transformation must enumerate dependent fields, team accounting and a policy
   for QBR/EPA/opportunity signals; none may silently leave stale derived values.
3. Deliberate extreme fixtures such as a sustained 100-point QB, plus paired
   cohort response reports. The donor sampler fails clearly when no qualifying
   window exists; it never relaxes a recipe or invents a correct future score.
4. Position-specific schemas and checks for RB/WR/TE (flat), then DST (team
   identities, opponent-offense stream) and K (nested per-kick history),
   including their distinct loading and scoring paths.

## Changelog

- 2026-09-10: Start versioned QB donor-history generation, provenance, consistency
  checks and production attention-history replay in a draft PR.
- 2026-09-18: Schema 2 exports the forecast game's production feature context,
  publishes per-family model-input readiness derived from exact-window cases,
  adds the `window: exact` recipe guarantee, and replays QB cohorts against
  saved checkpoints with recorded responses, a training-generation coherence
  gate and a tolerance-aware identity control.
