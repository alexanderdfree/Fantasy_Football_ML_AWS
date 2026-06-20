"""A/B: Stage-3 CONFIRM — re-run a chosen drop-set together on the PRODUCTION config.

Stage 1 ([ab_feature_screen.py](ab_feature_screen.py) /
[ab_feature_screen_extended.py](ab_feature_screen_extended.py)) ranks whole
families and Stage 2 ([ab_feature_subscreen.py](ab_feature_subscreen.py)) resolves
the flagged ones to sub-groups — both with PCA OFF (raw-feature Ridge) for robust,
PCA-invariant per-group attribution, and both Plackett-Burman main-effects designs
that assume the per-group effects are ADDITIVE. This spec is the final gate: it
re-runs the operator's chosen drop-set **together, as one arm**, on the **production
config (PCA-Ridge ON)** so the decision is faithful (production RB/WR/DST ship
PCA-Ridge) and so an interaction the additive screen assumed away shows up as a
combined effect that diverges from the sum of the per-group main effects.

It is a 2-variant A/B (``baseline`` vs ``drop_confirmed``) built by
[feature_groups.build_confirm_variants](feature_groups.py); the whole drop is one
synthetic group ``confirmed_drop`` so the same ``main_effects`` / report machinery
reports the combined MAE+RMSE effect per model.

Parametrized by (position, drop-columns) via env so ONE spec module serves every
position and is dispatchable to the Batch fleet (the launcher forwards the env):

    FF_CONFIRM_POSITION    position to confirm (required)
    FF_CONFIRM_DROP_COLS   comma-separated column names to drop together (required)

Run high-seed on the production metric path — skill stacks (``--seeds 42..65
--stacked-seeds``; production-faithful Ridge/LGBM, stacked-regime attention), K/DST
eager (``--seeds 42..49``). The driver's ``confirm`` subcommand prints the exact
command::

    FF_CONFIRM_POSITION=RB FF_CONFIRM_DROP_COLS=trend_targets,target_share_L3 \\
        python -m src.tuning.ab_feature_confirm --list
    python -m src.tuning.launch_ab --spec src.tuning.ab_feature_confirm \\
        --positions RB --env FF_CONFIRM_POSITION=RB \\
        --env FF_CONFIRM_DROP_COLS=trend_targets,target_share_L3 \\
        --seeds 42 43 ... 65 --stacked-seeds --max-cells 52
"""

from __future__ import annotations

import os

from src.tuning.ab_harness import ab_main
from src.tuning.feature_groups import build_confirm_variants, main_effects

CONFIRM_POSITION = os.environ.get("FF_CONFIRM_POSITION", "RB").upper()
_DROP_RAW = os.environ.get("FF_CONFIRM_DROP_COLS", "")
DROP_COLS = frozenset(c.strip() for c in _DROP_RAW.split(",") if c.strip())
if not DROP_COLS:
    raise ValueError(
        "FF_CONFIRM_DROP_COLS is empty; set it to a comma-separated list of column "
        "names to drop together (e.g. FF_CONFIRM_DROP_COLS=trend_targets,target_share_L3)."
    )

POSITIONS = [CONFIRM_POSITION]

VARIANTS, ROW_DROPS = build_confirm_variants(DROP_COLS)
BASELINE = "baseline"
# The whole drop is one synthetic group; the report resolves it as a single effect.
SCREENED_FAMILIES: list[str] = ["confirmed_drop"]
# Disambiguates artifacts/run-ids per position since the spec is reused.
AB_NAME = f"confirm_{CONFIRM_POSITION}"


def feature_main_effects(
    variant_seed_value: dict[str, dict[int, float]],
) -> dict[str, dict[str, float]]:
    """Combined drop-set main effect (MAE or RMSE values) on the production config."""
    return main_effects(variant_seed_value, ROW_DROPS, SCREENED_FAMILIES)


if __name__ == "__main__":
    ab_main(__spec__.name)
