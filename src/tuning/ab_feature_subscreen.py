"""A/B: within-FAMILY sub-group screen (Stage 2) — zoom into one family's columns.

Stage 1 ([ab_feature_screen.py](ab_feature_screen.py) /
[ab_feature_screen_extended.py](ab_feature_screen_extended.py) /
[ab_feature_screen_k.py](ab_feature_screen_k.py) /
[ab_feature_screen_dst.py](ab_feature_screen_dst.py)) ranks whole families. This
spec resolves the families Stage 1 flags neutral/borderline (or large +
heterogeneous) down to their sub-groups — by stat-root for ``rolling`` /
``prior_season`` (single columns sit below the 0.02 FP noise floor), by
individual column for the smaller families. See
[feature_groups.subfamily_groups](feature_groups.py).

Parametrized by (position, family) via env so ONE spec module serves every cell
and is dispatchable to the Batch fleet (the launcher forwards the env):

    FF_SUBSCREEN_POSITION   position to zoom (default ``RB``)
    FF_SUBSCREEN_FAMILY     family to zoom (default ``rolling``)

It is single-position by construction — sub-group names are derived from that
position's column names, so mixing positions would make a sub-group a no-op on
the position that lacks it and false-trip the Ridge sentinel. ``POSITIONS`` is
pinned to the env position; pass ``--positions`` to launch_ab to match.

Run (local authoring smoke only — training runs on the Batch fleet)::

    FF_SUBSCREEN_POSITION=RB FF_SUBSCREEN_FAMILY=rolling \\
        python -m src.tuning.ab_feature_subscreen --list
    python -m src.tuning.launch_ab --spec src.tuning.ab_feature_subscreen \\
        --positions RB --env FF_SUBSCREEN_FAMILY=rolling --stacked-seeds
"""

from __future__ import annotations

import os

from src.tuning.ab_harness import ab_main
from src.tuning.feature_groups import build_drop_variants, main_effects, subfamily_groups

SUBSCREEN_POSITION = os.environ.get("FF_SUBSCREEN_POSITION", "RB").upper()
SUBSCREEN_FAMILY = os.environ.get("FF_SUBSCREEN_FAMILY", "rolling")

POSITIONS = [SUBSCREEN_POSITION]

_GROUP_COLS = subfamily_groups(SUBSCREEN_POSITION, SUBSCREEN_FAMILY)
if not _GROUP_COLS:
    raise ValueError(
        f"no sub-groups for position={SUBSCREEN_POSITION!r} family={SUBSCREEN_FAMILY!r}; "
        "set FF_SUBSCREEN_POSITION / FF_SUBSCREEN_FAMILY to a populated (position, family)."
    )
SCREENED_FAMILIES: list[str] = list(_GROUP_COLS)  # sub-group names within the family

VARIANTS, ROW_DROPS = build_drop_variants(_GROUP_COLS)
BASELINE = "baseline"
# Disambiguates artifacts/run-ids per (position, family) since the spec is reused.
AB_NAME = f"subscreen_{SUBSCREEN_POSITION}_{SUBSCREEN_FAMILY}"


def feature_main_effects(
    variant_seed_value: dict[str, dict[int, float]],
) -> dict[str, dict[str, float]]:
    """High-minus-low main effect per sub-group (MAE or RMSE values)."""
    return main_effects(variant_seed_value, ROW_DROPS, SCREENED_FAMILIES)


if __name__ == "__main__":
    ab_main(__spec__.name)
