"""A/B: DST feature-GROUP screen — the flat-list analog of the skill core-8
family screen [ab_feature_screen.py](ab_feature_screen.py).

DST has no ``include_features`` category dict; its features are a flat
``all_features`` list (21 specific + 17 contextual = 38). [feature_groups.py](feature_groups.py)
partitions that list into nine named groups (defensive production / DST points /
points-yards allowed / trend, opponent offense, opposing QB, game context,
weather, prior season) — validated exhaustive against
``POSITION_CONFIG.all_features`` at import.

Nine groups → a 9-factor Plackett-Burman 12-run main-effects design. DST cannot
use frame injection (it builds its own splits) or the vmap stacked path
(own-splits ``run()``), so this screen is **cfg-mutator only, eager seeds**. Each
drop variant filters its group's columns out of ``get_feature_columns_fn`` and
``attn_static_features`` and declares ``expect_ridge_identical=False``.

Note: dropping a non-static group (rolling production / allowed / trend / opp_*)
moves Ridge/LightGBM only; the attention static branch reads just the explicit
``attn_static_features`` whitelist (game context / weather / prior season), so
dropping one of those groups moves the attention NN too — the per-model effect
table makes that split visible.

Run (local authoring smoke only — training runs on the Batch CPU/GPU fleet)::

    python -m src.tuning.ab_feature_screen_dst --list
    python -m src.tuning.launch_ab --spec src.tuning.ab_feature_screen_dst --positions DST
"""

from __future__ import annotations

from src.tuning.ab_harness import ab_main
from src.tuning.feature_groups import build_drop_variants, main_effects, special_family_columns

POSITIONS = ["DST"]
SEEDS = [42, 123, 7]  # eager 3-seed default (DST can't stack); bump with --seeds

_GROUP_COLS = special_family_columns("DST")
SCREENED_FAMILIES: list[str] = list(_GROUP_COLS)

VARIANTS, ROW_DROPS = build_drop_variants(_GROUP_COLS)
BASELINE = "baseline"


def feature_main_effects(
    variant_seed_value: dict[str, dict[int, float]],
) -> dict[str, dict[str, float]]:
    """High-minus-low main effect per DST group (MAE or RMSE values)."""
    return main_effects(variant_seed_value, ROW_DROPS, SCREENED_FAMILIES)


if __name__ == "__main__":
    ab_main(__spec__.name)
