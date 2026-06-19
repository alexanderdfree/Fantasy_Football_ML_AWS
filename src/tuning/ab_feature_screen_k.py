"""A/B: K (kicker) feature-GROUP screen — the flat-list analog of the skill
core-8 family screen [ab_feature_screen.py](ab_feature_screen.py).

K has no ``include_features`` category dict; its features are a flat
``all_features`` list (12 specific + 7 contextual = 19). [feature_groups.py](feature_groups.py)
partitions that list into six named groups (FG volume / accuracy / distance /
trend, game context, weather) — validated exhaustive against
``POSITION_CONFIG.all_features`` at import, so a config edit that adds a column
fails loudly here rather than leaving it unscreened.

Six groups → a 6-factor Plackett-Burman 12-run main-effects design. K cannot use
frame injection (it builds its own splits inside ``run(seed, config)``) or the
vmap stacked path (nested per-kick trainer), so this screen is **cfg-mutator
only, eager seeds** — the harness runs K cells eager regardless of
``--stacked-seeds``. Each drop variant filters its group's columns out of
``get_feature_columns_fn`` and ``attn_static_features`` and declares
``expect_ridge_identical=False`` (a real drop MUST move Ridge).

Run (local authoring smoke only — training runs on the Batch CPU/GPU fleet)::

    python -m src.tuning.ab_feature_screen_k --list
    python -m src.tuning.launch_ab --spec src.tuning.ab_feature_screen_k --positions K
    python -m src.tuning.ab_feature_screen_k --seeds 42 123 7 --sequential
"""

from __future__ import annotations

from src.tuning.ab_harness import ab_main
from src.tuning.feature_groups import build_drop_variants, main_effects, special_family_columns

POSITIONS = ["K"]
SEEDS = [42, 123, 7]  # eager 3-seed default (K can't stack); bump with --seeds

_GROUP_COLS = special_family_columns("K")
SCREENED_FAMILIES: list[str] = list(_GROUP_COLS)

VARIANTS, ROW_DROPS = build_drop_variants(_GROUP_COLS)
BASELINE = "baseline"


def feature_main_effects(
    variant_seed_value: dict[str, dict[int, float]],
) -> dict[str, dict[str, float]]:
    """High-minus-low main effect per K group (MAE or RMSE values)."""
    return main_effects(variant_seed_value, ROW_DROPS, SCREENED_FAMILIES)


if __name__ == "__main__":
    ab_main(__spec__.name)
