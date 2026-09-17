"""Compare three selectors on identical QB/WR development trajectories."""

from functools import partial

from src.tuning.ab_harness import Variant, ab_main
from src.tuning.model_default_repair import configure, origin_frames
from src.tuning.model_default_repair import metric_fn as metric_fn

POSITIONS = ["QB", "WR"]
SEEDS = [42, 123, 7]
SUPPORTS_STACKED = False
VARIANTS = [
    Variant(
        "baseline",
        cfg_mutator=partial(configure, arm="baseline", mode="selection"),
        frame_injector=origin_frames,
    )
]


if __name__ == "__main__":
    ab_main("src.tuning.ab_selector_trajectories")
