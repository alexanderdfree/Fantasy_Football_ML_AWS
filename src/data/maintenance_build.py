"""The shared historical release producer used by CI and scheduled jobs."""

from __future__ import annotations


def build() -> dict:
    """Build in a clean working directory, preserving the production split policy."""
    from src.config import CACHE_DIR, SEASONS
    from src.data import nfl_source
    from src.data.cache_io import atomic_write_parquet
    from src.data.external_sources import _seasons_cache_signature
    from src.data.loader import load_raw_data
    from src.data.preprocessing import preprocess
    from src.data.release import prewarm_training_dependencies, seal_inputs
    from src.data.split import temporal_split
    from src.features.engineer import build_features

    injuries = nfl_source.injuries(list(SEASONS))
    rosters = nfl_source.rosters_weekly(list(SEASONS))
    signature = _seasons_cache_signature(SEASONS)
    atomic_write_parquet(injuries, f"{CACHE_DIR}/injuries_{signature}.parquet", index=False)
    atomic_write_parquet(rosters, f"{CACHE_DIR}/rosters_weekly_{signature}.parquet", index=False)
    max_weeks = int(rosters.groupby("season")["week"].nunique().max())
    if max_weeks < 5:
        raise ValueError(f"rosters frame is not weekly-granular ({max_weeks} distinct weeks)")
    temporal_split(
        build_features(preprocess(load_raw_data()), injuries_df=injuries, rosters_df=rosters)
    )
    prewarm_training_dependencies()
    return seal_inputs()


if __name__ == "__main__":
    build()
