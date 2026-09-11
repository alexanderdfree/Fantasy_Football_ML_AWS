"""Build a complete local training-data release in an unpinned producer directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_training_data() -> dict:
    import pandas as pd

    from src.config import CACHE_DIR, SEASONS
    from src.data.external_sources import _seasons_cache_signature
    from src.data.loader import load_raw_data
    from src.data.preprocessing import preprocess
    from src.data.providers.snapshot import capture_provider_sources
    from src.data.release import (
        prewarm_training_dependencies,
        seal_inputs,
        verify_historical_loader_inputs,
    )
    from src.data.split import temporal_split
    from src.features.engineer import build_features

    with capture_provider_sources(Path(CACHE_DIR) / "provider_sources"):
        raw = load_raw_data()
        verify_historical_loader_inputs(CACHE_DIR, SEASONS)
        signature = _seasons_cache_signature(SEASONS)
        injuries = pd.read_parquet(Path(CACHE_DIR) / f"injuries_{signature}.parquet")
        rosters = pd.read_parquet(Path(CACHE_DIR) / f"rosters_weekly_{signature}.parquet")
        full = build_features(preprocess(raw), injuries_df=injuries, rosters_df=rosters)
        temporal_split(full)
        prewarm_training_dependencies()
    return seal_inputs(raw_dir=CACHE_DIR, repo_root=Path(__file__).resolve().parents[2])


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    manifest = build_training_data()
    print(f"Sealed {len(manifest['files'])} training-data files; no data was uploaded.")


if __name__ == "__main__":
    main()
