"""End-to-end TE position model pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.shared.pipeline import run_pipeline
from src.shared.position_pipeline import build_pipeline_config
from src.te.config import POSITION_CONFIG

CONFIG = build_pipeline_config("TE", POSITION_CONFIG)


def run(train_df=None, val_df=None, test_df=None, seed=42, config=None):
    return run_pipeline("TE", config or CONFIG, train_df, val_df, test_df, seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use shrunk smoke-test config (from tests/_pipeline_e2e_utils)",
    )
    args = parser.parse_args()
    if args.tiny:
        from tests._pipeline_e2e_utils import build_tiny_config

        config = build_tiny_config("TE")
    else:
        config = CONFIG
    run(config=config)
