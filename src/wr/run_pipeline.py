"""End-to-end WR position model pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.shared.pipeline import run_cv_pipeline, run_pipeline
from src.shared.position_pipeline import build_pipeline_config
from src.wr.config import POSITION_CONFIG

CONFIG = build_pipeline_config("WR", POSITION_CONFIG)


def run(train_df=None, val_df=None, test_df=None, seed=42, config=None):
    return run_pipeline("WR", config or CONFIG, train_df, val_df, test_df, seed)


def run_cv(full_df=None, test_df=None, seed=42, config=None):
    return run_cv_pipeline("WR", config or CONFIG, full_df, test_df, seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true", help="Use expanding-window CV")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use shrunk smoke-test config (from tests/_pipeline_e2e_utils)",
    )
    args = parser.parse_args()
    if args.tiny:
        from tests._pipeline_e2e_utils import build_tiny_config

        config = build_tiny_config("WR")
    else:
        config = CONFIG
    if args.cv:
        run_cv(config=config)
    else:
        run(config=config)
