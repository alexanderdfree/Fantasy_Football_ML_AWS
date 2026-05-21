"""End-to-end QB position model pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.qb.config import POSITION_CONFIG
from src.shared.pipeline import run_cv_pipeline, run_pipeline
from src.shared.position_pipeline import build_pipeline_config
from src.shared.run_pipeline_factory import cli_main

CONFIG = build_pipeline_config("QB", POSITION_CONFIG)


def run(train_df=None, val_df=None, test_df=None, seed=42, config=None):
    return run_pipeline("QB", config or CONFIG, train_df, val_df, test_df, seed)


def run_cv(full_df=None, test_df=None, seed=42, config=None):
    return run_cv_pipeline("QB", config or CONFIG, full_df, test_df, seed)


if __name__ == "__main__":
    cli_main(
        position_name="QB",
        default_config=CONFIG,
        run_fn=run,
        run_cv_fn=run_cv,
    )
