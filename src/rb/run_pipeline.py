"""End-to-end RB position model pipeline.

The TD-model variant (ridge / two_stage / ordinal / gated_ordinal) is selected
by ``td_model_type`` on ``POSITION_CONFIG`` in :mod:`src.rb.config`; the factory
translates it into the ``two_stage_targets`` / ``classification_targets`` cfg keys.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.rb.config import POSITION_CONFIG
from src.shared.pipeline import run_cv_pipeline, run_pipeline
from src.shared.position_pipeline import build_pipeline_config
from src.shared.run_pipeline_factory import cli_main

CONFIG = build_pipeline_config("RB", POSITION_CONFIG)


def run(train_df=None, val_df=None, test_df=None, seed=42, config=None):
    return run_pipeline("RB", config or CONFIG, train_df, val_df, test_df, seed)


def run_cv(full_df=None, test_df=None, seed=42, config=None):
    return run_cv_pipeline("RB", config or CONFIG, full_df, test_df, seed)


if __name__ == "__main__":
    cli_main(
        position_name="RB",
        default_config=CONFIG,
        run_fn=run,
        run_cv_fn=run_cv,
    )
