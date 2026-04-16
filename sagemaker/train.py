"""SageMaker training entry point.

SageMaker invokes this as: python train.py --position RB --seed 42

Environment variables set by SageMaker:
  SM_CHANNEL_TRAINING  = /opt/ml/input/data/training/
  SM_MODEL_DIR         = /opt/ml/model/
  SM_NUM_GPUS          = 1
"""
import argparse
import os
import shutil
import sys

# SageMaker extracts source.tar.gz to /opt/ml/code/
sys.path.insert(0, os.environ.get("SAGEMAKER_SUBMIT_DIRECTORY", "/opt/ml/code"))
# Also ensure project root is on path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd


# Position registry: maps name to (module_path, runner_function_name, accepts_dataframes)
POSITIONS = {
    "QB": ("QB.run_qb_pipeline", "run_qb_pipeline", True),
    "RB": ("RB.run_rb_pipeline", "run_rb_pipeline", True),
    "WR": ("WR.run_wr_pipeline", "run_wr_pipeline", True),
    "TE": ("TE.run_te_pipeline", "run_te_pipeline", True),
    "K":  ("K.run_k_pipeline",   "run_k_pipeline",  False),
    "DST": ("DST.run_dst_pipeline", "run_dst_pipeline", False),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", required=True, choices=list(POSITIONS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pos = args.position
    data_dir = os.environ.get("SM_CHANNEL_TRAINING", "data/splits")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    # Override log interval for dense CloudWatch metrics
    log_every = int(os.environ.get("SM_LOG_EVERY", "1"))

    # Patch run_pipeline to inject nn_log_every BEFORE importing runners.
    # Runners use `from shared.pipeline import run_pipeline`, so the patch
    # must be in place before the runner module is first imported.
    import shared.pipeline as pipeline_mod
    _orig_run_pipeline = pipeline_mod.run_pipeline

    def _patched_run_pipeline(position, cfg, *a, **kw):
        cfg["nn_log_every"] = log_every
        return _orig_run_pipeline(position, cfg, *a, **kw)

    pipeline_mod.run_pipeline = _patched_run_pipeline

    mod_path, func_name, accepts_df = POSITIONS[pos]
    mod = __import__(mod_path, fromlist=[func_name])
    run_fn = getattr(mod, func_name)

    if accepts_df:
        # Standard positions: read parquet from SageMaker data channel
        train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
        val_df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
        test_df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
        print(f"Loaded data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        run_fn(train_df, val_df, test_df, seed=args.seed)
    else:
        # K/DST: self-contained data loading
        run_fn(seed=args.seed)

    # Copy model artifacts to SM_MODEL_DIR for SageMaker to upload to S3
    src_model_dir = os.path.join(pos, "outputs", "models")
    if os.path.isdir(src_model_dir):
        print(f"Copying model artifacts from {src_model_dir} to {model_dir}")
        shutil.copytree(src_model_dir, model_dir, dirs_exist_ok=True)
    else:
        print(f"WARNING: No model directory found at {src_model_dir}")


if __name__ == "__main__":
    main()
