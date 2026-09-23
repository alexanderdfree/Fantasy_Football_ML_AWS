"""Historical WR factorial: four fits per seed, two depth replays per fit.

Run in the diagnostic image based on b9d24f92 through launch_ab. The historical
production pipeline, configuration, scalers and model classes stay unchanged.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from src.analysis.wr_pr1564_inputs import require_batch, validate_numerical_source
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["WR"]
SEEDS = [42, 123, 7]
BASELINE = "r0a0"
MODELS = {"ridge": "Ridge", "nn": "NN", "attn_nn": "Attention NN", "lgbm": "LightGBM"}
KEYS = ["player_id", "season", "week"]
TARGETS = ["receiving_yards", "receiving_tds", "receptions", "fumbles_lost"]
_ROOT = None
_ACTIVE = None


def sha(content):
    return hashlib.sha256(content).hexdigest()


def _inputs():
    global _ROOT
    if _ROOT is not None:
        return _ROOT
    import boto3

    uri = os.environ["FF_WR1564_INPUT_URI"]
    if not uri.startswith("s3://"):
        raise ValueError("The experiment requires an immutable S3 input archive")
    bucket, key = uri[5:].split("/", 1)
    raw = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    if sha(raw) != os.environ["FF_WR1564_INPUT_SHA"]:
        raise ValueError("Input archive hash mismatch")
    root = Path(tempfile.mkdtemp(prefix="wr-pr1564-inputs-"))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError("Unexpected archive member")
            path = root.joinpath(*name.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
    manifest = json.loads((root / "input-manifest.json").read_text())
    if not manifest["depth_reuse_verified"]:
        raise ValueError("Training/validation equality does not permit depth reuse")
    for name, metadata in manifest["files"].items():
        content = (root / name).read_bytes()
        if sha(content) != metadata["sha256"] or len(content) != metadata["bytes"]:
            raise ValueError(f"Input content mismatch: {name}")
    for name, arm in manifest["arms"].items():
        (root / "arms" / name / "data/raw").symlink_to(root / f"raw/source{arm['source']}")
    _ROOT = root
    return root


def _activate(name):
    def configure(cfg):
        global _ACTIVE
        require_batch()
        root = _inputs()
        manifest = json.loads((root / "input-manifest.json").read_text())
        validate_numerical_source(manifest, Path(__file__).resolve().parents[2])
        if (
            os.environ.get("FF_AMP_DTYPE") != "fp32"
            or os.environ.get("FF_CUDA_GRAPH") not in {"0", "false"}
            or os.environ.get("FF_COMPILE", "0") not in {"0", "false"}
            or os.environ.get("FF_AB_STACKED", "0") == "1"
        ):
            raise ValueError("Historical experiment requires eager FP32 with graphs/compile off")
        _ACTIVE = name
        return cfg

    return configure


def _frames(name):
    import pandas as pd

    root = _inputs() / "arms" / name / "data/splits"
    return [pd.read_parquet(root / f"{split}.parquet") for split in ("train", "val", "test")]


def _inject(train, val, test):
    del train, val, test
    # The harness owns this temporary directory and its data symlink. Never
    # replace a real directory or a caller's persistent input root.
    path = Path("data")
    if not path.is_symlink():
        raise ValueError("Expected the legacy harness's private data symlink")
    path.unlink()
    path.symlink_to(_inputs() / "arms" / f"{_ACTIVE}d0" / "data")
    return tuple(_frames(f"{_ACTIVE}d0"))


VARIANTS = [
    Variant(name, cfg_mutator=_activate(name), frame_injector=_inject)
    for name in ("r0a0", "r0a1", "r1a0", "r1a1")
]


def _immutable_put(s3, bucket, key, content):
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=content, IfNoneMatch="*")
    except Exception as error:
        if getattr(error, "response", {}).get("Error", {}).get("Code") not in {
            "PreconditionFailed",
            "412",
        }:
            raise
        existing = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        if existing != content:
            raise ValueError(f"Refusing to replace existing evidence: {key}") from error


def _metrics(frame, mask):
    import numpy as np

    subset = frame.loc[mask]
    result = {"n": len(subset), "models": {}}
    for family, name in MODELS.items():
        error = subset[f"pred_{family}_total"].to_numpy() - subset.actual.to_numpy()
        result["models"][name] = {
            "mae": float(np.abs(error).mean()) if len(error) else None,
            "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else None,
            "bias": float(error.mean()) if len(error) else None,
        }
    return result


def metric_fn(result, position):
    import boto3
    import joblib
    import numpy as np
    import pandas as pd
    import torch

    from src.analysis.artifact_eval import build_test_df_from_artifacts
    from src.features.engineer import get_attn_static_columns
    from src.shared.aggregate_targets import predictions_to_fantasy_points
    from src.shared.comparison_scoring import score_actual_components
    from src.shared.evaluation_cohorts import build_cohorts, ranked_rows, reference_selection
    from src.shared.registry import INFERENCE_REGISTRY
    from src.wr.run_pipeline import CONFIG

    if position != "WR" or not torch.cuda.is_available():
        raise ValueError("This investigation requires WR on AWS CUDA")
    if os.environ.get("FF_AMP_DTYPE") != "fp32" or os.environ.get("FF_AB_STACKED", "0") == "1":
        raise ValueError("Unexpected numerical regime")
    seed = int(torch.initial_seed())
    if seed not in SEEDS:
        raise ValueError(f"Unexpected seed: {seed}")
    root = _inputs()
    truth = pd.read_parquet(root / "truth.parquet").set_index(KEYS)
    canonical = _frames("r1a1d1")
    canonical_prior = [CONFIG["compute_targets_fn"](CONFIG["filter_fn"](f)) for f in canonical[:2]]
    reference = pd.read_parquet(root / "raw/source1/weekly_evaluation_reference_v1.parquet")
    native = result["test_df"].set_index(KEYS)
    predictions = [f"pred_{family}_{target}" for family in MODELS for target in [*TARGETS, "total"]]
    s3 = boto3.client("s3")
    bucket = os.environ["S3_BUCKET"]
    prefix = f"{os.environ.get('FF_AB_S3_PREFIX', 'ab_runs')}/{os.environ['FF_AB_RUN_ID']}"
    manifest = {
        "arm": _ACTIVE,
        "seed": seed,
        "source": os.environ["FF_TRAIN_GIT_SHA"],
        "input_archive_sha256": os.environ["FF_WR1564_INPUT_SHA"],
        "gpu": torch.cuda.get_device_name(0),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "amp": False,
        "cuda_graph": os.environ.get("FF_CUDA_GRAPH"),
        "scoring": "ppr_shared_projected_components",
        "parity": {},
        "replays": {},
    }
    for depth in (0, 1):
        arm_name = f"{_ACTIVE}d{depth}"
        frames = _frames(arm_name)
        replay = build_test_df_from_artifacts(
            "WR", *frames, model_dir="wr/outputs/models", device=torch.device("cuda")
        )
        replay = replay.set_index(KEYS)
        if not replay.index.is_unique or not set(predictions).issubset(replay):
            raise ValueError("Missing model predictions or duplicate player-weeks")
        if not np.isfinite(replay[predictions].to_numpy()).all():
            raise ValueError("Non-finite predictions")
        if depth == 0:
            if not replay.index.equals(native.index):
                raise ValueError("Saved inference changed row identities/order")
            for col in predictions:
                difference = np.abs(replay[col].to_numpy() - native[col].to_numpy())
                manifest["parity"][col] = float(difference.max())
                if not np.allclose(replay[col], native[col], atol=1e-4, rtol=1e-6):
                    raise ValueError(f"Saved-model parity failed: {col}")
        aligned = truth.reindex(replay.index)
        if aligned["common"].isna().any():
            raise ValueError("Prediction outside frozen corrected truth")
        output = aligned[
            [*TARGETS, "common", "depth_old", "depth_new", "availability_old", "availability_new"]
        ].copy()
        output["actual"] = score_actual_components(aligned.reset_index(), "WR").to_numpy()
        if not np.isfinite(output.actual).all():
            raise ValueError("Unavailable corrected actual components")
        for col in predictions:
            output[col] = replay[col]
        for family in MODELS:
            expected = predictions_to_fantasy_points(
                "WR", {t: output[f"pred_{family}_{t}"].to_numpy() for t in TARGETS}
            )
            if not np.allclose(expected, output[f"pred_{family}_total"], atol=1e-6):
                raise ValueError("Raw predictions do not reconstruct scored totals")
        output = output.reset_index()
        common = output.common.astype(bool)
        if common.sum() != 2761:
            raise ValueError("Common evaluation cohort must contain 2761 rows")
        matched = aligned.reset_index().loc[common].copy()
        for col in predictions:
            matched[col] = output.loc[common, col].to_numpy()
        cohorts = build_cohorts("WR", matched, prior_frames=canonical_prior, reference=reference)
        for name in ("elite_top24", "weekly_reference_top24"):
            if cohorts[name]["status"] != "available":
                raise ValueError(f"Required cohort unavailable: {name}")
        ref_mask, _ = reference_selection("WR", output, reference, 24)
        output["weekly_reference_top24"] = ref_mask.to_numpy()
        prior = pd.concat(canonical_prior, ignore_index=True)
        prior["shared_points"] = score_actual_components(prior, "WR")
        means = prior.groupby(["player_id", "season"]).shared_points.mean()
        candidates = output.loc[common].drop_duplicates(["player_id", "season"]).copy()
        candidates["prior"] = means.reindex(
            pd.MultiIndex.from_arrays([candidates.player_id, candidates.season - 1])
        ).to_numpy()
        elite = ranked_rows(candidates, "prior", ["season"], 24)
        output["elite_top24"] = pd.MultiIndex.from_frame(output[["season", "player_id"]]).isin(
            pd.MultiIndex.from_frame(elite[["season", "player_id"]])
        )
        flat_features = CONFIG["get_feature_columns_fn"]()
        registry = INFERENCE_REGISTRY["WR"]
        static_features = (
            registry["attn_static_features"]
            if registry.get("attn_static_from_df")
            else get_attn_static_columns(flat_features, registry["attn_static_features"])
        )
        for filename, features in (
            ("nn_scaler.pkl", flat_features),
            ("attention_nn_scaler.pkl", static_features),
        ):
            scaler = joblib.load(f"wr/outputs/models/{filename}")
            if len(features) != len(scaler.mean_):
                raise ValueError("Scaler feature ordering/size mismatch")
            index = list(features).index("depth_chart_rank")
            raw_rank = replay.depth_chart_rank.to_numpy()
            output[filename + "_depth_z"] = (raw_rank - scaler.mean_[index]) / scaler.scale_[index]
        summaries = {"overall": _metrics(output, common), "added": _metrics(output, ~common)}
        masks = {
            "depth_changed": output.depth_old.ne(output.depth_new),
            "depth_unchanged": output.depth_old.eq(output.depth_new),
            "availability_changed": output.availability_old.ne(output.availability_new),
            "week1": output.week.eq(1),
            "weekly_reference_top24": output.weekly_reference_top24,
            "elite_top24": output.elite_top24,
        }
        for name, mask in masks.items():
            summaries[name] = _metrics(output, common & mask)
        buffer = io.BytesIO()
        output.to_parquet(buffer, index=False)
        raw = buffer.getvalue()
        key = f"{prefix}/rows/{sha(raw)}.parquet"
        _immutable_put(s3, bucket, key, raw)
        manifest["replays"][arm_name] = {
            "rows": {"key": key, "sha256": sha(raw), "bytes": len(raw)},
            "metrics": summaries,
            "cohorts": cohorts,
        }
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    _immutable_put(s3, bucket, f"{prefix}/evidence/{_ACTIVE}-{seed}.json", raw)
    return manifest["replays"][f"{_ACTIVE}d0"]["metrics"]["overall"]["models"]


if __name__ == "__main__":
    ab_main(__spec__.name)
