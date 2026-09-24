"""Verify #1602's data/constructor equivalence without fitting any model.

Run this file with --baseline/--candidate checkout roots, --raw-root containing
the four native DST caches, --manifest for their immutable release manifest,
and a new --output directory. Each checkout runs in its own subprocess and
output directory. Native pandas fills run; estimator/scaler/trainer fitting,
optimizer steps, network requests and feature-cache writes are prohibited.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

INPUTS = (
    "weekly_2012_2025.parquet",
    "schedules_2012_2025.parquet",
    "team_stats_2012_2025.parquet",
    "dst_scoring_pbp_v1_2012_2025.parquet",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: normalize(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, partial):
        return {
            "partial": normalize(value.func),
            "args": normalize(value.args),
            "keywords": normalize(value.keywords),
        }
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(normalize(v) for v in value)
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unrecorded recipe value: {type(value).__name__}")


def forbidden(*args, **kwargs):
    raise RuntimeError(
        "The metric-inert verifier prohibits fitting, optimizer steps and network I/O"
    )


def worker(args):
    # Resolve imports from the selected checkout even when this verifier lives
    # in the candidate. Environment configuration precedes every project import.
    sys.path.insert(0, str(args.worktree))
    os.environ.update(FF_CACHE_DIR=str(args.raw_root), FF_FEATURE_CACHE_DISABLE="1")
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.preprocessing import StandardScaler

    from src.data import nfl_source
    from src.data.release import require_cached_sources
    from src.dst import data as dst_data
    from src.dst.run_pipeline import CONFIG, provide_dataset
    from src.k.config import POSITION_CONFIG
    from src.shared import models
    from src.shared.neural_net import MultiHeadNetWithNestedHistory
    from src.shared.pipeline import _prepare_position_data
    from src.shared.registry import ALL_POSITIONS, get_config, get_inference_spec
    from src.shared.training import MultiHeadTrainer

    manifest = json.loads(args.manifest.read_text())
    inputs = {}
    for name in INPUTS:
        path = args.raw_root / name
        expected = manifest["files"][f"raw/{name}"]
        assert path.stat().st_size == expected["bytes"] and digest(path) == expected["sha256"]
        inputs[name] = expected
    schedules = pd.read_parquet(args.raw_root / INPUTS[1])
    regular = schedules.loc[schedules.game_type.eq("REG")]
    affected = int(regular[["home_score", "away_score"]].isna().any(axis=1).sum())
    with ExitStack() as guards:
        guards.enter_context(require_cached_sources(args.raw_root))
        guards.enter_context(patch("socket.socket.connect", forbidden))
        guards.enter_context(patch.object(StandardScaler, "fit", forbidden))
        guards.enter_context(patch.object(StandardScaler, "partial_fit", forbidden))
        guards.enter_context(patch.object(MultiHeadTrainer, "train", forbidden))
        for cls in set(v for v in vars(models).values() if isinstance(v, type)):
            if "fit" in cls.__dict__:
                guards.enter_context(patch.object(cls, "fit", forbidden))
        for cls in (torch.optim.Optimizer, torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
            guards.enter_context(patch.object(cls, "step", forbidden))
        # Logos are display-only, fetched outside the frozen numerical release,
        # and absent from every feature whitelist. Both arms use empty logos.
        guards.enter_context(patch.object(nfl_source, "teams", lambda: pd.DataFrame()))
        frames = {"dst_built": dst_data.build_data(allow_scoring_fetch=False)}
        dataset = provide_dataset(CONFIG)
        prepared = _prepare_position_data("DST", CONFIG, *dataset.frames)
        arrays = {}
        for split in ("train", "val", "test"):
            frames[f"dst_native_{split}"] = getattr(dataset, split)
            frames[f"dst_prepared_{split}"] = getattr(prepared, split)
            arrays[f"X_{split}"] = getattr(prepared, f"X_{split}")
            arrays.update({f"y_{split}_{k}": v for k, v in getattr(prepared, f"y_{split}").items()})
        inference = {pos: normalize(get_inference_spec(pos)) for pos in ALL_POSITIONS}
        recipes = {pos: normalize(get_config(pos)) for pos in ALL_POSITIONS}
        spec = get_inference_spec("K")
        torch.manual_seed(42)
        net = MultiHeadNetWithNestedHistory(
            static_dim=len(spec["attn_static_features"]),
            kick_dim=len(spec["attn_kick_stats"]),
            target_names=list(spec["targets"]),
            **spec["attn_nn_kwargs_static"],
        ).eval()
        arrays.update(
            {f"k_state_{k}": v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
        )
        batch, games, kicks = 2, spec["attn_max_games"], spec["attn_max_kicks_per_game"]
        with torch.inference_mode():
            predictions = net.predict_numpy(
                np.zeros((batch, len(spec["attn_static_features"])), dtype=np.float32),
                np.zeros((batch, games, kicks, len(spec["attn_kick_stats"])), dtype=np.float32),
                np.ones((batch, games), dtype=bool),
                np.ones((batch, games, kicks), dtype=bool),
                torch.device("cpu"),
                X_game_history=np.zeros(
                    (batch, games, len(spec["attn_history_stats"])), dtype=np.float32
                ),
            )
        arrays.update({f"k_prediction_{k}": v for k, v in predictions.items()})
    for name, expected in inputs.items():
        assert digest(args.raw_root / name) == expected["sha256"], (
            "Input changed during verification"
        )
    result = {
        "frames": frames,
        "arrays": arrays,
        "inference": inference,
        "recipes": recipes,
        "feature_columns": prepared.feature_columns,
        "metadata": {
            "code_sha": subprocess.check_output(
                ["git", "-C", str(args.worktree), "rev-parse", "HEAD"], text=True
            ).strip(),
            "release_id": digest(args.manifest),
            "inputs": inputs,
            "regular_schedule_rows": len(regular),
            "affected_schedule_rows": affected,
            "k_head_hidden_overrides": normalize(POSITION_CONFIG.nn_head_hidden_overrides),
            "runtime": {
                "python": sys.version.split()[0],
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
        },
    }
    with (args.output / "values.pkl").open("wb") as stream:
        pickle.dump(result, stream)


def compare(args):
    import numpy as np
    import pandas as pd

    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    for root in (args.baseline, args.candidate):
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "src/",
            ],
            text=True,
        )
        assert not dirty, f"Source must match its recorded commit: {root}"
    baseline_sha = subprocess.check_output(
        ["git", "-C", str(args.baseline), "rev-parse", "HEAD"], text=True
    ).strip()
    candidate_sha = subprocess.check_output(
        ["git", "-C", str(args.candidate), "rev-parse", "HEAD"], text=True
    ).strip()
    changed = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.candidate),
            "diff",
            "--name-only",
            baseline_sha,
            candidate_sha,
            "--",
            "src/",
        ],
        text=True,
    ).splitlines()
    changed = [name for name in changed if name != "src/analysis/verify_kdst_inert.py"]
    assert set(changed) == {
        "src/dst/data.py",
        "src/prediction/upcoming_special_teams.py",
        "src/shared/registry.py",
    }, "Unexpected production-code differences"
    source_delta = {
        name: {"baseline": digest(args.baseline / name), "candidate": digest(args.candidate / name)}
        for name in changed
    }

    def run_arm(arm):
        path = args.output / arm
        path.mkdir()
        command = [
            sys.executable,
            str(script),
            "--worker",
            "--worktree",
            str(getattr(args, arm)),
            "--raw-root",
            str(args.raw_root),
            "--manifest",
            str(args.manifest),
            "--output",
            str(path),
        ]
        with (path / "run.log").open("w") as log:
            subprocess.run(command, cwd=path, stdout=log, stderr=subprocess.STDOUT, check=True)
        with (path / "values.pkl").open("rb") as stream:
            return pickle.load(stream)

    with ThreadPoolExecutor(max_workers=2) as pool:
        baseline, candidate = list(pool.map(run_arm, ("baseline", "candidate")))
    for key in ("recipes", "inference", "feature_columns"):
        assert baseline[key] == candidate[key], f"{key} differs"
    assert (
        baseline["metadata"]["affected_schedule_rows"]
        == candidate["metadata"]["affected_schedule_rows"]
        == 0
    )
    assert (
        baseline["metadata"]["k_head_hidden_overrides"]
        == candidate["metadata"]["k_head_hidden_overrides"]
        == {}
    )
    frame_receipts = {}
    for name, left in baseline["frames"].items():
        right = candidate["frames"][name]
        pd.testing.assert_frame_equal(left, right, check_exact=True)
        frame_receipts[name] = {
            "rows": len(left),
            "columns": len(left.columns),
            "exact_equal": True,
            "row_values_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(left, index=True).to_numpy().tobytes()
            ).hexdigest(),
            "schema_sha256": hashlib.sha256(
                json.dumps(
                    [[str(column), str(dtype)] for column, dtype in left.dtypes.items()]
                ).encode()
            ).hexdigest(),
        }
    array_receipts = {}
    for name, left in baseline["arrays"].items():
        right = candidate["arrays"][name]
        assert left.dtype == right.dtype and left.shape == right.shape, name
        assert np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes(), name
        array_receipts[name] = {
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(left).tobytes()).hexdigest(),
        }
    receipt = {
        "schema": "kdst-inert-proof/v1",
        "ok": True,
        "verifier_sha256": digest(script),
        "production_source_delta": source_delta,
        "baseline": baseline["metadata"],
        "candidate": candidate["metadata"],
        "frames": frame_receipts,
        "arrays": array_receipts,
        "byte_equal_arrays": len(baseline["arrays"]),
        "dst_feature_columns": list(baseline["feature_columns"]),
        "all_six_training_recipes_equal": True,
        "all_six_inference_specs_equal": True,
        "training_recipes_sha256": hashlib.sha256(
            json.dumps(baseline["recipes"], sort_keys=True).encode()
        ).hexdigest(),
        "inference_specs_sha256": hashlib.sha256(
            json.dumps(baseline["inference"], sort_keys=True).encode()
        ).hexdigest(),
        "method": "Native DST data provider and unscaled preparation, including deterministic pandas fills; seed-42 untrained K constructor/state/forward parity.",
        "limits": "No fitted-model benchmark, accuracy result, scaler fit or GPU claim. Display-only team logos stubbed identically. Proof is specific to this frozen release and current production configs; edge cases are covered separately by regression tests.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"ok": True, "receipt": str(args.output / "receipt.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "worktree", "raw-root", "manifest", "output"):
        parser.add_argument(
            f"--{name}",
            type=lambda p: Path(p).resolve(),
            required=name in {"raw-root", "manifest", "output"},
        )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.worktree is None:
            parser.error("--worker requires --worktree")
        worker(args)
    else:
        if args.baseline is None or args.candidate is None:
            parser.error("--baseline and --candidate are required")
        compare(args)


if __name__ == "__main__":
    main()
