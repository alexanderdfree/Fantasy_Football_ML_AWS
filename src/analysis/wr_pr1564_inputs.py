"""Build archived WR interventions on Batch, including scaler-fit fingerprints."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.wr_pr1564_recovery import FIXED_RELEASE, relative_path

BASELINE = "6406cf213cc0a5fbcd1e56e449e3d7f82804dc01"
RECIPE = "b9d24f9259c7fd261ab1a4e77d4212d821726420"
KEYS = ["player_id", "season", "week"]
AVAILABILITY = ["is_top_available", "inherited_opportunity"]
SPLITS = ("train", "val", "test")


def require_batch():
    if not os.environ.get("AWS_BATCH_JOB_ID"):
        raise ValueError("Historical preparation and fitting require AWS Batch, including scalers")


def validate_numerical_source(manifest, root):
    """Reject reuse of the historical inputs with a different numerical recipe."""
    if manifest.get("recipe") != RECIPE or not manifest.get("numerical_files"):
        raise ValueError("Missing historical numerical-source provenance")
    mismatch = []
    for name, expected in manifest["numerical_files"].items():
        path = root / name
        if not path.is_file() or sha(path.read_bytes()) != expected:
            mismatch.append(name)
    if mismatch:
        raise ValueError("Historical numerical source mismatch: " + ", ".join(mismatch[:6]))


def historical_numerical_files(root):
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", RECIPE], cwd=root, text=True
    ).splitlines()
    selected = [
        name
        for name in names
        if name in {"requirements.txt", "src/config.py"}
        or (
            name.endswith(".py")
            and name.startswith(("src/data/", "src/features/", "src/shared/", "src/wr/"))
        )
    ]
    expected = {
        name: sha(subprocess.check_output(["git", "show", f"{RECIPE}:{name}"], cwd=root))
        for name in selected
    }
    validate_numerical_source({"recipe": RECIPE, "numerical_files": expected}, root)
    return expected


def verify_recovered_inputs(recovered):
    """Bind preparation to the receipt and reject changed or unrecorded inputs."""
    raw_receipt = (recovered / "recovery.json").read_bytes()
    receipt = json.loads(raw_receipt)
    records = receipt["records"]
    if (
        not records
        or receipt.get("missing")
        or receipt.get("recovered") != len(records)
        or receipt.get("fixed_release") != FIXED_RELEASE
        or not all(record.get("ok") for record in records)
    ):
        raise ValueError("Historical input recovery is incomplete or uses a different release")
    release = (recovered / "fixed-release-manifest.json").read_bytes()
    if sha(release) != FIXED_RELEASE:
        raise ValueError("Recovered fixed-release manifest checksum mismatch")
    expected = set()
    for record in records:
        name = record["destination"]
        if not name.startswith(("baseline/data/", "fixed/data/")) or name in expected:
            raise ValueError("Unexpected or duplicate recovered input destination")
        expected.add(name)
        path = recovered / relative_path(name)
        content = path.read_bytes()
        if sha(content) != record["sha256"] or len(content) != record["bytes"]:
            raise ValueError(f"Recovered input changed after verification: {name}")
    actual = {
        str(path.relative_to(recovered))
        for arm in ("baseline", "fixed")
        for path in (recovered / arm / "data").rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ValueError("Unrecorded or missing recovered input files")
    return sha(raw_receipt)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def frame_hash(frame):
    return sha(
        json.dumps([(str(c), str(d)) for c, d in frame.dtypes.items()]).encode()
        + pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    )


def array_hash(value):
    value = np.ascontiguousarray(value)
    return sha(str((value.shape, value.dtype)).encode() + value.tobytes())


def legacy_availability():
    """Load the exact original function, not an independently rewritten proxy."""
    from src.features import engineer

    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:src/features/engineer.py"], text=True
    )
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_inheritance_features"
    )
    namespace = dict(engineer.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), BASELINE, "exec"), namespace)
    return namespace[node.name], sha(ast.dump(node).encode())


def depth_lookup(raw, frames):
    depth = pd.read_parquet(raw)
    depth = depth[depth.formation.eq("Offense")].copy()
    depth["depth_team"] = pd.to_numeric(depth.depth_team, errors="coerce")
    depth = depth.rename(columns={"gsis_id": "player_id"})
    lookup = depth.groupby(KEYS).depth_team.min().fillna(-1).clip(-1, 10)
    # Saved split values also bind source-date and join semantics. Their values
    # override the raw lookup; raw depth supplies only added player-weeks.
    saved = pd.concat(frames, ignore_index=True)
    saved = saved[saved.player_id.notna()].drop_duplicates(KEYS).set_index(KEYS)
    lookup = lookup.reindex(lookup.index.union(saved.index))
    lookup.loc[saved.index] = saved.depth_chart_rank
    return lookup


def apply_depth(frame, lookup):
    result = frame.copy()
    mask = result.position.eq("WR") & result.season.ge(2025) & result.player_id.notna()
    keys = pd.MultiIndex.from_frame(result.loc[mask, KEYS])
    result.loc[mask, "depth_chart_rank"] = lookup.reindex(keys).fillna(-1).to_numpy()
    return result


def selected_inventory(left, right, columns):
    a = left.set_index(KEYS, drop=False)
    b = right.set_index(KEYS, drop=False)
    common = a.index.intersection(b.index)
    changed = {}
    for col in columns:
        x = a.loc[common, col].to_numpy(float)
        y = b.loc[common, col].to_numpy(float)
        bad = ~np.isclose(x, y, atol=1e-9, rtol=0, equal_nan=True)
        if bad.any():
            changed[col] = int(bad.sum())
    return {
        "left_rows": len(a),
        "right_rows": len(b),
        "common_rows": len(common),
        "added": [list(key) for key in b.index.difference(a.index)],
        "removed": [list(key) for key in a.index.difference(b.index)],
        "changed": changed,
    }


def build(recovered: Path, output: Path):
    require_batch()
    numerical_files = historical_numerical_files(Path(__file__).resolve().parents[2])
    recovery_sha256 = verify_recovered_inputs(recovered.resolve())
    from sklearn.preprocessing import StandardScaler

    from src.data.loader import load_raw_data
    from src.data.preprocessing import preprocess
    from src.data.release import require_cached_sources
    from src.features.engineer import _build_inheritance_features, build_game_history_arrays
    from src.shared.pipeline import _prepare_position_data_uncached
    from src.wr.run_pipeline import CONFIG

    recovered, output = recovered.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        r: [pd.read_parquet(recovered / r / "data/splits" / f"{s}.parquet") for s in SPLITS]
        for r in ("baseline", "fixed")
    }
    old, function_hash = legacy_availability()
    raw = recovered / "fixed/data/raw"
    rosters = pd.read_parquet(raw / "rosters_weekly_2012_2025.parquet")
    injuries = pd.read_parquet(raw / "injuries_2012_2025.parquet")
    with require_cached_sources(raw):
        context = preprocess(load_raw_data(cache_dir=str(raw)))
    context = context[context.position.eq("WR") & context.season.eq(2012)]
    lookups = [
        depth_lookup(
            recovered / "baseline/data/raw/depth_charts_v2_2012_2025.parquet", frames["baseline"]
        ),
        depth_lookup(raw / "depth_charts_v3_2012_2025.parquet", frames["fixed"]),
    ]
    report = {
        "recipe": RECIPE,
        "numerical_files": numerical_files,
        "recovery_sha256": recovery_sha256,
        "baseline_source": BASELINE,
        "legacy_availability_ast_sha256": function_hash,
        "scope": "Reconstruction from archived production data, not original local CPU output bytes",
        "availability_controls": {},
        "compatibility_adapter": "baseline team_stats marker added without changing numerical columns",
        "context2012_sha256": frame_hash(context),
        "arms": {},
        "files": {},
    }

    def record(path):
        content = path.read_bytes()
        report["files"][str(path.relative_to(output))] = {
            "sha256": sha(content),
            "bytes": len(content),
        }

    prepared = {}
    cwd = Path.cwd()
    for source, r in enumerate(("baseline", "fixed")):
        full = pd.concat(
            [context, *[f[f.position.eq("WR") & f.player_id.notna()] for f in frames[r]]],
            ignore_index=True,
        )
        policies = [
            func(full.copy(), injuries, rosters) for func in (old, _build_inheritance_features)
        ]
        original = full[full.season.ge(2013)]
        check = policies[source].loc[original.index]
        discrepancy = np.abs(check[AVAILABILITY].to_numpy() - original[AVAILABILITY].to_numpy())
        if np.nanmax(discrepancy) > 1e-9:
            raise ValueError(f"{r}: availability does not reproduce archived inputs")
        report["availability_controls"][r] = {
            "rows": len(original),
            "max_abs_difference": float(np.nanmax(discrepancy)),
        }

        raw_out = output / f"raw/source{source}"
        raw_out.mkdir(parents=True, exist_ok=True)
        for path in (recovered / r / "data/raw").glob("*.parquet"):
            destination = raw_out / path.name
            destination.write_bytes(path.read_bytes())
            if path.name == "team_stats_2012_2025.parquet" and source == 0:
                table = pd.read_parquet(destination)
                table["_team_stats_schema_v2"] = True
                table.to_parquet(destination, index=False)
            record(destination)
        # Both arms are scored against one archived pregame reference.
        reference = raw_out / "weekly_evaluation_reference_v1.parquet"
        reference.write_bytes((raw / reference.name).read_bytes())
        record(reference)

        for availability in (0, 1):
            mapping = policies[availability].set_index(KEYS)[AVAILABILITY]
            modified = []
            for frame in frames[r]:
                frame = frame.copy()
                mask = frame.position.eq("WR") & frame.player_id.notna()
                keys = pd.MultiIndex.from_frame(frame.loc[mask, KEYS])
                frame.loc[mask, AVAILABILITY] = mapping.reindex(keys).to_numpy()
                modified.append(frame)
            for depth in (0, 1):
                name = f"r{source}a{availability}d{depth}"
                arm = output / "arms" / name
                (arm / "data/splits").mkdir(parents=True, exist_ok=True)
                link = arm / "data/raw"
                if not link.exists():
                    link.symlink_to(raw_out, target_is_directory=True)
                elif not link.is_symlink() or link.resolve() != raw_out:
                    raise ValueError("Unexpected existing raw directory")
                selected = [apply_depth(frame, lookups[depth]) for frame in modified]
                for split, frame in zip(SPLITS, selected, strict=True):
                    path = arm / "data/splits" / f"{split}.parquet"
                    frame.to_parquet(path, index=False)
                    record(path)
                try:
                    os.chdir(arm)
                    with require_cached_sources(raw_out):
                        values = _prepare_position_data_uncached("WR", CONFIG, *selected)
                finally:
                    os.chdir(cwd)
                xs, ys, dfs, features = values[:3], values[3:6], values[6:9], values[9]
                scaler = StandardScaler().fit(xs[0])
                entry = {
                    "source": source,
                    "availability": availability,
                    "depth": depth,
                    "features": features,
                    "splits": {},
                }
                for split, x, y, frame in zip(SPLITS, xs, ys, dfs, strict=True):
                    hist, mask = build_game_history_arrays(
                        frame,
                        history_stats=CONFIG["attn_history_stats"],
                        max_seq_len=CONFIG["attn_max_seq_len"],
                    )
                    entry["splits"][split] = {
                        "rows": len(frame),
                        "x": array_hash(x),
                        "targets": {k: array_hash(v) for k, v in y.items()},
                        "frame": frame_hash(frame),
                        "history": array_hash(hist),
                        "mask": array_hash(mask),
                    }
                entry["scaler"] = {
                    key: array_hash(getattr(scaler, key)) for key in ("mean_", "scale_", "var_")
                }
                report["arms"][name] = entry
                prepared[name] = dfs

    report["depth_reuse_verified"] = all(
        report["arms"][f"r{r}a{a}d0"][key] == report["arms"][f"r{r}a{a}d1"][key]
        for r in (0, 1)
        for a in (0, 1)
        for key in ("scaler", "features")
    ) and all(
        report["arms"][f"r{r}a{a}d0"]["splits"][split]
        == report["arms"][f"r{r}a{a}d1"]["splits"][split]
        for r in (0, 1)
        for a in (0, 1)
        for split in ("train", "val")
    )
    report["prepared_changes"] = {
        split: selected_inventory(
            prepared["r0a0d0"][i], prepared["r1a1d1"][i], CONFIG["get_feature_columns_fn"]()
        )
        for i, split in enumerate(SPLITS)
    }
    test0, test1 = prepared["r0a0d0"][2], prepared["r1a1d1"][2]
    common = pd.MultiIndex.from_frame(test0[KEYS]).intersection(
        pd.MultiIndex.from_frame(test1[KEYS])
    )
    if len(common) != 2761 or len(test1) != 2768:
        raise ValueError("Historical WR cohort counts changed")
    truth = test1.copy()
    truth["common"] = pd.MultiIndex.from_frame(truth[KEYS]).isin(common)
    for name, field, source in (
        ("depth_old", "depth_chart_rank", "r1a1d0"),
        ("depth_new", "depth_chart_rank", "r1a1d1"),
        ("availability_old", "is_top_available", "r1a0d1"),
        ("availability_new", "is_top_available", "r1a1d1"),
    ):
        lookup = prepared[source][2].set_index(KEYS)[field]
        truth[name] = lookup.reindex(pd.MultiIndex.from_frame(truth[KEYS])).to_numpy()
    truth.to_parquet(output / "truth.parquet", index=False)
    record(output / "truth.parquet")
    (output / "input-manifest.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "depth_reuse_verified": report["depth_reuse_verified"],
                "availability_controls": report["availability_controls"],
                "prepared_rows": {
                    s: {k: v for k, v in x.items() if k.endswith("_rows")}
                    for s, x in report["prepared_changes"].items()
                },
            }
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.recovered, args.output)


if __name__ == "__main__":
    main()
