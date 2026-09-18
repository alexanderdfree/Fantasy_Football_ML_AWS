"""Reproducible donor-based player histories for offline behavioral diagnostics.

The generator resamples whole observed game records so opaque signals (QBR,
opportunity, game context) travel with their box scores, optionally rewrites
them through declared transforms, and exports the real forecast game's unscaled
production feature row as the fixed static context. Together with a saved
checkpoint that is a complete attention-model input; families that read
windowed features are replayable for exact identity cohorts only. Synthetic
histories carry no observed outcome. No source is fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.synthetic_history_schema import (
    POSITION_HISTORY_SCHEMAS,
    PositionHistorySchema,
    position_schema,
)
from src.analysis.synthetic_transforms import OPAQUE_POLICIES, apply_transforms, parse_transform
from src.config import TRAIN_SEASONS
from src.features.engineer import build_game_history_arrays
from src.prediction.bundle import MODEL_FAMILIES, file_digest
from src.shared.aggregate_targets import predictions_to_fantasy_points

SCHEMA_VERSION = 2
KEYS = ["player_id", "season", "week"]
WINDOWS = ("any", "exact")
FLAT_FAMILIES = tuple(family for family in MODEL_FAMILIES if family != "attn_nn")
SAMPLING_IDENTITY_EXCLUDED = ("name", "transforms", "opaque_signal_policy")
RESAMPLED_REASON = (
    "windowed rolling/ewma/trend/share/specific features are not reconstructed from "
    "resampled histories; identity replay only"
)
TRANSFORMED_REASON = (
    "history transformed; the forecast row's windowed features describe the donor history, "
    "not the rewritten one"
)
TRUNCATED_REASON = (
    "{truncated} of {cases} cases truncate the real history (real_prior_games > "
    "history_games); the forecast row's windowed features describe the full real history; "
    "use window: exact"
)
CONTEXT_SEMANTICS = (
    "unscaled production feature row of the real forecast game, held fixed; the "
    "sequence-coupled columns listed in context_sequence_coupled_columns and every windowed "
    "column describe the real prior games, not the synthetic history"
)
STATIC_CONTEXT_POLICY = (
    "non-temporal static features (prior-season, matchup, contextual, weather/vegas) are held "
    "at the donor forecast row's context in context.parquet; transforms never rewrite them"
)
FORECAST_OUTCOME = (
    "none: synthetic histories carry no observed future score; recorded model outputs are "
    "responses, not accuracy"
)
SOURCE_VALUES_SCOPE = (
    "history projection, recomputed history points and every production feature column of "
    "the selected donor rows; a scoring or whitelist change alters it without a data change"
)


@dataclass(frozen=True)
class HistoryRecipe:
    """Select a cohort by observed history; never select using the forecast game."""

    name: str
    seed: int = 42
    cases: int = 20
    history_games: int = 8
    mode: str = "replay"
    block_games: int = 3
    window: str = "any"
    min_history_ppg: float | None = None
    max_history_ppg: float | None = None
    donor_seasons: tuple[int, ...] = tuple(TRAIN_SEASONS)
    position: str = "QB"
    schema_version: int = SCHEMA_VERSION
    transforms: tuple = ()
    opaque_signal_policy: str | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        for name in ("seed", "cases", "history_games", "block_games", "schema_version"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.schema_version != SCHEMA_VERSION or self.position not in POSITION_HISTORY_SCHEMAS:
            raise ValueError(
                f"schema version {SCHEMA_VERSION} supports "
                f"{sorted(POSITION_HISTORY_SCHEMAS)} histories only"
            )
        schema = position_schema(self.position)
        if self.seed < 0 or not 1 <= self.cases <= 1000:
            raise ValueError("seed must be nonnegative and cases must be between 1 and 1000")
        if not 1 <= self.history_games <= schema.max_history_games:
            raise ValueError("history_games must fit the production attention window")
        if self.block_games < 1 or (
            self.mode == "block_bootstrap" and self.block_games > self.history_games
        ):
            raise ValueError("block_games must be between 1 and history_games")
        if self.mode not in {"replay", "block_bootstrap"}:
            raise ValueError("mode must be replay or block_bootstrap")
        if self.window not in WINDOWS:
            raise ValueError("window must be any or exact")
        if not isinstance(self.donor_seasons, (list, tuple)):
            raise ValueError("donor_seasons must be a list of integer training seasons")
        seasons = tuple(self.donor_seasons)
        if not seasons or any(type(s) is not int for s in seasons):
            raise ValueError("donor_seasons must contain integer training seasons")
        if len(set(seasons)) != len(seasons) or not set(seasons) <= set(TRAIN_SEASONS):
            raise ValueError("donor_seasons must be unique and confined to TRAIN_SEASONS")
        object.__setattr__(self, "donor_seasons", tuple(sorted(seasons)))
        for name in ("min_history_ppg", "max_history_ppg"):
            value = getattr(self, name)
            if value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite or null")
            # 15 and 15.0 are one recipe; canonical floats keep the recipe hash stable.
            object.__setattr__(self, name, float(value) + 0.0)
        if (
            self.min_history_ppg is not None
            and self.max_history_ppg is not None
            and self.min_history_ppg > self.max_history_ppg
        ):
            raise ValueError("min_history_ppg must not exceed max_history_ppg")
        if not isinstance(self.transforms, (list, tuple)):
            raise ValueError("transforms must be a list of transform objects")
        ops = tuple(parse_transform(op, schema, self.history_games) for op in self.transforms)
        object.__setattr__(self, "transforms", ops)
        if ops and self.opaque_signal_policy is None:
            raise ValueError("opaque_signal_policy is required when transforms are present")
        if not ops and self.opaque_signal_policy is not None:
            raise ValueError("opaque_signal_policy has no effect without transforms")
        if ops and self.opaque_signal_policy not in OPAQUE_POLICIES:
            raise ValueError(f"opaque_signal_policy must be one of {OPAQUE_POLICIES}")

    @classmethod
    def from_dict(cls, value: dict) -> HistoryRecipe:
        if not isinstance(value, dict):
            raise ValueError("recipe must be a JSON object")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown recipe fields: {sorted(unknown)}")
        if "name" not in value:
            raise ValueError("recipe requires name")
        return cls(**value)


@dataclass
class HistoryCohort:
    games: pd.DataFrame
    cases: pd.DataFrame
    context: pd.DataFrame
    history: np.ndarray
    mask: np.ndarray
    manifest: dict
    donor_games: pd.DataFrame | None = None


@dataclass(frozen=True)
class ConsumedSource:
    """Everything generation reads from a source, with stable index labels.

    ``frame`` and ``context_rows`` share labels that are positions into
    ``source`` (the reset-index copy), so forecast rows can be selected in
    either and production builders can run on ``source.loc[frame.index]``.
    """

    frame: pd.DataFrame
    context_rows: pd.DataFrame
    source: pd.DataFrame
    values_sha256: str


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sampling_identity_hash(recipe: dict) -> str:
    """Identity of the sampled donors: the recipe without its name or transforms."""
    return _json_hash({k: v for k, v in recipe.items() if k not in SAMPLING_IDENTITY_EXCLUDED})


def _dtype_label(dtype) -> str:
    # object, str and arrow string columns hash the same rows; label them alike.
    return "str" if getattr(dtype, "kind", "O") in "OUS" else str(dtype)


def _frame_hash(frame: pd.DataFrame) -> str:
    """Fingerprint the canonical consumed values, including missingness and schema."""
    digest = hashlib.sha256()
    digest.update(json.dumps([(c, _dtype_label(frame[c].dtype)) for c in frame]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def consumed_values_hash(frame: pd.DataFrame, context_rows: pd.DataFrame) -> str:
    """Identity of everything generation reads: history projection plus static context."""
    return hashlib.sha256((_frame_hash(frame) + _frame_hash(context_rows)).encode()).hexdigest()


def runtime_versions(*packages: str) -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for package in ("pyarrow", *packages):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def code_hashes(relative_paths: Iterable[str]) -> dict[str, str]:
    """sha256 of implementation files, keyed by their repo-relative ``src/`` path."""
    src_root = Path(__file__).resolve().parents[1]
    return {f"src/{relative}": file_digest(src_root / relative) for relative in relative_paths}


def validate_history_frame(
    frame: pd.DataFrame, schema: PositionHistorySchema, *, stage: str = "source"
) -> None:
    """Value checks shared by donor validation and transformed histories.

    ``frame`` holds float64 history/target columns. Missing external signals
    are allowed; missing raw outcomes are not, and count/relation/bound
    violations name ``stage`` so a rewritten history fails as loudly as a bad
    source.
    """
    missing = [column for column in schema.validated_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{stage} is missing schema columns: {missing}")
    for column in schema.validated_columns:
        values = frame[column]
        if np.isinf(values).any():
            raise ValueError(f"{stage} {column} contains infinity")
        if values.abs().gt(np.finfo(np.float32).max).any():
            raise ValueError(f"{stage} {column} exceeds the production float32 range")
    # Missing external signals are retained and reported; missing raw outcomes
    # cannot stand in for observed zero production when defining a cohort.
    if frame[list(schema.must_observe)].isna().any().any():
        raise ValueError(f"{stage} raw outcomes and opportunity counts must be observed")
    for column in schema.count_columns:
        values = frame[column].dropna()
        if (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"{stage} {column} must contain nonnegative integer counts")
    for smaller, larger in schema.relations:
        if (frame[smaller] > frame[larger]).any():
            raise ValueError(f"{stage} violates {smaller} <= {larger}")
    for name, violated in schema.derived_checks:
        if violated(frame).any():
            raise ValueError(f"{stage} {name}")
    for column, low, high in schema.bounded_columns:
        if not frame[column].dropna().between(low, high).all():
            raise ValueError(f"{stage} {column} must be within [{low}, {high}]")


def _source_frame(
    source: pd.DataFrame,
    *,
    position: str,
    donor_seasons: Iterable[int],
    schema: PositionHistorySchema,
) -> pd.DataFrame:
    """Project the donor pool onto raw game signals; index labels survive."""
    history_columns = list(schema.history_columns)
    required = list(
        dict.fromkeys([*KEYS, *schema.identity_columns, *history_columns, *schema.targets])
    )
    missing = sorted(set(required) - set(source.columns))
    if missing:
        raise ValueError(f"source is missing production history columns: {missing}")
    # Project onto raw game signals only: cached rolling/static columns must not
    # survive resampling and masquerade as recomputed synthetic features.
    selected = source["position"].eq(position)
    if "season_type" in schema.identity_columns:
        selected &= source["season_type"].eq("REG")
    frame = source.loc[selected, required].copy()
    if frame[KEYS].isna().any().any():
        raise ValueError("source has missing player/season/week keys")
    team_columns = [c for c in ("recent_team", "opponent_team") if c in schema.identity_columns]
    if frame[team_columns].isna().any().any():
        raise ValueError("source has missing team identities")
    if not frame["player_id"].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise ValueError("source player_id must be a nonempty string")
    for column in ("season", "week"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(numeric).all() or not (numeric == np.floor(numeric)).all():
            raise ValueError(f"source {column} must contain finite integers")
        frame[column] = numeric.astype("int64")
    if not frame["week"].between(1, 18).all():
        raise ValueError("regular-season week must be between 1 and 18")
    frame = frame[frame["season"].isin(list(donor_seasons))]
    if frame.empty:
        raise ValueError(f"no {position} records in the requested training seasons")
    if frame.duplicated(KEYS).any():
        raise ValueError("source has duplicate player/season/week keys")
    for column in schema.validated_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    validate_history_frame(frame, schema, stage="source")
    frame["fantasy_points"] = predictions_to_fantasy_points(
        position, {name: frame[name].to_numpy() for name in schema.targets}
    )
    return frame.sort_values(KEYS, kind="stable")


def _context_rows(
    source: pd.DataFrame, index: pd.Index, schema: PositionHistorySchema
) -> pd.DataFrame:
    """The unscaled production feature columns of the selected rows.

    Every whitelisted column must exist and be finite: a frame without them is
    the raw or stale source the contract rejects, not a prepared one.
    """
    missing = sorted(set(schema.feature_columns) - set(source.columns))
    if missing:
        raise ValueError(f"source is missing production feature columns: {missing}")
    try:
        context = source.loc[index, list(schema.feature_columns)].astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("source feature columns must be numeric") from exc
    if not np.isfinite(context.to_numpy()).all():
        raise ValueError("source feature columns must be finite; run the production preparation")
    return context


def consume_source(
    source: pd.DataFrame,
    *,
    position: str,
    donor_seasons: Iterable[int],
    schema: PositionHistorySchema,
) -> ConsumedSource:
    """Read a source the way generation does; the replay's control does the same."""
    source = source.reset_index(drop=True)
    frame = _source_frame(source, position=position, donor_seasons=donor_seasons, schema=schema)
    context_rows = _context_rows(source, frame.index, schema)
    return ConsumedSource(frame, context_rows, source, consumed_values_hash(frame, context_rows))


def model_input_readiness(mode: str, cases: pd.DataFrame, *, transformed: bool = False) -> dict:
    """Which saved-model families a cohort can feed coherently, and why not.

    The attention NN is always ready: its static branch is the forecast row's
    non-temporal context and its history branch is the synthetic tensor. The
    flat families read the forecast row's windowed features, which describe the
    real history, so they are ready only when every case replays exactly the
    real, untransformed window.
    """
    readiness = {
        "attn_nn": {"ready": True, "inputs": ["history.npz", "context.parquet"], "reason": None}
    }
    truncated = int((~cases["exact_window"].astype(bool)).sum())
    if transformed:
        reason = TRANSFORMED_REASON
    elif mode != "replay":
        reason = RESAMPLED_REASON
    elif truncated:
        reason = TRUNCATED_REASON.format(truncated=truncated, cases=len(cases))
    else:
        reason = None
    for family in FLAT_FAMILIES:
        readiness[family] = {
            "ready": reason is None,
            "inputs": ["context.parquet"],
            "reason": reason,
        }
    return readiness


def generate_cohort(source: pd.DataFrame, recipe: HistoryRecipe) -> HistoryCohort:
    """Bootstrap isolated history cases, preserving whole donor-game records.

    Each eligible forecast has at least N observed games earlier in the same
    season (exactly N when ``window`` is ``exact``). The forecast row
    contributes its key and its pre-kickoff static context; its outcomes cannot
    select a case or enter a tensor. Sampling is with replacement. In block
    mode, both source blocks and all their per-game fields are sampled together
    from that case's eligible past; artificial joins are explicitly recorded.
    Declared transforms then rewrite the sampled games; the untransformed donor
    window is kept beside them.
    """
    schema = position_schema(recipe.position)
    consumed = consume_source(
        source, position=recipe.position, donor_seasons=recipe.donor_seasons, schema=schema
    )
    frame, context_rows = consumed.frame, consumed.context_rows
    history_columns = list(schema.history_columns)
    candidates = []
    n = recipe.history_games
    for (_, _), group in frame.groupby(["player_id", "season"], sort=True):
        indices = group.index.to_numpy()
        for offset in range(n, len(indices)):
            if recipe.window == "exact" and offset != n:
                continue
            past = indices[offset - n : offset]
            ppg = float(frame.loc[past, "fantasy_points"].mean())
            if recipe.min_history_ppg is not None and ppg < recipe.min_history_ppg:
                continue
            if recipe.max_history_ppg is not None and ppg > recipe.max_history_ppg:
                continue
            candidates.append((past, indices[offset], ppg, offset))
    if not candidates:
        raise ValueError("no eligible donor histories; cannot synthesize an unsupported archetype")
    source_hash = consumed.values_sha256
    recipe_dict = asdict(recipe)
    recipe_hash = _json_hash(recipe_dict)
    rng = np.random.default_rng(recipe.seed)
    game_parts, case_rows, forecast_indices = [], [], []
    for case_number, candidate in enumerate(rng.integers(len(candidates), size=recipe.cases)):
        past, forecast_idx, donor_ppg, offset = candidates[candidate]
        forecast = frame.loc[forecast_idx]
        offsets = np.arange(n)
        block_ids = np.zeros(n, dtype=int)
        if recipe.mode == "block_bootstrap":
            starts = rng.integers(
                n - recipe.block_games + 1, size=math.ceil(n / recipe.block_games)
            )
            offsets = np.concatenate(
                [np.arange(start, start + recipe.block_games) for start in starts]
            )[:n]
            block_ids = np.repeat(np.arange(len(starts)), recipe.block_games)[:n]
        case_id = f"{recipe_hash[:12]}-{source_hash[:12]}-{case_number:05d}"
        games = frame.loc[past[offsets]].copy().reset_index(drop=True)
        games = games.rename(columns={key: f"donor_{key}" for key in KEYS})
        games.insert(0, "case_id", case_id)
        games.insert(1, "history_step", np.arange(1, n + 1))
        games.insert(2, "block_id", block_ids)
        game_parts.append(games)
        forecast_indices.append(forecast_idx)
        case_rows.append(
            {
                "case_id": case_id,
                "case_index": case_number,
                "donor_player_id": forecast["player_id"],
                "donor_season": int(forecast["season"]),
                "forecast_week": int(forecast["week"]),
                "real_prior_games": int(offset),
                "exact_window": bool(offset == n),
                "donor_history_ppg": donor_ppg,
                "sampled_history_ppg": float(games["fantasy_points"].mean()),
                "unique_donor_games": int(games["donor_week"].nunique()),
            }
        )
    games = pd.concat(game_parts, ignore_index=True)
    cases = pd.DataFrame(case_rows)
    donor_games, transform_report = None, None
    if recipe.transforms:
        donor_games = games.copy()
        games, transform_report = apply_transforms(
            games, recipe.transforms, schema=schema, policy=recipe.opaque_signal_policy
        )
        validate_history_frame(games, schema, stage="transformed")
    cases["generated_history_ppg"] = (
        games.groupby("case_id", sort=False)["fantasy_points"]
        .mean()
        .loc[cases["case_id"]]
        .to_numpy()
    )
    # These are ordinal history slots, NOT a fabricated NFL calendar. Every
    # case has a separate identity, so duplicated donors cannot mix tokens.
    history_frames = []
    for case_id, season in zip(cases["case_id"], cases["donor_season"], strict=True):
        tensor_frame = games.loc[games["case_id"].eq(case_id), history_columns].copy()
        tensor_frame["player_id"] = case_id
        tensor_frame["season"] = int(season)
        tensor_frame["week"] = np.arange(1, n + 1)
        probe = {col: 0.0 for col in history_columns}
        probe.update(player_id=case_id, season=int(season), week=n + 1)
        history_frames.append(pd.concat([tensor_frame, pd.DataFrame([probe])], ignore_index=True))
    history_input = pd.concat(history_frames, ignore_index=True)
    arrays, masks = build_game_history_arrays(
        history_input, history_stats=history_columns, max_seq_len=schema.max_history_games
    )
    probe_indices = np.arange(n, len(history_input), n + 1)
    # The static branch is the real forecast game's own pre-kickoff row.
    team_columns = [c for c in ("recent_team", "opponent_team") if c in schema.identity_columns]
    forecast_rows = frame.loc[forecast_indices]
    context = pd.concat(
        [
            cases[["case_id", "case_index", "donor_player_id", "donor_season", "forecast_week"]],
            forecast_rows[team_columns].reset_index(drop=True),
            context_rows.loc[forecast_indices].reset_index(drop=True),
        ],
        axis=1,
    )
    history, mask = arrays[probe_indices], masks[probe_indices]
    transformed = transform_report is not None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "recipe": recipe_dict,
        "recipe_sha256": recipe_hash,
        "sampling_identity_sha256": sampling_identity_hash(recipe_dict),
        "source_values_sha256": source_hash,
        "source_values_scope": SOURCE_VALUES_SCOPE,
        "source_rows": len(frame),
        "eligible_windows": len(candidates),
        "unique_donor_windows": int(
            cases.drop_duplicates(["donor_player_id", "donor_season", "forecast_week"]).shape[0]
        ),
        "exact_window_cases": int(cases["exact_window"].sum()),
        "diagnostic_scope": "attention_history_and_forecast_context",
        "history_kind": "transformed" if transformed else "donor",
        "fixture": transformed,
        "forecast_outcome": FORECAST_OUTCOME,
        "model_input_readiness": model_input_readiness(recipe.mode, cases, transformed=transformed),
        "history_columns": history_columns,
        "history_shape": list(history.shape),
        "history_order": "newest_first",
        "history_scaling": "unscaled; the production attention model consumes raw history",
        "context_columns": list(schema.feature_columns),
        "context_sequence_coupled_columns": list(schema.sequence_coupled_context),
        "context_semantics": CONTEXT_SEMANTICS,
        "static_context_policy": STATIC_CONTEXT_POLICY,
        "scoring_format": "ppr",
        "scoring_scope": schema.scoring_scope,
        "external_signal_policy": "preserve_whole_donor_game; missingness preserved in parquet",
        "missing_history_values": {
            c: int(games[c].isna().sum()) for c in history_columns if games[c].isna().any()
        },
        "temporal_policy": "original_order"
        if recipe.mode == "replay"
        else "within_block_order_only; boundary transitions are synthetic",
        "transform_support": dict(schema.transform_support),
        "versions": runtime_versions(),
        "code_sha256": code_hashes(
            (
                "analysis/synthetic_history.py",
                "analysis/synthetic_history_schema.py",
                "analysis/synthetic_transforms.py",
                *schema.code_paths,
            )
        ),
    }
    if transformed:
        manifest.update(transform_report)
    return HistoryCohort(games, cases, context, history, mask, manifest, donor_games)


def publish_artifact_dir(
    output: Path,
    write_files: Callable[[Path], None],
    manifest: dict,
    *,
    manifest_name: str = "manifest.json",
) -> Path:
    """Publish a new local artifact directory; never replace existing results.

    ``write_files`` fills a temporary sibling directory; every file it wrote is
    hashed into ``manifest["files"]`` before the atomic rename.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        write_files(temporary)
        manifest = dict(manifest)
        manifest["files"] = {
            path.name: file_digest(path) for path in sorted(temporary.iterdir()) if path.is_file()
        }
        (temporary / manifest_name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def write_cohort(cohort: HistoryCohort, output: Path) -> Path:
    def _write(directory: Path) -> None:
        cohort.games.to_parquet(directory / "games.parquet", index=False)
        cohort.cases.to_parquet(directory / "cases.parquet", index=False)
        cohort.context.to_parquet(directory / "context.parquet", index=False)
        if cohort.donor_games is not None:
            cohort.donor_games.to_parquet(directory / "donor_games.parquet", index=False)
        np.savez_compressed(directory / "history.npz", history=cohort.history, mask=cohort.mask)

    return publish_artifact_dir(output, _write, cohort.manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Frozen, enriched, unscaled player-game parquet"
    )
    parser.add_argument("--recipe", type=Path, required=True, help="Versioned JSON recipe")
    parser.add_argument(
        "--output", type=Path, required=True, help="New artifact directory (never overwritten)"
    )
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        recipe = HistoryRecipe.from_dict(json.loads(args.recipe.read_text()))
        cohort = generate_cohort(pd.read_parquet(args.source), recipe)
        cohort.manifest["source_file_sha256"] = file_digest(args.source)
        output = write_cohort(cohort, args.output)
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"synthetic-history: {exc}\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "cases": len(cohort.cases),
                "eligible_windows": cohort.manifest["eligible_windows"],
                "exact_window_cases": cohort.manifest["exact_window_cases"],
                "history_kind": cohort.manifest["history_kind"],
                "model_input_readiness": {
                    family: entry["ready"]
                    for family, entry in cohort.manifest["model_input_readiness"].items()
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
