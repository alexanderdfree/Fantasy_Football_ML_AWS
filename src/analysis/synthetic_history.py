"""Reproducible donor-based QB histories for offline behavioral diagnostics.

This is an attention-history substrate, not a complete forecasting input. It
resamples whole observed game records so opaque signals (QBR, opportunity, game
context) travel with their box scores. Static features, scalers, and checkpoints
are deliberately outside this first version's contract. No source is fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass, fields
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TRAIN_SEASONS
from src.features.engineer import build_game_history_arrays
from src.qb.config import POSITION_CONFIG
from src.shared.aggregate_targets import predictions_to_fantasy_points

SCHEMA_VERSION = 1
KEYS = ["player_id", "season", "week"]
QB_TARGETS = tuple(POSITION_CONFIG.targets)
COUNTS = (
    "attempts",
    "completions",
    "carries",
    "passing_tds",
    "rushing_tds",
    "interceptions",
    "fumbles_lost",
    "sacks",
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
    min_history_ppg: float | None = None
    max_history_ppg: float | None = None
    donor_seasons: tuple[int, ...] = tuple(TRAIN_SEASONS)
    position: str = "QB"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        for name in ("seed", "cases", "history_games", "block_games", "schema_version"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.schema_version != SCHEMA_VERSION or self.position != "QB":
            raise ValueError("schema version 1 supports QB histories only")
        if self.seed < 0 or not 1 <= self.cases <= 1000:
            raise ValueError("seed must be nonnegative and cases must be between 1 and 1000")
        if not 1 <= self.history_games <= POSITION_CONFIG.attn_max_seq_len:
            raise ValueError("history_games must fit the production attention window")
        if self.block_games < 1 or (
            self.mode == "block_bootstrap" and self.block_games > self.history_games
        ):
            raise ValueError("block_games must be between 1 and history_games")
        if self.mode not in {"replay", "block_bootstrap"}:
            raise ValueError("mode must be replay or block_bootstrap")
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
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite or null")
        if (
            self.min_history_ppg is not None
            and self.max_history_ppg is not None
            and self.min_history_ppg > self.max_history_ppg
        ):
            raise ValueError("min_history_ppg must not exceed max_history_ppg")

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
    history: np.ndarray
    mask: np.ndarray
    manifest: dict


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    """Fingerprint the canonical consumed values, including missingness and schema."""
    digest = hashlib.sha256()
    digest.update(json.dumps([(c, str(frame[c].dtype)) for c in frame], sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _source_frame(source: pd.DataFrame, recipe: HistoryRecipe) -> pd.DataFrame:
    history_columns = list(POSITION_CONFIG.attn_history_stats)
    required = list(
        dict.fromkeys(
            [
                *KEYS,
                "position",
                "season_type",
                "recent_team",
                "opponent_team",
                *history_columns,
                *QB_TARGETS,
            ]
        )
    )
    missing = sorted(set(required) - set(source.columns))
    if missing:
        raise ValueError(f"source is missing production history columns: {missing}")
    # Project onto raw game signals only: cached rolling/static columns must not
    # survive resampling and masquerade as recomputed synthetic features.
    frame = source.loc[
        source["position"].eq("QB") & source["season_type"].eq("REG"), required
    ].copy()
    if frame[KEYS].isna().any().any():
        raise ValueError("source has missing player/season/week keys")
    if frame[["recent_team", "opponent_team"]].isna().any().any():
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
    frame = frame[frame["season"].isin(recipe.donor_seasons)]
    if frame.empty:
        raise ValueError("no QB records in the requested training seasons")
    if frame.duplicated(KEYS).any():
        raise ValueError("source has duplicate player/season/week keys")
    for column in dict.fromkeys([*history_columns, *QB_TARGETS]):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
        if np.isinf(frame[column]).any():
            raise ValueError(f"source {column} contains infinity")
        if frame[column].abs().gt(np.finfo(np.float32).max).any():
            raise ValueError(f"source {column} exceeds the production float32 range")
    # Missing external signals are retained and reported; missing raw outcomes
    # cannot stand in for observed zero production when defining a cohort.
    if frame[list(QB_TARGETS) + ["attempts", "completions", "carries"]].isna().any().any():
        raise ValueError("raw QB outcomes and opportunity counts must be observed")
    for column in COUNTS:
        values = frame[column].dropna()
        if (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"source {column} must contain nonnegative integer counts")
    for smaller, larger in (
        ("completions", "attempts"),
        ("passing_tds", "completions"),
        ("rushing_tds", "carries"),
    ):
        if (frame[smaller] > frame[larger]).any():
            raise ValueError(f"source violates {smaller} <= {larger}")
    if (frame["interceptions"] > frame["attempts"] - frame["completions"]).any():
        raise ValueError("source interceptions exceed incomplete attempts")
    observed_snaps = frame["snap_pct_raw"].dropna()
    if not observed_snaps.between(0, 1).all():
        raise ValueError("snap_pct_raw must be a fraction in [0, 1]")
    if not frame["qbr_total"].dropna().between(0, 100).all():
        raise ValueError("qbr_total must be in [0, 100]")
    if (frame["carries"] > frame["team_rush_attempts"]).any():
        raise ValueError("QB carries exceed team rushing attempts")
    frame["fantasy_points"] = predictions_to_fantasy_points(
        "QB", {name: frame[name].to_numpy() for name in QB_TARGETS}
    )
    return frame.sort_values(KEYS, kind="stable").reset_index(drop=True)


def generate_cohort(source: pd.DataFrame, recipe: HistoryRecipe) -> HistoryCohort:
    """Bootstrap isolated history cases, preserving whole donor-game records.

    Each eligible forecast has N observed games earlier in the same season.
    The forecast row contributes its key only; its outcomes cannot select a
    case or enter a tensor. Sampling is with replacement. In block mode, both
    source blocks and all their per-game fields are sampled together from that
    case's eligible past; artificial joins are explicitly recorded.
    """
    frame = _source_frame(source, recipe)
    history_columns = list(POSITION_CONFIG.attn_history_stats)
    candidates = []
    n = recipe.history_games
    for (_, _), group in frame.groupby(["player_id", "season"], sort=True):
        indices = group.index.to_numpy()
        for offset in range(n, len(indices)):
            past = indices[offset - n : offset]
            ppg = float(frame.loc[past, "fantasy_points"].mean())
            if recipe.min_history_ppg is not None and ppg < recipe.min_history_ppg:
                continue
            if recipe.max_history_ppg is not None and ppg > recipe.max_history_ppg:
                continue
            candidates.append((past, indices[offset], ppg))
    if not candidates:
        raise ValueError("no eligible donor histories; cannot synthesize an unsupported archetype")
    source_hash = _frame_hash(frame)
    recipe_hash = _json_hash(asdict(recipe))
    rng = np.random.default_rng(recipe.seed)
    game_parts, case_rows, history_frames = [], [], []
    for case_number, candidate in enumerate(rng.integers(len(candidates), size=recipe.cases)):
        past, forecast_idx, donor_ppg = candidates[candidate]
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
        case_rows.append(
            {
                "case_id": case_id,
                "donor_player_id": forecast["player_id"],
                "donor_season": int(forecast["season"]),
                "forecast_week": int(forecast["week"]),
                "donor_history_ppg": donor_ppg,
                "generated_history_ppg": float(games["fantasy_points"].mean()),
                "unique_donor_games": int(games["donor_week"].nunique()),
            }
        )
        # These are ordinal history slots, NOT a fabricated NFL calendar. Every
        # case has a separate identity, so duplicated donors cannot mix tokens.
        tensor_frame = games[history_columns].copy()
        tensor_frame["player_id"] = case_id
        tensor_frame["season"] = int(forecast["season"])
        tensor_frame["week"] = np.arange(1, n + 1)
        probe = {col: 0.0 for col in history_columns}
        probe.update(player_id=case_id, season=int(forecast["season"]), week=n + 1)
        history_frames.append(pd.concat([tensor_frame, pd.DataFrame([probe])], ignore_index=True))
    history_input = pd.concat(history_frames, ignore_index=True)
    arrays, masks = build_game_history_arrays(
        history_input, history_stats=history_columns, max_seq_len=POSITION_CONFIG.attn_max_seq_len
    )
    probe_indices = np.arange(n, len(history_input), n + 1)
    games = pd.concat(game_parts, ignore_index=True)
    cases = pd.DataFrame(case_rows)
    history, mask = arrays[probe_indices], masks[probe_indices]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "recipe": asdict(recipe),
        "recipe_sha256": recipe_hash,
        "source_values_sha256": source_hash,
        "source_rows": len(frame),
        "eligible_windows": len(candidates),
        "unique_donor_windows": int(
            cases.drop_duplicates(["donor_player_id", "donor_season", "forecast_week"]).shape[0]
        ),
        "diagnostic_scope": "attention_history_only",
        "full_model_input_ready": False,
        "history_columns": history_columns,
        "history_shape": list(history.shape),
        "history_order": "newest_first",
        "history_scaling": "unscaled; apply the checkpoint's fitted history preprocessing",
        "scoring_format": "ppr",
        "scoring_scope": "QB projected components only; excludes receiving and two-point conversions",
        "external_signal_policy": "preserve_whole_donor_game; missingness preserved in parquet",
        "missing_history_values": {
            c: int(games[c].isna().sum()) for c in history_columns if games[c].isna().any()
        },
        "temporal_policy": "original_order"
        if recipe.mode == "replay"
        else "within_block_order_only; boundary transitions are synthetic",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": version("pyarrow"),
        },
        "code_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1] / "features/engineer.py",
                Path(__file__).resolve().parents[1] / "qb/config.py",
                Path(__file__).resolve().parents[1] / "config.py",
                Path(__file__).resolve().parents[1] / "shared/aggregate_targets.py",
            )
        },
    }
    return HistoryCohort(games, cases, history, mask, manifest)


def write_cohort(cohort: HistoryCohort, output: Path) -> Path:
    """Publish a new local artifact directory; never replace existing results."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        cohort.games.to_parquet(temporary / "games.parquet", index=False)
        cohort.cases.to_parquet(temporary / "cases.parquet", index=False)
        np.savez_compressed(temporary / "history.npz", history=cohort.history, mask=cohort.mask)
        manifest = dict(cohort.manifest)
        manifest["files"] = {
            name: _file_hash(temporary / name)
            for name in ("games.parquet", "cases.parquet", "history.npz")
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


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
        recipe = HistoryRecipe.from_dict(json.loads(args.recipe.read_text()))
        cohort = generate_cohort(pd.read_parquet(args.source), recipe)
        cohort.manifest["source_file_sha256"] = _file_hash(args.source)
        output = write_cohort(cohort, args.output)
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"synthetic-history: {exc}\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "cases": len(cohort.cases),
                "eligible_windows": cohort.manifest["eligible_windows"],
                "full_model_input_ready": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
