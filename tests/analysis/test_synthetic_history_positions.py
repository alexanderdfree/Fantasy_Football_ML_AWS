"""RB, WR and TE cohorts through the position schema registry; no models, no real data."""

import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.synthetic_history import HistoryRecipe, generate_cohort, write_cohort
from src.analysis.synthetic_history_schema import position_schema
from src.analysis.synthetic_replay import load_cohort, replay_cohort
from src.features.engineer import build_game_history_arrays
from tests.analysis.conftest import LONG_WEEKS, poison_column, position_rows
from tests.analysis.fake_bundles import fake_artifacts

pytestmark = pytest.mark.unit

POSITIONS = ("RB", "WR", "TE")
RECIPES = Path(__file__).resolve().parents[2] / "src/analysis/synthetic_history_recipes"


def recipe(position, **kwargs):
    kwargs.setdefault("cases", 4)
    kwargs.setdefault("history_games", 3)
    kwargs.setdefault("block_games", 2)
    return HistoryRecipe(name="test", position=position, **kwargs)


@pytest.mark.parametrize("position", POSITIONS)
def test_whole_game_signals_travel_and_context_is_the_forecast_row(position):
    source = position_rows(position)
    cohort = generate_cohort(source, recipe(position, mode="block_bootstrap"))
    schema = position_schema(position)
    assert poison_column(position) not in cohort.games
    original = source.set_index(["player_id", "season", "week"], drop=False)
    cases = cohort.cases.set_index("case_id")
    for row in cohort.games.to_dict("records"):
        donor = original.loc[(row["donor_player_id"], row["donor_season"], row["donor_week"])]
        for column in schema.history_columns:
            assert row[column] == donor[column]
        assert row["donor_week"] < cases.loc[row["case_id"], "forecast_week"]
    for row in cohort.context.to_dict("records"):
        forecast = original.loc[(row["donor_player_id"], row["donor_season"], row["forecast_week"])]
        assert row[poison_column(position)] == 9999
        for column in schema.feature_columns:
            assert row[column] == float(forecast[column])
    assert cohort.manifest["scoring_scope"] == schema.scoring_scope
    assert cohort.manifest["context_sequence_coupled_columns"] == list(
        schema.sequence_coupled_context
    )
    assert "week" in schema.sequence_coupled_context


@pytest.mark.parametrize("position", POSITIONS)
def test_identity_replay_matches_production_arrays(position):
    source = position_rows(position)
    schema = position_schema(position)
    cohort = generate_cohort(source, recipe(position, cases=5, history_games=5, window="exact"))
    production, mask = build_game_history_arrays(
        source, list(schema.history_columns), schema.max_history_games
    )
    for i, row in cohort.cases.iterrows():
        index = source.index[
            source.player_id.eq(row.donor_player_id)
            & source.season.eq(row.donor_season)
            & source.week.eq(row.forecast_week)
        ][0]
        np.testing.assert_array_equal(cohort.history[i], production[index])
        np.testing.assert_array_equal(cohort.mask[i], mask[index])
    assert all(entry["ready"] for entry in cohort.manifest["model_input_readiness"].values())


@pytest.mark.parametrize("position", POSITIONS)
def test_reproducible_under_reordering_and_ppg_uses_position_scoring(position):
    source = position_rows(position)
    first = generate_cohort(source, recipe(position, mode="block_bootstrap"))
    second = generate_cohort(
        source.sample(frac=1, random_state=3), recipe(position, mode="block_bootstrap")
    )
    pd.testing.assert_frame_equal(first.games, second.games)
    assert first.manifest == second.manifest
    targets = import_module(f"src.{position.lower()}.config").POSITION_CONFIG.targets
    row = first.games.iloc[0]
    weights = {"receptions": 1.0, "receiving_yards": 0.1, "rushing_yards": 0.1}
    expected = sum(weights.get(t, 6.0 if t.endswith("_tds") else -2.0) * row[t] for t in targets)
    assert row["fantasy_points"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "position,change,error",
    [
        ("RB", {"receptions": 5}, "receptions <= targets"),
        ("RB", {"receiving_tds": 4}, "receiving_tds <= receptions"),
        ("RB", {"inside5_carries": 2}, "inside5_carries <= inside10_carries"),
        ("RB", {"redzone_carries": 15}, "redzone_carries <= carries"),
        ("RB", {"game_carry_share": 1.5}, "game_carry_share must be within"),
        ("RB", {"carries": 30}, "carries <= team_rush_attempts"),
        ("RB", {"receptions": 24, "targets": 30}, "receptions <= team_completions"),
        ("RB", {"rush_yards_gained_exp": np.inf}, "infinity"),
        ("RB", {"rushing_tds": 5}, "touchdowns exceed team points"),
        ("WR", {"receptions": 9}, "receptions <= targets"),
        ("WR", {"redzone_targets": 9}, "redzone_targets <= targets"),
        ("WR", {"game_opportunity_index": -0.1}, "game_opportunity_index must be within"),
        ("WR", {"targets": 40}, "targets <= team_pass_attempts"),
        ("WR", {"receptions": np.nan}, "must be observed"),
        ("TE", {"receiving_tds": 5}, "receiving_tds <= receptions"),
        ("TE", {"rushing_tds": 1}, "rushing_tds <= carries"),
    ],
)
def test_invalid_donor_data_fails_before_generation(position, change, error):
    source = position_rows(position)
    for column, value in change.items():
        source[column] = source[column].astype(object)
        source.loc[0, column] = value
    with pytest.raises(ValueError, match=error):
        generate_cohort(source, recipe(position))


def test_position_mismatch_between_recipe_and_source_fails():
    relabelled = position_rows("RB").assign(position="QB")
    with pytest.raises(ValueError, match="no RB records"):
        generate_cohort(relabelled, recipe("RB"))
    with pytest.raises(ValueError, match="missing production history columns"):
        generate_cohort(position_rows("QB"), recipe("RB"))
    with pytest.raises(ValueError, match="missing production history columns"):
        generate_cohort(position_rows("WR").drop(columns="game_opportunity_index"), recipe("WR"))


@pytest.mark.parametrize("position", POSITIONS)
def test_transforms_use_the_position_declarations(position):
    source = position_rows(position)
    schema = position_schema(position)
    stat = "targets"
    team_column, coefficient = schema.team_accounting[stat]
    transforms = [{"op": "scale", "stats": [stat], "factor": 2.0}]
    cohort = generate_cohort(
        source, recipe(position, transforms=transforms, opaque_signal_policy="mark_missing")
    )
    games = cohort.games
    base = source[stat].iloc[0]
    assert (games[stat] == 2 * base).all()
    assert (games[team_column] == source[team_column].iloc[0] + coefficient * base).all()
    for column in schema.opaque_columns:
        assert games[column].isna().all()
    assert cohort.manifest["opaque_columns"] == list(schema.opaque_columns)
    with pytest.raises(ValueError, match="opaque external signal"):
        recipe(
            position,
            transforms=[{"op": "scale", "stats": [schema.opaque_columns[0]], "factor": 2.0}],
            opaque_signal_policy="keep_donor",
        )


@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("mode", ["replay", "block_bootstrap"])
def test_shipped_recipes_generate_within_their_bands(position, mode):
    path = RECIPES / f"{position.lower()}_{mode}.json"
    loaded = HistoryRecipe.from_dict(json.loads(path.read_text()))
    assert loaded.position == position and loaded.mode == mode
    assert 0 < loaded.min_history_ppg < loaded.max_history_ppg
    cohort = generate_cohort(position_rows(position, LONG_WEEKS), loaded)
    assert cohort.manifest["eligible_windows"] > 0
    band = (loaded.min_history_ppg, loaded.max_history_ppg)
    assert cohort.cases["donor_history_ppg"].between(*band).all()


@pytest.mark.parametrize("position", POSITIONS)
def test_replay_through_fake_bundles_with_an_ambiguous_player_season(tmp_path, position):
    source = position_rows(position)
    # p2's 2023 week 1 is duplicated: no case may come from that player-season.
    row = source.index[source.player_id.eq("p2") & source.season.eq(2023)][:1]
    source = pd.concat([source, source.loc[row]], ignore_index=True)
    models = tmp_path / "models"
    fake_artifacts(models, position, families=("attn_nn", "ridge"))
    cohort = write_cohort(
        generate_cohort(source, recipe(position, cases=5, history_games=3, window="exact")),
        tmp_path / "cohort",
    )
    loaded = load_cohort(cohort)
    assert loaded.manifest["donor_pool_exclusions"]["duplicate_game_keys"] == [
        {"player_id": "p2", "season": 2023, "weeks": [1], "rows": 2}
    ]
    predictions, manifest = replay_cohort(loaded, str(models), ["attn_nn", "ridge"], source=source)
    assert manifest["position"] == position
    assert manifest["identity_control"]["status"] == "passed"
    assert manifest["families_excluded"] == {}
    targets = position_schema(position).targets
    for family in ("attn_nn", "ridge"):
        assert manifest["identity_control"]["families"][family]["compared_predictions"] == 5
        columns = [f"pred_{family}_{t}" for t in targets] + [f"pred_{family}_total"]
        assert np.isfinite(predictions[columns].to_numpy()).all()
    donors = set(zip(predictions.donor_player_id, predictions.donor_season, strict=True))
    assert ("p2", 2023) not in donors
