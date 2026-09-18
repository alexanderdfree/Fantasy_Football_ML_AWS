"""Declared history transforms: accounting, rounding, policies, fixtures; no models."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.synthetic_history import (
    HistoryRecipe,
    generate_cohort,
    sampling_identity_hash,
    write_cohort,
)
from src.analysis.synthetic_history_schema import position_schema
from src.analysis.synthetic_transforms import (
    TransformUnsupported,
    parse_transform,
    scoring_weights,
)
from src.shared.aggregate_targets import predictions_to_fantasy_points

pytestmark = pytest.mark.unit

RECIPES = Path("src/analysis/synthetic_history_recipes")
HISTORY = ["case_id", "history_step", "block_id"]


def recipe(transforms=(), policy=None, **kwargs):
    kwargs.setdefault("history_games", 3)
    kwargs.setdefault("cases", 4)
    kwargs.setdefault("window", "exact")
    return HistoryRecipe(
        name="test", transforms=list(transforms), opaque_signal_policy=policy, **kwargs
    )


def scale(stats, factor, steps=None):
    return {"op": "scale", "stats": list(stats), "factor": factor, "steps": steps}


def test_scale_changes_only_targeted_steps_and_records_the_op(qb_source):
    baseline = generate_cohort(qb_source, recipe())
    cohort = generate_cohort(
        qb_source, recipe([scale(["passing_yards"], 1.25, [1, 2])], "keep_donor")
    )
    # Case ids carry the recipe hash, so compare the sampled donor rows themselves.
    pd.testing.assert_frame_equal(
        cohort.donor_games.drop(columns="case_id"), baseline.games.drop(columns="case_id")
    )
    keys = ["donor_player_id", "donor_season", "forecast_week"]
    pd.testing.assert_frame_equal(cohort.cases[keys], baseline.cases[keys])
    early = cohort.games["history_step"] <= 2
    np.testing.assert_allclose(
        cohort.games.loc[early, "passing_yards"], 1.25 * baseline.games.loc[early, "passing_yards"]
    )
    np.testing.assert_array_equal(
        cohort.games.loc[~early, "passing_yards"], baseline.games.loc[~early, "passing_yards"]
    )
    assert cohort.games["transformed"].eq(early).all()
    manifest = cohort.manifest
    assert manifest["history_kind"] == "transformed" and manifest["fixture"] is True
    assert manifest["transforms"][0]["fields_changed"] == ["passing_yards"]
    assert manifest["transforms"][0]["rows_targeted"] == 8
    assert manifest["transforms"][0]["team_accounting"] == []
    assert manifest["opaque_signal_policy"] == "keep_donor"
    assert manifest["sampling_identity_sha256"] == baseline.manifest["sampling_identity_sha256"]
    assert manifest["recipe_sha256"] != baseline.manifest["recipe_sha256"]
    assert (cohort.cases["generated_history_ppg"] > cohort.cases["sampled_history_ppg"]).all()
    for family in ("ridge", "nn", "lgbm"):
        entry = manifest["model_input_readiness"][family]
        assert entry["ready"] is False and "transformed" in entry["reason"]
    assert manifest["model_input_readiness"]["attn_nn"]["ready"] is True
    field = list(position_schema("QB").history_columns).index("passing_yards")
    np.testing.assert_allclose(
        cohort.history[0, :3, field],
        cohort.games.loc[cohort.games.case_id.eq(cohort.cases.case_id[0]), "passing_yards"].iloc[
            ::-1
        ],
    )


def test_team_accounting_propagates_and_points_are_recomputed(qb_source):
    cohort = generate_cohort(
        qb_source, recipe([scale(["carries", "rushing_yards", "rushing_tds"], 2.0)], "keep_donor")
    )
    games = cohort.games
    assert (games["carries"] == 8).all() and (games["rushing_yards"] == 40).all()
    assert (games["team_rush_attempts"] == 29).all()
    assert (games["team_rushing_yards"] == 120).all()
    assert (games["team_points_scored"] == 24).all()  # rushing_tds stayed zero
    assert (games["opp_team_points_scored"] == 21).all()
    expected = predictions_to_fantasy_points(
        "QB", {t: games[t].to_numpy() for t in position_schema("QB").targets}
    )
    np.testing.assert_allclose(games["fantasy_points"], expected)
    accounting = cohort.manifest["transforms"][0]["team_accounting"]
    assert accounting == [
        "team_rush_attempts += 1 * delta carries",
        "team_rushing_yards += 1 * delta rushing_yards",
        "team_points_scored += 6 * delta rushing_tds",
    ]


def test_counts_round_half_to_even_and_rounding_artifacts_are_capped(qb_source):
    source = qb_source.copy()
    source[["attempts", "completions", "interceptions"]] = [15, 13, 2]
    cohort = generate_cohort(
        source, recipe([scale(["attempts", "completions", "interceptions"], 0.3)], "keep_donor")
    )
    games = cohort.games
    assert (games["attempts"] == 4).all()  # 4.5 rounds to even
    assert (games["completions"] == 4).all()
    assert (games["interceptions"] == 0).all()  # 0.6 -> 1 exceeded 4 - 4 incompletions
    report = cohort.manifest["transforms"][0]
    assert report["cells_rounded"] == 3 * len(games)
    assert report["cells_capped_by_relation"] == len(games)
    assert cohort.manifest["count_rounding"].startswith("half_to_even")


def test_unit_interval_columns_are_clamped_and_counted(qb_source):
    cohort = generate_cohort(qb_source, recipe([scale(["snap_pct_raw"], 1.6)], "keep_donor"))
    assert (cohort.games["snap_pct_raw"] == 1.0).all()
    assert cohort.manifest["transforms"][0]["cells_clamped"] == len(cohort.games)


def test_extreme_production_factor_without_usage_fails_loud(qb_source):
    with pytest.raises(ValueError, match="passing_tds <= completions .* matched by usage"):
        generate_cohort(qb_source, recipe([scale(["passing_tds"], 12.0)], "keep_donor"))


def test_set_history_ppg_hits_the_target_within_rounding_and_records_factors(qb_source):
    stats = [
        "passing_yards",
        "rushing_yards",
        "passing_tds",
        "rushing_tds",
        "attempts",
        "completions",
    ]
    op = {"op": "set_history_ppg", "target_ppg": 100.0, "stats": stats}
    cohort = generate_cohort(qb_source, recipe([op], "mark_missing"))
    weights = scoring_weights(position_schema("QB"))
    bound = 0.5 * (weights["passing_tds"] + weights["rushing_tds"])
    assert (abs(cohort.cases["generated_history_ppg"] - 100.0) <= bound).all()
    assert (cohort.cases["sampled_history_ppg"] < 30).all()
    factors = cohort.manifest["transforms"][0]["ppg_target"]["factors"]
    assert set(factors) == set(cohort.cases["case_id"]) and all(f > 1 for f in factors.values())
    assert cohort.manifest["scoring_weights"]["passing_yards"] == pytest.approx(0.04)


def test_unreachable_ppg_target_fails(qb_source):
    op = {"op": "set_history_ppg", "target_ppg": -50.0, "stats": ["passing_yards"]}
    with pytest.raises(ValueError, match="not reachable"):
        generate_cohort(qb_source, recipe([op], "keep_donor"))


def test_mark_missing_blanks_opaque_signals_on_targeted_rows_only(qb_source):
    cohort = generate_cohort(
        qb_source, recipe([scale(["passing_yards"], 1.1, [1, 1])], "mark_missing")
    )
    games = cohort.games
    first = games["history_step"] == 1
    for column in position_schema("QB").opaque_columns:
        assert games.loc[first, column].isna().all()
        assert games.loc[~first, column].notna().all()
    assert cohort.manifest["opaque_cells_marked_missing"] == 4 * 7
    assert cohort.manifest["missing_history_values"]["qbr_total"] == 4
    assert np.isfinite(cohort.history).all()
    kept = generate_cohort(qb_source, recipe([scale(["passing_yards"], 1.1, [1, 1])], "keep_donor"))
    assert kept.games["qbr_total"].notna().all()
    assert kept.manifest["opaque_cells_marked_missing"] == 0


@pytest.mark.parametrize(
    "transforms,policy,error",
    [
        ([scale(["passing_yards"], 1.1)], None, "opaque_signal_policy is required"),
        ([], "keep_donor", "has no effect without transforms"),
        ([scale(["passing_yards"], 1.1)], "ignore", "opaque_signal_policy must be one of"),
        ([{"op": "warp", "stats": ["passing_yards"]}], "keep_donor", "unknown transform op"),
        ([{**scale(["passing_yards"], 1.1), "why": "x"}], "keep_donor", "unknown scale fields"),
        ([scale(["qbr_total"], 1.1)], "keep_donor", "opaque external signal"),
        ([scale(["team_rush_attempts"], 1.1)], "keep_donor", "team-accounting column"),
        ([scale(["implied_team_total"], 1.1)], "keep_donor", "held fixed"),
        ([scale(["nonsense"], 1.1)], "keep_donor", "not a transformable"),
        ([scale(["passing_yards", "passing_yards"], 1.1)], "keep_donor", "unique column names"),
        ([scale(["passing_yards"], 0.0)], "keep_donor", "factor must be positive"),
        ([scale(["passing_yards"], 1.1, [0, 2])], "keep_donor", "steps must be"),
        ([scale(["passing_yards"], 1.1, [2, 9])], "keep_donor", "steps must be"),
        (
            [{"op": "set_history_ppg", "target_ppg": 50.0, "stats": ["attempts"]}],
            "keep_donor",
            "at least one scoring target",
        ),
    ],
)
def test_invalid_transform_recipes_are_rejected(transforms, policy, error):
    with pytest.raises(ValueError, match=error):
        recipe(transforms, policy)
    with pytest.raises(ValueError, match="must be a list"):
        HistoryRecipe(name="x", transforms="scale", opaque_signal_policy="keep_donor")


def test_declined_ops_fail_before_sampling():
    schema = dataclasses.replace(position_schema("QB"), transform_support={"scale": "declined"})
    with pytest.raises(TransformUnsupported, match="does not support scale: declined"):
        parse_transform(scale(["passing_yards"], 1.1), schema, 3)
    with pytest.raises(TransformUnsupported, match="not declared"):
        parse_transform(
            {"op": "set_history_ppg", "target_ppg": 1.0, "stats": ["passing_yards"]}, schema, 3
        )


def test_recipe_round_trips_through_its_manifest_dict(qb_source):
    cohort = generate_cohort(
        qb_source, recipe([scale(["passing_yards"], 1.1, [1, 2])], "keep_donor")
    )
    again = HistoryRecipe.from_dict(json.loads(json.dumps(cohort.manifest["recipe"])))
    assert dataclasses.asdict(again) == cohort.manifest["recipe"]
    identity = sampling_identity_hash(cohort.manifest["recipe"])
    assert identity == generate_cohort(qb_source, recipe()).manifest["sampling_identity_sha256"]


def test_write_cohort_records_donor_games_and_fixture_fields(tmp_path, qb_source):
    cohort = generate_cohort(qb_source, recipe([scale(["passing_yards"], 1.1)], "keep_donor"))
    output = write_cohort(cohort, tmp_path / "fixture")
    manifest = json.loads((output / "manifest.json").read_text())
    assert "donor_games.parquet" in manifest["files"]
    donors = pd.read_parquet(output / "donor_games.parquet")
    assert "transformed" not in donors and len(donors) == len(cohort.games)
    assert manifest["fixture"] is True
    assert manifest["forecast_outcome"].startswith("none")
    assert manifest["static_context_policy"].startswith("non-temporal")
    plain = write_cohort(generate_cohort(qb_source, recipe()), tmp_path / "plain")
    assert "donor_games.parquet" not in json.loads((plain / "manifest.json").read_text())["files"]


@pytest.mark.parametrize(
    "name", ["qb_sustained_100pt.json", "qb_usage_step_up.json", "qb_efficiency_up.json"]
)
def test_shipped_presets_load_and_generate(qb_source_long, name):
    loaded = HistoryRecipe.from_dict(json.loads((RECIPES / name).read_text()))
    assert loaded.transforms and loaded.window == "exact"
    cohort = generate_cohort(qb_source_long, loaded)
    assert cohort.manifest["history_kind"] == "transformed"
    assert cohort.cases["exact_window"].all()
    if name == "qb_sustained_100pt.json":
        assert (abs(cohort.cases["generated_history_ppg"] - 100.0) <= 5.0).all()
    if name == "qb_usage_step_up.json":
        early = cohort.games["history_step"] <= 4
        assert (cohort.games.loc[early, "attempts"] == 9).all()
        assert (cohort.games.loc[~early, "attempts"] == 30).all()
