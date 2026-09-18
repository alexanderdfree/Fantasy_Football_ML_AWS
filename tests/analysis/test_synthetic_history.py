"""Boundary, provenance, and production-history replay checks; no model training."""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.synthetic_history import (
    HistoryRecipe,
    consume_source,
    generate_cohort,
    main,
    validate_history_frame,
    write_cohort,
)
from src.analysis.synthetic_history_schema import position_schema
from src.features.engineer import build_game_history_arrays
from src.qb.config import POSITION_CONFIG
from src.qb.features import get_feature_columns
from tests.analysis.conftest import SHORT_WEEKS

pytestmark = pytest.mark.unit

STRING_COLUMNS = ["player_id", "position", "season_type", "recent_team", "opponent_team"]


@pytest.fixture
def source(qb_source):
    """The shared prepared-QB fixture under the name these tests were written with."""
    return qb_source


def recipe(**kwargs):
    return HistoryRecipe(name="test", cases=4, history_games=3, block_games=2, **kwargs)


def test_reproducible_under_input_reordering_and_does_not_mutate_source(source):
    before = source.copy(deep=True)
    first = generate_cohort(source, recipe(mode="block_bootstrap"))
    second = generate_cohort(source.sample(frac=1, random_state=9), recipe(mode="block_bootstrap"))
    pd.testing.assert_frame_equal(first.games, second.games)
    pd.testing.assert_frame_equal(first.cases, second.cases)
    pd.testing.assert_frame_equal(first.context, second.context)
    np.testing.assert_array_equal(first.history, second.history)
    assert first.manifest == second.manifest
    pd.testing.assert_frame_equal(source, before)
    different = generate_cohort(source, recipe(seed=123, mode="block_bootstrap"))
    assert not first.games["donor_week"].equals(different.games["donor_week"])


@pytest.mark.parametrize("mode", ["replay", "block_bootstrap"])
def test_whole_game_signals_survive_together_and_forecast_is_excluded(source, mode):
    cohort = generate_cohort(source, recipe(mode=mode))
    assert "rolling_mean_passing_yards_L3" not in cohort.games
    original = source.set_index(["player_id", "season", "week"])
    cases = cohort.cases.set_index("case_id")
    for row in cohort.games.to_dict("records"):
        donor = original.loc[(row["donor_player_id"], row["donor_season"], row["donor_week"])]
        for column in POSITION_CONFIG.attn_history_stats:
            assert row[column] == donor[column]
        assert row["donor_week"] < cases.loc[row["case_id"], "forecast_week"]
        assert row["donor_season"] < 2024
    readiness = cohort.manifest["model_input_readiness"]
    assert readiness["attn_nn"]["ready"] is True
    for family in ("ridge", "nn", "lgbm"):
        assert readiness[family]["ready"] is False
        assert ("resampled" in readiness[family]["reason"]) is (mode == "block_bootstrap")
        assert ("truncate" in readiness[family]["reason"]) is (mode == "replay")


def test_exact_window_is_a_recipe_guarantee_and_unlocks_flat_families(source):
    exact = generate_cohort(source, recipe(window="exact"))
    assert exact.cases["exact_window"].all()
    assert (exact.cases["real_prior_games"] == 3).all()
    assert exact.manifest["exact_window_cases"] == 4
    assert all(entry["ready"] for entry in exact.manifest["model_input_readiness"].values())
    # Six games per season: only the forecast at the fourth game replays exactly.
    assert exact.manifest["eligible_windows"] == 4
    loose = generate_cohort(source, recipe())
    assert loose.manifest["eligible_windows"] == 12
    truncated = loose.cases[~loose.cases["exact_window"]]
    assert (truncated["real_prior_games"] > 3).all()
    reason = loose.manifest["model_input_readiness"]["ridge"]["reason"]
    assert reason.startswith(f"{len(truncated)} of {len(loose.cases)} cases truncate")


def test_context_is_the_forecast_row_only(source):
    cohort = generate_cohort(source, recipe(mode="block_bootstrap"))
    assert list(cohort.context["case_id"]) == list(cohort.cases["case_id"])
    assert list(cohort.context["case_index"]) == list(range(len(cohort.cases)))
    assert cohort.manifest["context_columns"] == get_feature_columns()
    assert not set(cohort.manifest["context_columns"]) & set(POSITION_CONFIG.targets)
    assert set(cohort.manifest["context_sequence_coupled_columns"]) <= set(get_feature_columns())
    original = source.set_index(["player_id", "season", "week"], drop=False)
    for row in cohort.context.to_dict("records"):
        forecast = original.loc[(row["donor_player_id"], row["donor_season"], row["forecast_week"])]
        assert row["week"] == row["forecast_week"]
        assert row["rolling_mean_passing_yards_L3"] == 9999
        for column in get_feature_columns():
            assert row[column] == float(forecast[column])


def test_replay_matches_production_arrays_for_original_forecast_rows(source):
    # Use all prior games so identity replay can be compared directly with the
    # real calendar, including the week-3 gap, on the shared production builder.
    cfg = HistoryRecipe(name="identity", cases=5, history_games=5, block_games=3)
    cohort = generate_cohort(source, cfg)
    production, mask = build_game_history_arrays(
        source, POSITION_CONFIG.attn_history_stats, POSITION_CONFIG.attn_max_seq_len
    )
    for i, row in cohort.cases.iterrows():
        index = source.index[
            source.player_id.eq(row.donor_player_id)
            & source.season.eq(row.donor_season)
            & source.week.eq(row.forecast_week)
        ][0]
        np.testing.assert_array_equal(cohort.history[i], production[index])
        np.testing.assert_array_equal(cohort.mask[i], mask[index])


def test_cases_do_not_share_history_and_padding_has_false_mask(source):
    cohort = generate_cohort(source, recipe())
    assert cohort.history.shape == (4, 17, len(POSITION_CONFIG.attn_history_stats))
    assert cohort.mask[:, :3].all()
    assert not cohort.mask[:, 3:].any()
    assert not cohort.history[:, 3:].any()
    field = POSITION_CONFIG.attn_history_stats.index("qbr_total")
    for i, case_id in enumerate(cohort.cases.case_id):
        games = cohort.games[cohort.games.case_id.eq(case_id)]
        np.testing.assert_array_equal(cohort.history[i, :3, field], games.qbr_total.iloc[::-1])


def test_missing_external_signal_is_preserved_and_reported(source):
    source["qbr_total"] = np.nan
    cohort = generate_cohort(source, recipe())
    assert cohort.games.qbr_total.isna().all()
    assert cohort.manifest["missing_history_values"]["qbr_total"] == 12
    assert np.isfinite(cohort.history).all()
    assert not cohort.history[:, :, POSITION_CONFIG.attn_history_stats.index("qbr_total")].any()


def test_history_filter_uses_shared_scoring_and_never_final_forecast_outcomes(source):
    cfg = recipe(min_history_ppg=15, max_history_ppg=30)
    first = generate_cohort(source, cfg)
    source.loc[source.week.eq(7), "passing_yards"] = 9999
    second = generate_cohort(source, cfg)
    np.testing.assert_array_equal(first.history, second.history)
    pd.testing.assert_frame_equal(
        first.cases.drop(columns="case_id"), second.cases.drop(columns="case_id")
    )
    row = first.games.iloc[0]
    assert row.fantasy_points == pytest.approx(row.passing_yards * 0.04 + 8 - 2 + 2)


def test_recipe_hash_is_invariant_to_bound_spelling(source):
    integer = generate_cohort(source, recipe(min_history_ppg=15, max_history_ppg=30))
    floating = generate_cohort(source, recipe(min_history_ppg=15.0, max_history_ppg=30.0))
    assert integer.manifest["recipe_sha256"] == floating.manifest["recipe_sha256"]
    assert list(integer.cases["case_id"]) == list(floating.cases["case_id"])
    assert integer.manifest["recipe"]["min_history_ppg"] == 15.0


def test_consumed_values_hash_ignores_string_dtype_spelling(source, tmp_path):
    schema = position_schema("QB")
    seasons = tuple(range(2013, 2024))
    as_object = source.astype({column: object for column in STRING_COLUMNS})
    source.to_parquet(tmp_path / "source.parquet", index=False)
    round_tripped = pd.read_parquet(tmp_path / "source.parquet")
    hashes = {
        consume_source(frame, position="QB", donor_seasons=seasons, schema=schema).values_sha256
        for frame in (source, as_object, round_tripped)
    }
    assert len(hashes) == 1


def test_unsupported_archetype_is_reported_instead_of_relaxing_recipe(source):
    with pytest.raises(ValueError, match="no eligible donor histories"):
        generate_cohort(source, recipe(min_history_ppg=100))


@pytest.mark.parametrize(
    "change,error",
    [
        ({"completions": 31}, "completions <= attempts"),
        ({"passing_tds": 21}, "passing_tds <= completions"),
        ({"rushing_tds": 5}, "rushing_tds <= carries"),
        ({"interceptions": 11}, "interceptions exceed"),
        ({"carries": 2.5}, "integer counts"),
        ({"passing_yards": np.inf}, "infinity"),
        ({"passing_yards": 1e100}, "float32 range"),
        ({"fumbles_lost": np.nan}, "must be observed"),
        ({"snap_pct_raw": 90}, "snap_pct_raw must be within"),
        ({"qbr_total": 101}, "qbr_total must be within"),
        ({"team_rush_attempts": 3}, "carries <= team_rush_attempts"),
        ({"week": 1.5}, "finite integers"),
        ({"player_id": ""}, "nonempty string"),
        ({"prior_season_mean_passing_yards": np.nan}, "feature columns must be finite"),
        ({"depth_chart_rank": "starter"}, "feature columns must be numeric"),
    ],
)
def test_invalid_donor_data_fails_before_generation(source, change, error):
    for column, value in change.items():
        source[column] = source[column].astype(object)
        source.loc[0, column] = value
    with pytest.raises(ValueError, match=error):
        generate_cohort(source, recipe())


def test_missing_columns_fail(source):
    with pytest.raises(ValueError, match="missing production history columns"):
        generate_cohort(source.drop(columns="qbr_total"), recipe())
    with pytest.raises(ValueError, match="missing production feature columns"):
        generate_cohort(source.drop(columns="rolling_mean_passing_yards_L3"), recipe())


def test_duplicate_game_keys_exclude_the_player_season_and_are_recorded(source):
    twelve = HistoryRecipe(name="test", cases=12, history_games=3, block_games=2)
    clean = generate_cohort(source, twelve)
    assert clean.manifest["donor_pool_exclusions"]["duplicate_game_keys"] == []
    # p1's 2022 week 1 appears twice: that player-season's sequence is ambiguous.
    duplicated = pd.concat([source, source.iloc[:1]], ignore_index=True)
    cohort = generate_cohort(duplicated, twelve)
    exclusions = cohort.manifest["donor_pool_exclusions"]
    assert exclusions["duplicate_game_keys"] == [
        {"player_id": "p1", "season": 2022, "weeks": [1], "rows": 2}
    ]
    assert "never merged" in exclusions["policy"]
    assert cohort.manifest["source_rows"] == clean.manifest["source_rows"] - len(SHORT_WEEKS)
    donors = set(zip(cohort.cases.donor_player_id, cohort.cases.donor_season, strict=True))
    assert ("p1", 2022) not in donors and donors
    assert not (cohort.games.donor_player_id.eq("p1") & cohort.games.donor_season.eq(2022)).any()
    # The exported source is never repaired: every player-season ambiguous fails loud.
    with pytest.raises(ValueError, match="every QB player-season .* duplicate"):
        generate_cohort(pd.concat([source, source], ignore_index=True), recipe())


def test_validate_history_frame_is_reusable_on_generated_games(source):
    cohort = generate_cohort(source, recipe())
    schema = position_schema("QB")
    validate_history_frame(cohort.games, schema, stage="generated")
    broken = cohort.games.copy()
    broken.loc[0, "completions"] = broken.loc[0, "attempts"] + 1
    with pytest.raises(ValueError, match="generated violates completions <= attempts"):
        validate_history_frame(broken, schema, stage="generated")
    with pytest.raises(ValueError, match="generated is missing schema columns"):
        validate_history_frame(cohort.games.drop(columns="sacks"), schema, stage="generated")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": "K"},
        {"schema_version": 1},
        {"cases": 0},
        {"cases": True},
        {"seed": -1},
        {"history_games": 18},
        {"block_games": 9, "mode": "block_bootstrap"},
        {"mode": "scale"},
        {"window": "sometimes"},
        {"donor_seasons": [2025]},
        {"donor_seasons": 2022},
        {"donor_seasons": [2022, 2022]},
        {"min_history_ppg": float("nan")},
        {"min_history_ppg": 30, "max_history_ppg": 20},
    ],
)
def test_recipe_rejects_unsupported_or_ambiguous_requests(kwargs):
    with pytest.raises(ValueError):
        HistoryRecipe(name="invalid", **kwargs)


def test_artifacts_round_trip_hashes_no_overwrite_and_cli(tmp_path, source, capsys):
    import hashlib

    src = tmp_path / "source.parquet"
    source.to_parquet(src, index=False)
    config = tmp_path / "recipe.json"
    config.write_text(json.dumps({"name": "cli", "cases": 3, "history_games": 3}))
    output = tmp_path / "run"
    assert main(["--source", str(src), "--recipe", str(config), "--output", str(output)]) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["cases"] == 3
    cases = pd.read_parquet(output / "cases.parquet")
    assert stdout["exact_window_cases"] == int(cases["exact_window"].sum())
    assert stdout["model_input_readiness"]["attn_nn"] is True
    assert stdout["model_input_readiness"]["ridge"] is bool(cases["exact_window"].all())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_file_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert set(manifest["files"]) == {
        "games.parquet",
        "cases.parquet",
        "context.parquet",
        "history.npz",
    }
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with np.load(output / "history.npz", allow_pickle=False) as arrays:
        assert arrays["history"].shape[0] == 3
    assert len(pd.read_parquet(output / "games.parquet")) == 9
    assert len(pd.read_parquet(output / "context.parquet")) == 3
    with pytest.raises(FileExistsError):
        write_cohort(generate_cohort(source, recipe()), output)
    with pytest.raises(SystemExit) as exit_info:
        main(["--source", str(src), "--recipe", str(config), "--output", str(output)])
    assert exit_info.value.code == 2
    assert not list(tmp_path.glob(".run-*"))


def test_unknown_recipe_field_fails():
    with pytest.raises(ValueError, match="unknown recipe fields"):
        HistoryRecipe.from_dict({"name": "typo", "case": 5})


def test_blocks_are_contiguous_in_observed_game_order(source):
    cohort = generate_cohort(source, recipe(mode="block_bootstrap"))
    observed = [1, 2, 4, 5, 6, 7]
    for _, block in cohort.games.groupby(["case_id", "block_id"]):
        positions = [observed.index(week) for week in block.donor_week]
        assert all(b == a + 1 for a, b in zip(positions, positions[1:], strict=False))


def test_one_game_replay_and_negative_observed_yards_are_supported(source):
    source["rushing_yards"] = -2
    cohort = generate_cohort(source, HistoryRecipe(name="one-game", history_games=1, cases=1))
    assert cohort.mask.sum() == 1
    assert cohort.games.rushing_yards.iloc[0] == -2
