"""Boundary, provenance, and production-history replay checks; no model training."""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.synthetic_history import (
    HistoryRecipe,
    generate_cohort,
    main,
    write_cohort,
)
from src.features.engineer import build_game_history_arrays
from src.qb.config import POSITION_CONFIG

pytestmark = pytest.mark.unit


@pytest.fixture
def source():
    rows = []
    for player in ("p1", "p2"):
        for season in (2022, 2023, 2025):
            for week in (1, 2, 4, 5, 6, 7):
                row = dict.fromkeys(POSITION_CONFIG.attn_history_stats, 0.0)
                row.update(
                    player_id=player,
                    season=season,
                    week=week,
                    position="QB",
                    season_type="REG",
                    recent_team="KC",
                    opponent_team="BUF",
                    attempts=30,
                    completions=20,
                    passing_yards=180 + week * 10 + (20 if player == "p2" else 0),
                    passing_tds=2,
                    interceptions=1,
                    carries=4,
                    rushing_yards=20,
                    rushing_tds=0,
                    fumbles_lost=0,
                    snap_pct_raw=0.9,
                    # Deliberately distinctive correlated opaque fields.
                    qbr_total=week * 10,
                    pts_added=week,
                    pass_yards_gained_exp=150 + week,
                    team_rush_attempts=25,
                    team_rushing_yards=100,
                    team_points_scored=24,
                    opp_team_points_scored=21,
                    # Must never be copied into generated histories.
                    rolling_mean_passing_yards_L3=9999,
                )
                rows.append(row)
    return pd.DataFrame(rows)


def recipe(**kwargs):
    return HistoryRecipe(name="test", cases=4, history_games=3, block_games=2, **kwargs)


def test_reproducible_under_input_reordering_and_does_not_mutate_source(source):
    before = source.copy(deep=True)
    first = generate_cohort(source, recipe(mode="block_bootstrap"))
    second = generate_cohort(source.sample(frac=1, random_state=9), recipe(mode="block_bootstrap"))
    pd.testing.assert_frame_equal(first.games, second.games)
    pd.testing.assert_frame_equal(first.cases, second.cases)
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
    assert cohort.manifest["full_model_input_ready"] is False


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
        ({"snap_pct_raw": 90}, "fraction"),
        ({"qbr_total": 101}, "qbr_total"),
        ({"team_rush_attempts": 3}, "team rushing attempts"),
        ({"week": 1.5}, "finite integers"),
        ({"player_id": ""}, "nonempty string"),
    ],
)
def test_invalid_donor_data_fails_before_generation(source, change, error):
    for column, value in change.items():
        source[column] = source[column].astype(object)
        source.loc[0, column] = value
    with pytest.raises(ValueError, match=error):
        generate_cohort(source, recipe())


def test_missing_column_and_duplicate_keys_fail(source):
    with pytest.raises(ValueError, match="missing production history columns"):
        generate_cohort(source.drop(columns="qbr_total"), recipe())
    with pytest.raises(ValueError, match="duplicate"):
        generate_cohort(pd.concat([source, source.iloc[:1]]), recipe())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": "RB"},
        {"schema_version": 2},
        {"cases": 0},
        {"cases": True},
        {"seed": -1},
        {"history_games": 18},
        {"block_games": 9, "mode": "block_bootstrap"},
        {"mode": "scale"},
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
    assert json.loads(capsys.readouterr().out)["cases"] == 3
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_file_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with np.load(output / "history.npz", allow_pickle=False) as arrays:
        assert arrays["history"].shape[0] == 3
    assert len(pd.read_parquet(output / "games.parquet")) == 9
    with pytest.raises(FileExistsError):
        write_cohort(generate_cohort(source, recipe()), output)
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
