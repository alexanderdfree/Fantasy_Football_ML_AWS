"""DST cohorts: team donors, tier scoring and the never-resampled opponent stream."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import synthetic_history_sources as sources
from src.analysis.synthetic_history import (
    HistoryRecipe,
    generate_cohort,
    validate_opponent_per_game,
    write_cohort,
)
from src.analysis.synthetic_history_schema import position_schema
from src.analysis.synthetic_replay import load_cohort, replay_cohort
from src.analysis.synthetic_replay import main as replay_main
from src.features.engineer import build_game_history_arrays, build_opp_defense_history_arrays
from tests.analysis.conftest import (
    DST_POISON,
    LONG_WEEKS,
    dst_rows,
    fake_schedules,
    opponent_per_game_rows,
    opponent_weekly_rows,
    position_rows,
)
from tests.analysis.fake_bundles import fake_artifacts

pytestmark = pytest.mark.unit

RECIPES = Path(__file__).resolve().parents[2] / "src/analysis/synthetic_history_recipes"
SCHEMA = position_schema("DST")


def recipe(**kwargs):
    kwargs.setdefault("cases", 4)
    kwargs.setdefault("history_games", 3)
    kwargs.setdefault("block_games", 2)
    return HistoryRecipe(name="test", position="DST", **kwargs)


def test_generation_requires_the_opponent_frame_and_refuses_it_elsewhere():
    with pytest.raises(ValueError, match="need the opponent per-game frame"):
        generate_cohort(dst_rows(), recipe())
    with pytest.raises(ValueError, match="no opponent stream"):
        generate_cohort(
            position_rows("QB"),
            HistoryRecipe(name="qb", cases=2, history_games=3),
            opponent_per_game=opponent_per_game_rows(),
        )


def test_team_donors_carry_their_own_stream_and_static_context():
    source, per_game = dst_rows(), opponent_per_game_rows()
    cohort = generate_cohort(source, recipe(mode="block_bootstrap"), opponent_per_game=per_game)
    assert DST_POISON not in cohort.games
    assert cohort.manifest["donor_identity"] == "team"
    assert cohort.manifest["fantasy_points_policy"] == "assert_equal"
    assert cohort.manifest["opponent_history_columns"] == list(SCHEMA.opponent_history_columns)
    assert cohort.manifest["opponent_history_shape"] == [4, 17, 7]
    assert "never resampled" in cohort.manifest["opponent_stream_policy"]
    assert cohort.manifest["opponent_per_game_rows"] == len(per_game)
    original = source.set_index(["player_id", "season", "week"], drop=False)
    for row in cohort.games.to_dict("records"):
        donor = original.loc[(row["donor_player_id"], row["donor_season"], row["donor_week"])]
        for column in SCHEMA.history_columns:
            assert row[column] == donor[column]
        assert row["fantasy_points"] == donor["fantasy_points"]
    for row in cohort.context.to_dict("records"):
        forecast = original.loc[(row["donor_player_id"], row["donor_season"], row["forecast_week"])]
        assert row[DST_POISON] == 9999 and row["opponent_team"] == forecast["opponent_team"]
        assert row["week"] == forecast["week"] and row["rest_days"] == forecast["rest_days"]
    assert cohort.manifest["context_sequence_coupled_columns"] == ["week", "rest_days"]


def test_opponent_stream_matches_production_and_never_resamples():
    source, per_game = dst_rows(LONG_WEEKS), opponent_per_game_rows()
    exact = generate_cohort(
        source, recipe(cases=6, history_games=5, window="exact"), opponent_per_game=per_game
    )
    forecast = exact.cases[["donor_player_id", "donor_season", "forecast_week"]]
    keyed = source.set_index(["player_id", "season", "week"])
    lookup = keyed.loc[
        list(
            zip(
                forecast.donor_player_id, forecast.donor_season, forecast.forecast_week, strict=True
            )
        )
    ].reset_index()
    # Production's builder on the same forecast games gives the same tensors.
    expected, expected_mask = build_opp_defense_history_arrays(
        lookup[["opponent_team", "season", "week"]],
        per_game,
        list(SCHEMA.opponent_history_columns),
        17,
    )
    np.testing.assert_array_equal(exact.opponent_history, expected)
    np.testing.assert_array_equal(exact.opponent_mask, expected_mask)
    np.testing.assert_array_equal(
        exact.cases["opponent_prior_games"].to_numpy(), expected_mask.sum(axis=1)
    )
    # The player history also matches production on exact windows.
    production, mask = build_game_history_arrays(source, list(SCHEMA.history_columns), 17)
    for i, row in exact.cases.iterrows():
        index = source.index[
            source.player_id.eq(row.donor_player_id)
            & source.season.eq(row.donor_season)
            & source.week.eq(row.forecast_week)
        ][0]
        np.testing.assert_array_equal(exact.history[i], production[index])
        np.testing.assert_array_equal(exact.mask[i], mask[index])
    # A bootstrap or transformed cohort resamples the defense's games but keeps
    # each forecast's real opponent stream.
    bootstrap = generate_cohort(
        source, recipe(cases=6, history_games=5, mode="block_bootstrap"), opponent_per_game=per_game
    )
    keyed = {
        (r.donor_player_id, r.donor_season, r.forecast_week): i for i, r in exact.cases.iterrows()
    }
    for i, row in bootstrap.cases.iterrows():
        key = (row.donor_player_id, row.donor_season, row.forecast_week)
        if key in keyed:
            np.testing.assert_array_equal(
                bootstrap.opponent_history[i], exact.opponent_history[keyed[key]]
            )
    # The exported opponent games are the tensor rows, newest first.
    games = exact.opponent_games
    for i in range(len(exact.cases)):
        rows = games[games["case_index"].eq(i)].sort_values("history_slot")
        assert len(rows) == exact.cases.loc[i, "opponent_prior_games"]
        np.testing.assert_array_equal(
            exact.opponent_history[i, : len(rows)],
            rows[list(SCHEMA.opponent_history_columns)].to_numpy(dtype=np.float32),
        )
        assert rows["week"].is_monotonic_decreasing


def test_reproducible_under_reordering_and_points_are_the_shared_scoring():
    source, per_game = dst_rows(), opponent_per_game_rows()
    first = generate_cohort(source, recipe(mode="block_bootstrap"), opponent_per_game=per_game)
    second = generate_cohort(
        source.sample(frac=1, random_state=5),
        recipe(mode="block_bootstrap"),
        opponent_per_game=per_game.sample(frac=1, random_state=6),
    )
    pd.testing.assert_frame_equal(first.games, second.games)
    assert first.manifest == second.manifest
    np.testing.assert_array_equal(first.opponent_history, second.opponent_history)
    # 3 sacks + 2 ints + 2 fumble rec + 1 forced + tier(17 PA) 1 + tier(yards) per week.
    row = first.games.iloc[0]
    yards_bonus = {True: 0.0, False: -1.0}[row["yards_allowed"] < 350]
    assert row["fantasy_points"] == pytest.approx(3 + 2 + 2 + 1 + 1 + yards_bonus)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"player_id": "KCX"}, "player_id must equal recent_team"),
        ({"points_allowed": -1}, "nonnegative integer"),
        ({"def_sacks": 2.5}, "nonnegative integer"),
        ({"yards_allowed": 1200}, "yards_allowed must be within"),
        ({"opp_qb_epa": np.inf}, "infinity"),
        ({"def_ints": np.nan}, "must be observed"),
        ({"fantasy_points": 99.0}, "disagree with the shared DST scoring"),
    ],
)
def test_invalid_donor_data_fails_before_generation(change, error):
    source = dst_rows()
    for column, value in change.items():
        source[column] = source[column].astype(object)
        source.loc[0, column] = value
    with pytest.raises(ValueError, match=error):
        generate_cohort(source, recipe(), opponent_per_game=opponent_per_game_rows())


def test_source_without_a_points_column_is_rejected():
    with pytest.raises(ValueError, match="missing production history columns"):
        generate_cohort(
            dst_rows().drop(columns="fantasy_points"),
            recipe(),
            opponent_per_game=opponent_per_game_rows(),
        )


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda f: f.drop(columns="off_pts_scored"), "missing columns"),
        (lambda f: f.assign(week=[np.nan] + list(f["week"][1:])), "missing team/season/week"),
        (lambda f: pd.concat([f, f.iloc[:1]]), "duplicate team/season/week"),
        (lambda f: f.assign(off_ints=[np.nan] + list(f["off_ints"][1:])), "observed and finite"),
        (lambda f: f.assign(week=[1.5] + list(f["week"][1:])), "finite integers"),
        (lambda f: f.iloc[:0], "frame is empty"),
        (
            lambda f: f.assign(off_pts_scored=0.0),
            "touchdowns in a game recorded with zero points",
        ),
        (
            lambda f: f.assign(off_pts_scored=[0.0] + list(f["off_pts_scored"][1:])),
            "touchdowns in a game recorded with zero points",
        ),
    ],
)
def test_invalid_opponent_frames_fail(mutate, error):
    with pytest.raises(ValueError, match=error):
        validate_opponent_per_game(mutate(opponent_per_game_rows()), SCHEMA)


def test_observed_zero_turnover_histories_are_preserved():
    per_game = opponent_per_game_rows().assign(off_ints=0.0, off_fumbles_lost=0.0)
    validated = validate_opponent_per_game(per_game, SCHEMA)
    assert validated[["off_ints", "off_fumbles_lost"]].eq(0).all().all()
    cohort = generate_cohort(dst_rows(), recipe(), opponent_per_game=per_game)
    assert cohort.cases["opponent_prior_games"].gt(0).all()
    for column in ("off_ints", "off_fumbles_lost"):
        index = SCHEMA.opponent_history_columns.index(column)
        assert (cohort.opponent_history[:, :, index] == 0).all()
        with pytest.raises(ValueError, match="missing columns"):
            validate_opponent_per_game(per_game.drop(columns=column), SCHEMA)


def test_dst_export_rejects_incompatible_team_cache_before_build(monkeypatch, tmp_path):
    team_path = tmp_path / "team_stats_2023.parquet"
    pd.DataFrame({"team": ["KC"], "season": [2023]}).to_parquet(team_path)
    before = team_path.read_bytes()
    monkeypatch.setattr(sources, "dst_raw_cache_files", lambda: {"team_stats": team_path})
    monkeypatch.setattr(
        "src.dst.data.build_data",
        lambda **kwargs: pytest.fail("incompatible cache reached the native build"),
    )
    with pytest.raises(ValueError, match="team_stats.*incompatible"):
        sources.export_dst_source()
    assert team_path.read_bytes() == before


def test_dst_export_enforces_cache_only_during_native_build(monkeypatch, tmp_path):
    from src.data import loader
    from src.data.release import DataReleaseError

    team_path = tmp_path / "team_stats_2023.parquet"
    pd.DataFrame({"_team_stats_schema_v2": [True]}).to_parquet(team_path)
    before = team_path.read_bytes()
    monkeypatch.setattr(sources, "dst_raw_cache_files", lambda: {"team_stats": team_path})
    monkeypatch.setattr(
        loader.nfl_source,
        "team_week_stats_release",
        lambda season: pytest.fail("cache-only export attempted a source fetch"),
    )
    # A newly requested producer dependency must not recover from the network,
    # even when the preflight cache was compatible and has no release seal.
    monkeypatch.setattr(
        "src.dst.data.build_data",
        lambda **kwargs: loader.load_team_week_stats([2022], cache_dir=str(tmp_path)),
    )
    with pytest.raises(DataReleaseError, match="missing or incompatible"):
        sources.export_dst_source()
    assert team_path.read_bytes() == before
    assert not (tmp_path / "team_stats_2022.parquet").exists()


@pytest.mark.parametrize("score_state", ["missing_row", "null_score", "observed_zero"])
def test_dst_export_requires_observed_schedule_scores(monkeypatch, tmp_path, score_state):
    from types import SimpleNamespace

    sources.get_config("DST")  # Resolve runner callbacks before patching their modules.
    source = dst_rows()
    weekly = opponent_weekly_rows().assign(passing_tds=0.0, rushing_tds=0.0)
    weekly = weekly.loc[weekly.recent_team.isin(["LV", "DEN"])]
    schedules = fake_schedules()
    schedules = schedules.loc[schedules.home_team.eq("LV")].assign(away_team="DEN")
    # Both offenses have no TDs, so a missing schedule cannot be detected from
    # touchdown counts. Exercise the production relocation mapping as well.
    schedules["home_team"] = schedules["home_team"].replace({"LV": "OAK"})
    selected = schedules.home_team.eq("OAK") & schedules.season.eq(2022) & schedules.week.eq(1)
    schedules.loc[selected, ["home_score", "away_score"]] = 0.0
    if score_state == "missing_row":
        schedules = schedules.loc[~selected]
    elif score_state == "null_score":
        schedules.loc[selected, "home_score"] = np.nan
    caches = {
        name: tmp_path / f"{name}_2012_2025.parquet"
        for name in ("weekly", "schedules", "team_stats")
    }
    weekly.to_parquet(caches["weekly"])
    schedules.to_parquet(caches["schedules"])
    pd.DataFrame({"_team_stats_schema_v2": [True]}).to_parquet(caches["team_stats"])
    before = {name: path.read_bytes() for name, path in caches.items()}
    monkeypatch.setattr(sources, "dst_raw_cache_files", lambda: caches)
    monkeypatch.setattr("src.dst.data.build_data", lambda **kwargs: source.copy())
    monkeypatch.setattr("src.dst.targets.compute_targets", lambda frame: frame)
    monkeypatch.setattr("src.dst.features.compute_features", lambda frame: None)
    monkeypatch.setattr(
        sources,
        "_prepare_position_data",
        lambda *args: SimpleNamespace(
            train=source, feature_columns=SCHEMA.feature_columns, data_id="dst-score-coverage"
        ),
    )
    monkeypatch.setattr(
        "src.shared.weather_features._load_schedules",
        lambda: pd.read_parquet(caches["schedules"]),
    )
    output = tmp_path / "export"
    if score_state == "observed_zero":
        sources.write_sources("DST", output)
        per_game = pd.read_parquet(output / "dst_opponent_per_game.parquet")
        observed = per_game.loc[
            per_game.opponent_team.eq("LV") & per_game.season.eq(2022) & per_game.week.eq(1)
        ]
        assert len(observed) == 1 and observed.off_pts_scored.iloc[0] == 0.0
    else:
        with pytest.raises(ValueError, match="opponent weekly games lack observed schedule scores"):
            sources.write_sources("DST", output)
        assert not output.exists()
    assert {name: path.read_bytes() for name, path in caches.items()} == before


def test_forecast_opponents_missing_from_the_frame_fail_loudly():
    per_game = opponent_per_game_rows()
    thin = per_game[~per_game["opponent_team"].eq("DEN")]
    with pytest.raises(ValueError, match=r"no games for forecast opponents \[\('DEN'"):
        generate_cohort(dst_rows(), recipe(cases=8), opponent_per_game=thin)
    # A covered opponent with no game before the forecast week is legitimate.
    early = per_game[per_game["week"].ge(6)]
    cohort = generate_cohort(dst_rows(), recipe(cases=8), opponent_per_game=early)
    assert (cohort.cases["opponent_prior_games"] == 0).any()


def test_transforms_scale_counts_without_accounting_and_decline_ppg_targets():
    source, per_game = dst_rows(), opponent_per_game_rows()
    scaled = generate_cohort(
        source,
        recipe(
            transforms=[{"op": "scale", "stats": ["def_sacks", "points_allowed"], "factor": 2.0}],
            opaque_signal_policy="mark_missing",
        ),
        opponent_per_game=per_game,
    )
    assert (scaled.games["def_sacks"] == 6).all() and (scaled.games["points_allowed"] == 34).all()
    assert scaled.games["opp_qb_epa"].isna().all()
    assert scaled.manifest["transforms"][0]["team_accounting"] == []
    assert scaled.manifest["scoring_weights"] is None  # tiered points have no unit weights
    # Points are recomputed through the shared tier scoring (34 allowed = -1).
    assert scaled.games["fantasy_points"].iloc[0] == pytest.approx(
        6 + 2 + 2 + 1 - 1 + (0.0 if scaled.games["yards_allowed"].iloc[0] < 350 else -1.0)
    )
    np.testing.assert_array_equal(
        scaled.opponent_history,
        generate_cohort(source, recipe(), opponent_per_game=per_game).opponent_history,
    )
    assert scaled.manifest["model_input_readiness"]["ridge"]["ready"] is False
    with pytest.raises(ValueError, match="tier bonuses"):
        recipe(
            transforms=[{"op": "set_history_ppg", "target_ppg": 20.0, "stats": ["def_sacks"]}],
            opaque_signal_policy="keep_donor",
        )


@pytest.mark.parametrize("mode", ["replay", "block_bootstrap"])
def test_shipped_recipes_generate_within_their_bands(mode):
    loaded = HistoryRecipe.from_dict(json.loads((RECIPES / f"dst_{mode}.json").read_text()))
    assert loaded.position == "DST" and loaded.mode == mode
    cohort = generate_cohort(
        dst_rows(LONG_WEEKS), loaded, opponent_per_game=opponent_per_game_rows()
    )
    assert cohort.manifest["eligible_windows"] > 0
    band = (loaded.min_history_ppg, loaded.max_history_ppg)
    assert cohort.cases["donor_history_ppg"].between(*band).all()


def test_replay_streams_the_opponent_and_the_control_rebuilds_it(tmp_path, dst_schedules):
    source, per_game, weekly = (
        dst_rows(LONG_WEEKS),
        opponent_per_game_rows(),
        opponent_weekly_rows(),
    )
    models = tmp_path / "models"
    fake_artifacts(models, "DST", families=("attn_nn", "ridge"))
    cohort = write_cohort(
        generate_cohort(
            source, recipe(cases=5, history_games=5, window="exact"), opponent_per_game=per_game
        ),
        tmp_path / "cohort",
    )
    loaded = load_cohort(cohort)
    assert {"opponent_history", "opponent_mask"} <= set(loaded.arrays)
    assert "opponent_games.parquet" in loaded.manifest["files"]
    predictions, manifest = replay_cohort(
        loaded, str(models), ["attn_nn", "ridge"], source=source, opponent_weekly=weekly
    )
    control = manifest["identity_control"]
    assert control["status"] == "passed" and manifest["opponent_stream"] is True
    assert control["families"]["attn_nn"]["checks"] == [
        "static_values",
        "opponent_stream",
        "history_prefix",
        "mask",
        "predictions_on_exact_windows",
    ]
    assert control["families"]["ridge"]["checks"] == [
        "static_values",
        "predictions_on_exact_windows",
    ]
    assert np.isfinite(predictions[[c for c in predictions if c.startswith("pred_")]]).all().all()
    # Tier scoring is format-invariant: every total column agrees.
    np.testing.assert_array_equal(
        predictions["pred_attn_nn_total"], predictions["pred_attn_nn_total_standard"]
    )
    pinned = control["schedules_cache_sha256"]  # the local raw cache, when present
    assert pinned is None or len(pinned) == 64
    assert manifest["families"]["attn_nn"]["opponent_history"] == list(
        SCHEMA.opponent_history_columns
    )
    assert (
        manifest["opponent_per_game_values_sha256"]
        == loaded.manifest["opponent_per_game_values_sha256"]
    )
    # A wrong or missing weekly frame breaks the control loudly, but only the
    # attention family needs it.
    with pytest.raises(ValueError, match="needs the opponent weekly"):
        replay_cohort(loaded, str(models), ["attn_nn"], source=source)
    _, ridge_only = replay_cohort(loaded, str(models), ["ridge"], source=source)
    assert ridge_only["identity_control"]["status"] == "passed"
    assert ridge_only["identity_control"]["opponent_weekly_file_sha256"] is None
    assert ridge_only["identity_control"]["schedules_cache_sha256"] is None
    with pytest.raises(ValueError, match="weekly frame lacks columns: \\['sack_fumbles_lost'\\]"):
        replay_cohort(
            loaded,
            str(models),
            ["attn_nn"],
            source=source,
            opponent_weekly=weekly.drop(columns="sack_fumbles_lost"),
        )
    altered = weekly.assign(passing_yards=weekly["passing_yards"] + 1.0)
    with pytest.raises(ValueError, match="opponent_stream"):
        replay_cohort(loaded, str(models), ["attn_nn"], source=source, opponent_weekly=altered)


def test_replay_refuses_mismatched_streams(tmp_path, qb_source):
    qb_models, dst_models = tmp_path / "qb", tmp_path / "dst"
    fake_artifacts(qb_models, "QB")
    fake_artifacts(dst_models, "DST")
    dst_cohort = write_cohort(
        generate_cohort(dst_rows(), recipe(), opponent_per_game=opponent_per_game_rows()),
        tmp_path / "dst-cohort",
    )
    qb_cohort = write_cohort(
        generate_cohort(qb_source, HistoryRecipe(name="qb", cases=3, history_games=3)),
        tmp_path / "qb-cohort",
    )
    with pytest.raises(ValueError, match="position"):
        replay_cohort(load_cohort(dst_cohort), str(qb_models), ["attn_nn"])
    with pytest.raises(ValueError, match="position"):
        replay_cohort(load_cohort(qb_cohort), str(dst_models), ["attn_nn"])
    with pytest.raises(ValueError, match="no opponent stream; do not pass"):
        replay_cohort(
            load_cohort(qb_cohort),
            str(qb_models),
            ["attn_nn"],
            source=qb_source,
            opponent_weekly=opponent_weekly_rows(),
        )
    # A DST cohort written without its stream is refused at load time.
    naked = dataclasses.replace(
        generate_cohort(dst_rows(), recipe(), opponent_per_game=opponent_per_game_rows()),
        opponent_history=None,
        opponent_mask=None,
        opponent_games=None,
    )
    stripped = write_cohort(naked, tmp_path / "stripped")
    with pytest.raises(ValueError, match="opponent stream disagrees"):
        load_cohort(stripped)


def test_cli_round_trip_with_the_opponent_frame(tmp_path, dst_schedules, capsys):
    from src.analysis.synthetic_history import main as generate_main

    source, per_game, weekly = (
        dst_rows(LONG_WEEKS),
        opponent_per_game_rows(),
        opponent_weekly_rows(),
    )
    source.to_parquet(tmp_path / "dst.parquet", index=False)
    per_game.to_parquet(tmp_path / "per_game.parquet", index=False)
    weekly.to_parquet(tmp_path / "weekly.parquet", index=False)
    (tmp_path / "recipe.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "name": "cli",
                "position": "DST",
                "cases": 3,
                "history_games": 5,
                "window": "exact",
            }
        )
    )
    argv = [
        "--source",
        str(tmp_path / "dst.parquet"),
        "--recipe",
        str(tmp_path / "recipe.json"),
        "--output",
        str(tmp_path / "cohort"),
    ]
    with pytest.raises(SystemExit) as exit_info:
        generate_main(argv)
    assert exit_info.value.code == 2 and "opponent per-game" in capsys.readouterr().err
    assert generate_main([*argv, "--opponent-per-game", str(tmp_path / "per_game.parquet")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["opponent_stream"] is True and summary["cases"] == 3
    manifest = json.loads((tmp_path / "cohort" / "manifest.json").read_text())
    assert manifest["opponent_per_game_file_sha256"] == sources.file_digest(
        tmp_path / "per_game.parquet"
    )
    models = tmp_path / "models"
    fake_artifacts(models, "DST")
    assert (
        replay_main(
            [
                "--cohort",
                str(tmp_path / "cohort"),
                "--output",
                str(tmp_path / "replay"),
                "--model-dir",
                str(models),
                "--source",
                str(tmp_path / "dst.parquet"),
                "--opponent-weekly",
                str(tmp_path / "weekly.parquet"),
            ]
        )
        == 0
    )
    replay_manifest = json.loads((tmp_path / "replay" / "replay_manifest.json").read_text())
    assert replay_manifest["identity_control"]["status"] == "passed"
    assert replay_manifest["identity_control"][
        "opponent_weekly_file_sha256"
    ] == sources.file_digest(tmp_path / "weekly.parquet")


def test_dst_export_publishes_the_stream_inputs(monkeypatch, tmp_path, dst_schedules):
    from types import SimpleNamespace

    # get_config lazily imports the native runner and binds its data/feature
    # functions. Resolve those real callbacks before installing export doubles
    # so the cached runner cannot retain a no-op after monkeypatch teardown.
    sources.get_config("DST")
    source, weekly = dst_rows().assign(headshot_url="https://logo"), opponent_weekly_rows()
    calls = []
    monkeypatch.setattr("src.dst.data.build_data", lambda **kwargs: calls.append(kwargs) or source)
    monkeypatch.setattr("src.dst.targets.compute_targets", lambda frame: frame)
    monkeypatch.setattr("src.dst.features.compute_features", lambda frame: None)

    def prepare(position, config, train, val, test=None):
        calls.append((position, sorted(train["season"].unique()), sorted(val["season"].unique())))
        return SimpleNamespace(
            train=train, feature_columns=SCHEMA.feature_columns, data_id="dst-id"
        )

    monkeypatch.setattr(sources, "_prepare_position_data", prepare)
    weekly_path = tmp_path / "weekly_2012_2025.parquet"
    pd.concat([weekly, weekly.iloc[:2].assign(season_type="POST")]).to_parquet(weekly_path)
    missing = tmp_path / "team_stats_2012_2025.parquet"
    monkeypatch.setattr(
        sources, "dst_raw_cache_files", lambda: {"weekly": weekly_path, "team_stats": missing}
    )
    with pytest.raises(ValueError, match="raw caches missing"):
        sources.write_sources("DST", tmp_path / "dst")
    assert calls == []  # refused before build_data could fetch anything
    pd.DataFrame({"_team_stats_schema_v2": [True]}).to_parquet(missing)
    schedules_path = tmp_path / "schedules_2012_2025.parquet"
    fake_schedules().to_parquet(schedules_path)
    caches = {"weekly": weekly_path, "team_stats": missing, "schedules": schedules_path}
    monkeypatch.setattr(sources, "dst_raw_cache_files", lambda: caches)
    output = sources.write_sources("DST", tmp_path / "dst")
    manifest = json.loads((output / "sources.json").read_text())
    assert calls[0] == {"allow_scoring_fetch": False}
    assert calls[1] == ("DST", [2022, 2023], [])
    assert manifest["source"] == "dst.parquet" and manifest["splits"] == {}
    assert manifest["opponent_weekly_rows"] == len(weekly)  # the POST rows are dropped
    per_game = pd.read_parquet(output / "dst_opponent_per_game.parquet")
    pd.testing.assert_frame_equal(
        per_game, validate_opponent_per_game(opponent_per_game_rows(), SCHEMA)
    )
    assert manifest["opponent_per_game_rows"] == len(per_game)
    assert manifest["raw_caches"] == {
        name: sources.file_digest(path) for name, path in caches.items()
    }
    assert manifest["excluded_columns"] == ["headshot_url"]
    assert "headshot_url" not in pd.read_parquet(output / "dst.parquet").columns
    assert "src/shared/weather_features.py" in manifest["code_sha256"]
    assert "src/dst/targets.py" in manifest["code_sha256"]
    assert manifest["prepared_data_id"] == "dst-id"
    with pytest.raises(ValueError, match="not a skill position"):
        sources.export_skill_source("DST")
    # A loader that rewrites a cache mid-export (a stale-schema refresh) is refused.

    def rewriting_build(**kwargs):
        weekly.iloc[:1].to_parquet(weekly_path)
        return source

    monkeypatch.setattr("src.dst.data.build_data", rewriting_build)
    with pytest.raises(ValueError, match="changed during the export"):
        sources.write_sources("DST", tmp_path / "dst-rewritten")
