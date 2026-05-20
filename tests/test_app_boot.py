"""Production disk-loading path for ``app.py::_load_base_data_locked``.

The function reads ``data/splits/{train,val,test}.parquet`` from disk, calls
``_load_k_splits`` + ``_load_dst_splits`` to fetch the position-specific K/DST
frames, concatenates the test rows, and initializes per-model per-format
prediction columns in the cached ``results`` DataFrame. Everything else in the
test suite monkeypatches ``_load_base_data_locked`` itself or injects synthetic
data straight into ``_cache``, leaving this on-boot codepath at 0% coverage.

The risk that motivates the new tests is documented in ``TODO.md``'s archive:

    [FIXED] ``kicker_week_split`` does not exist - app.py crashed on import

The same shape of bug (column-name mismatch, K/DST index collision, missing
``season_type`` filter) would slip past CI today because nothing exercises the
real disk-load path. These tests do.

Strategy
--------

The skill-position splits are written as tiny parquets under ``tmp_path`` and
the test ``chdir``s there so the hard-coded ``"data/splits/*.parquet"``
literals resolve. ``_load_k_splits`` and ``_load_dst_splits`` are monkeypatched
to return synthetic frames (they normally call into ``nfl_data_py`` and the
PBP cache, which is not practical to fake at the parquet level inside a unit
test). This is documented as a simplification - the test still exercises the
parquet read, ``season_type`` filter, ``_compute_scoring_formats`` injection,
K/DST concat with index offsets, per-model per-format column initialisation,
and the legacy unsuffixed alias columns. The K/DST loader internals are
covered by ``tests/k/`` and ``tests/dst/``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Synthetic split builders. Schemas mirror what the real ``data/splits/*.parquet``
# files produced by ``src/data/split.py`` contain - enough columns for
# ``_compute_scoring_formats`` (via ``compute_fantasy_points``) to run without
# KeyError. Sizes are intentionally tiny so the test runs in well under a second.
# ---------------------------------------------------------------------------


def _make_skill_position_frame(seed: int, season: int, n_players: int = 5) -> pd.DataFrame:
    """Build a skill-position weekly frame mirroring the train/val/test parquet shape.

    Includes the columns ``_compute_scoring_formats`` reads (it sums weighted
    raw stats into ``fantasy_points_standard`` / ``fantasy_points_half_ppr``)
    and the identifier columns ``_load_base_data_locked`` lifts into
    ``results``. Provides ``season_type`` rows of both ``REG`` and ``POST`` so
    the ``_load_reg`` filter has something to drop - this catches the silent
    "all rows are POST" regression class.
    """
    rng = np.random.default_rng(seed)
    positions = ["QB", "RB", "WR", "TE"]
    rows = []
    for pos in positions:
        for i in range(n_players):
            for week in (1, 2, 3):
                # 1 in 6 rows is POST so we can assert the season_type filter
                # actually drops something.
                season_type = "POST" if (week == 3 and i == 0 and pos == "QB") else "REG"
                rows.append(
                    {
                        "player_id": f"{pos}{i:03d}",
                        "player_display_name": f"{pos} Player {i}",
                        "position": pos,
                        "recent_team": "KC",
                        "season": season,
                        "week": week,
                        "season_type": season_type,
                        "headshot_url": f"https://example.com/{pos}{i}.png",
                        # Raw stats needed by compute_fantasy_points
                        "passing_yards": float(rng.uniform(0, 300) if pos == "QB" else 0.0),
                        "passing_tds": float(rng.integers(0, 3) if pos == "QB" else 0),
                        "interceptions": float(rng.integers(0, 2) if pos == "QB" else 0),
                        "rushing_yards": float(rng.uniform(0, 80)),
                        "rushing_tds": float(rng.integers(0, 2)),
                        "receiving_yards": float(rng.uniform(0, 80)) if pos != "QB" else 0.0,
                        "receiving_tds": float(rng.integers(0, 2)) if pos != "QB" else 0.0,
                        "receptions": float(rng.integers(0, 6)) if pos != "QB" else 0.0,
                        # Fumble components - compute_fantasy_points sums these three
                        "sack_fumbles_lost": 0.0,
                        "rushing_fumbles_lost": 0.0,
                        "receiving_fumbles_lost": 0.0,
                        # PPR fantasy_points - the canonical actual; the other two
                        # formats are computed by _compute_scoring_formats
                        "fantasy_points": float(rng.uniform(5, 25)),
                    }
                )
    return pd.DataFrame(rows)


def _make_k_split(seed: int, season: int, n_kickers: int = 3) -> pd.DataFrame:
    """K test split - shape matches what ``_load_k_splits`` would return.

    ``_load_base_data_locked`` only copies the ``keep_cols`` it finds, falls
    back to ``fantasy_points`` for the half-ppr / standard cells, and ignores
    everything else, so we only need the identifier columns and ``fantasy_points``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_kickers):
        for week in (1, 2):
            rows.append(
                {
                    "player_id": f"K{i:03d}",
                    "player_display_name": f"Kicker {i}",
                    "position": "K",
                    "recent_team": "KC",
                    "season": season,
                    "week": week,
                    "fantasy_points": float(rng.uniform(5, 15)),
                }
            )
    return pd.DataFrame(rows)


def _make_dst_split(seed: int, season: int, n_teams: int = 3) -> pd.DataFrame:
    """DST test split - schema mirrors what ``_load_dst_splits`` returns.

    Same minimal-columns approach as the K builder; the cache assembly only
    reads identifier columns + ``fantasy_points``.
    """
    rng = np.random.default_rng(seed)
    teams = ["KC", "SF", "BUF"][:n_teams]
    rows = []
    for team in teams:
        for week in (1, 2):
            rows.append(
                {
                    "player_id": team,
                    "player_display_name": f"{team} D/ST",
                    "position": "DST",
                    "recent_team": team,
                    "season": season,
                    "week": week,
                    "fantasy_points": float(rng.uniform(5, 20)),
                }
            )
    return pd.DataFrame(rows)


def _make_k_kicks(n: int = 10) -> pd.DataFrame:
    """Per-kick records dataframe - the fourth return value of ``_load_k_splits``.

    Required by the K nested-attention inference path (see
    ``_apply_position_models`` and the ``"K nested attention requires
    kicks_df"`` runtime guard). Shape matches ``src/k/data.py::_KICKS_SCHEMA``.
    """
    return pd.DataFrame(
        {
            "player_id": [f"K{i % 3:03d}" for i in range(n)],
            "season": [2025] * n,
            "week": [(i % 2) + 1 for i in range(n)],
            "play_id": list(range(n)),
            "is_fg": [1] * n,
            "is_xp": [0] * n,
            "kick_distance": [35.0] * n,
            "kick_made": [1] * n,
            "fg_prob": [0.8] * n,
            "is_q4": [0] * n,
            "score_diff": [0.0] * n,
            "game_wind": [0.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# Fixture: build the on-disk splits + monkeypatch the K/DST loaders.
# ---------------------------------------------------------------------------


@pytest.fixture
def boot_env(tmp_path, monkeypatch):
    """Write synthetic parquets under ``tmp_path/data/splits/`` + stub K/DST loaders.

    Yields ``(app_mod, tmp_path)`` so each test can assert on the cache and
    the temp dir. ``os.chdir`` is restored in teardown so a failing assertion
    doesn't poison neighbouring tests' working directory.
    """
    import src.serving.app as app_mod

    splits_dir = tmp_path / "data" / "splits"
    splits_dir.mkdir(parents=True)

    train_df = _make_skill_position_frame(seed=42, season=2023)
    val_df = _make_skill_position_frame(seed=43, season=2024)
    test_df = _make_skill_position_frame(seed=44, season=2025)

    train_df.to_parquet(splits_dir / "train.parquet", index=False)
    val_df.to_parquet(splits_dir / "val.parquet", index=False)
    test_df.to_parquet(splits_dir / "test.parquet", index=False)

    # Stash row counts pre-filter so a test can assert the season_type filter
    # actually fires (POST rows must be dropped from the cached splits).
    n_test_rows_raw = len(test_df)
    n_test_post = int((test_df["season_type"] == "POST").sum())

    # K + DST loaders fan out into nfl_data_py and the PBP cache; stub them to
    # return synthetic DataFrames so the test stays hermetic. The integration
    # of these loaders with the rest of the pipeline is covered by tests/k/ and
    # tests/dst/.
    k_test = _make_k_split(seed=45, season=2025)
    dst_test = _make_dst_split(seed=46, season=2025)
    k_kicks = _make_k_kicks()

    monkeypatch.setattr(
        app_mod,
        "_load_k_splits",
        lambda: (
            _make_k_split(seed=10, season=2023),
            _make_k_split(seed=11, season=2024),
            k_test,
            k_kicks,
        ),
    )
    monkeypatch.setattr(
        app_mod,
        "_load_dst_splits",
        lambda: (
            _make_dst_split(seed=20, season=2023),
            _make_dst_split(seed=21, season=2024),
            dst_test,
        ),
    )

    # Reset the module cache so _load_base_data_locked actually runs (the
    # early-return on _cache["base_loaded"] would otherwise skip everything).
    monkeypatch.setattr(app_mod, "_cache", {})

    # Hardcoded "data/splits/*.parquet" literals in _load_base_data_locked
    # mean we have to point the process's cwd at tmp_path. monkeypatch.chdir
    # restores the original cwd at teardown.
    monkeypatch.chdir(tmp_path)

    return {
        "app": app_mod,
        "tmp_path": tmp_path,
        "n_test_rows_raw": n_test_rows_raw,
        "n_test_post": n_test_post,
        "n_k_test": len(k_test),
        "n_dst_test": len(dst_test),
        "n_k_kicks": len(k_kicks),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadBaseDataLocked:
    """End-to-end exercise of ``_load_base_data_locked`` with on-disk parquets."""

    def test_base_loaded_flag_flips_true(self, boot_env):
        """After the loader runs the cache must announce it is done so
        ``_ensure_base_data`` short-circuits subsequent threads."""
        app_mod = boot_env["app"]
        assert app_mod._cache.get("base_loaded") is not True

        app_mod._load_base_data_locked()

        assert app_mod._cache["base_loaded"] is True

    def test_results_frame_non_empty_and_has_expected_columns(self, boot_env):
        """The cached ``results`` frame must carry the identifier columns
        ``_records_to_player_rows`` reads plus every per-model per-format
        pred column the API endpoints touch."""
        app_mod = boot_env["app"]
        app_mod._load_base_data_locked()

        results = app_mod._cache["results"]
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

        # Identifier + actual columns lifted from the test split + K/DST appendix
        required_identifier_cols = {
            "player_id",
            "player_display_name",
            "position",
            "recent_team",
            "season",
            "week",
            "headshot_url",
            "fantasy_points",
            "fantasy_points_half_ppr",
            "fantasy_points_standard",
        }
        assert required_identifier_cols.issubset(results.columns), (
            f"missing identifier cols: {required_identifier_cols - set(results.columns)}"
        )

        # Per-model per-format pred columns initialized in _load_base_data_locked
        for prefix in ("ridge", "nn", "attn_nn", "lgbm"):
            for fmt in ("ppr", "half_ppr", "standard"):
                col = f"{prefix}_pred_{fmt}"
                assert col in results.columns, f"missing pred column {col}"

        # Legacy unsuffixed alias columns (PPR aliases kept for old tests/code)
        for col in ("ridge_pred", "nn_pred", "attn_nn_pred", "lgbm_pred"):
            assert col in results.columns

    def test_pred_columns_initialized_to_expected_sentinel_values(self, boot_env):
        """Ridge / NN init to 0.0 (every row gets a value once the position
        loads); attn_nn / lgbm init to NaN so K/DST rows - which have no such
        models - are correctly excluded from overall MAE."""
        app_mod = boot_env["app"]
        app_mod._load_base_data_locked()

        results = app_mod._cache["results"]
        for fmt in ("ppr", "half_ppr", "standard"):
            assert (results[f"ridge_pred_{fmt}"] == 0.0).all()
            assert (results[f"nn_pred_{fmt}"] == 0.0).all()
            assert results[f"attn_nn_pred_{fmt}"].isna().all()
            assert results[f"lgbm_pred_{fmt}"].isna().all()
        assert (results["ridge_pred"] == 0.0).all()
        assert (results["nn_pred"] == 0.0).all()
        assert results["attn_nn_pred"].isna().all()
        assert results["lgbm_pred"].isna().all()

    def test_splits_dict_keys_cover_all_six_positions(self, boot_env):
        """``_cache["splits"]`` is keyed by every position; each value is a
        ``(train, val, test)`` 3-tuple. ``_apply_position_models`` reads
        ``_cache["splits"][pos]`` so a missing key would crash inference."""
        app_mod = boot_env["app"]
        app_mod._load_base_data_locked()

        splits = app_mod._cache["splits"]
        assert set(splits.keys()) == {"QB", "RB", "WR", "TE", "K", "DST"}

        for pos, val in splits.items():
            assert isinstance(val, tuple) and len(val) == 3, (
                f"splits[{pos}] not a 3-tuple: {type(val)}"
            )
            train, va, test = val
            assert isinstance(train, pd.DataFrame)
            assert isinstance(va, pd.DataFrame)
            assert isinstance(test, pd.DataFrame)

    def test_k_dst_rows_appended_with_position_set(self, boot_env):
        """All six positions must appear in ``results["position"]`` after the
        K/DST concat. The TODO archive's ``kicker_week_split`` bug would
        manifest as K rows missing entirely (the import crashed before any K
        data landed in results)."""
        app_mod = boot_env["app"]
        app_mod._load_base_data_locked()

        results = app_mod._cache["results"]
        positions_seen = set(results["position"].dropna().unique())
        assert positions_seen == {"QB", "RB", "WR", "TE", "K", "DST"}, (
            f"missing positions in results: {positions_seen}"
        )

    def test_concat_shape_matches_sum_of_test_splits(self, boot_env):
        """``results`` length should equal the skill-position test row count
        (minus POST-filtered rows) plus the K/DST test row counts."""
        app_mod = boot_env["app"]
        app_mod._load_base_data_locked()

        results = app_mod._cache["results"]
        expected = (
            boot_env["n_test_rows_raw"]
            - boot_env["n_test_post"]
            + boot_env["n_k_test"]
            + boot_env["n_dst_test"]
        )
        assert len(results) == expected, (
            f"results has {len(results)} rows, expected {expected} "
            f"(skill_test={boot_env['n_test_rows_raw']} - post={boot_env['n_test_post']} "
            f"+ k={boot_env['n_k_test']} + dst={boot_env['n_dst_test']})"
        )

    def test_season_type_filter_drops_post_rows(self, boot_env):
        """``_load_reg`` strips out playoff rows before concat. The fixture
        plants exactly one POST row in the skill-position test split; assert
        no row with the matching player_id + week survives."""
        app_mod = boot_env["app"]
        app_mod._load_base_data_locked()

        # Fixture plants POST at: position=QB, player_id=QB000, week=3
        post_match = app_mod._cache["results"].query(
            "position == 'QB' and player_id == 'QB000' and week == 3"
        )
        assert post_match.empty, "POST row leaked into results - season_type filter regressed"

    def test_k_kicks_df_populated(self, boot_env):
        """``k_kicks_df`` is cached so the K attention NN can build nested
        kick history at inference. Required by the runtime guard in
        ``_apply_position_models`` line 706-708."""
        app_mod = boot_env["app"]
        app_mod._load_base_data_locked()

        kicks_df = app_mod._cache.get("k_kicks_df")
        assert kicks_df is not None
        assert isinstance(kicks_df, pd.DataFrame)
        assert len(kicks_df) == boot_env["n_k_kicks"]

    def test_positions_loaded_initialized_as_empty_set(self, boot_env):
        """``positions_loaded`` tracks which positions have applied their
        per-position models. ``_load_base_data_locked`` only fills the base
        DataFrame, not models, so the set must start empty - otherwise
        ``_ensure_position_loaded`` would skip the model load."""
        app_mod = boot_env["app"]
        app_mod._load_base_data_locked()

        assert app_mod._cache["positions_loaded"] == set()

    def test_k_dst_index_offset_no_collision_with_skill_rows(self, boot_env):
        """K and DST rows are appended with ``offset = results.index.max() + 1``
        before each concat. Assert the resulting indexes are unique - an off-by-
        one in the offset calculation would mean duplicate index values, which
        breaks ``results.loc[pos_index, col] = ...`` writes in
        ``_apply_position_models``."""
        app_mod = boot_env["app"]
        app_mod._load_base_data_locked()

        results = app_mod._cache["results"]
        assert results.index.is_unique, "results.index has duplicate values"

        # Specifically: the K split's index should match indices in results
        # where position == "K".
        k_train, k_val, k_test = app_mod._cache["splits"]["K"]
        k_indices_in_results = set(results.index[results["position"] == "K"])
        assert set(k_test.index) == k_indices_in_results, (
            "K test-split index does not align with results K-row indices"
        )

    def test_half_ppr_and_standard_fantasy_points_populated_for_skill_positions(self, boot_env):
        """``_compute_scoring_formats`` injects ``fantasy_points_standard`` and
        ``fantasy_points_half_ppr`` into the skill-position frames before they
        are lifted into ``results``. K/DST rows fall back to ``fantasy_points``
        (their scoring is format-invariant), so the value should still be
        finite for every skill-position row."""
        app_mod = boot_env["app"]
        app_mod._load_base_data_locked()

        results = app_mod._cache["results"]
        skill_mask = results["position"].isin(["QB", "RB", "WR", "TE"])
        assert skill_mask.any()
        assert results.loc[skill_mask, "fantasy_points_half_ppr"].notna().all()
        assert results.loc[skill_mask, "fantasy_points_standard"].notna().all()

    def test_ensure_base_data_only_runs_once(self, boot_env):
        """``_ensure_base_data`` short-circuits on the ``base_loaded`` flag so
        Flask's per-request handler doesn't re-read parquets every call. This
        catches a regression of the ``base_loaded`` flag not being set after
        ``_load_base_data_locked`` returns."""
        app_mod = boot_env["app"]
        call_count = {"n": 0}
        real_loader = app_mod._load_base_data_locked

        def counting_loader():
            call_count["n"] += 1
            return real_loader()

        with mock.patch.object(app_mod, "_load_base_data_locked", side_effect=counting_loader):
            app_mod._ensure_base_data()
            app_mod._ensure_base_data()
            app_mod._ensure_base_data()

        assert call_count["n"] == 1, (
            f"_ensure_base_data ran the loader {call_count['n']} times; "
            "expected exactly 1 after base_loaded flips True"
        )

    def test_k_split_missing_headshot_and_team_falls_back_correctly(self, tmp_path, monkeypatch):
        """K real-world stubs: ``reconstruct_kicker_weekly_from_pbp`` doesn't
        emit ``headshot_url`` and may drop ``recent_team`` on rare merge edge
        cases. The K/DST appendix loop falls back to ``""`` for ``headshot_url``
        and ``NaN`` for any other missing column. Exercise both fallback
        branches so a future column-rename doesn't accidentally NaN headshots
        (would surface as blank avatars in the UI)."""
        import src.serving.app as app_mod

        splits_dir = tmp_path / "data" / "splits"
        splits_dir.mkdir(parents=True)
        for name, seed in (("train", 1), ("val", 2), ("test", 3)):
            _make_skill_position_frame(seed=seed, season=2023 + seed).to_parquet(
                splits_dir / f"{name}.parquet", index=False
            )

        # K stub without headshot_url AND without recent_team — exercises
        # both line 931 (headshot_url default) and line 933 (NaN fallback).
        def minimal_k():
            base = pd.DataFrame(
                {
                    "player_id": ["K001", "K002"],
                    "player_display_name": ["Kicker 1", "Kicker 2"],
                    "position": ["K", "K"],
                    "season": [2025, 2025],
                    "week": [1, 2],
                    "fantasy_points": [8.0, 12.0],
                }
            )
            return base.copy(), base.copy(), base.copy(), _make_k_kicks()

        monkeypatch.setattr(app_mod, "_load_k_splits", minimal_k)
        monkeypatch.setattr(
            app_mod,
            "_load_dst_splits",
            lambda: (
                _make_dst_split(seed=4, season=2023),
                _make_dst_split(seed=5, season=2024),
                _make_dst_split(seed=6, season=2025),
            ),
        )
        monkeypatch.setattr(app_mod, "_cache", {})
        monkeypatch.chdir(tmp_path)

        app_mod._load_base_data_locked()

        results = app_mod._cache["results"]
        k_rows = results[results["position"] == "K"]
        assert len(k_rows) == 2
        # headshot_url falls back to empty string for K (line 931)
        assert (k_rows["headshot_url"] == "").all()
        # recent_team is missing from the K frame and isn't a fantasy_points
        # format column, so it falls back to NaN (line 933)
        assert k_rows["recent_team"].isna().all()


# ---------------------------------------------------------------------------
# Bonus: targeted coverage for ``_round_or_none`` (lines 168, 171-172).
# Cheap and they live near the disk-loader in the missing-lines report.
# ---------------------------------------------------------------------------


class TestRoundOrNoneEdgeCases:
    """``_round_or_none`` is the formatter applied to actual + pred values
    before JSON serialization. Its None / non-numeric / non-finite branches
    were not covered by any existing test."""

    def test_none_returns_none(self):
        from src.serving.app import _round_or_none

        assert _round_or_none(None) is None

    def test_string_returns_none(self):
        """``float("foo")`` raises ValueError; the except returns None."""
        from src.serving.app import _round_or_none

        assert _round_or_none("not a number") is None

    def test_nan_returns_none(self):
        """``np.isfinite(NaN)`` is False; the guard returns None."""
        from src.serving.app import _round_or_none

        assert _round_or_none(float("nan")) is None

    def test_inf_returns_none(self):
        from src.serving.app import _round_or_none

        assert _round_or_none(float("inf")) is None

    def test_finite_rounds_to_two_dp(self):
        from src.serving.app import _round_or_none

        assert _round_or_none(1.23456) == 1.23
