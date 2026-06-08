import pandas as pd

from src.config import POSITIONS
from src.data.loader import compute_all_scoring_formats


def impute_snap_pct(df: pd.DataFrame, *, fit_on: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fill missing ``snap_pct`` values with the ``(position, week)`` median.

    Parameters
    ----------
    df : pd.DataFrame
        The frame whose ``snap_pct`` column should be imputed (mutated in
        place on a copy; the input frame is not modified).
    fit_on : pd.DataFrame, optional
        The frame from which to compute the per-(position, week) medians.
        If None, defaults to ``df`` itself — note that this leaks
        distribution information across rows when ``df`` spans train + val
        + test. Callers operating on split frames should pass
        ``fit_on=train_df`` to avoid val/test signal leaking into the
        median used to fill train rows (and conversely, ensure val/test
        rows are filled with train-only medians, matching the cross-season
        generalization assumption of the temporal split).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with imputed ``snap_pct`` column. Rows whose
        ``(position, week)`` group has no observations in ``fit_on`` fall
        back to 0.0 (matches the legacy behaviour).

    Notes
    -----
    No-op if ``snap_pct`` is absent from ``df`` — older fixtures lacking
    the column flow through unchanged.
    """
    df = df.copy()
    if "snap_pct" not in df.columns:
        return df

    source = fit_on if fit_on is not None else df

    # Compute (position, week) medians from ``source`` only. We use a
    # groupby + lookup instead of ``transform`` because ``transform`` would
    # require ``source`` and ``df`` to share an index, which is false in the
    # cross-split case.
    if "snap_pct" not in source.columns:
        # Defensive: if the source frame doesn't carry snap_pct, fall back to 0.
        df["snap_pct"] = df["snap_pct"].fillna(0)
        return df

    medians = source.dropna(subset=["snap_pct"]).groupby(["position", "week"])["snap_pct"].median()
    # Vectorised lookup: map each row's (position, week) to the median.
    keys = pd.MultiIndex.from_arrays([df["position"], df["week"]])
    lookup = medians.reindex(keys).values
    df["snap_pct"] = df["snap_pct"].fillna(pd.Series(lookup, index=df.index))
    df["snap_pct"] = df["snap_pct"].fillna(0)
    return df


def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean raw NFL data for modeling."""
    df = raw_df.copy()

    # Filter to regular season. ``season_type`` is part of the nflverse
    # weekly schema (via the ``src.data.nfl_source`` shim), so its absence
    # indicates a malformed upstream frame
    # rather than an older-dataset path that should be tolerated. Fail loudly.
    assert "season_type" in df.columns, (
        "preprocess() requires 'season_type'; upstream loader (src/data/loader.py) "
        "must emit it. nflverse weekly data carries this column on every release."
    )
    df = df[df["season_type"] == "REG"].copy()

    # Step 1: Filter to all 6 modeled positions (QB/RB/WR/TE/K/DST). K and
    # DST land here too even though they have their own data-loading paths
    # downstream — POSITIONS is the global allowlist.
    df = df[df["position"].isin(POSITIONS)].copy()

    # Step 2: Remove rows where player didn't play. The stat allowlist must
    # cover every modeled position so K and DST rows aren't silently dropped
    # via the (all_zero & no_snaps) mask — kickers have no snap_pct (they're
    # not on the roster snapshot the snap-counts feed comes from) AND zero
    # skill-position stats. DST stats source from src/dst/targets.py
    # (def_sacks/def_ints/def_fumble_rec/def_fumbles_forced/def_safeties/
    # def_tds/def_blocked_kicks/special_teams_tds + points_allowed/
    # yards_allowed). K stats source from src/k/targets.py
    # (fg_att/pat_att/fg_made/pat_made/fg_yards_made).
    stat_cols = [
        # Skill-position raw stats (QB/RB/WR/TE)
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "targets",
        "carries",
        "completions",
        "attempts",
        # K raw counts — see src/k/targets.py
        "fg_att",
        "pat_att",
        # DST raw counts — see src/dst/targets.py. points_allowed/yards_allowed
        # are NOT included here because their "zero" values are meaningful
        # (a defense holding the opponent to 0 points), not "didn't play".
        "def_sacks",
        "def_ints",
        "def_fumble_rec",
        "def_fumbles_forced",
        "def_safeties",
        "def_tds",
        "def_blocked_kicks",
        "special_teams_tds",
    ]
    existing_stat_cols = [c for c in stat_cols if c in df.columns]
    all_zero = df[existing_stat_cols].fillna(0).sum(axis=1) == 0
    no_snaps = (
        df["snap_pct"].isna() if "snap_pct" in df.columns else pd.Series(True, index=df.index)
    )
    df = df[~(all_zero & no_snaps)].copy()

    # Step 3: Fill missing stat columns with 0
    fill_zero_cols = [
        "passing_yards",
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "rushing_tds",
        "carries",
        "receiving_yards",
        "receiving_tds",
        "receptions",
        "targets",
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "attempts",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
        "rushing_first_downs",
        "receiving_first_downs",
        "rushing_epa",
        "receiving_epa",
        "receiving_yards_after_catch",
        "receiving_air_yards",
        "passing_first_downs",
        "passing_yards_after_catch",
        "sack_yards",
        "special_teams_tds",
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    # Step 4 (removed): no longer impute snap_pct here. The previous in-line
    # ``impute_snap_pct(df)`` fitted position-week medians on ``df`` itself —
    # a frame that spans train + val + test — so val/test snap_pct leaked
    # into the training-row imputation via the shared median, AND it pre-empted
    # the split-aware train-only fill in
    # ``src/data/split.py::{temporal_split,expanding_window_folds}`` (those
    # calls only ``fillna`` NaNs, so a frame whose NaNs were already filled
    # here saw them as no-ops). The post-split ``impute_snap_pct(frame,
    # fit_on=train_df)`` in split.py is the train-only fill for any frame that
    # still carries raw NaN snap_pct — but in the canonical flow it is a NO-OP:
    # build_features zero-fills snap_pct before the split (see below), so impute
    # never sees a NaN there. (Whether a genuinely-missing week should be 0 vs
    # the position-week median is a separate modeling call, not a leak this
    # machinery prevents — #396.)
    #
    # No path is left with un-imputed snap_pct: the canonical production flow
    # (refresh-splits.yml + tests/qb/test_pipeline_e2e.py) is
    # ``preprocess -> build_features -> temporal_split``, and
    # ``src/features/engineer.py`` lags snap_pct with
    # ``groupby(...).shift(1).fillna(0)`` (zeroing every remaining NaN —
    # both first-week rows and genuinely-missing weeks) before
    # ``temporal_split`` runs. The serving layer reads the already-imputed
    # split parquets and never calls ``preprocess`` directly.

    # Step 5: Compute fantasy points for all scoring formats
    df = compute_all_scoring_formats(df)

    # Validate against nflverse pre-computed
    if "fantasy_points_ppr" in df.columns:
        discrepancy = (df["fantasy_points_ppr"] - df["fantasy_points"]).abs()
        n_mismatch = (discrepancy > 0.5).sum()
        if n_mismatch > 0:
            print(f"WARNING: {n_mismatch} rows differ from nflverse PPR points by > 0.5")

    return df
