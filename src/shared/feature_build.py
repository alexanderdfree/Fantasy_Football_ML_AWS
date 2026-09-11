"""Shared feature-building for training and inference paths.

Both ``src/shared/pipeline.py::_prepare_position_data`` (training) and
``src/serving/core.py::_apply_position_models`` (serving) need the same per-position
feature-engineering pipeline. They drifted in the past — TODO.md archive
entry "Weather/Vegas features missing at inference in ``app.py``" was the
most recent recurrence: a training-side feature wasn't mirrored at serving
time, so 12 weather/Vegas features were silently zeroed at inference.
Centralizing the shared block here makes that drift class impossible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.shared.team_box_score import merge_team_box_score_features
from src.shared.weather_features import merge_schedule_features

# Clip bounds applied after every ``StandardScaler`` op. Guards against
# extreme z-scores when test-distribution features wander well outside the
# training distribution — see TODO.md archive "No feature clipping after
# StandardScaler". Kept at (-4, 4): catches ~0.3-0.4% of values under a
# Gaussian assumption, well below the extreme tails that cause catastrophic
# NN extrapolation.
FEATURE_CLIP: tuple[float, float] = (-4.0, 4.0)

# Bounded ordinal "flag" columns → their semantic ``(center, half_width)``
# domain, used to override a fitted ``StandardScaler``'s stats.
#
# Both are injury-report codes on a fixed, semantically-bounded domain
# (``src/data/loader.py``'s ``status_map``), NOT samples from a continuous
# distribution:
#   game_status      1.0 healthy / 0.5 Questionable / 0.1 Doubtful / 0.0 Out
#   practice_status  2.0 full / 1.0 limited / 0.0 DNP
#
# z-scoring them is the wrong transform. Out/Doubtful players self-eliminate
# (preprocessing drops no-play rows), so ``game_status`` is ~96% constant at 1.0
# and its train std collapses to ~0.099 — a Questionable row lands at z=-4.83,
# which FEATURE_CLIP then truncates to -4.0. That single coordinate carries
# ~34% of the expected squared norm of the whole standardized static vector, and
# the clip flattens Questionable/Doubtful/Out into one indistinguishable value.
# On the 2026-06-11 splits the clip fires on 4.0% (RB) / 5.3% (WR) / 4.1% (TE) /
# 2.2% (QB) of rows — essentially every questionable row. LightGBM is
# scale-invariant and never sees this; it is an NN-path artifact.
#
# The override rewrites ``(mean_, scale_)`` so the column's semantic domain maps
# onto ``[-target_range, +target_range]``: ``mean_ = center`` and
# ``scale_ = half_width / target_range``. Ordinal-preserving and, for any
# ``target_range <= FEATURE_CLIP``, provably never clipped.
#
# ``target_range`` is a real experimental axis, not a cosmetic constant, because
# unit variance and a clip-free range are mutually exclusive for a rare binary:
# forcing unit variance on a 4%-prevalence column *necessarily* puts the
# minority value ~4.8 sigma out. So the two candidate arms trade off:
#   range=1.0  bounded and conservative, but post-scale std ~0.15-0.23 —
#              ~5x quieter than a genuinely standardized feature.
#   range=4.0  the domain mapped onto exactly +/-FEATURE_CLIP: the largest
#              provably clip-free magnitude, post-scale std ~0.58-0.90 (near
#              unit variance).
# Running both against the z-scored baseline is what separates "the clip was the
# problem" from "the magnitude was the problem" — a single arm confounds them.
#
# Expressing this as scaler state rather than a separate transform is
# deliberate: the fitted scaler is persisted and reused verbatim at serving time
# (``src/serving/core.py::_apply_position_models``), so training and inference
# cannot drift.
#
# Only genuinely bounded ordinal codes belong here. The other high-|z| static
# features (``opp_*_pts_allowed_to_pos``, ``prior_season_*``, ``contract_*``)
# are heavy-tailed continuous columns with hundreds of distinct values, where
# the clip is doing its intended job — leave those to StandardScaler.
BOUNDED_FLAG_DOMAINS: dict[str, tuple[float, float]] = {
    "game_status": (0.5, 0.5),  # domain [0, 1]
    "practice_status": (1.0, 1.0),  # domain [0, 2]
}


def apply_bounded_flag_scaling(
    scaler: StandardScaler,
    feature_cols: list[str],
    *,
    target_range: float = 1.0,
) -> list[str]:
    """Overwrite ``scaler``'s fitted stats for bounded flag columns, in place.

    Maps each flag column's semantic domain onto ``[-target_range,
    +target_range]``. Must be called on an already-fitted ``scaler`` whose
    column order is ``feature_cols``.

    Returns the columns actually overridden. Callers MUST treat an empty return
    as a failure when the knob is on: an A/B arm that silently overrides nothing
    (a renamed column, a position whose whitelist omits the flags) otherwise
    reads as a genuine "no effect" result, and the Ridge-invariance sentinel is
    structurally blind to it because Ridge never sees NN config either way.
    """
    if getattr(scaler, "mean_", None) is None:
        raise ValueError("apply_bounded_flag_scaling requires a fitted StandardScaler")
    if len(feature_cols) != scaler.n_features_in_:
        raise ValueError(
            f"feature_cols has {len(feature_cols)} entries but the scaler was fit on "
            f"{scaler.n_features_in_} features — column order would be misaligned."
        )
    if not target_range > 0:
        raise ValueError(f"target_range must be positive, got {target_range!r}")
    if target_range > FEATURE_CLIP[1]:
        raise ValueError(
            f"target_range={target_range} exceeds FEATURE_CLIP={FEATURE_CLIP[1]}, which would "
            "reintroduce the truncation this override exists to remove."
        )
    touched: list[str] = []
    for i, col in enumerate(feature_cols):
        domain = BOUNDED_FLAG_DOMAINS.get(col)
        if domain is None:
            continue
        center, half_width = domain
        scale = half_width / target_range
        scaler.mean_[i] = center
        scaler.scale_[i] = scale
        scaler.var_[i] = scale**2
        touched.append(col)
    return touched


class MagnitudePreservingScaler(StandardScaler):
    """Keep sparse continuous magnitudes distinguishable within the NN bounds.

    Ordinary columns retain StandardScaler semantics. Selected columns use
    ``4 * x / (median_nonzero_abs_train + abs(x))``: zero stays zero and the
    mapping is monotone, bounded, and independent of the frequency of zeros.
    The selected indices, positive scale, and bound are fitted/pickled with
    the model. Legacy StandardScaler artifacts keep their original behavior.
    """

    def __init__(self, magnitude_indices: tuple[int, ...] = (), *, copy: bool = True):
        super().__init__(copy=copy)
        self.magnitude_indices = magnitude_indices

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        indices = tuple(self.magnitude_indices)
        if len(set(indices)) != len(indices) or any(
            i < 0 or i >= self.n_features_in_ for i in indices
        ):
            raise ValueError("magnitude_indices must be unique indices within the feature matrix")
        values = np.asarray(X)
        self.magnitude_scales_ = np.array(
            [
                np.median(np.abs(values[:, i][np.isfinite(values[:, i]) & (values[:, i] != 0)]))
                if np.any(np.isfinite(values[:, i]) & (values[:, i] != 0))
                else 1.0
                for i in indices
            ]
        )
        self.magnitude_bound_ = float(FEATURE_CLIP[1])
        return self

    def transform(self, X, copy=None):
        # Preserve raw values before StandardScaler's optional in-place write.
        raw = np.asarray(X)[:, self.magnitude_indices].copy()
        result = super().transform(X, copy=copy)
        result[:, self.magnitude_indices] = self.magnitude_bound_ * (
            raw / (self.magnitude_scales_ + np.abs(raw))
        )
        return result

    def inverse_transform(self, X, copy=None):
        bounded = np.asarray(X)[:, self.magnitude_indices].copy()
        result = super().inverse_transform(X, copy=copy)
        with np.errstate(divide="ignore", invalid="ignore"):
            result[:, self.magnitude_indices] = (
                self.magnitude_scales_ * bounded / (self.magnitude_bound_ - np.abs(bounded))
            )
        return result


def make_nn_scaler(feature_cols=None, magnitude_features=()) -> StandardScaler:
    """Build a scaler for the exact ordered NN inputs, before fitting on train."""
    selected = set(magnitude_features)
    indices = tuple(i for i, name in enumerate(feature_cols or ()) if name in selected)
    return MagnitudePreservingScaler(indices) if indices else StandardScaler()


def build_position_features(
    pos_train: pd.DataFrame,
    pos_val: pd.DataFrame,
    pos_test: pd.DataFrame | None,
    cfg: dict,
    feature_cols: list[str],
    full_train: pd.DataFrame | None = None,
    *,
    fitted_state: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Merge schedule features, add position-specific features, backfill missing
    whitelist columns, and clean inf/NaN values.

    Callers are responsible for ``filter_fn`` and ``compute_targets_fn`` up to
    this point; this helper starts from already-filtered, target-computed
    splits so both training and serving can route through the same block.
    """
    dfs = [pos_train, pos_val] + ([pos_test] if pos_test is not None else [])
    practice_mean = pos_train["practice_status"].mean() if "practice_status" in pos_train else None
    split_labels = ["train", "val", "test"][: len(dfs)]

    # Schedule merge first — downstream ``add_features_fn`` may read the
    # merged weather/Vegas columns.
    for label, df in zip(split_labels, dfs, strict=True):
        merge_schedule_features(df, label=label)
        # Per-game team and opponent box-score columns. Used by the RB
        # attention NN's history sequence so each historical token carries
        # the realised game environment in addition to the player's own
        # stats; other positions ignore the columns via their own
        # ``ATTN_HISTORY_STATS`` whitelist.
        merge_team_box_score_features(df, label=label)
    # ``full_train`` is the pre-min-games-filter train: RB/WR compute their
    # per-game team-total / share / HHI / career features over it (so dropped
    # low-volume players don't undercount those aggregates) and return only the
    # filtered rows. Those rows must carry the same schedule + box-score columns,
    # so merge them here too. label="train" — it IS train data. Other positions
    # ignore ``full_train``. fill_nans + the scaler still fit on the FILTERED
    # train below, preserving #569. (#574/#531)
    if full_train is not None:
        merge_schedule_features(full_train, label="train")
        merge_team_box_score_features(full_train, label="train")

    # Position-specific feature engineering + fill_nans. Both take three
    # splits; when test is None, pass an empty stub so the signatures line up.
    if pos_test is not None:
        pos_train, pos_val, pos_test = cfg["add_features_fn"](
            pos_train, pos_val, pos_test, full_train=full_train
        )
        if fitted_state is None:
            pos_train, pos_val, pos_test = cfg["fill_nans_fn"](
                pos_train, pos_val, pos_test, cfg["specific_features"]
            )
    else:
        empty = pos_val.iloc[:0].copy()
        pos_train, pos_val, _ = cfg["add_features_fn"](
            pos_train, pos_val, empty, full_train=full_train
        )
        if fitted_state is None:
            pos_train, pos_val, _ = cfg["fill_nans_fn"](
                pos_train, pos_val, empty, cfg["specific_features"]
            )

    # Whitelist columns the pipeline didn't produce mean either build_features
    # never ran (the regression behind PR fixing 8c46b59 — refresh-splits.yml
    # was missing the build_features() call, so 150 cols were absent and
    # silently zero-filled, masking a real metric regression) or that an
    # add_features_fn / merge step failed to materialise an expected column.
    # Either way, training on constant-zero columns produces undetectable
    # silent drift; fail loudly instead. Mirrors the same fail-loud contract
    # build_game_history_arrays adopted in c06568a.
    dfs = [pos_train, pos_val] + ([pos_test] if pos_test is not None else [])
    if fitted_state is not None:
        for df in dfs:
            for col, value in fitted_state["fill_values"].items():
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(value)
    missing = [c for c in feature_cols if c not in pos_train.columns]
    if missing:
        raise KeyError(
            f"build_position_features: {len(missing)} whitelisted feature "
            f"columns are missing from pos_train. This usually means "
            f"build_features() was not run on the upstream splits parquet, "
            f"or an add_features_fn / merge step failed to produce expected "
            f"columns. Missing: {missing}"
        )

    # Remediate the depth_chart_rank "-1 = no depth-chart data" sentinel set in
    # src/data/loader.py. Left raw, -1 is out-of-band vs legitimate ranks (>=1),
    # so StandardScaler maps it to a large negative z-score and the attention NN
    # extrapolates badly; for seasons/players with no depth-chart coverage EVERY
    # row hits the sentinel and the metric regresses. (2025 was such a gap until
    # the ESPN-format adapter src/data/loader.py::_normalize_espn_depth (PR #370)
    # closed it — 2025 is now covered; only all-sentinel positions like K remain.)
    # Impute to the train mean of real ranks (leakage-safe) so missing -> ~0
    # after standardization (neutral). loader.py defers this consumer-side fix
    # by design. Positions whose ranks are *all* sentinel (e.g. K — kickers are
    # never on depth charts) yield a NaN mean; the guard leaves them constant,
    # which StandardScaler's zero-variance handling already neutralizes to 0.
    depth_fill = None
    if "depth_chart_rank" in pos_train.columns:
        fill_val = (
            fitted_state.get("depth_chart_rank_fill")
            if fitted_state is not None
            else pos_train["depth_chart_rank"].replace(-1, np.nan).mean()
        )
        if pd.notna(fill_val):
            depth_fill = float(fill_val)
            for df in dfs:
                df["depth_chart_rank"] = df["depth_chart_rank"].replace(-1, fill_val)

    for df in dfs:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    from src.shared.comparison_truth import restore_comparison_actuals

    for df in dfs:
        restore_comparison_actuals(df)

    pos_train.attrs["preprocessing_state"] = fitted_state or {
        "fill_values": pos_train.attrs.get("fitted_fill_values", {}),
        "depth_chart_rank_fill": depth_fill,
        "clip": list(FEATURE_CLIP),
        "missing_value": 0.0,
        "dtype": "float32",
        "practice_status_mean": float(practice_mean) if pd.notna(practice_mean) else None,
    }

    return pos_train, pos_val, pos_test


def scale_and_clip(
    scaler: StandardScaler,
    X: np.ndarray,
    *,
    fit: bool = False,
) -> np.ndarray:
    """Scale X with ``scaler`` (fit first if ``fit=True``) and clip to ``FEATURE_CLIP``."""
    X = scaler.fit_transform(X) if fit else scaler.transform(X)
    return np.clip(X, *FEATURE_CLIP)


def safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Return ``num / denom`` with all ill-defined results replaced by 0.

    Handles the three pathological cases inline so callers don't need the
    ``(a/b).fillna(0); df.loc[b == 0, col] = 0`` pair:

    - ``0 / 0``  → NaN   → 0
    - ``x / 0``  → ±inf  → 0  (``fillna`` alone wouldn't catch this)
    - ``x / NaN`` or ``NaN / x`` → NaN → 0
    """
    return (num / denom).replace([np.inf, -np.inf], 0).fillna(0)


def rolling_agg(
    df: pd.DataFrame,
    col: str,
    groupby: str | list[str],
    window: int,
    *,
    agg: str = "sum",
    shift: int = 1,
    min_periods: int = 1,
    fill: float | None = None,
) -> pd.Series:
    """Grouped, shifted rolling aggregation.

    ``shift`` defaults to 1 to prevent current-week leakage into rolling
    features — the same convention every position's feature code follows.
    ``groupby`` is explicit because K deliberately uses ``["player_id"]``
    only (cross-season windows) while skill positions use
    ``["player_id", "season"]``. See TODO.md "K features use cross-season
    rolling windows".

    ``fill`` optionally replaces the leading NaN produced by ``shift`` (and
    any NaN the input itself carries). K and DST pass ``fill=0`` because
    their features are used directly — skill positions usually leave
    ``fill=None`` because their rolling outputs feed into ``safe_divide``,
    which already maps NaN to 0.
    """
    result = df.groupby(groupby)[col].transform(
        lambda x: getattr(x.shift(shift).rolling(window, min_periods=min_periods), agg)()
    )
    if fill is not None:
        result = result.fillna(fill)
    return result


def fill_nans_with_train_means(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace inf → NaN in ``cols`` across all splits, then backfill NaNs
    with the training-set column means.

    Using train-set statistics for every split is the leakage-safe contract
    every position's ``fill_*_nans`` already follows; lifting the loop here
    keeps the ``cfg["fill_nans_fn"]`` signature intact.

    Two failure modes that used to be silent are now explicit:

    - A column listed in ``cols`` but missing from ``train_df`` raises
      ``KeyError`` with the offending columns named (used to surface as a
      cryptic pandas KeyError on ``train_df[cols]`` access).
    - A column entirely NaN in ``train_df`` has ``train_means[col] = NaN``,
      so the per-column ``fillna`` would be a no-op and the NaN would only
      get caught by ``build_position_features``'s catch-all ``.fillna(0)``
      with no signal that anything went wrong. We now log a warning and
      substitute 0 for those columns, matching the catch-all's behavior
      but making the silent zero-feature visible.
    """
    missing = [c for c in cols if c not in train_df.columns]
    if missing:
        raise KeyError(f"fill_nans_with_train_means: cols not in train_df: {missing}")

    for split_df in (train_df, val_df, test_df):
        split_df[cols] = split_df[cols].replace([np.inf, -np.inf], np.nan)

    train_means = train_df[cols].mean()
    all_nan_cols = [c for c in cols if pd.isna(train_means[c])]
    if all_nan_cols:
        print(
            f"  WARNING: {len(all_nan_cols)} feature(s) entirely NaN in training "
            f"set; filling with 0: {all_nan_cols}"
        )
        train_means[all_nan_cols] = 0.0

    for split_df in (train_df, val_df, test_df):
        for col in cols:
            split_df[col] = split_df[col].fillna(train_means[col])
    train_df.attrs["fitted_fill_values"] = {col: float(train_means[col]) for col in cols}
    return train_df, val_df, test_df
