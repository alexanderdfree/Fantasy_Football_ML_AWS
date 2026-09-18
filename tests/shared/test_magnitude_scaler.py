"""Behavioral regressions for the magnitude-preserving NN scaler (#1575, component 1).

WR's positive ``inherited_opportunity`` values collapsed to +4 under z-score +
clip. ``MagnitudePreservingScaler`` keeps them distinguishable; every other
position (``nn_magnitude_features=()``) must take the legacy ``StandardScaler``
path byte-for-byte.
"""

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.shared.feature_build import MagnitudePreservingScaler, make_nn_scaler, scale_and_clip
from src.shared.pipeline import _scale_xs
from src.shared.registry import ALL_POSITIONS, get_config

pytestmark = pytest.mark.unit


def _sparse_training():
    # Both positive observations collapse to +4 under the old all-row z-score.
    x = np.zeros((100, 2), dtype=np.float64)
    x[:, 0] = np.arange(100)
    x[-2:, 1] = [5, 10]
    return x


def test_positive_magnitudes_survive_clipping_and_zero_stays_zero():
    train = _sparse_training()
    future = np.array([[1, 0], [1, 5.833333], [1, 20.297619], [1, 10000]])
    old = StandardScaler().fit(train)
    assert np.array_equal(scale_and_clip(old, future)[1:, 1], [4, 4, 4])
    scaler, (scaled_train, scaled_future) = _scale_xs(
        train,
        future,
        feature_cols=["ordinary", "inherited_opportunity"],
        magnitude_features=["inherited_opportunity"],
    )
    assert isinstance(scaler, MagnitudePreservingScaler)
    assert scaler.magnitude_scales_.tolist() == [7.5]
    assert scaled_future[0, 1] == 0
    assert np.all(np.diff(scaled_future[:, 1]) > 0)
    assert np.all(scaled_future[:, 1] < 4)
    np.testing.assert_array_equal(scaled_train[:, 0], scale_and_clip(old, train)[:, 0])
    np.testing.assert_array_equal(scaled_future[:, 0], scale_and_clip(old, future)[:, 0])


def test_scaler_roundtrip_train_only_and_in_place(tmp_path):
    train = _sparse_training()
    original = train.copy()
    scaler = MagnitudePreservingScaler((1,)).fit(train)
    future = np.array([[30.0, 6.0], [50.0, 20.0]])
    expected = scaler.transform(future)
    np.testing.assert_array_equal(train, original)
    np.testing.assert_allclose(scaler.inverse_transform(expected), future)
    in_place = future.copy()
    np.testing.assert_array_equal(scaler.transform(in_place, copy=False), expected)
    np.testing.assert_allclose(scaler.inverse_transform(expected.copy(), copy=False), future)
    # Inference data never enters the fitted magnitude scale.
    scaler.transform(np.array([[0.0, 1e9]]))
    assert scaler.magnitude_scales_.tolist() == [7.5]
    path = tmp_path / "scaler.pkl"
    joblib.dump(scaler, path)
    np.testing.assert_array_equal(scale_and_clip(joblib.load(path), future), expected)
    # Historical artifacts remain ordinary StandardScalers.
    old = StandardScaler().fit(train)
    joblib.dump(old, path)
    np.testing.assert_array_equal(
        scale_and_clip(joblib.load(path), future), np.clip(old.transform(future), -4, 4)
    )


def test_absent_features_and_zero_only_training():
    assert type(make_nn_scaler(["ordinary"], ["inherited_opportunity"])) is StandardScaler
    assert type(make_nn_scaler(["inherited_opportunity"], [])) is StandardScaler
    scaler = MagnitudePreservingScaler((0,)).fit(np.zeros((4, 1)))
    values = scaler.transform(np.array([[0], [1], [2]], dtype=float))[:, 0]
    assert values[0] == 0 and np.all(np.diff(values) > 0)
    with pytest.raises(ValueError, match="exact input columns"):
        _scale_xs(
            np.zeros((3, 2)),
            feature_cols=["inherited_opportunity"],
            magnitude_features=["inherited_opportunity"],
        )
    # A requested magnitude column with no column list would otherwise degrade
    # silently to a plain StandardScaler (the #1534 no-silent-no-op rule).
    with pytest.raises(ValueError, match="exact input columns"):
        _scale_xs(np.zeros((3, 2)), magnitude_features=["inherited_opportunity"])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scale_xs_without_magnitude_features_is_legacy_fit_transform(dtype):
    """No magnitude columns and no flag policy == main's plain fit_transform path."""
    train = _sparse_training().astype(dtype)
    future = np.array([[1, 0], [1, 5.833333], [1, 20.297619], [1, 10000]], dtype=dtype)
    legacy_scaler = StandardScaler()
    legacy = [
        scale_and_clip(legacy_scaler, train, fit=True),
        scale_and_clip(legacy_scaler, future),
    ]
    cols = ["ordinary", "inherited_opportunity"]
    for kwargs in (
        {},
        {"magnitude_features": ()},
        {"feature_cols": cols},
        {"feature_cols": cols, "magnitude_features": []},
        {"cfg": {"nn_bounded_flag_range": None}, "feature_cols": cols},
        {"cfg": {}, "feature_cols": cols, "magnitude_features": ()},
    ):
        scaler, scaled = _scale_xs(train, future, **kwargs)
        assert type(scaler) is StandardScaler
        np.testing.assert_array_equal(scaled[0], legacy[0])
        np.testing.assert_array_equal(scaled[1], legacy[1])
        np.testing.assert_array_equal(scaler.mean_, legacy_scaler.mean_)
        np.testing.assert_array_equal(scaler.scale_, legacy_scaler.scale_)
        np.testing.assert_array_equal(scaler.var_, legacy_scaler.var_)
        np.testing.assert_array_equal(scaler.n_samples_seen_, legacy_scaler.n_samples_seen_)


def test_scale_xs_composes_flag_override_and_magnitude_scaling():
    """Both policies on: flag stats are rewritten, magnitude columns bypass them."""
    rng = np.random.default_rng(0)
    n = 500
    game_status = np.where(rng.random(n) < 0.04, 0.5, 1.0)
    inherited = np.zeros(n)
    inherited[-3:] = [2.0, 5.0, 10.0]
    X_train = np.column_stack([rng.standard_normal(n), game_status, inherited])
    X_test = np.array([[0.0, 0.5, 0.0], [0.0, 0.5, 5.0], [0.0, 1.0, 1e6]])
    cols = ["other", "game_status", "inherited_opportunity"]
    scaler, (train_s, test_s) = _scale_xs(
        X_train,
        X_test,
        cfg={"nn_bounded_flag_range": 1.0},
        feature_cols=cols,
        magnitude_features=("inherited_opportunity",),
    )
    assert isinstance(scaler, MagnitudePreservingScaler)
    assert scaler.magnitude_indices == (2,)
    assert scaler.magnitude_scales_.tolist() == [5.0]
    # Flag override (range 1.0): Questionable 0.5 -> 0.0, healthy 1.0 -> +1.0.
    assert scaler.mean_[1] == 0.5 and scaler.scale_[1] == 0.5
    np.testing.assert_allclose(test_s[:, 1], [0.0, 0.0, 1.0], atol=1e-12)
    # Magnitude column: zero stays zero, 5 -> 4*5/(5+5) = 2, huge -> just under 4.
    np.testing.assert_allclose(test_s[:, 2], [0.0, 2.0, 4 * 1e6 / (5 + 1e6)])
    assert np.all(test_s[:, 2] < 4)
    # Ordinary column keeps plain standardization.
    np.testing.assert_allclose(scaler.mean_[0], X_train[:, 0].mean(), atol=1e-12)
    np.testing.assert_allclose(
        train_s[:, 0], np.clip((X_train[:, 0] - scaler.mean_[0]) / scaler.scale_[0], -4, 4)
    )


@pytest.mark.parametrize("position", ALL_POSITIONS)
def test_only_wr_enables_magnitude_scaling(position):
    """Scope containment: the production scaler class changes for WR alone."""
    cfg = get_config(position)
    expected = ("inherited_opportunity",) if position == "WR" else ()
    assert tuple(cfg["nn_magnitude_features"]) == expected
    cols = cfg["get_feature_columns_fn"]()
    scaler = make_nn_scaler(cols, cfg["nn_magnitude_features"])
    assert isinstance(scaler, MagnitudePreservingScaler) == (position == "WR")
    if position == "WR":
        # Both NN branches see the feature, so both fitted scalers carry the policy.
        assert "inherited_opportunity" in cols
        assert "inherited_opportunity" in cfg["attn_static_features"]
        assert scaler.magnitude_indices == (cols.index("inherited_opportunity"),)
