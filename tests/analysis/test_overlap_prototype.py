"""Import-smoke + signature-drift guard for the base∥attn overlap prototype.

The harness (``src/analysis/overlap_base_attn_prototype.py``) can't run
end-to-end without ``data/splits`` + CUDA, so this guards the cheap surface:
the module imports, the fingerprint helper is correct, and the pipeline
functions the worker depends on still have the parameters it passes — so an
upstream signature refactor fails the unit shard instead of a GPU benchmark run.
"""

import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_module_imports():
    import src.analysis.overlap_base_attn_prototype as m

    assert callable(m.main)
    assert callable(m._train_one_worker)
    assert callable(m._spawn_and_collect)


def test_fingerprint_is_per_target_float_sum():
    from src.analysis.overlap_base_attn_prototype import _fingerprint

    fp = _fingerprint({"a": np.array([1.0, 2.0, 3.0]), "b": np.array([0.5, 0.5])})
    assert fp == {"a": 6.0, "b": 1.0}


def test_depended_pipeline_signatures_unchanged():
    """The worker calls these positionally / with these kwargs — guard drift."""
    from src.shared.pipeline import (
        _prepare_position_data,
        _train_attention_holdout,
        _train_nn,
    )
    from src.shared.registry import get_config

    assert callable(get_config)
    # _train_nn(X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict, cfg, targets, seed)
    assert len(inspect.signature(_train_nn).parameters) >= 9
    # _prepare_position_data(position, cfg, train_df, val_df, test_df)
    prep = list(inspect.signature(_prepare_position_data).parameters)
    assert prep[:2] == ["position", "cfg"]
    # _train_attention_holdout: the worker passes 14 positionals + the
    # opp_source_frames kwarg. Guard the positional prefix (an insertion like the
    # trial-memo ``position`` param shifts every arg — the kwarg-only check missed
    # it) and bind the exact call shape (catches arity / new-required-param drift).
    sig = inspect.signature(_train_attention_holdout)
    assert list(sig.parameters)[:14] == [
        "position",
        "cfg",
        "targets",
        "seed",
        "X_train",
        "X_val",
        "X_test",
        "y_train_dict",
        "y_val_dict",
        "y_test_dict",
        "pos_train",
        "pos_val",
        "pos_test",
        "feature_cols",
    ]
    sig.bind(*range(14), opp_source_frames=None)
