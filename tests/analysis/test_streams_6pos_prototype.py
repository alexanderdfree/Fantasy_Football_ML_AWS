"""Import-smoke + signature/contract guards for the 6-position CUDA-streams prototype.

The harness (``src/analysis/streams_6pos_prototype.py``) can't run end-to-end without
``data/splits`` + CUDA, so this guards the cheap surface that would otherwise only fail
on a GPU benchmark run: the module imports, the fingerprint helper is correct, the
pipeline functions it constructs through still have the expected parameters, and — most
importantly — the **monkeypatch capture contract** holds (``MultiHeadTrainer.train``
keeps its ``(self, train_loader, val_loader, n_epochs)`` signature and no attention
subclass overrides it, since the prototype patches the base method and relies on
inheritance to intercept every attention trainer).
"""

import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_module_imports():
    import src.analysis.streams_6pos_prototype as m

    assert callable(m.main)
    assert callable(m._build_states)
    assert callable(m._train_one_step)
    assert callable(m._train_streams_roundrobin)
    assert callable(m._train_position_solo)
    assert callable(m._fingerprint_arm)


def test_fingerprint_is_per_target_float_sum():
    from src.analysis.streams_6pos_prototype import _fingerprint

    fp = _fingerprint({"a": np.array([1.0, 2.0, 3.0]), "b": np.array([0.5, 0.5])})
    assert fp == {"a": 6.0, "b": 1.0}


def test_depended_pipeline_signatures_unchanged():
    """``_build_states`` / ``main`` call these — guard positional/kwarg drift."""
    from src.shared.pipeline import (
        _prepare_position_data,
        _read_split,
        _train_attention_holdout,
    )
    from src.shared.registry import get_config

    assert callable(get_config)
    assert callable(_read_split)
    # _prepare_position_data(position, cfg, train_df, val_df, test_df)
    assert list(inspect.signature(_prepare_position_data).parameters)[:2] == ["position", "cfg"]
    # _train_attention_holdout(..., opp_source_frames=...) — called positionally + this kwarg
    assert "opp_source_frames" in inspect.signature(_train_attention_holdout).parameters


def test_capture_contract_train_signature_and_no_subclass_override():
    """The monkeypatch patches the BASE ``train`` and relies on inheritance.

    If a subclass grows its own ``train`` or the signature changes, capture would
    silently miss / mis-bind — this is the guard that turns that into a unit failure.
    """
    from src.shared.training import (
        MultiHeadHistoryTrainer,
        MultiHeadHistoryWithOppTrainer,
        MultiHeadNestedHistoryTrainer,
        MultiHeadTrainer,
    )

    assert list(inspect.signature(MultiHeadTrainer.train).parameters) == [
        "self",
        "train_loader",
        "val_loader",
        "n_epochs",
    ]
    for cls in (
        MultiHeadHistoryTrainer,
        MultiHeadHistoryWithOppTrainer,
        MultiHeadNestedHistoryTrainer,
    ):
        assert "train" not in cls.__dict__, (
            f"{cls.__name__} overrides train — capture would miss it"
        )


def test_step_body_uses_only_existing_trainer_methods():
    """``_train_one_step`` / ``_fingerprint_arm`` call these trainer methods."""
    from src.shared.training import MultiHeadTrainer

    for name in ("_autocast", "_forward_batch", "_maybe_graph_model", "train"):
        assert callable(getattr(MultiHeadTrainer, name, None)), name


def test_capture_trainers_patches_captures_and_restores():
    """End-to-end of the capture machinery without a model/GPU/data.

    The stub never touches ``self``'s attributes, so a sentinel object exercises the
    full append-and-return-{} path; the context manager must restore the original.
    """
    import src.analysis.streams_6pos_prototype as m
    from src.shared.training import MultiHeadTrainer

    original = MultiHeadTrainer.train
    m._CAPTURED.clear()
    with m._capture_trainers():
        assert MultiHeadTrainer.train is not original
        sentinel = object()
        out = MultiHeadTrainer.train(sentinel, "TL", "VL", 7)
        assert out == {}
        assert len(m._CAPTURED) == 1
        cap = m._CAPTURED[0]
        assert cap["trainer"] is sentinel
        assert (cap["train_loader"], cap["val_loader"], cap["n_epochs"]) == ("TL", "VL", 7)
    assert MultiHeadTrainer.train is original  # restored even though we captured
