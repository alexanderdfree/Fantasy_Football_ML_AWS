"""Per-worker trial-data memo (cfg["trial_data_memo"], PR D1).

Asserts the two pipeline hooks on the RB tiny pipeline:

* hook 2 (attn arrays): a second same-config run with a shared memo skips the
  history/opp array rebuild entirely AND produces bit-identical attention
  metrics — the memo is a pure fixed-cost remover, never a numerics change.
* hook 1 (prepared frames): with frames read from (a tmp) SPLITS_DIR, a second
  run skips both the parquet reads and ``_prepare_position_data``; bumping a
  split file's mtime invalidates via the ``_splits_stat_key`` guard.

Budget: ~4 tiny pipeline runs, well under the e2e timeout.
"""

from __future__ import annotations

import os
import tempfile

import pytest

import src.shared.pipeline as pipeline_mod
from tests._pipeline_e2e_utils import build_tiny_config

# Attention ON (tiny dims) — _TINY_OVERRIDES disables it, but the memo's
# array hook lives on the attention path.
_ATTN_TINY = {
    "train_attention_nn": True,
    "attn_d_model": 8,
    "attn_n_heads": 1,
    "attn_encoder_hidden_dim": 0,
    "attn_dropout": 0.0,
    "attn_lr": 1e-3,
    "attn_batch_size": 64,
}


def _tiny_attn_config(memo: dict | None) -> dict:
    cfg = build_tiny_config("RB")
    cfg.update(_ATTN_TINY)
    cfg["cv_split_column"] = "week"
    if memo is not None:
        cfg["trial_data_memo"] = memo
    return cfg


def _find_data_raw_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        candidate = os.path.join(here, "data", "raw")
        if os.path.isdir(candidate):
            return candidate
        here = os.path.dirname(here)
    raise FileNotFoundError("Could not locate data/raw/ relative to test file")


def _run_in_tmp(cfg, workdir, *, train_df=None, val_df=None, test_df=None, seed=42):
    original_cwd = os.getcwd()
    data_raw_src = _find_data_raw_dir()
    try:
        os.chdir(workdir)
        os.makedirs("src/rb/outputs/models", exist_ok=True)
        os.makedirs("src/rb/outputs/figures", exist_ok=True)
        dst = os.path.join(workdir, "data", "raw")
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(data_raw_src, dst)
        return pipeline_mod.run_pipeline(
            "RB",
            cfg,
            train_df=None if train_df is None else train_df.copy(),
            val_df=None if val_df is None else val_df.copy(),
            test_df=None if test_df is None else test_df.copy(),
            seed=seed,
        )
    finally:
        os.chdir(original_cwd)


def _count_calls(monkeypatch, module, name):
    counter = {"n": 0}
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return counter


@pytest.mark.e2e
@pytest.mark.timeout(300)
def test_attn_array_memo_skips_rebuild_and_is_bit_identical(synthetic_splits, monkeypatch):
    train_df, val_df, test_df = synthetic_splits
    hist_calls = _count_calls(monkeypatch, pipeline_mod, "build_game_history_arrays")
    opp_calls = _count_calls(monkeypatch, pipeline_mod, "build_opp_defense_history_arrays")

    memo: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        r1 = _run_in_tmp(
            _tiny_attn_config(memo), tmp, train_df=train_df, val_df=val_df, test_df=test_df
        )
    first_hist, first_opp = hist_calls["n"], opp_calls["n"]
    assert first_hist == 3  # train/val/test
    assert ("attn_arrays", "RB") in memo

    with tempfile.TemporaryDirectory() as tmp:
        r2 = _run_in_tmp(
            _tiny_attn_config(memo), tmp, train_df=train_df, val_df=val_df, test_df=test_df
        )
    assert hist_calls["n"] == first_hist  # zero additional builds
    assert opp_calls["n"] == first_opp

    # Numerics-inert: same seed, cached arrays -> bit-identical attention metrics.
    assert r1["attn_nn_metrics"] == r2["attn_nn_metrics"]


@pytest.mark.e2e
@pytest.mark.timeout(300)
def test_attn_array_memo_fp_mismatch_recomputes(synthetic_splits, monkeypatch):
    train_df, val_df, test_df = synthetic_splits
    hist_calls = _count_calls(monkeypatch, pipeline_mod, "build_game_history_arrays")

    memo: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        _run_in_tmp(_tiny_attn_config(memo), tmp, train_df=train_df, val_df=val_df, test_df=test_df)
    assert hist_calls["n"] == 3
    # Corrupt the stored fingerprint -> the guard must force a rebuild.
    memo[("attn_arrays", "RB")]["fp"] = {"stale": True}
    with tempfile.TemporaryDirectory() as tmp:
        _run_in_tmp(_tiny_attn_config(memo), tmp, train_df=train_df, val_df=val_df, test_df=test_df)
    assert hist_calls["n"] == 6


@pytest.mark.e2e
@pytest.mark.timeout(300)
def test_prepared_memo_skips_split_reads_and_stat_guard_invalidates(
    synthetic_splits, monkeypatch, tmp_path
):
    train_df, val_df, test_df = synthetic_splits
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    # The synthetic frames carry duplicated pos_* one-hots (base frame +
    # build_features both add them) which parquet rejects; the real splits
    # have unique columns, so dedupe for the round-trip.
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        df.loc[:, ~df.columns.duplicated()].to_parquet(splits_dir / f"{name}.parquet")
    monkeypatch.setattr(pipeline_mod, "SPLITS_DIR", str(splits_dir))

    read_calls = _count_calls(monkeypatch, pipeline_mod, "_read_split")
    prepare_calls = _count_calls(monkeypatch, pipeline_mod, "_prepare_position_data")

    memo: dict = {}
    cfg = _tiny_attn_config(memo)
    with tempfile.TemporaryDirectory() as tmp:
        _run_in_tmp(cfg, tmp)  # train_df=None -> disk path
    assert read_calls["n"] == 3
    assert prepare_calls["n"] == 1
    assert ("prepared", "RB") in memo

    with tempfile.TemporaryDirectory() as tmp:
        _run_in_tmp(_tiny_attn_config(memo), tmp)
    assert read_calls["n"] == 3  # no re-read
    assert prepare_calls["n"] == 1  # no re-prepare

    # Stat guard: rewriting a split file invalidates the memo entry.
    train_df.loc[:, ~train_df.columns.duplicated()].to_parquet(splits_dir / "train.parquet")
    with tempfile.TemporaryDirectory() as tmp:
        _run_in_tmp(_tiny_attn_config(memo), tmp)
    assert read_calls["n"] == 6
    assert prepare_calls["n"] == 2


@pytest.mark.unit
def test_splits_stat_key_tracks_file_changes(monkeypatch, tmp_path):
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(pipeline_mod, "SPLITS_DIR", str(tmp_path))
    key1 = pipeline_mod._splits_stat_key()
    assert len(key1) == 3
    assert pipeline_mod._splits_stat_key() == key1
    (tmp_path / "val.parquet").write_bytes(b"xy")  # size change
    assert pipeline_mod._splits_stat_key() != key1
