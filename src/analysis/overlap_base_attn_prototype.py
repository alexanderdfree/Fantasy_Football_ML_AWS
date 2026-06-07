"""Prototype + benchmark: train the base NN ∥ attention NN concurrently.

Lever 2 from the launch-bound investigation (todo/gpu_launch_bound_levers.md).
The production pipeline (``src/shared/pipeline.py`` ``_gpu_branch``) trains the
base ``MultiHeadNet`` and then the attention NN **sequentially** on the GPU.
They are **independent** (the base NN's predictions are not an input to the
attention NN — no stacking), and each is **launch-bound** (the ~69K-param model
leaves the GPU ~90% idle waiting on host-side CUDA launch dispatch). So running
the two trainings as **two processes sharing one GPU** lets their launch streams
interleave and fill each other's idle gaps, collapsing the GPU branch from
``nn_train + attn_nn_train`` toward ``max(nn_train, attn_nn_train)``.

Why two PROCESSES, not threads: the bottleneck is Python/host-side launch
dispatch, which is GIL-bound — two threads in one process would serialise on the
GIL (the very thing that's the bottleneck). Two processes each get their own GIL
*and* their own RNG, so (a) host dispatch truly parallelises across cores and
(b) ``seed_everything`` in each child can't race the other's global RNG — which
is the correctness invariant this harness verifies (per-target prediction
fingerprints must match the solo runs).

This is a **standalone prototype** — it imports the pipeline's real training
functions so the benchmark is faithful, but it does NOT change the production
pipeline (no ``src/shared`` edit ⇒ no 6-position retrain). If the measured win
holds, the follow-up wires the overlap into ``_gpu_branch`` behind an ``FF_*``
flag.

The overlap is **GPU-arch-independent** (it fills idle gaps on any launch-bound
GPU), so it is measurable on the current T4 Batch fleet — it does not require the
L4 migration, and it composes with CUDA graphs (graph each model AND overlap
them).

Usage (needs ``data/splits/*.parquet`` and ideally CUDA — on CPU the "concurrent"
arm just contends for cores and is NOT representative of the GPU-overlap win):

    python -m src.analysis.overlap_base_attn_prototype --position QB --seed 42
"""

from __future__ import annotations

import argparse
import sys
import time


def _fingerprint(preds: dict) -> dict:
    """Deterministic per-target summary of test predictions for parity checks.

    Concurrency must not change the trained models: each child re-seeds its own
    process, so solo and concurrent runs should produce identical predictions
    (modulo cuDNN autotune nondeterminism under GPU contention — which this
    fingerprint surfaces rather than hides). Float64 sum per target catches any
    divergence at ~4 decimals.
    """
    import numpy as np

    return {t: round(float(np.asarray(v, dtype=np.float64).sum()), 4) for t, v in preds.items()}


def _train_one_worker(which: str, position: str, seed: int, result_q) -> None:
    """Child entry point: train ONE model (``base`` or ``attn``) and report.

    Top-level (picklable) so it works under the spawn start method CUDA needs.
    Loads its own data (the feature cache makes the 2nd load cheap), trains, and
    puts ``{which, train_sec, fingerprint, device}`` — or ``{which, error}`` — on
    the queue. ``train_sec`` times ONLY the training call (excludes data prep), so
    the solo-vs-concurrent comparison isolates the GPU-branch overlap.
    """
    try:
        import torch

        from src.shared.pipeline import (
            _prepare_position_data,
            _read_split,
            _train_attention_holdout,
            _train_nn,
        )
        from src.shared.registry import get_config

        cfg = get_config(position)
        targets = cfg["targets"]
        splits_dir = __import__("src.config", fromlist=["SPLITS_DIR"]).SPLITS_DIR
        train_df = _read_split(f"{splits_dir}/train.parquet")
        val_df = _read_split(f"{splits_dir}/val.parquet")
        test_df = _read_split(f"{splits_dir}/test.parquet")

        (X_tr, X_v, X_te, y_tr, y_v, y_te, pos_tr, pos_v, pos_te, feat_cols) = (
            _prepare_position_data(position, cfg, train_df, val_df, test_df)
        )

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

        t0 = time.monotonic()
        if which == "base":
            _model, _scaler, test_preds, _metrics, _hist = _train_nn(
                X_tr, X_v, X_te, y_tr, y_v, y_te, cfg, targets, seed
            )
        elif which == "attn":
            _model, _scaler, test_preds, _metrics, _hist, _cols = _train_attention_holdout(
                cfg,
                targets,
                seed,
                X_tr,
                X_v,
                X_te,
                y_tr,
                y_v,
                y_te,
                pos_tr,
                pos_v,
                pos_te,
                feat_cols,
                opp_source_frames=(train_df, val_df, test_df),
            )
        else:
            raise ValueError(f"unknown which={which!r}")
        train_sec = time.monotonic() - t0

        result_q.put(
            {
                "which": which,
                "train_sec": round(train_sec, 2),
                "fingerprint": _fingerprint(test_preds),
                "device": device,
            }
        )
    except Exception as e:  # noqa: BLE001 — report instead of hanging the parent
        import traceback

        result_q.put({"which": which, "error": repr(e), "traceback": traceback.format_exc()})


def _spawn_and_collect(ctx, position: str, seed: int, whichs: list[str]) -> dict:
    """Spawn one process per entry in ``whichs`` (concurrently), join, collect.

    ``whichs=["base"]`` / ``["attn"]`` gives the solo baselines; ``["base",
    "attn"]`` runs both at once (the overlap arm). Returns ``{which: result}``.
    """
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_train_one_worker, args=(w, position, seed, q), name=f"train-{w}")
        for w in whichs
    ]
    for p in procs:
        p.start()
    # Collect before join: the child blocks on the queue feeder until the parent
    # drains it for large payloads (not an issue for these tiny dicts, but the
    # drain-then-join order is the robust pattern).
    results = {}
    for _ in whichs:
        r = q.get()
        results[r["which"]] = r
    for p in procs:
        p.join()
    return results


def _fmt(x) -> str:
    return f"{x:6.1f}" if isinstance(x, (int, float)) else f"{str(x):>6}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", default="QB", help="Position to benchmark (default: QB)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pos = args.position.upper()

    import torch
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")

    cuda = torch.cuda.is_available()
    if not cuda:
        print(
            "WARNING: no CUDA visible. This harness measures GPU-branch OVERLAP; "
            "on CPU the 'concurrent' arm just contends for cores and the result is "
            "NOT representative of the GPU win. Run on a CUDA box (L4/T4/5080).\n"
        )

    print(f"=== base-NN ∥ attention-NN overlap prototype — {pos}, seed={args.seed} ===\n")

    # Solo baselines (each model alone on the GPU) — this is the current
    # sequential pipeline's per-phase cost.
    print("Running solo base-NN ...", flush=True)
    base_solo = _spawn_and_collect(ctx, pos, args.seed, ["base"])["base"]
    print("Running solo attention-NN ...", flush=True)
    attn_solo = _spawn_and_collect(ctx, pos, args.seed, ["attn"])["attn"]
    # Concurrent (the lever): both models train at once, sharing the GPU.
    print("Running base-NN ∥ attention-NN (concurrent) ...", flush=True)
    conc = _spawn_and_collect(ctx, pos, args.seed, ["base", "attn"])

    # Surface any worker failure loudly.
    for label, r in (
        ("base-solo", base_solo),
        ("attn-solo", attn_solo),
        ("base-conc", conc.get("base", {})),
        ("attn-conc", conc.get("attn", {})),
    ):
        if r.get("error"):
            print(f"\nWORKER FAILED ({label}): {r['error']}\n{r.get('traceback', '')}")
            return 1

    bs, as_ = base_solo["train_sec"], attn_solo["train_sec"]
    bc, ac = conc["base"]["train_sec"], conc["attn"]["train_sec"]
    seq_total = bs + as_  # sequential GPU branch (current pipeline)
    conc_wall = max(bc, ac)  # concurrent GPU branch (the lever)
    speedup = seq_total / conc_wall if conc_wall else float("nan")

    print(f"\ndevice: {base_solo.get('device')}\n")
    print(f"{'model':10} | {'solo (s)':>9} | {'concurrent (s)':>14} | {'contention ×':>12}")
    print("-" * 56)
    print(f"{'base NN':10} | {_fmt(bs)}    | {_fmt(bc)}         | {_fmt(bc / bs if bs else 0)}")
    print(f"{'attn NN':10} | {_fmt(as_)}    | {_fmt(ac)}         | {_fmt(ac / as_ if as_ else 0)}")
    print("-" * 56)
    print(f"sequential GPU branch (base + attn) : {seq_total:6.1f} s")
    print(f"concurrent GPU branch (max)         : {conc_wall:6.1f} s")
    print(f"OVERLAP SPEEDUP                      : {speedup:6.2f}×")

    # Correctness: concurrency must not change the trained models.
    base_ok = base_solo["fingerprint"] == conc["base"]["fingerprint"]
    attn_ok = attn_solo["fingerprint"] == conc["attn"]["fingerprint"]
    print(
        f"\nprediction parity (concurrent == solo): base={'OK' if base_ok else 'DRIFT'}, "
        f"attn={'OK' if attn_ok else 'DRIFT'}"
    )
    if not (base_ok and attn_ok):
        print(
            "  NOTE: drift under contention is usually cuDNN autotune picking "
            "different kernels when two contexts share the GPU; re-check with "
            "FF_DETERMINISTIC=1 before treating it as a real regression."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
