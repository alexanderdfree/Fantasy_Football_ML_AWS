"""Prototype + benchmark: single process, ONE thread, per-position CUDA streams.

**Lever B** from the launch-bound investigation
([todo/gpu_launch_bound_levers.md](../../todo/gpu_launch_bound_levers.md), lines 105-119).
The production local trainer runs the 6 positions as 6 OS subprocesses time-slicing
one GPU (``python -m src.benchmarking.parallel_train -j 6``). Each position's
attention NN is a tiny ~69K-param model that is **GPU launch/host-bound** — it fires
hundreds of thousands of microsecond kernels and the GPU sits ~80% idle between
launches. Lever B asks: collapse the 6 subprocesses into **one process** that drives
all 6 positions on per-position ``torch.cuda.Stream``s, so their kernels co-reside in a
single CUDA context and fill each other's idle gaps (the substitute for NVIDIA MPS,
which is unavailable on the WSL2/Windows 5080 this targets).

**The crux (why one thread, not six):** Python's GIL serialises host-side launches
across threads — the very thing that's the bottleneck — so this drives a **single
thread that round-robins one training step per position per round onto each position's
own stream**. The GPU scheduler then overlaps the *device-side* execution across
streams. This is "a real restructure of the training loop, not a wrapper" (the doc),
so we interleave at *step* granularity; the trainer's own ``train()`` owns an epoch
loop and can't be reused as a black box for that.

**How it stays faithful with minimal copying:** rather than re-implement ~150 lines of
tensor-building + model/optimizer/scheduler/criterion construction, we run the *real*
``src.shared.pipeline._train_attention_holdout`` (the same call the sibling
``overlap_base_attn_prototype.py`` uses) with ``MultiHeadTrainer.train`` **monkeypatched
to capture** the fully-built trainer + GPU-resident loaders instead of training them.
The only copied production code is the ~30-line per-step body (``_train_one_step``,
tagged KEEP-IN-SYNC). This reuses ALL real construction verbatim and is strictly more
faithful than replicating it.

**Standalone — no production edit:** lives only in ``src/analysis/`` and imports the
pipeline's real functions; it does NOT change ``src/shared``, ``src/{pos}``, or
``src/batch`` (⇒ no path-based 6-position retrain). If the measured win holds, the
follow-up wires the overlap into ``_gpu_branch`` behind an ``FF_*`` flag (default off).

**Parity (correctness guard):** a true single-process round-robin shares the global RNG
across positions, so interleaving would change each position's dropout/shuffle draw
order vs a sequential run (production isolates this by using separate *processes*). We
restore per-position determinism by **re-seeding the global RNG per (position, epoch)
before each batcher ``__iter__``** (the batcher draws its shuffle from the CPU global
RNG, training.py:387). With ``--force-dropout-zero`` (default on) the only per-step RNG
consumer is gone, so the streams arm and the sequential arm train **bit-identical**
models — the fingerprint asserts CUDA streams don't corrupt the math. Re-seeding
deviates from production's continuous-RNG flow but does NOT affect the wall-clock
measurement (compute cost is batch-order-independent).

Usage (needs ``data/splits/*.parquet`` and CUDA — on CPU the streams arm is meaningless):

    # headline A/B (graphed seq vs streams), 3 seeds:
    for s in 42 1337 2024; do
      python -m src.analysis.streams_6pos_prototype --seed $s --fixed-epochs 30
    done

    # strict bit-parity isolation (eager + deterministic + dropout-zero):
    python -m src.analysis.streams_6pos_prototype --no-graph --deterministic

    # production-realistic absolute times (dropout on):
    python -m src.analysis.streams_6pos_prototype --no-force-dropout-zero

Then compare ``streams wall-clock`` against the ``-j6`` subprocess reference:

    python -m src.benchmarking.parallel_train -j 6     # prints 'total wall-clock'
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from dataclasses import dataclass

# Per-position seed offset (prime) so re-seeds for distinct positions/epochs don't
# collide. CAPTURE_TAG is a distinct epoch sentinel so graph-capture's warmup batch
# is deterministic per position and identical across the two arms (its BatchNorm
# warmup perturbation must match for bit-parity).
_POS_SEED_STRIDE = 100_003
_CAPTURE_TAG = 999_999

# Filled by the monkeypatched MultiHeadTrainer.train (see _capture_trainers).
_CAPTURED: list[dict] = []


@dataclass
class PositionState:
    position: str
    index: int  # stable per-position offset for the re-seed scheme
    trainer: object  # the real (untrained) MultiHead*Trainer, fully constructed
    train_loader: object  # _GPUResidentBatcher (shuffle=True, drop_last=True)
    val_loader: object  # _GPUResidentBatcher (shuffle=False) — for the parity forward
    n_epochs: int
    targets: list[str]
    graphed: bool = False


def _fingerprint(preds: dict) -> dict:
    """Deterministic per-target summary of model outputs for parity checks.

    Verbatim from ``overlap_base_attn_prototype.py:45-56`` — float64 sum per target,
    rounded to 4 decimals, catches divergence at ~4 dp.
    """
    import numpy as np

    return {t: round(float(np.asarray(v, dtype=np.float64).sum()), 4) for t, v in preds.items()}


# ---------------------------------------------------------------------------
# Capture machinery: reuse the REAL construction path, intercept .train()
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _capture_trainers():
    """Patch ``MultiHeadTrainer.train`` to stash the built trainer and skip training.

    Subclasses (History / HistoryWithOpp / NestedHistory) inherit ``train`` — they
    override only ``_forward_batch`` / ``_graph_inputs`` / ``_maybe_graph_model`` — so
    patching the base method intercepts every attention trainer. The stub records the
    fully-constructed trainer + its GPU-resident loaders and returns an empty history
    so the surrounding ``_train_attention_nn`` finishes harmlessly (its post-train
    ``predict_numpy`` on the untrained model is ignored).
    """
    from src.shared.training import MultiHeadTrainer

    real_train = MultiHeadTrainer.train

    def _capturing_train(self, train_loader, val_loader, n_epochs):
        _CAPTURED.append(
            {
                "trainer": self,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "n_epochs": n_epochs,
            }
        )
        return {}

    MultiHeadTrainer.train = _capturing_train
    try:
        yield
    finally:
        MultiHeadTrainer.train = real_train


def _build_states(positions: list[str], seed: int, splits, fixed_epochs: int) -> dict:
    """Construct one untrained PositionState per position via the real pipeline.

    Calls the real ``_train_attention_holdout`` (faithful tensor-building + model /
    optimizer / scheduler / criterion / trainer construction) with ``.train()``
    intercepted, so we get production-identical trainers without copying construction.
    """
    from src.shared.pipeline import _prepare_position_data, _train_attention_holdout
    from src.shared.registry import get_config

    train_df, val_df, test_df = splits
    states: dict[str, PositionState] = {}
    for idx, pos in enumerate(positions):
        cfg = get_config(pos)
        targets = cfg["targets"]
        prepared = _prepare_position_data(pos, cfg, train_df, val_df, test_df)
        (X_tr, X_v, X_te, y_tr, y_v, y_te, pos_tr, pos_v, pos_te, feat_cols) = prepared

        _CAPTURED.clear()
        with _capture_trainers():
            # seed_everything(seed) is called INSIDE _train_attention_nn, so each
            # build re-seeds → models are identical across arms regardless of order.
            _train_attention_holdout(
                pos,
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
        if len(_CAPTURED) != 1:
            raise RuntimeError(
                f"{pos}: expected exactly one captured trainer, got {len(_CAPTURED)}"
            )
        cap = _CAPTURED[0]
        n_epochs = fixed_epochs if fixed_epochs > 0 else int(cap["n_epochs"])
        states[pos] = PositionState(
            position=pos,
            index=idx,
            trainer=cap["trainer"],
            train_loader=cap["train_loader"],
            val_loader=cap["val_loader"],
            n_epochs=n_epochs,
            targets=targets,
        )
    return states


# ---------------------------------------------------------------------------
# The per-step body (the ONLY copied production code) + the drivers
# ---------------------------------------------------------------------------
def _train_one_step(trainer, batch) -> None:
    """One optimizer step.

    === KEEP IN SYNC with src/shared/training.py MultiHeadTrainer.train (866-906) ===
    Replicates: zero_grad → autocast(forward + loss + dormant entropy) →
    scaler.scale().backward() → unscale_ → clip_grad_norm_(1.0) → scaler.step →
    scaler.update → (per-batch scheduler).
    Intentionally omits: the FP32 GPU-resident loss accumulator (a sync we don't need
    for fixed-work timing and that would serialise streams), the scaler-trace /
    fixed-scale diagnostic branches (912-937), and the per-batch get_scale()
    skip-detection (only used when scheduler_per_batch=True). The parity assertion
    catches any *numeric* drift from this copy; it cannot catch a *new sync* added to
    production — re-check this block if the source loop changes.
    """
    import torch

    trainer.optimizer.zero_grad(set_to_none=True)
    with trainer._autocast():
        preds, y_batch = trainer._forward_batch(batch)
        loss, _ = trainer.criterion(preds, y_batch)
        entropy_fn = getattr(trainer.model, "attention_entropy_loss", None)
        if entropy_fn is not None:
            entropy_term = entropy_fn()
            if entropy_term is not None:
                loss = loss + entropy_term
    trainer._scaler.scale(loss).backward()
    trainer._scaler.unscale_(trainer.optimizer)
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
    trainer._scaler.step(trainer.optimizer)
    trainer._scaler.update()
    # Default cosine_warm_restarts is per-epoch (stepped by the drivers at epoch
    # boundaries). A scheduler_per_batch config (e.g. OneCycle) steps here; we step
    # unconditionally (skip detection omitted) — flagged in the report if it occurs.
    if trainer.scheduler_per_batch:
        trainer.scheduler.step()


def _reseed(base_seed: int, index: int, epoch: int) -> None:
    """Make the next batcher __iter__'s shuffle deterministic per (position, epoch).

    The GPU-resident batcher draws its permutation from the CPU global RNG at
    __iter__ (training.py:387); re-seeding here makes batch order independent of
    round-robin interleaving so the two arms are bit-comparable.
    """
    import torch

    torch.manual_seed(base_seed + index * _POS_SEED_STRIDE + epoch)


def _maybe_capture_graph(state: PositionState, base_seed: int, stream) -> None:
    """Capture the position's CUDA graph (if enabled) on its stream, before timing.

    Re-seeds first so capture's warmup sample-batch (and thus its BatchNorm warmup
    perturbation) is identical across the sequential and streams arms. K's nested
    trainer no-ops capture by design (training.py:1448); failures fall back to eager.
    """
    import torch

    _reseed(base_seed, state.index, _CAPTURE_TAG)
    ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    try:
        with ctx:
            state.trainer._maybe_graph_model(state.train_loader)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        state.graphed = getattr(state.trainer, "_graphed", False)
    except Exception as e:  # noqa: BLE001 — graph+stream capture is the fragile bit
        print(f"  WARN: graph capture failed for {state.position}: {e!r}; eager fallback")
        state.graphed = False


def _train_position_solo(state: PositionState, base_seed: int) -> float:
    """Sequential baseline: train one position to completion on the default stream.

    Times ONLY the training loop (graph capture excluded), mirroring the sibling
    prototype's "time the training call only" discipline.
    """
    import torch

    state.trainer.model.train()
    _maybe_capture_graph(state, base_seed, stream=None)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for epoch in range(state.n_epochs):
        _reseed(base_seed, state.index, epoch)
        for batch in state.train_loader:
            _train_one_step(state.trainer, batch)
        if not state.trainer.scheduler_per_batch:
            state.trainer.scheduler.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _train_streams_roundrobin(states: dict, base_seed: int) -> tuple[float, dict]:
    """The lever: one thread, round-robin one step per position per round on streams.

    Returns (total_wall_sec, per_position_host_sec). Per-position host time brackets
    the position's first step to when it's dropped from the round (host-side, not a
    sync) for a rough contention factor.
    """
    import torch

    pos_list = list(states.values())
    for s in pos_list:
        s.trainer.model.train()
    # Real streams on CUDA; None (→ default stream, no overlap) off-CUDA so the
    # round-robin logic still runs on a CPU box for validation (the caller already
    # warned that CPU results are meaningless for the overlap measurement).
    cuda = torch.cuda.is_available()
    streams = {s.position: (torch.cuda.Stream() if cuda else None) for s in pos_list}

    # Capture all graphs up front, serially, each on its own stream.
    for s in pos_list:
        _maybe_capture_graph(s, base_seed, stream=streams[s.position])

    iters = {}
    epochs = {s.position: 0 for s in pos_list}
    for s in pos_list:
        _reseed(base_seed, s.index, 0)
        iters[s.position] = iter(s.train_loader)
    active = list(pos_list)
    host_t0 = {s.position: None for s in pos_list}
    host_sec = {s.position: 0.0 for s in pos_list}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    while active:
        for s in list(active):
            if host_t0[s.position] is None:
                host_t0[s.position] = time.perf_counter()
            try:
                batch = next(iters[s.position])
            except StopIteration:
                # End of this position's epoch: step its per-epoch scheduler.
                if not s.trainer.scheduler_per_batch:
                    s.trainer.scheduler.step()
                epochs[s.position] += 1
                if epochs[s.position] >= s.n_epochs:
                    host_sec[s.position] = time.perf_counter() - host_t0[s.position]
                    active.remove(s)
                    continue
                _reseed(base_seed, s.index, epochs[s.position])
                iters[s.position] = iter(s.train_loader)
                batch = next(iters[s.position])
            # Issue this step onto the position's stream; NO cross-stream sync in the
            # round — the GPU scheduler overlaps device-side kernels across streams.
            stream = streams[s.position]
            ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with ctx:
                _train_one_step(s.trainer, batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0, host_sec


def _fingerprint_arm(states: dict) -> dict:
    """Per-position parity fingerprint = summed eval-mode outputs over the val set.

    eval() routes a graphed model back to eager forward (make_graphed_callables
    auto-dispatch), dropout is off, and BatchNorm uses running stats, so the forward
    is deterministic given the trained weights — effectively a weight fingerprint via
    the model's own (subclass-correct) ``_forward_batch``.
    """
    import torch

    out = {}
    for s in states.values():
        s.trainer.model.eval()
        sums = {t: 0.0 for t in s.targets}
        with torch.no_grad():
            for batch in s.val_loader:
                preds, _ = s.trainer._forward_batch(batch)
                for t in s.targets:
                    sums[t] += float(preds[t].detach().float().sum().cpu())
        out[s.position] = {t: round(v, 4) for t, v in sums.items()}
    return out


def _fmt(x) -> str:
    return f"{x:8.1f}" if isinstance(x, (int, float)) else f"{str(x):>8}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    import src.config as config

    parser.add_argument(
        "--positions",
        nargs="*",
        default=list(config.POSITIONS),
        help="Positions to pack (default: all six).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-epochs",
        type=int,
        default=30,
        help="Train exactly N epochs per position for a clean A/B (0 = each cfg's "
        "nn_epochs). Default 30.",
    )
    parser.add_argument(
        "--graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force CUDA graph on/off (default: autodetect via cuda_graph_enabled()). "
        "--no-graph sets FF_CUDA_GRAPH=0 for a bit-comparable eager run.",
    )
    parser.add_argument(
        "--force-dropout-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero all NN dropout (default: on) so the two arms are bit-comparable "
        "and the seq→streams speedup is faithful (affects both arms equally). "
        "--no-force-dropout-zero for production-realistic absolute times.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set FF_DETERMINISTIC=1 (disables cuDNN autotune + TF32) to isolate "
        "parity drift. Default off for realistic per-kernel timing.",
    )
    parser.add_argument(
        "--skip-nested",
        action="store_true",
        help="Drop nested-history positions (K) — useful if graph+stream capture is problematic.",
    )
    args = parser.parse_args()

    # Env knobs must be set BEFORE the pipeline reads them (all read at call-time:
    # cuda_graph_enabled / _nn_device / _maybe_force_dropout_zero).
    if args.graph is False:
        os.environ["FF_CUDA_GRAPH"] = "0"
    elif args.graph is True:
        os.environ.pop("FF_CUDA_GRAPH", None)  # let autodetect (sm_80+) decide ON
    if args.force_dropout_zero:
        os.environ["FF_FORCE_DROPOUT_ZERO"] = "1"
    if args.deterministic:
        os.environ["FF_DETERMINISTIC"] = "1"

    import torch

    from src.shared.pipeline import _read_split
    from src.shared.utils import cuda_graph_enabled

    positions = [p.upper() for p in args.positions]
    if args.skip_nested:
        # K is the only nested-history position; drop it by config flag rather than
        # hard-coding the name.
        from src.shared.registry import get_config

        positions = [
            p for p in positions if get_config(p).get("attn_history_structure") != "nested"
        ]

    cuda = torch.cuda.is_available()
    if not cuda:
        print(
            "WARNING: no CUDA visible. This harness measures GPU per-position STREAM "
            "OVERLAP; on CPU there are no streams and the 'streams' arm is meaningless. "
            "Run on a CUDA box (5080/WSL2 or L4/g6).\n"
        )

    graph_on = cuda_graph_enabled()
    print("=== single-process + per-position CUDA streams prototype (Lever B) ===")
    print(
        f"positions={positions} seed={args.seed} fixed_epochs={args.fixed_epochs} "
        f"graph={'ON' if graph_on else 'off'} "
        f"dropout_zero={'ON' if args.force_dropout_zero else 'off'} "
        f"deterministic={'ON' if args.deterministic else 'off'}\n"
    )

    splits_dir = config.SPLITS_DIR
    splits = (
        _read_split(f"{splits_dir}/train.parquet"),
        _read_split(f"{splits_dir}/val.parquet"),
        _read_split(f"{splits_dir}/test.parquet"),
    )

    # --- Arm A: sequential single-process (baseline + parity reference) ---
    print("Building + training SEQUENTIAL arm (one position at a time) ...", flush=True)
    seq_states = _build_states(positions, args.seed, splits, args.fixed_epochs)
    device_name = torch.cuda.get_device_name(0) if cuda else "cpu"
    solo_sec = {p: _train_position_solo(s, args.seed) for p, s in seq_states.items()}
    fp_seq = _fingerprint_arm(seq_states)
    del seq_states  # free GPU memory before the streams arm builds fresh models
    if cuda:
        torch.cuda.empty_cache()

    # --- Arm B: round-robin per-position streams (the lever) ---
    print("Building + training STREAMS arm (round-robin on per-position streams) ...", flush=True)
    str_states = _build_states(positions, args.seed, splits, args.fixed_epochs)
    streams_wall, host_sec = _train_streams_roundrobin(str_states, args.seed)
    fp_str = _fingerprint_arm(str_states)
    per_batch_sched = [p for p, s in str_states.items() if s.trainer.scheduler_per_batch]
    eager_in_graph = [p for p, s in str_states.items() if graph_on and not s.graphed]

    # --- Report ---
    seq_total = sum(solo_sec.values())
    speedup = seq_total / streams_wall if streams_wall else float("nan")
    print(f"\ndevice: {device_name}\n")
    print(f"{'position':9} | {'solo (s)':>9} | {'streams (s)':>11} | {'contention ×':>12} | parity")
    print("-" * 70)
    for p in positions:
        if p not in solo_sec:
            continue
        cont = host_sec.get(p, 0.0) / solo_sec[p] if solo_sec[p] else 0.0
        parity = "OK" if fp_seq.get(p) == fp_str.get(p) else "DRIFT"
        print(
            f"{p:9} | {_fmt(solo_sec[p])} | {_fmt(host_sec.get(p, 0.0))} | {_fmt(cont)} | {parity}"
        )
    print("-" * 70)
    print(f"sequential total (sum of solo)   : {seq_total:8.1f} s")
    print(f"streams wall-clock (round-robin) : {streams_wall:8.1f} s")
    print(f"SEQUENTIAL → STREAMS SPEEDUP     : {speedup:8.2f}×")
    if eager_in_graph:
        print(f"\nnote: graph ON but eager (no-op capture) for: {eager_in_graph} (nested trainer)")
    if per_batch_sched:
        print(
            f"note: scheduler_per_batch positions (step-skip detection omitted): {per_batch_sched}"
        )

    all_ok = all(fp_seq.get(p) == fp_str.get(p) for p in solo_sec)
    if not all_ok:
        print(
            "\nPARITY DRIFT — streams-trained models differ from sequential.\n"
            "  Expected when --no-force-dropout-zero (shared-RNG interleaving changes\n"
            "  dropout draws) or when cuDNN autotune picks different kernels under\n"
            "  co-residency. For the authoritative bit-parity check, run:\n"
            "    --force-dropout-zero --deterministic --no-graph\n"
            "  Persistent DRIFT there points to a real bug in the copied step body."
        )
    print(
        "\nreference: `python -m src.benchmarking.parallel_train -j 6` prints the\n"
        "'-j6' subprocess total wall-clock — the production local baseline. Compare it\n"
        "to the streams wall-clock above (caveat: -j6 runs the FULL pipeline incl. base\n"
        "NN + LightGBM + early-stop, so it's a directional ceiling check, not an\n"
        "apples-to-apples fixed-epoch attn-only A/B). Honest expectation: the win is\n"
        "bounded — CUDA graphs already collapsed the launch storm and the single thread\n"
        "only overlaps device-side work (GIL serialises launches); and -j6 already fills\n"
        "some idle gaps. Run 3 seeds and report mean±std."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
