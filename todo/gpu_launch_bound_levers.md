# GPU launch-bound optimization — levers & plans (not yet built)

Planning doc for making **local 6-position parallel training faster**, written after the
core-pool work (#670) established that the bottleneck is the GPU, not the CPU. Nothing here
is built — this is the pick-up plan for a future focused session. Both levers are opt-in,
must clear a per-position A/B (inertness Δ=0 MAE + measured speedup) before shipping, and are
gated like the existing `FF_COMPILE` per-arch speed knobs (ADR-0017).

## Diagnosis: the parallel local trainer is GPU launch/host-bound, not CPU-bound

Measured on the WSL2 / 9950X3D / RTX 5080 box, all 6 positions, pool ON:

- The core pool (#670) cut the **LightGBM stage 15–90×** (RB 92s→6s, TE 91s→3s, DST 71s→1s) but **total wall-clock was unchanged (~242s).**
- `-j` sweep (total wall-clock): **`-j6` 242s < `-j3` 244s < `-j2` 269s** — *more* concurrency is *faster*. If the GPU were compute-saturated this would tie or hurt; instead each process leaves the GPU ~80% idle and stacking processes fills the gaps.
- Per-position wall-clock = `max(CPU branch ≈ 6–11s, GPU branch ≈ 200s)`. The **attention-NN** training dominates and is **launch-bound**: a tiny model (~69K params, `attn_batch_size=256`, ~18K steps/position) fires hundreds of thousands of microsecond kernels; the GPU idles between launches waiting on CPU/Python dispatch. The 5080's TFLOPS are irrelevant — the limit is launch/host overhead, not occupancy.

**Implication:** the lever for total wall-clock is **reducing GPU launch/host overhead or packing kernels**, not CPU allocation. The core pool was a correctness/cleanliness win (and shipped the auto-`total_wall_sec` + measured dispatch order), but it cannot move this wall.

### Already done (don't redo)
- `#305` — train-branch per-batch sync removal (GPU-resident loss accumulation). **The val branch is also already sync-free** (`training.py:751-811`, one sync/epoch) — verified 2026-05-31, nothing to remove there.
- `#655` — fused AdamW + fewer `GradScaler.get_scale()` syncs (−15.6%).
- `#309` — GPU-resident batcher (no per-batch H2D copy; `drop_last=True` → fixed train-batch shape).
- FP16 AMP default + TF32 auto (sm_80+).

### Stop-rule (don't relitigate without a benchmark)
- **`torch.compile` is measured-rejected** (`#641`, **+169% on the 5080**) — dynamic-shape recompiles. A hand-rolled CUDA graph sidesteps that (train shapes are static via `drop_last`), but anything touching this area must clear a per-position A/B.

## Why not MPS (researched 2026-05-31, rejected on both OSes)
NVIDIA MPS would give true multi-process kernel co-residency — but it is **Linux/QNX-only**. MPS Overview r590 (Dec 2025), verbatim: *"MPS is only supported on the Linux and QNX operating systems. The MPS server will fail to start when launched on an operating system other than Linux."*
- **WSL2:** binaries not shipped, `/dev/nvidiactl` absent (only `/dev/dxg` paravirt). NVIDIA moderator (Apr 2026) confirms WDDM blocks it.
- **Native Windows 11:** no Windows MPS build exists on any GPU; **TCC** is not available on consumer GeForce (and wouldn't help — MPS is OS-gated, not driver-gated); WDDM only time-slices; MIG is datacenter-only.
- **Only bare-metal Linux** unlocks MPS on the 5080 — not worth dual-booting (reintroduces the native-Windows `OPENBLAS_NUM_THREADS=1` segfault, and WSL2 is the tuned toolchain). So the substitute for MPS is the in-process **CUDA-streams** lever below, which works identically on WSL2/Windows.

---

## Lever A — CUDA graphs (per-position; collapse the per-step launch storm)

**Idea:** capture the per-step forward+backward kernel sequence once and `replay()` it, turning thousands of host launches into one. Directly attacks the launch overhead.

**Approach:** opt-in `FF_CUDA_GRAPH` env flag (off by default, sm_80+ gate, mirroring `FF_COMPILE` in `pipeline.py`). When on, `MultiHeadTrainer.train` / `MultiHeadHistoryTrainer.train` graph **only forward+backward** via `torch.cuda.make_graphed_callables` (or a hand-rolled warmup→capture→replay), leaving the GradScaler/optimizer step *outside* the graph.

**Three friction points to solve (the real work):**
1. **GradScaler (FP16 default on the 5080).** `scaler.step`/`update` have data-dependent inf/nan control flow (`training.py:694-717`) — not graph-safe. Keep them outside the captured region; graph only fwd+bwd. (Optimizer step is already a single fused-AdamW launch since #655, so little is lost.)
2. **Side-effecting entropy regulariser.** `attention_entropy_loss()` reads attn weights cached as a side effect of the last forward (`neural_net.py:609`, `training.py:679`). Graph replay writes the *same* static buffers each step, so the entropy term must read the post-replay buffer (wire it as a graph output / read the static tensor after `replay()`), not a fresh Python value.
3. **Static input buffers.** Allocate static `(static_x, static_h, static_mask, static_y)`; copy each batch in, `replay()`. `drop_last=True` guarantees fixed train-batch shape; the GPU-resident batcher already holds data on-GPU, so this is a copy into the captured buffers. Val pass (drop_last=False, variable last batch) stays un-graphed — it's eval-only and cheaper.

**Benchmark gate:** single position (start with a heavy one, e.g. DST/TE), `FF_CUDA_GRAPH` off vs on. Assert **inertness** (per-target MAE Δ=0.0000 — graphs change launch mechanics, not math) and report **attn_nn_train wall-clock delta**. Ship behind the flag only if it wins; if it regresses or walls on the entropy/scaler frictions, record the result here and stop (the torch.compile precedent says this is a real risk).

**Effort:** ~1 focused session + 1 training A/B run. Per-position, doesn't disturb the orchestrator.

---

## Lever B — single process + per-position CUDA streams (the MPS substitute)

**Idea:** the thing MPS would have done. Collapse the 6 subprocesses into **one process** running the 6 positions on separate `torch.cuda.Stream`s, so their kernels co-reside in one CUDA context and fill each other's idle gaps inside the scheduler — without a cross-process server.

**Approach:** a new in-process orchestrator that builds all 6 positions' data + models, then drives their training steps on per-position streams. The launch-bound nature means the GPU has headroom to overlap independent streams' kernels.

**Frictions / risks:**
- **GIL.** Python's GIL serialises the *host-side* launches across threads, which is the very thing that's the bottleneck. So **one thread issuing to multiple streams** (round-robin a step per position per stream), not 6 Python threads. This is a real restructure of the training loop, not a wrapper.
- **Supersedes recent work.** It largely *replaces* the subprocess model + the just-merged core pool (#670) + the AF_UNIX coordinator — those exist to share *CPU* cores across *processes*; a single process shares the GPU via streams and the CPU via in-process thread/`n_jobs` control. Don't build B without deciding the core pool's fate (B would retire it for the local path).
- **Blast radius.** One crash kills all 6 (vs. the subprocess model's isolation); shared CUDA OOM risk (6 models resident at once — fine at ~69K params each); much higher implementation complexity.
- **CUDA graphs compose with streams** — B and A are not mutually exclusive (graph each position's step, replay on its stream).

**Benchmark gate:** total wall-clock of the single-process-streams run vs the current `-j6` subprocess run (~242s), same inertness assertion. Only pursue if Lever A's per-position win is insufficient and the total-wall-clock ceiling is worth the refactor.

**Effort:** multi-session; architectural. Highest ceiling, highest risk.

---

## Recommended sequencing
1. **Lever A first** — lower risk, per-position, leaves the orchestrator intact; a clean win shippable behind `FF_CUDA_GRAPH`. Start with the feasibility probe (does `make_graphed_callables` capture the real attention model + autocast given the entropy/scaler frictions?).
2. **Lever B only if A is insufficient** — it's the bigger MPS-substitute hammer but supersedes the core pool + subprocess model; weigh that explicitly before committing.

Both are **launch-overhead** levers (the measured bottleneck), not occupancy/FLOP levers (the 5080 has those to spare).
