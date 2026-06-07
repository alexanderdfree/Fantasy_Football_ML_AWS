# GPU launch-bound optimization — CUDA graph built; streams planned

Planning doc for making **local 6-position parallel training faster**, written after the
core-pool work (#670) established that the bottleneck is the GPU, not the CPU. **Lever A is
now built & measured** (2026-05-31, see below); Lever B is not. Both levers are opt-in and
gated like the existing `FF_COMPILE` per-arch speed knobs (ADR-0017). The intended gate was a
per-position A/B (inertness Δ=0 MAE + speedup) — Lever A cleared the speedup but **not** the
strict Δ=0 inertness, and shipped as a documented opt-in speed knob anyway (details below).

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

## Why not MPS locally (researched 2026-05-31, rejected on both OSes)
NVIDIA MPS would give true multi-process kernel co-residency — but it is **Linux/QNX-only**. MPS Overview r590 (Dec 2025), verbatim: *"MPS is only supported on the Linux and QNX operating systems. The MPS server will fail to start when launched on an operating system other than Linux."*
- **WSL2:** binaries not shipped, `/dev/nvidiactl` absent (only `/dev/dxg` paravirt). NVIDIA moderator (Apr 2026) confirms WDDM blocks it.
- **Native Windows 11:** no Windows MPS build exists on any GPU; **TCC** is not available on consumer GeForce (and wouldn't help — MPS is OS-gated, not driver-gated); WDDM only time-slices; MIG is datacenter-only.
- **Only bare-metal Linux** unlocks MPS on the 5080 — not worth dual-booting (reintroduces the native-Windows `OPENBLAS_NUM_THREADS=1` segfault, and WSL2 is the tuned toolchain). So the substitute for MPS is the in-process **CUDA-streams** lever below, which works identically on WSL2/Windows.

## AWS Batch tuning exception — NVIDIA MPS is available on g6/Linux

The local rejection above does **not** apply to the active AWS Batch tune host:
g6.xlarge runs Linux on an L4 (`sm_89`). The NN tuner now has a true
process-backed MPS backend for that environment:

- `src.tuning.launch_tune` defaults Batch tune jobs to
  `--parallel-backend auto --n-jobs 3`, preserving the outer six-position
  fan-out (one g6.xlarge per position). The container resolves `auto` through
  `detect_platform()`: native-Linux L4/g6 -> MPS; Mac/MPS, WSL/native 5080, and
  non-L4 CUDA hosts -> the existing thread backend.
- `src.tuning.tune_nn` starts `nvidia-cuda-mps-control` inside the container,
  launches worker subprocesses via `spawn`, and uses the existing core pool to
  lease CPU affinity for each whole trial while BLAS stays capped at one thread.
- Optuna SQLite studies use WAL + a longer busy timeout in MPS mode; S3
  checkpoints are parent-owned and uploaded from SQLite backup snapshots so
  concurrent workers do not race the checkpoint file.
- Batch tune jobs set `FF_CUDA_GRAPH=1`, `FF_DEVICE=cuda`,
  `FF_AMP_DTYPE=auto`, and `FF_COMPILE=0`. The graph-on study namespace is
  `scheduler_v2_mps_graph`, separate from eager `scheduler_v2`.
- K's nested-history attention trainer explicitly no-ops CUDA graph capture;
  flat-history positions use graph capture, and all six positions can run under
  the same graph-on tune workflow.

**Production now autodetects graphs ON (2026-06-05, owner decision).**
`cuda_graph_enabled()` defaults ON for **any** CUDA sm_80+ box (g6/L4 `sm_89`,
5080 `sm_120`), so the production Batch fan-out and local sm_80+ runs are graphed
with no env opt-in; `FF_CUDA_GRAPH=0` is the force-off override. `train-batch.yml`
still threads the `FF_BATCH_CUDA_GRAPH` repo variable as an optional fleet
override (set it to `0` to force eager), and labels Batch benchmark rows
`g6.xlarge (Spot, CUDA-graph)` so the graphed era stays separate from the
pre-cutover eager baseline. This was shipped **without** a pre-merge A/B (owner
chose speed-now); the first post-merge 6-position retrain *is* the graphed
rebaseline. K's nested trainer still no-ops capture, and CPU/CI/T4 stay eager
(byte-identical). To run a bit-comparable eager A/B locally, set `FF_CUDA_GRAPH=0`
(~3 seeds per AGENTS.md).

---

## Lever A — CUDA graphs (per-position; collapse the per-step launch storm) — BUILT & MEASURED (2026-05-31)

**Idea:** capture the per-step forward+backward kernel sequence once and replay it, turning thousands of host launches into one.

**Shipped:** opt-in `FF_CUDA_GRAPH` (off by default, sm_80+ gate, mirrors `FF_COMPILE`; `cuda_graph_enabled()` in `utils.py`). `MultiHeadTrainer._maybe_graph_model` wraps the model with `torch.cuda.make_graphed_callables` at the top of `train()`, leaving GradScaler/optimizer step *outside* the captured fwd+bwd. The nested-K trainer is deliberately **not** graphed (its `x_game_history=` kwarg + `None`-leaf inputs violate the tensor-only `sample_args` contract). The three original "friction points" mostly evaporated against the real torch 2.11 `make_graphed_callables`: it is **pytree-native** (the dict-returning forward round-trips with no adapter), **auto-dispatches** train→graph / eval→eager (so the ragged val pass is untouched, no manual plumbing), and the **entropy regulariser is dormant** (`attn_entropy_coeff=0` in every config, so the side-effect never fires).

**Result on RB (single position, 5080, dropout-0 unless noted):**
- **Speedup: 1.84× on `attn_nn_train`** (23.8s→13.1s); ~1.54× total pipeline. Real — the model is *partly* launch-bound (but only partly: FP32 is 2× slower than FP16, so tensor-core compute matters too).
- **NOT bit-inert.** Graph-vs-eager attn_nn drifts **+0.13% aggregate / ~0.5% worst-target** (`receiving_yards`), while **eager-vs-eager is Δ=0.0000** (fully reproducible). So the drift is graph-attributable, not noise.

**Root cause (fully bisected):** the forward output is **bitwise identical**, and a **single** fwd+bwd step is **bitwise identical** (trivial loss *and* real `MultiTargetLoss` + FP16 ×65536 GradScaler scaling — grad Δ=0). The drift is purely **multi-step**: the **FP16 + GradScaler** path amplifies sub-ULP graph kernel-ordering differences over thousands of steps. The same fixed-step A/B in **FP32 keeps the divergence 30–100× smaller** (0.0014 vs 0.044 weight-space), confirming amplification rather than a bug. It appears in both the learnable weights *and* the BatchNorm running-stat buffers, and **early-stopping further amplifies it** (a perturbed val curve picks a different best epoch — LN swings the stop 13 epochs).

**Fix levers tried — none gives fast AND bit-inert:**

| Lever | Bit-inert? | Fast vs AMP-eager? | Verdict |
|---|---|---|---|
| LayerNorm backbone (`FF_NN_NORM`) | No — *worse* per-target (0.27) | 2.05× | flatter val curve → early-stop swings 13 epochs; aggregate ≈ BN |
| FP32 (`FF_AMP_DTYPE=fp32`) | ≈ yes | **No — 0.87×** (slower than baseline) | FP32 forfeits the FP16 tensor-core 2×; graph collapse doesn't overcome it |
| Deterministic stop (`FF_NN_FIXED_EPOCHS=N`) | No — 0.077 worst | 1.84× | drift is model-state divergence, not best-epoch *selection* |
| BN running-stat recalibration (implied) | No | — | learnable weights also diverge (0.074), not just BN buffers |

**Decision (2026-05-31):** ship `FF_CUDA_GRAPH` as an **opt-in local-iteration speed knob** with the non-inertness documented — per-step math is exact and the model is equivalent quality, but it is **not** suitable for bit-comparable benchmark A/Bs against eager baselines. Off by default ⇒ AWS / CI / production byte-identical (the commit is training-skippable; Ridge MAE unchanged). The investigation knobs (`FF_NN_NORM`, `FF_FORCE_DROPOUT_ZERO`, `FF_NN_FIXED_EPOCHS`) are **kept** for a follow-up. Note `FF_NN_NORM` overlaps [src/tuning/ablate_backbone_norm.py](../src/tuning/ablate_backbone_norm.py) (which monkeypatches the same BN→LN swap) but composes with the graph env knobs in a single `benchmark` invocation.

**Decision SUPERSEDED (2026-06-05, owner call):** `cuda_graph_enabled()` now **autodetects ON for sm_80+** (graphs are the default on g6/L4 + 5080); `FF_CUDA_GRAPH` is demoted to a force-off override. This deliberately makes the sm_80+ training path non-byte-identical to CPU/CI — the launch-bound speedup was prioritised over benchmark comparability, and benchmark history rebaselines graphed-vs-graphed from the cutover (Batch rows self-label `g6.xlarge (Spot, CUDA-graph)`). Shipped **without** a pre-merge A/B; the first post-merge retrain is the new graphed baseline. The "off by default ⇒ production byte-identical" property above no longer holds for sm_80+ — use `FF_CUDA_GRAPH=0` to recover the eager path for a bit-comparable A/B. See ADR-0017.

**Benchmarkability follow-up (2026-06-01):** [src/analysis/cuda_graph_gradscale.py](../src/analysis/cuda_graph_gradscale.py) resolved the GradScaler question. Graph-vs-graph is clean (identical metrics + scale schedule), BN warmup snapshot/restore is inert, and eager-vs-graph first diverges in `GradScaler` at step 1. Fixed normal-scale mode is invalid because the default initial scale 65536 overflows at step 0.

**Fixed-scale follow-up (2026-06-01):** lower explicit scales were tested on RB with fixed 30 epochs, dropout disabled, deterministic mode on, and graph BN warmup restored. `init_scale=2048` and `1024` still overflowed in the graph arm; `512` completed both graph and eager with identical scale/skip traces (2130 steps, 0 scale changes) but still produced a graph-vs-eager MAE delta (`4.044157` vs `4.060941`). That isolates the remaining difference to the expected multi-step FP16 trajectory drift from graph replay/kernel ordering. Fixed scale is not a bit-comparable bridge to eager and is its own worse-quality training regime, so the decision is: **graphed runs compare to graphed runs, not to eager baselines**; use a graphed local rebaseline for `FF_CUDA_GRAPH=1` A/Bs.

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
1. ~~**Lever A first**~~ — **DONE** (shipped behind `FF_CUDA_GRAPH`, 1.84×, off by default; not bit-inert — see the Lever A result above).
2. **Lever B only if A is insufficient** — it's the bigger MPS-substitute hammer but supersedes the core pool + subprocess model; weigh that explicitly before committing. (A's per-position 1.84× is a real local-iteration win, so B is lower priority now.)

Both are **launch-overhead** levers (the measured bottleneck), not occupancy/FLOP levers (the 5080 has those to spare).
