# GPU launch-bound optimization — CUDA graph built; streams prototyped

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
  `--parallel-backend auto --n-jobs auto` (auto n_jobs = CPU count, RAM-clamped
  for mps → 4 on the g6.xlarge shape), preserving the outer six-position
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

### Thread-vs-MPS backend A/B (2026-06-11) — MPS wins by disqualification; thread+graph is concurrency-UNSAFE

Two 20-target-trial RB tune jobs on identical g6.xlarge Spot hosts/image, `--n-jobs 4`:
- **mps (processes): SUCCEEDED** — 45 trials (22 complete / 23 pruned / 0 fail) in **331 s**
  (~8.2 trials/min incl. pruned), studies in `scheduler_v2_mps_graph`.
- **thread: FAILED** — `torch.AcceleratorError: CUDA error: the operation cannot be performed
  in the present state` (`cudaErrorIllegalState`) in `_maybe_graph_model` →
  `make_graphed_callables` → `capture_begin`, a few trials in. Root cause is structural,
  not a flake: `torch.cuda.graph` captures in the default **"global" capture mode**, which
  errors when ANY other thread in the process has in-flight GPU work — with n_jobs>1
  threads, one trial's capture races every other trial's kernel launches. Processes (MPS
  backend) isolate capture state per trial; threads cannot, short of upstream
  `capture_error_mode="thread_local"` support in `make_graphed_callables`.

**Verdict: `auto` stays mps on Batch.** Corollary: LOCAL thread-backend tuning
with `n_jobs>1` on an sm_80+ box (the 5080's default `--n-jobs 2` + autodetect-ON graphs)
has been a latent crash lottery since the 2026-06-05 cutover.

**Guard shipped (2026-06-11):** `tune_nn.py::_force_eager_for_concurrent_thread_trials` —
when the resolved backend is `thread` AND resolved `n_jobs>1` AND `cuda_graph_enabled()`
(the trainer's actual capture decision, not the raw env), it forces `FF_CUDA_GRAPH=0` +
`FF_CUDA_GRAPH_FULL=0` (Lever A2 shares the constraint and gates on the base resolver
anyway) **before** `_resolve_storage_version`, so the study lands in the eager
`scheduler_v2` namespace the trials actually train under, and prints a one-liner citing
`cudaErrorIllegalState`. `n_jobs==1` thread mode and the mps backend stay graphed. A
capture lock was considered and rejected: it only serializes captures, but global-mode
capture also conflicts with other threads' *ordinary* kernel launches mid-capture, so
eager is the only safe thread-concurrent configuration (short of the upstream
`thread_local` support above). **A/B consequence:** a graphed thread-concurrent arm is
not a valid configuration at all — the backend A/B is thread-eager-vs-MPS-graphed
(different training regimes, separate namespaces), or thread `n_jobs=1` graphed vs MPS
graphed for a same-regime comparison.

### Lever A2 — FULL-STEP capture (`FF_CUDA_GRAPH_FULL`) — BUILT 2026-06-11; gates i/iv/v measured PASS, ii/iii pending

Lever A collapsed the launches *inside* the model, but the L4 step remained ~7.5 ms of
host dispatch for ≲0.1 ms of GPU math (measured 2026-06-10: per-step time is
batch-size-invariant — bs512 0.25 s/epoch over 33 steps ≈ bs256 0.47 s over 66 — and the
eager remainder is gather + MultiTargetLoss (~dozens of launches) + GradScaler/clip/AdamW +
the Python loop). A2 widens the capture: `_GraphedTrainStep` (training.py) graphs
gather+forward+combined-loss as one callable (branch-free hurdle dispatch via
`compute_combined_capturable`; the train loop feeds bare idx tensors from
`_GPUResidentBatcher.index_batches()`), leaving only the idx handoff + scaler/optimizer
tail eager. **Opt-in** (`cuda_graph_full_enabled()`: `FF_CUDA_GRAPH_FULL` truthy AND the
base sm_80+ gate) — training default-off (one approved rebaseline, not two); tune jobs
default-on via `launch_tune --cuda-graph-full` with studies isolated in `*_graphfull`
namespaces; capture failure falls back to the model-only graph; K no-ops. Ship gates (run
on Batch): step-ms ≥2× vs model-only graph, graph-vs-graph same-seed Δ=0, tune smoke
≥1.7× trials/min with pruning live, flat `memory_reserved` across ≥30 sequential trials
per worker. Context: under the 24-vCPU Spot quota (~1 launch-bound trial per vCPU), host-CPU
per trial is the only lever that raises fleet tune throughput.

**Gate results (2026-06-11, RB on g6.xlarge, mps n_jobs=4, 20-complete-trial smokes):**
- **First smoke caught a capture-aborting bug** (job `97a8cf3c`): every trial logged
  `Cannot copy between CPU and CUDA tensors during CUDA graph capture` and fell back to the
  model-only graph — `torch.tensor(0.0, device=cuda)` in the loss accumulators stages a
  Python scalar through pageable CPU memory. The fallback held perfectly (52 trials, 0
  failures, correct namespace/provenance) — but the run measured the fallback, not the
  feature, and its study was deleted as regime-mislabeled. Fixed with device-native
  `torch.zeros((), dtype=float32)` at both accumulator sites (also removes a hidden
  per-step pageable H2D from the eager path); dtype pinned by test.
- **(i) step-ms: PASS ~2.4×** (job `0174f61c`, zero capture failures across 58 trials).
  Same-config trial pairs across the fallback/fixed runs (identical worker seed streams from
  empty studies): bs512-class 0.40–0.42 → **0.16–0.18 s/epoch**; bs256-class 0.76–0.80 →
  0.30–0.37; bs128-class 1.3–1.6 → 0.61–0.69. ≈5 ms/step at bs512 under 4-way contention
  (vs ~12 contended / ~7.3 uncontended on model-only graphs).
- **(iv) trials/min: borderline PASS 1.64–1.68×** (58 trials/260.1 s vs 45/331 baseline and
  52/392.8 same-seed fallback) — diluted vs the per-step win by Amdahl on short trials:
  per-trial capture (~1–2 s), prepare_data (~0.5 s), the eager per-epoch val pass, and a
  prune-heavier mix (37 vs 23). Best val 58.71 ≈ fallback 58.69 ≈ baseline 58.78 — no
  regime catastrophe. Verdict: **tune default stays ON**.
- **(v) memory: no concern at current scale** — `peak_mem_gb` 0.10→0.37 ratchet per worker
  (lifetime-max metric + per-trial graph pools), ≲2% of the 24 GB L4 at n_jobs=4. Caveat
  stands for 100+-trial-per-worker runs (consider `empty_cache()` between trials).
- **(ii) graph-vs-graph same-seed Δ=0 and (iii) 3-seed regime sanity (FF_NN_FIXED_EPOCHS=30,
  FF_FORCE_DROPOUT_ZERO=1, vs model-only graph within the ~0.5% worst-target envelope):
  PENDING** — required before proposing the knob anywhere beyond tuning (training stays
  default-off regardless until an owner-approved rebaseline). 5080 recipe (Batch
  `--mode train` would pollute prod artifacts, so run locally):
  `FF_CUDA_GRAPH_FULL=1 FF_NN_FIXED_EPOCHS=30 FF_FORCE_DROPOUT_ZERO=1 python -m src.rb.run_pipeline`
  twice same-seed → diff attn metrics for (ii); 3 seeds × `FF_CUDA_GRAPH_FULL={1,0}`
  (same fixed-epoch knobs) → worst-target attn FP-MAE delta ≤ ~0.5% for (iii).
  Note the graphfull regime now INCLUDES the graphed val pass (D2 below).

### Round 2 — per-trial fixed costs (D1 #1127) + graphed val pass (D2 #1128), 2026-06-11

The 1.65× trials/min vs 2.4× step-ms gap was Amdahl: per-trial fixed costs + the eager
per-epoch val pass. D1 added `cfg["trial_data_memo"]` (tune-only, injected post-deepcopy):
trials 2+ of a worker skip the split parquet re-read + frame hashing + the attention
history/opp array rebuild (`prepare_data 0.0s`, `[trial_memo]` log lines) — plus
`empty_cache()` between MPS worker trials. D2 captures the val pass as ONE
`torch.cuda.CUDAGraph` over all K full-size val batches (baked `narrow` views, in-graph
FP32 accumulation, pred output buffers; ragged tail eager; permanent eager fallback on
build failure) — graphfull regime := graphed train step + graphed val (the day-old smoke
studies were wiped at the redefinition; no production studies existed).

**Measured (RB, 20-complete-target, mps n_jobs=4, fresh studies each):**

| stage | trials (complete) | wall | trials/min | epoch_sec bs512 / bs256 |
|---|---|---|---|---|
| 2026-06-10 morning (pre-everything) | 10 (6) | 117.7 s | 5.1 | 0.25 / 0.47 |
| Tier-1..3 base (model-only graphs + sync+sampler fixes) | 45 (22) | 331 s | 8.2 | — |
| + full-step train capture (#1122/#1124) | 58 (21) | 260.1 s | 13.4 | 0.16–0.18 / 0.30–0.37 |
| + D1 trial-data memo (#1127, COLD feature cache: 4×~40 s first-trial misses) | 67 (23) | 242.3 s | 16.6 | — |
| + D2 graphed val (#1128) | 60 (23) | 192.2 s | **18.7** | **0.14–0.16 / 0.26–0.27** |

D1+D2 = **+40% trials/min** over the full-step base (plan predicted +20–30%), zero capture
failures train or val, best-val stable (58.7–59.2 across runs). Net for the day: **~3.7×
trials per GPU-hour** on identical hardware. Remaining levers are the rejected
whole-iteration capture — this closes the planned throughput rounds; next bottleneck is
per-trial capture build (~1–2 s) + Optuna/study churn, not worth chasing.

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

**Prototype (2026-06-07) — runs the benchmark gate, no production change:** a standalone,
no-retrain measurement harness for the single-process round-robin is built —
[src/analysis/streams_6pos_prototype.py](../src/analysis/streams_6pos_prototype.py). It
reuses the *real* per-position construction (it runs `_train_attention_holdout` with
`MultiHeadTrainer.train` monkeypatched to **capture** the fully-built trainer + GPU-resident
loaders instead of training them — so only the ~30-line per-step body is copied, not the
construction) and drives the 6 attention NNs **one step per position per round on per-position
`torch.cuda.Stream`s** from a single thread. It reports the sequential→streams speedup, a
per-position contention factor, and per-position prediction parity (re-seeding per
(position, epoch) + `--force-dropout-zero` keeps the two arms bit-comparable despite the
shared global RNG). It composes with CUDA graphs (`--graph`/`--no-graph`) and degrades to a
serial no-stream run off-CUDA. The gate itself is still **unrun** (no local GPU in this
environment / CI); run it on a CUDA box:
`for s in 42 1337 2024; do python -m src.analysis.streams_6pos_prototype --seed $s --fixed-epochs 30; done`,
then compare its `streams wall-clock` to `python -m src.benchmarking.parallel_train -j 6`.
**Honest expectation:** the win is bounded — graphs already collapsed the launch storm, the
single thread only overlaps *device-side* work (the GIL serialises launches), and `-j6`
already fills some idle gaps; so the incremental win over graphed `-j6` may be modest. The
harness exists to replace that estimate with a number.

### Lever B′ — within-position overlap (base NN ∥ attention NN) — PROTOTYPE (2026-06-07)

A smaller, lower-risk slice of B scoped to **one position**: the pipeline trains
the base NN then the attention NN *sequentially* in `_gpu_branch`, but they are
independent (no stacking) and each is launch-bound, so overlapping them as **two
processes sharing one GPU** collapses the GPU branch from `nn_train + attn_nn_train`
→ `max(...)` (~2× for balanced positions like DST 98+94 / QB 36+36). Two
processes (not threads) because host launch dispatch is GIL-bound, and each child
re-seeds its own RNG so solo and concurrent runs stay bitwise-identical (the
harness asserts per-target prediction-fingerprint parity). **GPU-arch-independent**
(fills idle gaps on T4/L4/5080 alike — measurable on the current T4 fleet, no L4
migration needed) and **composes with CUDA graphs** (graph each model AND overlap).

Standalone benchmark harness (does NOT touch the production pipeline ⇒ no
retrain): [src/analysis/overlap_base_attn_prototype.py](../src/analysis/overlap_base_attn_prototype.py).
Run on a CUDA box with `data/splits`:
`python -m src.analysis.overlap_base_attn_prototype --position QB --seed 42` — it
reports solo-vs-concurrent per-model train time, the contention factor, the
overlap speedup, and prediction parity. If the win holds, productionize by
overlapping the two trainers in `_gpu_branch` behind an `FF_*` flag (default off).

**AWS cousin:** the same "pack all six on one GPU" idea, but via **real NVIDIA MPS on a
warm Linux L4** (MPS is available there, unlike WSL2/Windows) instead of in-process CUDA
streams, is sketched in [proposed-adr-warm-mps-packed-training.md](proposed-adr-warm-mps-packed-training.md)
(Option B; start/stop warm host, benchmark-gated before any build). The two could share a
per-position worker entry.

---

## Recommended sequencing
1. ~~**Lever A first**~~ — **DONE** (shipped behind `FF_CUDA_GRAPH`, 1.84×, off by default; not bit-inert — see the Lever A result above).
2. **Lever B only if A is insufficient** — it's the bigger MPS-substitute hammer but supersedes the core pool + subprocess model; weigh that explicitly before committing. (A's per-position 1.84× is a real local-iteration win, so B is lower priority now.)

Both are **launch-overhead** levers (the measured bottleneck), not occupancy/FLOP levers (the 5080 has those to spare).
