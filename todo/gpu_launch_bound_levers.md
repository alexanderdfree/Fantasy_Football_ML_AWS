# GPU launch-bound optimization — CUDA graph built; streams measured-negative

Planning doc for making **local 6-position parallel training faster**, written after the
core-pool work (#670) established that the bottleneck is the GPU, not the CPU. **Lever A is
now built & measured** (2026-05-31, see below); **Lever B is now MEASURED-NEGATIVE**
(2026-06-22, 5080 sm_120 — no cross-stream overlap; lever closed, see below). Both levers are opt-in and
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
- FP32+TF32 default (AMP off; TF32 auto sm_80+) — flipped from FP16-AMP-default 2026-06-22 (FP16 now `FF_AMP_DTYPE=fp16` opt-in); the autocast-removal lever below, shipped.

### Stop-rule (don't relitigate without a benchmark)
- **`torch.compile` is measured-rejected** (`#641`, **+169% on the 5080**) — dynamic-shape recompiles. A hand-rolled CUDA graph sidesteps that (train shapes are static via `drop_last`), but anything touching this area must clear a per-position A/B.

## Precision & quantization levers (FP16 / TF32 / AMP / quant) — SHIPPED: FP32+TF32 is now the default (2026-06-22; autocast-removal speedup measured + shipped — see UPDATE/SHIPPED below)

Asked whether fp16-vs-tf32, mixed precision, or quantization offer an easy win. They don't:
the model is **launch-bound** (above), so any lever that targets math throughput or memory
can't move wall-clock. Current state, for the record so it isn't re-investigated:

- **FP16 vs TF32 — already optimal, orthogonal, both on.** Autocast downcasts to FP16 tensor
  cores; residual FP32 GEMMs use TF32 via the modern `torch.set_float32_matmul_precision("high")`
  in `_nn_device()` ([src/shared/pipeline.py](../src/shared/pipeline.py)), sm_80+-effective,
  `not deterministic`-gated, across all 4 NN training paths + the CV path — not the legacy
  `allow_tf32` booleans. Nothing to tune; `"medium"` would alter the deliberately-frozen
  FP32-GEMM metric path. (But see the UPDATE below: the measured speed lever is *removing*
  autocast, not tuning the TF32 `set_float32_matmul_precision` setting.)
- **Mixed precision (AMP) — textbook, mature; now OFF by default (FP16 opt-in).** Since 2026-06-22 the
  default is AMP-off FP32+TF32 on every CUDA GPU (`amp_dtype` returns `None` for `auto`,
  [src/shared/utils.py](../src/shared/utils.py)); the FP16 path below remains as `FF_AMP_DTYPE=fp16`. When
  FP16 is opted in, GradScaler is FP16-only, CUDA-only, kept OUTSIDE the
  CUDA graph; autocast wraps train + val forward; inference (`predict_numpy`,
  [src/shared/neural_net.py](../src/shared/neural_net.py)) is plain FP32 — the correct choice (eval
  in higher precision; CPU serving is FP32 too, so no train→serve skew). BF16 is measured-rejected
  (#640: QB passing_yards +2.2–3.1%) + T4 hang (#293/#301); a per-arch training-dtype default is an
  AGENTS.md stop-rule.
- **Quantization — none in the repo, not worth adding.** GPU INT8/FP8 can't help a launch-bound
  model (no math saturation to relieve); the CPU serving bottleneck is feature-build (moved to a CI
  job; serving only downloads the S3 artifact, ADR-0018), not the ~276 KB NN forward. Calibration +
  accuracy risk for zero measurable gain.

**UPDATE (2026-06-22, measured — the "precision can't move wall-clock" framing above was too strong).** An owner-requested A/B (QB + RB + WR, 5080/sm_120, via an `ab_harness` spec toggling `nn_use_amp=False`, i.e. `FF_AMP_DTYPE=fp32`) found that **dropping FP16 autocast entirely** — FP32 storage + TF32 matmuls instead of FP16 + autocast + GradScaler — **moves wall-clock**: NN-wall **−27% QB / −11% RB / −13% WR**, and is **accuracy-neutral on the served attention NN** (n=8, graphs-off / per-step bit-exact; every |Δ|/SE < 1; no high-magnitude-head regression, opposite of BF16; Ridge+LGBM bit-identical → clean NN-only change). This does **not** contradict "launch-bound" — it *confirms* it: the win is a **launch** lever, not a throughput one. Autocast inserts per-op FP16↔FP32 **cast kernels**; on this tiny launch-bound model those extra launches cost more than FP16's tensor-core throughput saves, while TF32 keeps a single dtype (FP32 storage, matmul uses TF32 internally) and launches fewer kernels. A 3-arm attribution isolated the cause: **GradScaler is NOT it** — FP16-without-GradScaler is **−3.5% (slightly *slower*)** and the scaler is near-inert (dynamic scale settles to 4.0, 1.43% inf-skips confined to warmup, removing it leaves served-model accuracy unchanged); the entire win is the `fp16-no-scaler → tf32` (autocast-removal) step. **SHIPPED (2026-06-22, owner-approved):** FP32+TF32 is now the **default** training path (`amp_dtype()`→`None` for `auto`; FP16 autocast removed from the default path), with FP16+GradScaler retained as the `FF_AMP_DTYPE=fp16` opt-in. It is a **metric-path change**, so the rebaseline happens via the first post-merge **6-position retrain** (ADR-0017 updated, owner sign-off gated on K/DST validation). The absolute seconds are tiny (~1–2.4 s/position) and near-invisible on the orchestration-bound Batch critical path, and the local magnitude is contended/position-dependent — but the change is accuracy-neutral and removes the autocast cast-kernel launches, so it ships. **Net: the conclusion ("don't switch for speed alone") is superseded — the autocast-removal IS the switch, shipped as the new default; the old stated reason ("precision can't move wall-clock") was wrong** — removing the autocast machinery does.

The largest *dedicated* un-shipped GPU lever is still **launch**-side — Lever B/B′ (CUDA streams), below — but "launch, not precision" is a false dichotomy: the precision-*path* change in the UPDATE above is itself a launch lever (autocast cast-kernel removal).

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
  `detect_platform()`: native-Linux L4/g6 or A10G/g5 (sm_89/sm_86) -> MPS;
  Mac/MPS, WSL/native 5080, and other CUDA hosts -> the existing thread backend.
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

**Benchmarkability follow-up (2026-06-01):** the `cuda_graph_gradscale` harness resolved the GradScaler question. Graph-vs-graph is clean (identical metrics + scale schedule), BN warmup snapshot/restore is inert, and eager-vs-graph first diverges in `GradScaler` at step 1. Fixed normal-scale mode is invalid because the default initial scale 65536 overflows at step 0. **(Tooling removed 2026-06-22:** the `cuda_graph_gradscale` harness + the `FF_AMP_FIXED_SCALE` / `FF_AMP_INIT_SCALE` / `FF_GRADSCALER_TRACE_*` instrumentation were deleted once the FP32+TF32 default (#1311) dropped the `GradScaler` from the default path — the FP16-graph bit-comparability question is now moot on the default path; the conclusion above stands for the opt-in FP16 path.)

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
per worker. Context: under the Spot G+VT quota (~1 launch-bound trial per vCPU; 24 vCPU
when written, raised to 64 on 2026-06-11), host-CPU per trial is the only lever that
raises per-host tune throughput — the quota raise multiplies hosts, not trials-per-host.

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
  VERIFIED on the local 5080 (sm_120), 2026-06-22 — results below.** (Was the open gate
  before proposing the knob beyond tuning; full-step shipped as the production default on
  2026-06-15 / #1171 off the L4 Batch validation, and these 5080 numbers corroborate it.)
  Reproduction (Batch `--mode train` would pollute prod artifacts, so run locally):
  `FF_CUDA_GRAPH_FULL=1 FF_NN_FIXED_EPOCHS=30 FF_FORCE_DROPOUT_ZERO=1 python -m src.rb.run_pipeline`
  twice same-seed → diff attn metrics for (ii); seeds × `FF_CUDA_GRAPH_FULL={1,0}`
  (same fixed-epoch knobs) → full-step-vs-model-only attn FP-MAE delta for (iii).
  Note the graphfull regime now INCLUDES the graphed val pass (D2 below).
  - **(ii) PASS, exactly.** Two same-seed `FF_CUDA_GRAPH_FULL=1` RB runs gave attn FP-MAE
    `4.134755` vs `4.134755`, |Δ| = 0.00e+00 (deterministic Ridge FP-MAE identical too → same
    data path). The graphed full-step path is bit-deterministic run-to-run.
  - **(iii) PASS in substance (8 seeds, RB, 30 fixed epochs, dropout-0).** Paired same-seed
    Δ(full−model) total FP-MAE = **−0.001 ± 0.041 (−0.02% of the 4.13 base) = 0.02× the
    seed-to-seed spread** (model-only std 0.043 = 1.0% of base, full-step std 0.020 = 0.5%) —
    full-step and model-only are statistically indistinguishable, no systematic regression.
    Per-head signed mean drift is ≤0.25% on the yardage/reception heads; only `rushing_tds`
    shows +0.82% mean (~0.003 absolute, sign-varying, doesn't reach the aggregate). **Metric
    caveat (same trap as the Lever C parity gate):** a naive worst-target *relative* delta
    reads 50–136% — entirely `fumbles_lost` (model MAE ~0.04, near-zero clamped head); exclude
    sub-0.05-MAE heads or judge in FP space. A first 3-seed run looked 1–2% sign-varying; that
    was seed noise that averaged to ~0 at 8 seeds (the AGENTS ≥5–8-seed rule).

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
trials per GPU-hour** on identical hardware. The next per-step lever is the
whole-iteration capture (the optimizer tail) — built as A3 below; this closes the planned
throughput rounds, and the remaining tune-loop bottleneck is per-trial capture build
(~1–2 s) + Optuna/study churn, not worth chasing.

> **Correction (2026-06-22):** earlier notes called whole-iteration (optimizer-tail)
> capture "rejected / not worth chasing." That judgment was made under the FP16-autocast
> default (GradScaler's data-dependent inf/NaN skip branch makes the optimizer step
> non-capturable) and framed purely as tune throughput. The FP32+TF32 default (#1311)
> removed GradScaler, which UNBLOCKS the capture, and the optimizer tail is a real
> production step-time fraction (24.2% QB / 12.7% WR / 8.6% RB). A3 below builds it and
> ships it autodetect-ON, strictly inert.

**Oversubscription re-probed post-D2 (2026-06-11, n_jobs=5 on 4 vCPUs, fresh study): still
~1 useful trial per vCPU.** 69 trials (22C/47P) in 218.4 s = 19.0 trials/min vs 18.7 at
n_jobs=4 (+1.6%, trial-mix noise) while completes/min DROPPED 7.18 → 6.04 (−16%) and
bs512-class epochs slowed 0.14–0.16 → 0.24–0.27 s — a super-proportional ~1.7× per-trial
penalty at 1.25× oversubscription (5 workers on 2 physical cores). The graphed loop is
still host-paced; `--n-jobs auto` = vCPU count stays correct, and the concurrency
question is closed on measurement (the pre-D2 8-on-4 result was not stale after all).

### Lever A3 — OPTIMIZER-TAIL capture (`FF_CUDA_GRAPH_OPT`) — BUILT & GATE-PASSED 2026-06-22 (local 5080)

A2 graphs {gather + forward + backward + combined-loss} but leaves the per-step EAGER TAIL
eager: `zero_grad → clip_grad_norm_ → AdamW.step → loss accumulate`. Measured tail fraction
of attn-NN step time: **24.2% QB / 12.7% WR / 8.6% RB**. A3 captures that tail too.

**Why now (the FP32 unblock).** `make_graphed_callables` can't include the optimizer, so A3
is a MANUAL `torch.cuda.CUDAGraph` over the whole iteration (the `_GraphedValPass` style:
static I/O buffers, side-stream warmup, `torch.cuda.graph(...)`). Under the old FP16-autocast
default, GradScaler's data-dependent inf/NaN skip branch (`scaler.step` may skip
`optimizer.step()`) made the step non-capturable. The FP32+TF32 default (#1311) removed
GradScaler, so the step is now branch-free and capturable.

**Architecture (`_GraphedFullStep` in training.py).** A3 gates ON TOP of A2
(`cuda_graph_opt_enabled()`: `FF_CUDA_GRAPH_OPT` truthy on top of `cuda_graph_full_enabled()`
— A3 ⊆ A2 ⊆ base sm_80+ gate). Two non-obvious design points the implementation had to solve
(both would have made it NOT inert; both caught by the local Δ=0 gate before any Batch run):

1. **You cannot re-capture A2's `make_graphed_callables` output inside an outer graph** —
   calling its replay during an active capture raises *"Cannot prepare for replay during
   capturing stage."* So A3 captures the step EAGERLY: a fresh, un-graphed `_GraphedTrainStep`
   (the identical gather+fwd+loss ops) + `loss.backward()` + clip + `optimizer.step()`. A2's
   `make_graphed_callables` build still runs first — A3 reuses it ONLY to reproduce A2-only's
   BatchNorm warmup perturbation (so step-0 BN matches), then replays its own graph instead of
   A2's `_graphed_step`.
2. **`capturable=True` LR must be a DEVICE TENSOR, not a float** (`_run_nn_training` sets
   `AdamW(capturable=True)` only when A3-eligible — fused, per-EPOCH scheduler, gate on). Fused
   capturable AdamW happily reads a Python-float `lr`, and the graph then bakes that VALUE as a
   constant → replays freeze on the build-time LR and the cosine schedule silently no-ops, a
   ~1% trajectory fork vs A2-only (`build()` forces `param_group['lr']` to a device tensor;
   `refresh_lr_from_scheduler()` writes the post-`scheduler.step()` float into that tensor IN
   PLACE each epoch). Warmup also runs REAL optimizer steps, so `build()` snapshots params + BN
   and resets the Adam moments to step-0 after capture. K's nested trainer no-ops capture.

**STRICTLY INERT (no rebaseline) — local end-to-end Δ=0 gate, 2026-06-22 (5080 sm_120).**
Per position, two same-seed runs (`FF_NN_FIXED_EPOCHS=30 FF_FORCE_DROPOUT_ZERO=1`,
`FF_CUDA_GRAPH_OPT={1,0}`), attention-NN test FP-MAE compared; deterministic Ridge FP-MAE
identical throughout (data-identity tell). The owner skipped Batch validation, so this local
gate is the proof. **Engagement is a true positive, not a Δ=0-because-it-never-fired false
pass** (independently verified): with `FF_CUDA_GRAPH_OPT=1` the optimizer is built
`capturable=True`, `_graphed_opt` is set, and the trainer logs `[cuda-graph] optimizer-tail
capture engaged (Lever A3)`; with `FF_CUDA_GRAPH_OPT=0` the optimizer is `capturable=False`,
`_graphed_opt` is None, and A2's full-step graph still engages (the A2-only baseline).

| position | A3-on attn FP-MAE | A2-only attn FP-MAE | \|Δ\| |
|---|---|---|---|
| RB (seed 42) | 4.161292110313279 | 4.161292110313279 | **0.000000** |
| QB (seed 42) | 5.958248304188605 | 5.958248304188605 | **0.000000** |
| WR (seed 42) | 4.053313073959706 | 4.053313073959706 | **0.000000** |
| DST (seed 42) | 6.050014096147874 | 6.050014096147874 | **0.000000** |
| RB seed 1337 | 4.164618728178873 | 4.164618728178873 | **0.000000** |
| RB seed 2024 | 4.136827474537323 | 4.136827474537323 | **0.000000** |

RB same-seed determinism (A3-on run twice): `4.161292110313279` both times, |Δ|=0. Every
Ridge FP-MAE was bit-identical within each pair (data-identity tell). All twelve cells: **Δ=0
exactly.**

The verified mechanisms: `AdamW(fused=True)` vs `fused=True, capturable=True` over identical
grads → max|Δ|=0 (capturable is math-inert); a CUDA-graph replay of the capturable step vs
eager → max|Δ|=0; a tensor-LR updated in place reproduces an eager LR schedule through the
graph bit-for-bit. A naive first cut (float LR + identity-discovery) forked ~1% from epoch 1;
the 1-epoch slice stayed Δ=0 (per-step capture correct), which localized it to the
between-epoch LR refresh — fixed by the device-tensor LR above.

**Speedup (sanity, not a gate) — honest ~2%, NOT 30%.** The per-step eager-tail fractions
measured on origin/main are RB 8.6% / WR 12.7% / QB 24.2% of the attn step — that is the
size of the tail A3 captures. The *net production* gain is modest: A3 step ≈1.159 ms vs the
REAL A2-only baseline (`capturable=False`) ≈1.183 ms on RB, **~2%**. A naive 30%-class
reading is INFLATED — it timed A3 against an A2 tail running on a `capturable=True`
optimizer (capturable mode adds eager per-step overhead that the graph then reclaims), not
against the true `capturable=False` baseline. The 5080's low launch overhead and the
one-time capture build also swamp the small RB-tail saving at this scale (the phase timer
reads ~1.8 s either way; whole-pipeline wall ~20.6 vs ~21.5 s is dominated by non-attn
work). The win scales with the tail fraction (largest on QB at 24.2%) and the L4's higher
launch overhead vs the 5080, and helps LOCAL ITERATION only — the default tuner runs
stacked-graphs-OFF and production is orchestration-bound. The value of A3 is its inertness
(strictly bit-identical), so it ships autodetect-ON without a speedup gate.


---

## Lever C — vmap seed-ensembles — REJECTED then REVERSED as opt-in for comparative pipelines (both 2026-06-11, owner calls)

The oversubscription probes (8-on-4 eager, 5-on-4 post-D2) proved ~1 trial per vCPU for
HETEROGENEOUS jobs — but the multi-seed A/B workload (≥8 seeds × same config) is
homogeneous, which is the one case where true CPU multiplexing works:
[src/tuning/ab_ensemble_seeds.py](../src/tuning/ab_ensemble_seeds.py) stacks N seeds'
models (`torch.func.stack_module_state` + `functional_call` + `vmap`) so ONE host thread
trains all N with ~1× launch cost. A/B-ONLY regime, enforced in-module: `FF_NN_NORM=layer`
(kills the backbone BN — the model's only BN site — leaving a buffer-free vmap-clean
model; LN-vs-BN measured noise at 8 seeds), FP32 (no GradScaler → no cross-member
inf-coupling, provable parity), `FF_NN_FIXED_EPOCHS`, shared batch order (CRN variance
reduction), dropout live via `randomness='different'`. Construction is captured from the
REAL pipeline per seed (streams-prototype monkeypatch trick); the loss is the
`compute_combined_capturable` the CUDA-graph paths already use; the one new math is the
per-member grad clip (a global clip would couple members). CPU tests pin per-member-clip ≡
`clip_grad_norm_`, step-0 forward parity (1e-6), single-batch post-clip grad parity
(1e-5), and trained-end parity (1e-2 — fp32 vmap kernel-order compounding, same
amplification physics as the graph drift).

**Adam-step parity, fixed 2026-06-11.** The naive post-Adam-step param comparison was
ill-conditioned, not wrong: at step 1 Adam's update is `lr·g/(|g|+eps)` (v̂=g²), i.e.
sign-like near g=0, so ulp-level vmapped-vs-eager kernel-order grad noise flips signs
into full-lr param diffs. Restored tight coverage via a well-conditioned decomposition:
(1) `test_adam_step_parity_with_injected_grads` — identical injected grads ⇒ ONE stacked
foreach-AdamW must equal N independent AdamWs **bitwise** (rtol=0/atol=0 passes; AdamW is
elementwise so stacking is purely layout) across steps + lr changes, pinning per-member
state independence; (2) the grad-parity test now also steps both arms' optimizers and
compares params on the well-conditioned mask `|g| > 1e-4` (atol=1e-7). Together: grads
tight + optimizer exact ⇒ the real-path step is covered everywhere except the inherent
sign discontinuity, which tier (c)'s spread-scaled trained-end gate bounds.

**Batch route (no 5080 needed).** The training image's ENTRYPOINT is pinned to
`src.batch.train` and editing it fires a 6-position retrain, so the harness rides the
tune dispatch: submit a `--mode tune` job with `FF_TUNE_ENSEMBLE_AB=1` in the container
env — `tune_nn.main` dispatches to `ab_ensemble_seeds.run_batch_entry` right after
argparse (env knobs `FF_ENSEMBLE_SEEDS`/`FF_ENSEMBLE_FIXED_EPOCHS`/`FF_ENSEMBLE_PARITY`;
parity forces `FF_FORCE_DROPOUT_ZERO=1`, report → stdout +
`s3://$S3_BUCKET/ensemble_ab/{POS}/report.json`, parity drift exits non-zero).

**First Batch gate run (L4, 2026-06-11): speedup 4.4× ✅; the original parity gate was
ill-posed and got redefined.** Capture 33 s (memo: `prepare_data 0.0s` on seeds 2–8),
stacked 23.1 s vs sequential 101.4 s for 8×30 epochs. The per-seed trajectory gate
(`worst diff < 1% of pred spread`) reported 1.86e5 — diagnosed on CPU with the real RB
config: (1) the scalar's spread-clamp explodes on near-constant clamped heads
(`fumbles_lost` pinned at 0 → seq std ≈ 0 → 1e-6 clamp), and (2) per-seed trajectory
bit-parity is inherently ill-posed under fp32+Adam — vmapped-vs-eager sub-ULP kernel
diffs hit Adam's sign-like step-1 (`lr·g/(|g|+eps)`, v̂=g²) on near-zero-grad params and
deterministically FORK the trajectory. The fork follows the SEED, not the member slot
(verified by seed-order swap: identical tables, indices flipped; seed 42 stays
bit-faithful to ~2e-4 over 3 epochs, seeds 43/44 fork to ~1e-2 of spread) — both arms
are valid fp32 evaluations of the same algorithm, the ADR-0017 accepted-divergence
physics at trajectory scale. **Redefined gate (decision-level):** per-member fork in
FANTASY-POINT space (`RMS(fp_stacked_i − fp_seq_i)`) must stay under 0.2× the
seed-to-seed FP spread (`mean over pairs RMS(fp_seq_i − fp_seq_j)`) — the quantity the
A/B actually averages over. Measured CPU (RB, 3 seeds × 3 epochs): forks
[1e-5, 0.015, 0.024] vs seed noise 0.62 → ratio 0.039, ~5× headroom; a systematic bug
(coupled clip/optimizer, wrong lr) lands at O(1) and still fails. The raw per-key
worst-over-spread table ships in the report as diagnostics only.
**Second Batch gate run (L4, 2026-06-11) and VERDICT: REJECTED.** Speedup confirmed
4.58× (capture 35.4 s / stacked 25.1 s / sequential 114.9 s), but at 8 seeds × 30 GPU
epochs **every member forks**: per-member FP fork RMS [0.43, 0.77, 0.68, 0.77, 0.43,
0.40, 0.39, 0.33] vs seed-to-seed FP spread 1.21 → worst ratio 0.64 (gate 0.2). Uniform
magnitudes across all 8 slots again rule out a positional bug — this is trajectory
chaos growing with step count (900 steps vs the CPU diag's 90) and CUDA kernel
diversity. Estimated effect on the decision statistic was small (per-member MAE shift
≈ 0.8·f²/(2σₑ) ≈ 0.02 FP, common-mode across variants), and a decision-level
MAE-equivalence gate was designed — **but the owner rejected the redefinition: the
instrument requirement is SAME-SEED DETERMINISM, including across concurrent execution
modes ("runs of the same seed should stay deterministic"), which stacked-vs-eager
training structurally cannot satisfy** (vmapped batched kernels ≠ eager kernels at the
sub-ULP level; fp32+Adam amplifies into macroscopic forks). Verdict: keep the
**1 trial : 1 vCPU** doctrine for everything (tune already runs `--n-jobs auto` → 1:1;
multi-seed A/Bs stay process-parallel eager via ab_harness, which IS per-seed
deterministic). E2 (`ab_harness --stacked-seeds`) is NOT built.
Reports from both gate runs: `s3://ff-predictor-training/ensemble_ab/RB/report.json`.
Archive entry: [todo/fixed-archive.md](fixed-archive.md) "[TESTED, REJECTED] vmap
stacked seed-ensemble training (Lever C)".

**REVERSAL (same day, owner call): sanctioned as OPT-IN for the comparative pipelines.**
After the throughput framing (4.4–4.6× per host thread; ~32 vs 4 concurrent seeds on a
4-vCPU host), the owner directed wiring stacked mode into the A/B and NN-tuning Batch
pipelines. The determinism finding stands — a stacked run is NOT seed-comparable to an
eager run (and depends on stack width N) — so the integration ships with hard scoping:
**(1)** `ab_harness --stacked-seeds` (E2): the unit becomes a (position, variant) GROUP —
Phase A one real eager run at seeds[0] with attention OFF (non-attention models +
reference test_df + the Ridge sentinel), Phase B captures all seeds and trains the
attention ensemble under `ensemble_env` (restored on exit), per-member test preds
overwrite `pred_attn_nn_total` per seed; aggregation/sentinel contracts unchanged;
QB/RB/WR/TE only (K/DST fall back to eager cells); compare stacked runs ONLY against
stacked runs. **(2)** NN tuning: `FF_TUNE_STACKED_SEEDS=N` (env rides the fixed
ENTRYPOINT) makes each Optuna trial evaluate its config as a stacked N-seed ensemble —
per-epoch stacked val pass reports the across-member MEAN val loss (seed-averaged
objective; the "single-seed NN MAE is noise" fix) — in a `_ens{N}x{E}`-suffixed study
namespace (fixed-epochs ensemble regime ≠ the eager early-stop objective). **(3)**
Production training stays eager; nothing changes there.

**VALIDATED on Batch + peak compute measured (2026-06-11, image cbad0860, L4).**
Shipped in #1150 (E2 + shared surface) and #1151 (stacked tune objective, ab_batch/launch_ab
pass-through, [src/tuning/resource_probe.py](../src/tuning/resource_probe.py)). Three tests:
**(a) local E2 smoke** (RB, 2 seeds × 2 epochs, CPU, `ab_example --only nn_dropout=0`):
contracts hold — attention rows show per-seed variation, non-attention rows std=0, Ridge
sentinel verified `expect=identical` with Δ=+0.00000, env restored between groups. CPU is
correctness-only — stacking has no CPU throughput win (compute-bound; no idle engine to
hide host cost in); CPU multi-seed A/Bs stay process-parallel eager.
**(b) Batch stacked tune smoke** (job aea5aeec: RB, `--stacked-seeds 4`, target 8 completes,
n_jobs=4): SUCCEEDED in 513.7 s — 8 complete + 27 Hyperband-pruned trials (per-epoch
seed-averaged val reports drive the pruner correctly), namespace
`scheduler_v2_mps_ens4x30`, provenance records stacked_seeds/epochs + graphs-off.
Resources: `cpu_util_cores 3.82/4` (≈0.95 core/lane at 4 concurrent lanes — the 1-core/lane
model holds under contention), `cgroup_peak 6.91 GiB / 14.65` (≈1.7 GiB/lane all-in; worker
peak RSS 2.09 GiB), parent-process GPU counters read 0 (KNOWN probe limitation: workers own
the CUDA contexts; per-lane VRAM comes from (c)).
**(c) instrumented ensemble run** (job 1e35bd8d: 8 seeds × 30 epochs, parity off):
SUCCEEDED — capture 33.9 s + stacked train 23.2 s; **one 8-seed lane = `cpu_util_cores`
1.0 flat, cgroup peak 1.53 GiB, GPU 0.64 alloc / 0.83 GiB reserved of 24**.
**Packing math (g6.xlarge):** cores bind first (4 lanes × 1.0 core), RAM second
(~1.7-2.5 GiB/lane → ~6-8 lanes if oversubscription ever paid, which the n_jobs=5 probe says
it doesn't), VRAM never at doctrine widths.

**Width-scaling curve MEASURED (2026-06-11, jobs b12a4b6a/330a698a, RB × 30 epochs, L4,
one lane):**

| N (seeds) | stacked train | s/seed | CPU cores | cgroup RAM | GPU reserved |
|---|---|---|---|---|---|
| 1 (eager, derived) | ~14.4 s | 14.4 | 1.0 | ~1.5 GiB | ~0.5 GiB |
| 8 | 23.2 s | 2.90 | 1.0 | 1.53 GiB | 0.83 GiB |
| 24 | 24.3 s | **1.01** | 1.0 | 1.53 GiB | 2.53 GiB |
| 50 | 57.5 s | 1.15 | 1.0 | 1.54 GiB | 5.16 GiB |

8 → 24 is FREE (+5% wall for 3× seeds — the launch-bound lunch extends to 24); the knee is
in (24, 50]: N=50 costs 2.37× N=24's wall for 2.08× the seeds (the lane goes GPU/
bandwidth-bound; 2×24 sequential beats 1×50 — 48.6 s/48 seeds vs 57.5 s/50). **Per-seed
optimum ≈ N=24 (~1.0 s/seed, 14× eager); CPU and RAM are width-INVARIANT (1.0 core,
1.53 GiB at every N); GPU memory is the only width cost (~0.10 GiB/seed reserved).**
Host packing at the optimum: 4 lanes × 24 = **96 concurrent seeds per g6.xlarge**
(~10 GiB VRAM of 24, ~7 GiB RAM, 4 cores); fleet ceiling at the 64-vCPU Spot quota ≈
1,500 concurrent seeds. The *useful* width for decisions stays 8-16 (σ/√N returns; A/B
doctrine) — the headroom to 24 is free sharpening when wanted. Trials still cannot stack
(per-trial architectures differ; vmap needs one module structure) — width multiplies
seeds-per-config, lanes/hosts multiply configs. Eager-vs-stacked tuning rate: 35 trials (8C/27P) in 8.6 min at 4 seeds/trial
(eager-regime vmapped, no graphs) vs 18.7 single-seed trials/min in the graphfull eager
namespace — stacked trials cost ~2× an eager-regime trial for 4× the seeds.
**DEFAULT-ON at N=24 for NN tuning + the ab_harness A/B path (owner call, 2026-06-15).**
Given the crossover (eager-FP16+full-graph wins ≤8 seeds; stacking wins ≥~9, measured 2.68
vs 1.0 s/seed at N=24), stacking is now the GPU default at the per-seed optimum.

**Local 5080 (sm_120) head-to-head — stacking wins EARLIER and HARDER than on the L4**
(measured 2026-06-22, RB, 30 fixed epochs, `ab_ensemble_seeds --compare`, single lane, 1 core):
the **crossover sits at N≈6** — a sub-8 width sweep pins it. Stacked total wall is **flat ~11 s at
every width** (fixed launch+capture, ~0 marginal/seed), so stacked s/seed ≈ 11/N against an
eager-FP16+full-graph **~1.75 s/seed launch-bound floor** (the 5080's FP16 tensor cores don't help a
launch-bound model — `full_step_graph_active=True, amp=float16` confirmed in every report). Stacked
per-seed speedup by width:

| N | eager s/seed | stacked s/seed | speedup |
|---|---|---|---|
| 2 | 1.90 | 5.51 | 0.34× |
| 4 | 1.78 | 2.78 | 0.64× |
| 6 | 1.75 | 1.80 | 0.97× (≈ tie) |
| 8 | 1.75 | 1.35 | 1.29× |
| 16 | ~1.75 | 0.68 | 2.58× |
| 24 | ~1.65 | 0.47 | 3.51× |

— eager wins ≤6 seeds, stacking wins ≥7 (vs the L4's ~9-seed crossover / 2.68× at N=24). The
intuition that a faster GPU would favour the FP16-graph arm is **backwards** — a faster GPU is *more*
launch-bound, so it favours the launch-amortizing stacked arm even more (L4→5080 speeds eager ~1.6×
but stacked@24 ~2.1×). So the N=24 default is even better-placed on the 5080 than on the L4 it was
decided on; no per-arch retune needed (and the *useful* A/B width stays 8–16 by σ/√N returns, well
above the N≈6 crossover — the default is never in the eager-wins zone).

**Scope:
tune + every `ab_*.py` spec (and any `ab_harness`-based ablation) — NOT the legacy
`ablate_*` scripts on `ablation_runner`**, which stay eager: only ~3 of 8 are
stacking-compatible and several break under the LN/FP32/NN-only regime (`backbone_norm`
forces LN; `ridge_pca` is a Ridge ablation; `min_games`/`injury_features` are data
ablations). Owner chose "leave ablate_* eager — ab_harness already covers it."
**Follow-up (2026-06-15, re-confirmed):** the two stackable NN ablations were then ported to
`ab_harness` specs — `attn_arch` → `src.tuning.ab_attn_arch` (drops the `entropy` arm, a vmap
side-channel) and `scheduler_type` → `src.tuning.ab_scheduler_type` (drops `plateau`, which
`train_stacked` rejects) — so they now inherit the N=24 default. `rb_gate` (its per-target
head-MAE + gate-AUC decision rule isn't surfaced by the stacked harness, which only exposes
`pred_attn_nn_total`; D/E are the reverted `hurdle_poisson`) and `batch_lr` (a throughput
ablation the FP32/vmap/fixed-epochs regime structurally can't measure) were evaluated and
**deliberately left eager** — don't "finish" the job by porting those two.
`resolve_default_stacked_seeds()` ([src/tuning/ab_ensemble_seeds.py](../src/tuning/ab_ensemble_seeds.py)):
`cuda_enabled()` → 24, else 0 (eager) — CPU/MPS must fall back because the FP32 stack is
*slower* there. Wiring: tune `--stacked-seeds` default = the resolver (was 0);
`launch_tune` default = 24, resolved per-position (K/DST → eager, no `_ens` suffix, so the
predicted namespace matches the container fallback); `ab_harness --stacked-seeds` is now a
GPU-gated tri-state (`--no-stacked-seeds` forces eager) and, when on with no explicit seeds,
defaults the grid to the wide 24-seed list while the eager path keeps the lean 3-seed
default (no CPU/CI regression). Stacked artifacts land in `_ens{N}x{E}` namespaces that
**coexist** with eager-regime history (owner chose coexist over replace). Production
training unchanged. The user accepted 24-seed tune objectives (σ/√24, ~9× fewer configs
per budget) as an explicit precision-over-breadth call. (The earlier note that the
"useful width stays 8-16" is superseded for the *default*: 24 is the throughput optimum
and the chosen objective width; 8-16 remains the statistical sweet spot if a leaner run is
wanted via `--stacked-seeds N`.)

**Latent bug — RESOLVED:** `launch_tune --n-jobs auto` (the #1119 default) used to die in
the container because `src/batch/train.py`'s `--n-jobs` is `type=int` and prior submissions
happened to pass a number. Fixed via the env channel (not argv): `launch_tune` now routes
`n_jobs` through `FF_TUNE_N_JOBS`, which `train.py` reads as the `--n-jobs` default, so the
`auto` sentinel never hits `type=int`. No train.py edit was needed (it would have fired a
6-position retrain); an explicit `--n-jobs N` still overrides.

## Lever B — single process + per-position CUDA streams (the MPS substitute) — MEASURED-NEGATIVE (2026-06-22, 5080 sm_120); CLOSED

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

**MEASURED — NEGATIVE (2026-06-22, RTX 5080 / sm_120 / WSL2; PR #1309).** The prototype
was first refreshed to **full-step-graph parity** (its per-step body, capture, and batch
iterators now mirror production's autodetect-ON full-step CUDA graph: `_maybe_graph_full_step`
→ `_graphed_step`, `index_batches()` iterators, `_compute_loss_components`). Then the gate was
run on QB/RB/WR/TE, 3 seeds (42/1337/2024), `--fixed-epochs 30`, with **all four positions
confirmed captured `full`**:

- **Graphed full-step A/B: seq→streams = 1.033 ± 0.005×** (fresh re-run, the headline; an
  earlier graphed run gave 1.010 ± 0.008× — both ≈1.0× / no meaningful overlap, run-to-run
  variation on the shared GPU). i.e. **zero overlap**. The eager sanity arm
  (`--no-graph --deterministic`) is also 1.00×. The single-thread round-robin streams arm's
  wall-clock ≈ the sequential sum of solos.
- **Why no overlap:** the lever's two regimes both fail to fill the GPU's idle gaps. (1) *Eager:*
  the GIL serialises the host-side kernel **launches** — the exact bottleneck the diagnosis
  identified — so one thread round-robining steps onto N streams still issues launches one at a
  time. (2) *Graphed:* full-step capture has already collapsed the launch storm into one cheap
  replay per step, so there is little host-launch idle left to overlap, and the FP16+GradScaler
  per-step inf/NaN-check sync plus each replay's own GPU occupancy leave no device-side slack for
  a neighbouring stream to slot into. This **matches the section's own "honest expectation"**
  above (graphs already collapsed the launches; the single thread only overlaps device-side work;
  `-j6` already fills some gaps).
- **Limitations (don't over-read):** measured on the **4 skill positions only** — K and DST are
  unbuildable from this environment's `data/splits` (their targets `fg_yards_made` /
  `points_allowed` are computed from raw kicker/defense PBP downstream of the shared splits, a
  pre-existing prototype/data limitation, not a streams issue). They share the same launch-bound
  attention arch, so the overlap physics is representative. Separately, the prototype's
  `--no-graph --deterministic` "**parity OK**" precondition does **not** currently hold even for
  the *unmodified* (pre-edit) code — both arms DRIFT identically (proven by re-running pre-edit
  `HEAD~1`); the cause is the round-robin's cuDNN-autotune-under-co-residency / shared-RNG draw,
  not the copied step body, and it does **not** affect the batch-order-independent **timing**.
- **Conclusion — Lever B CLOSED.** No productionization; the `parallel_train -j6` subprocess
  model stays (it keeps process isolation + the core pool), and **Lever A (CUDA graphs) remains
  the local-iteration win**. The prototype is now refreshed to full-step-graph parity (PR
  #1309), so any future re-measure on a different GPU / regime (e.g. an L4 with real MPS, or a
  larger model where device-side overlap could exist) starts from a faithful baseline.

### Lever B′ — within-position overlap (base NN ∥ attention NN) — IN-PROCESS variant MEASURED-REJECTED (2026-06-22, 5080 sm_120); process+MPS variant still separate

**MEASURED-REJECTED for the in-process implementation (2026-06-22, RB / 5080 sm_120, draft PR #1332, FF_NN_OVERLAP).** The doc's suggested productionization — "overlap the two trainers in `_gpu_branch` behind an `FF_*` flag" — was built as in-process threads on two CUDA streams and A/B'd on the production path. Verdict: **reject, three independent reasons.**

1. **Mutually exclusive with the shipped CUDA-graph path.** On the production default (graph capture autodetect-ON, sm_80+) the overlap run **crashes**: `CUDA error: operation not permitted when stream is capturing` (`cudaErrorStreamCaptureUnsupported` → `StreamCaptureInvalidated`). One stream capturing a graph forbids any other stream from launching work, so two trainers capturing concurrently in the **same process/context** invalidate each other. This corrects the "**composes with CUDA graphs**" claim below — it does **not**, for the in-process variant. (The *process*-based prototype above sidesteps this: separate processes = separate CUDA contexts = independent capture.)
2. **Strictly dominated even where it runs.** Graphs-OFF (`FF_CUDA_GRAPH=0`) the overlap is real and **not** GIL-strangled — RB `train_models_total` 79.6s → **50.4s (1.58×)** on the GPU branch (so, unlike Lever B's 1.03×, the mechanism works). But graphs and B′ reclaim the **same** host-dispatch idle, and graphs do it far better: graphs-ON **sequential** = **7.5s** vs graphs-OFF **overlap** = 50.4s (~6.7×). You would never trade graphs for overlap.
3. **Off the critical path + non-deterministic.** NN training is ~7.5s of a ~23s RB run (`prepare_data` 10.6s dominates), and production wall-clock is orchestration-bound (warm-AMI / Batch lifecycle), so the GPU branch barely moves it. Separately the in-process threads share the **global RNG**, so the overlap path is **not** byte-identical (the process prototype's per-child RNG re-seed avoids this, but the in-process flag can't).

`FF_NN_OVERLAP` is unshipped (default-off only); don't re-propose the in-process variant. The **process-based** prototype below and its **AWS-MPS** cousin are a *different* track (separate contexts dodge reason 1; per-child RNG dodges reason 3) — still untested here and gated on MPS for real GPU overlap (unavailable on WSL2/Windows), so they live with the warm-MPS-packed-training proposal, not the local launch-bound effort.

---

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
2. ~~**Lever B only if A is insufficient**~~ — **CLOSED, MEASURED-NEGATIVE** (2026-06-22, 5080): the single-process round-robin streams arm gave **1.03×** (1.033 ± 0.005×, no cross-stream overlap) even with full-step graphs engaged — the GIL serialises launches and graphs already collapsed the launch storm, so there's no idle to fill. The `-j6` subprocess model + core pool stay; A's per-position 1.84× remains the local-iteration win. (See the Lever B section for the measurement.)

Both are **launch-overhead** levers (the measured bottleneck), not occupancy/FLOP levers (the 5080 has those to spare).
