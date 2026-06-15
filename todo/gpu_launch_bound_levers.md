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

**Oversubscription re-probed post-D2 (2026-06-11, n_jobs=5 on 4 vCPUs, fresh study): still
~1 useful trial per vCPU.** 69 trials (22C/47P) in 218.4 s = 19.0 trials/min vs 18.7 at
n_jobs=4 (+1.6%, trial-mix noise) while completes/min DROPPED 7.18 → 6.04 (−16%) and
bs512-class epochs slowed 0.14–0.16 → 0.24–0.27 s — a super-proportional ~1.7× per-trial
penalty at 1.25× oversubscription (5 workers on 2 physical cores). The graphed loop is
still host-paced; `--n-jobs auto` = vCPU count stays correct, and the concurrency
question is closed on measurement (the pre-D2 8-on-4 result was not stale after all).

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
vs 1.0 s/seed at N=24), stacking is now the GPU default at the per-seed optimum. **Scope:
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

**Latent bug surfaced (chip filed):** `launch_tune --n-jobs auto` (the #1119 default) dies in
the container — `src/batch/train.py`'s `--n-jobs` is `type=int` and every prior submission
happened to pass a number. Fix = env channel (`FF_TUNE_N_JOBS`) in src/tuning, NOT a
train.py edit (6-position retrain); explicit `--n-jobs 4` is the interim.

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
