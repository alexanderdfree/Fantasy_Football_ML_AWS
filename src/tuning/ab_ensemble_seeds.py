"""Stacked seed-ensemble training for multi-seed attention-NN A/Bs.

TESTED, REJECTED 2026-06-11 — SHELF PROTOTYPE, wired into nothing by default.
Two L4 Batch gate runs measured 4.4-4.6x speedup and the machinery is proven
correct (bitwise stacked-AdamW parity, fork follows seed not slot), but
fp32+Adam trajectory chaos forks same-seed stacked-vs-eager runs (all 8
members at 0.3-0.8 FP RMS by 30 GPU epochs), which violates the project
requirement that same-seed runs stay deterministic across execution modes.
Multi-seed A/Bs stay 1 trial : 1 vCPU, process-parallel eager. Verdict +
measurements: todo/gpu_launch_bound_levers.md (Lever C) and the fixed-archive
entry "[TESTED, REJECTED] vmap stacked seed-ensemble training". Do not wire
into ab_harness without revisiting that decision.

One host thread trains N seeds of the SAME config simultaneously via
``torch.func`` (``stack_module_state`` + ``functional_call`` + ``vmap``), so
the launch-bound per-step host cost is paid ~once instead of N times. This is
the one true "multiplex the CPU across GPU jobs" mechanism — it requires the
N trainings to share an architecture, which the multi-seed A/B doctrine
(>=8 seeds per variant, same config) satisfies and Optuna tuning (per-trial
architectures) does not.

A/B-HARNESS-ONLY REGIME — never production. Enforced by ``apply_ensemble_env``:

* ``FF_NN_NORM=layer`` — the backbone's BatchNorm1d (the model's ONLY BN site)
  swaps to LayerNorm, leaving a buffer-free model that vmaps cleanly. The
  8-seed LN-vs-BN A/B measured +0.007±0.034 (noise), so LN-regime *deltas*
  transfer to the BN production regime.
* FP32 (``FF_AMP_DTYPE=fp32``) — no GradScaler: a shared FP16 scaler would
  couple members (an inf in any member skips all), and FP32 makes
  stacked-vs-sequential parity provable. The ~N× launch amortization dwarfs
  FP32's ~2× compute on this ~69K-param model.
* ``FF_NN_FIXED_EPOCHS`` — no early stopping (per-member best-epoch selection
  doesn't vectorize; fixed epochs is the established A/B-isolation regime).
* Shared batch order across members (one permutation per epoch) — a
  common-random-numbers design that *reduces* the variance of cross-arm
  deltas; deliberate, not a limitation.
* Dropout stays live via ``vmap(randomness="different")`` (independent draws
  per member); parity checks additionally require ``FF_FORCE_DROPOUT_ZERO=1``.
* ``FF_CUDA_GRAPH=0`` / ``FF_CUDA_GRAPH_FULL=0`` — belt-and-suspenders; the
  hand-rolled loop below never calls ``trainer.train()`` so capture never
  engages anyway.

Scope v1: flat-history positions (QB/RB/WR/TE). K's nested trainer and DST's
own-splits ``run()`` are excluded by guard.

Usage (GPU box — the 5080 is the primary target):

    FF_FORCE_DROPOUT_ZERO=1 python -m src.tuning.ab_ensemble_seeds \\
        --position RB --seeds 8 --fixed-epochs 30 --parity-check
    python -m src.tuning.ab_ensemble_seeds --position RB --seeds 8 \\
        --fixed-epochs 30          # speedup measurement, dropout live

AWS Batch (one-off GPU run without a 5080): the training image's ENTRYPOINT
is fixed to ``src.batch.train``, so submit a ``--mode tune`` job with
``FF_TUNE_ENSEMBLE_AB=1`` in the container environment —
``src.tuning.tune_nn.main`` dispatches into :func:`run_batch_entry` before
any Optuna work. See ``run_batch_entry`` for the env knobs.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch

ENSEMBLE_POSITIONS = ("QB", "RB", "WR", "TE")
_CLIP_MAX_NORM = 1.0  # mirrors the hardcoded clip_grad_norm_(1.0) in MultiHeadTrainer
_EPOCH_SEED_STRIDE = 9973  # prime; shared-batch-order reseed = base + stride * epoch


def apply_ensemble_env(fixed_epochs: int) -> None:
    """Force the documented ensemble regime (see module docstring)."""
    os.environ["FF_NN_NORM"] = "layer"
    os.environ["FF_AMP_DTYPE"] = "fp32"
    os.environ["FF_CUDA_GRAPH"] = "0"
    os.environ["FF_CUDA_GRAPH_FULL"] = "0"
    os.environ["FF_NN_FIXED_EPOCHS"] = str(int(fixed_epochs))
    if os.environ.get("FF_COMPILE", "0").strip() not in {"", "0", "false", "off"}:
        raise SystemExit(
            "FF_COMPILE must be unset/0 for ensemble mode: a compiled "
            "OptimizedModule cannot be stacked by torch.func."
        )


# ---------------------------------------------------------------------------
# Construction capture — reuse the REAL pipeline construction per seed
# (streams_6pos_prototype precedent: patch train()/predict to capture, not run)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def capture_attention_construction(captures: list, test_capture: dict):
    """Patch ``MultiHeadTrainer.train`` + ``MultiHeadNetWithHistory.predict_numpy``
    so a full position ``run()`` builds everything (model with faithful
    per-seed init, loaders, criterion, optimizer hyperparams) and trains
    NOTHING. ``captures`` receives one dict per run; ``test_capture`` receives
    the scaled test arrays from the first ``predict_numpy`` call.
    """
    from src.shared import neural_net
    from src.shared.training import MultiHeadTrainer

    orig_train = MultiHeadTrainer.train
    orig_predict = neural_net.MultiHeadNetWithHistory.predict_numpy

    def _train_stub(self, train_loader, val_loader, n_epochs):
        captures.append(
            {
                "trainer": self,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "n_epochs": n_epochs,
            }
        )
        return {}

    def _predict_stub(
        self,
        X_static,
        X_history,
        history_mask,
        device,
        X_opp_history=None,
        opp_history_mask=None,
    ):
        test_capture.setdefault(
            "args", (X_static, X_history, history_mask, X_opp_history, opp_history_mask)
        )
        # Learn the exact output key set (incl. gated-head aux keys) from a
        # 1-row real forward, then zero-fill full length so the surrounding
        # compute_target_metrics completes harmlessly.
        sample = orig_predict(
            self,
            X_static[:1],
            X_history[:1],
            history_mask[:1],
            device,
            None if X_opp_history is None else X_opp_history[:1],
            None if opp_history_mask is None else opp_history_mask[:1],
        )
        n = X_static.shape[0]
        return {k: np.zeros(n, dtype=np.float32) for k in sample}

    MultiHeadTrainer.train = _train_stub
    neural_net.MultiHeadNetWithHistory.predict_numpy = _predict_stub
    try:
        yield
    finally:
        MultiHeadTrainer.train = orig_train
        neural_net.MultiHeadNetWithHistory.predict_numpy = orig_predict


def capture_seeds(position: str, seeds: list[int], base_cfg: dict) -> tuple[list, dict]:
    """Run the position's REAL ``run()`` once per seed with every non-attention
    branch disabled, capturing the fully-built attention trainer per seed.
    """
    from src.shared.registry import get_config, get_runner

    if position not in ENSEMBLE_POSITIONS:
        raise SystemExit(f"ensemble mode supports {ENSEMBLE_POSITIONS}, got {position}")

    runner = get_runner(position)
    captures: list = []
    test_capture: dict = {}
    memo: dict = {}
    with capture_attention_construction(captures, test_capture):
        for s in seeds:
            cfg = copy.deepcopy(base_cfg if base_cfg is not None else get_config(position))
            if cfg.get("attn_history_structure", "flat") == "nested":
                raise SystemExit("nested history (K) is excluded from ensemble mode v1")
            if cfg.get("attn_entropy_coeff", 0.0):
                raise SystemExit(
                    "attn_entropy_coeff != 0: the entropy side-channel write is a "
                    "vmap side effect; disable it for ensemble runs"
                )
            cfg["train_ridge"] = False
            cfg["train_elasticnet"] = False
            cfg["train_lightgbm"] = False
            cfg["train_base_nn"] = False
            cfg["trial_data_memo"] = memo  # reuse prepared frames across seed captures
            runner(seed=s, config=cfg)
    if len(captures) != len(seeds):
        raise RuntimeError(
            f"expected {len(seeds)} captured trainers, got {len(captures)} — "
            "is train_attention_nn enabled for this position?"
        )
    # Data prep is seed-invariant; the loaders must therefore be identical.
    # The resident-tensor identity check needs the CUDA _GPUResidentBatcher
    # (`.features`); the CPU DataLoader path (tests/debug runs) skips it —
    # the loop itself still works there via _batch_to_device.
    if hasattr(captures[0]["train_loader"], "features"):
        f0 = captures[0]["train_loader"].features
        for c in captures[1:]:
            for a, b in zip(f0, c["train_loader"].features, strict=True):
                if not torch.equal(a, b):
                    raise RuntimeError("seed captures disagree on training data — aborting")
    return captures, test_capture


# ---------------------------------------------------------------------------
# Stacking + the hand-rolled per-step body
# ---------------------------------------------------------------------------


def stack_models(models: list, device: torch.device):
    """Stack N same-architecture models into (template, params, buffers)."""
    template = copy.deepcopy(models[0]).to("meta")
    params, buffers = torch.func.stack_module_state([m.to(device) for m in models])
    return template, params, buffers


def clip_per_member_(grads: list[torch.Tensor], max_norm: float = _CLIP_MAX_NORM) -> None:
    """Per-member global-norm clip over stacked grads ``[N, *shape]``.

    A plain ``clip_grad_norm_`` over the stacked params would compute ONE
    norm across all members and couple them; this reproduces
    ``clip_grad_norm_(max_norm)`` independently per member (same
    ``max_norm / (norm + 1e-6)`` coefficient, clamped at 1).
    """
    if not grads:
        return
    n_members = grads[0].shape[0]
    total_sq = torch.zeros(n_members, device=grads[0].device, dtype=torch.float32)
    for g in grads:
        total_sq += g.float().reshape(n_members, -1).pow(2).sum(dim=1)
    coef = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(coef.view(-1, *([1] * (g.dim() - 1))).to(g.dtype))


def _batch_to_device(batch, device):
    *feats, y = batch
    feats = tuple(t.to(device, non_blocking=True) for t in feats)
    y = {k: v.to(device, non_blocking=True) for k, v in y.items()}
    return feats, y


def _optimizer_hyperparams(trainer) -> dict:
    group = trainer.optimizer.param_groups[0]
    return {
        "lr": group["lr"],
        "weight_decay": group["weight_decay"],
        "betas": group["betas"],
        "eps": group["eps"],
    }


def train_stacked(
    captures: list,
    cfg: dict,
    device: torch.device,
    n_epochs: int,
    base_order_seed: int = 0,
) -> tuple[dict, dict, object]:
    """Train all captured seeds simultaneously; returns (params, buffers, template).

    KEEP-IN-SYNC: the per-step body mirrors ``MultiHeadTrainer.train``'s train
    branch (zero_grad → fwd+loss → backward → clip(1.0) → step → per-batch
    scheduler) under the documented FP32/no-scaler ensemble regime. The
    capturable combined loss is the same function the full-step CUDA graph
    uses, so loss math shares one source of truth with production.
    """
    from src.shared.pipeline import _build_scheduler

    trainer0 = captures[0]["trainer"]
    criterion = trainer0.criterion
    train_loader = captures[0]["train_loader"]
    models = [c["trainer"].model for c in captures]
    template, params, buffers = stack_models(models, device)
    template.train()

    def member_loss(p, b, feats, y):
        preds = torch.func.functional_call(template, (p, b), feats)
        return criterion.compute_combined_capturable(preds, y)

    vstep = torch.vmap(member_loss, in_dims=(0, 0, None, None), randomness="different")

    opt = torch.optim.AdamW(list(params.values()), foreach=True, **_optimizer_hyperparams(trainer0))
    scheduler, per_batch = _build_scheduler(opt, cfg, train_loader, scheduler_prefix="attn_")
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        raise SystemExit("plateau scheduler is unsupported in ensemble mode")

    for epoch in range(n_epochs):
        # Shared batch order: one permutation feeds every member (CRN design).
        torch.manual_seed(base_order_seed + _EPOCH_SEED_STRIDE * epoch)
        for batch in train_loader:
            feats, y = _batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            losses = vstep(params, buffers, feats, y)
            losses.sum().backward()
            clip_per_member_([p.grad for p in params.values() if p.grad is not None])
            opt.step()
            if per_batch:
                scheduler.step()
        if not per_batch:
            scheduler.step()
    return params, buffers, template


def train_sequential(
    captures: list,
    cfg: dict,
    device: torch.device,
    n_epochs: int,
    base_order_seed: int = 0,
) -> list:
    """Reference arm: the SAME per-step body, unstacked, one model at a time.

    Serves as both the speedup baseline and the parity oracle (same regime,
    same shared batch order, per-model optimizers/schedulers).
    """
    from src.shared.pipeline import _build_scheduler

    criterion = captures[0]["trainer"].criterion
    train_loader = captures[0]["train_loader"]
    trained = []
    for c in captures:
        model = copy.deepcopy(c["trainer"].model).to(device)
        model.train()
        opt = torch.optim.AdamW(
            model.parameters(), foreach=True, **_optimizer_hyperparams(c["trainer"])
        )
        scheduler, per_batch = _build_scheduler(opt, cfg, train_loader, scheduler_prefix="attn_")
        for epoch in range(n_epochs):
            torch.manual_seed(base_order_seed + _EPOCH_SEED_STRIDE * epoch)
            for batch in train_loader:
                feats, y = _batch_to_device(batch, device)
                opt.zero_grad(set_to_none=True)
                preds = model(*feats)
                loss = criterion.compute_combined_capturable(preds, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), _CLIP_MAX_NORM)
                opt.step()
                if per_batch:
                    scheduler.step()
            if not per_batch:
                scheduler.step()
        trained.append(model)
    return trained


# ---------------------------------------------------------------------------
# Eval: per-member test predictions
# ---------------------------------------------------------------------------


def predict_stacked(template, params, buffers, test_capture: dict, device) -> list[dict]:
    """Vmapped eval forward over the captured (scaled) test arrays.

    Returns one ``{key: np.ndarray}`` dict per member — the same contract as
    ``MultiHeadNetWithHistory.predict_numpy``.
    """
    X_static, X_history, history_mask, X_opp, opp_mask = test_capture["args"]
    template.eval()
    feats = [
        torch.FloatTensor(X_static).to(device),
        torch.FloatTensor(X_history).to(device),
        torch.BoolTensor(history_mask).to(device),
    ]
    if X_opp is not None:
        feats.append(torch.FloatTensor(X_opp).to(device))
        feats.append(torch.BoolTensor(opp_mask).to(device))
    feats = tuple(feats)

    def member_fwd(p, b):
        return torch.func.functional_call(template, (p, b), feats)

    with torch.no_grad():
        stacked = torch.vmap(member_fwd, in_dims=(0, 0))(params, buffers)
    n_members = next(iter(stacked.values())).shape[0]
    out = []
    for i in range(n_members):
        out.append({k: v[i].cpu().numpy() for k, v in stacked.items()})
    template.train()
    return out


def predict_single(model, test_capture: dict, device) -> dict:
    X_static, X_history, history_mask, X_opp, opp_mask = test_capture["args"]
    return model.predict_numpy(X_static, X_history, history_mask, device, X_opp, opp_mask)


# ---------------------------------------------------------------------------
# CLI: speedup measurement + parity check
# ---------------------------------------------------------------------------


# Gate: every member's stacked-vs-sequential fork (RMS in FP space) must stay
# under this fraction of the seed-to-seed FP spread the A/B averages over.
# Measured CPU (RB real config, 3 seeds x 3 epochs): forks [1e-5, 0.015,
# 0.024] vs seed noise 0.62 -> ratio 0.039, so 0.2 has ~5x headroom over the
# inherent fork (GPU kernels over 30 epochs may amplify somewhat) while a
# systematic training bug (coupled members, wrong lr, coupled clip) lands at
# O(1) and still fails.
_PARITY_FORK_GATE = 0.2


def _parity_report(position: str, seq_preds: list[dict], stacked_preds: list[dict]) -> dict:
    """Decision-level parity gate + per-key diagnostics.

    Per-seed trajectory BIT-parity is ill-posed under fp32+Adam: vmapped
    batched kernels round differently than eager mm (sub-ULP), and Adam's
    step-1 update ``lr*g/(|g|+eps)`` is sign-like near g=0, so an ulp-level
    grad diff on a near-zero-grad param becomes a full-lr param diff that
    forks the whole trajectory. The fork is deterministic and SEED-dependent
    (member-order invariant — verified by swapping seeds): RB seed 42 matched
    to ~2e-4 of spread after 3 epochs while seed 43 forked to ~1e-2. Both
    arms are valid fp32 evaluations of the same algorithm — the same
    accepted-divergence physics as the CUDA-graph rebaseline (ADR-0017).

    What the A/B consumes is the seed-ensemble DISTRIBUTION (mean±std over
    seeds), so the gate is decision-level: each member's fork in FANTASY-
    POINT space (``fork_i = RMS_rows(fp_stacked_i − fp_seq_i)``) must be
    small relative to the seed-to-seed FP spread
    (``seed_noise = mean over pairs RMS_rows(fp_seq_i − fp_seq_j)``).

    The raw per-key worst-diff-over-spread stays as DIAGNOSTICS only: a
    near-constant clamped count head (e.g. RB fumbles_lost pinned at 0,
    seq std ≈ 0 → 1e-6 scale clamp) explodes that ratio even when the FP
    impact is negligible — the first Batch run reported 1.86e5 exactly so.
    """
    from src.shared.aggregate_targets import predictions_to_fantasy_points

    if len(seq_preds) < 2:
        raise SystemExit("parity gate needs >=2 seeds (seed-noise denominator)")

    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x))))

    fp_seq = [
        np.asarray(predictions_to_fantasy_points(position, p), dtype=np.float64) for p in seq_preds
    ]
    fp_stk = [
        np.asarray(predictions_to_fantasy_points(position, p), dtype=np.float64)
        for p in stacked_preds
    ]
    forks = [_rms(a - b) for a, b in zip(fp_seq, fp_stk, strict=True)]
    pairs = [
        _rms(fp_seq[i] - fp_seq[j]) for i in range(len(fp_seq)) for j in range(i + 1, len(fp_seq))
    ]
    seed_noise = float(np.mean(pairs))
    ratio = max(forks) / max(seed_noise, 1e-9)

    offenders = []
    for i, (sp, vp) in enumerate(zip(seq_preds, stacked_preds, strict=True)):
        for k in sp:
            a = np.asarray(sp[k], dtype=np.float64)
            b = np.asarray(vp[k], dtype=np.float64)
            d = float(np.max(np.abs(a - b)))
            offenders.append((d / max(float(a.std()), 1e-6), d, float(a.std()), i, k))
    offenders.sort(reverse=True)

    return {
        "fp_fork_rms_per_member": [round(f, 5) for f in forks],
        "fp_seed_noise_rms": round(seed_noise, 5),
        "fp_fork_over_seed_noise": round(ratio, 5),
        "ok": ratio < _PARITY_FORK_GATE,
        "raw_key_diagnostics_top": [
            {
                "ratio_over_spread": round(r, 3),
                "max_abs_diff": round(d, 6),
                "seq_std": round(s, 6),
                "member": i,
                "key": k,
            }
            for r, d, s, i, k in offenders[:5]
        ],
    }


def run_ensemble_ab(position: str, n_seeds: int, fixed_epochs: int, parity_check: bool) -> dict:
    """Run the stacked ensemble (and optionally the sequential parity arm);
    return the report dict. Shared by the CLI and the Batch entry."""
    apply_ensemble_env(fixed_epochs)
    if parity_check and os.environ.get("FF_FORCE_DROPOUT_ZERO", "0") != "1":
        raise SystemExit(
            "--parity-check requires FF_FORCE_DROPOUT_ZERO=1 (dropout "
            "draws are not comparable across arms otherwise)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(42, 42 + n_seeds))
    print(f"[ensemble] capturing {len(seeds)} seed constructions for {position}...", flush=True)
    t0 = time.perf_counter()
    captures, test_capture = capture_seeds(position, seeds, base_cfg=None)
    from src.shared.platform_detect import detect_platform
    from src.shared.registry import get_config

    cfg = get_config(position)
    capture_sec = time.perf_counter() - t0
    n_epochs = fixed_epochs

    print(f"[ensemble] training stacked N={len(seeds)} for {n_epochs} epochs...", flush=True)
    t0 = time.perf_counter()
    params, buffers, template = train_stacked(captures, cfg, device, n_epochs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    stacked_sec = time.perf_counter() - t0
    stacked_preds = predict_stacked(template, params, buffers, test_capture, device)

    report = {
        "position": position,
        "n_seeds": len(seeds),
        "n_epochs": n_epochs,
        "device": str(device),
        "gpu_name": detect_platform().gpu_name,
        "capture_sec": round(capture_sec, 2),
        "stacked_train_sec": round(stacked_sec, 2),
    }

    if parity_check:
        print("[ensemble] training sequential reference arm...", flush=True)
        t0 = time.perf_counter()
        seq_models = train_sequential(captures, cfg, device, n_epochs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seq_sec = time.perf_counter() - t0
        seq_preds = [predict_single(m, test_capture, device) for m in seq_models]
        report["sequential_train_sec"] = round(seq_sec, 2)
        report["speedup"] = round(seq_sec / stacked_sec, 2)
        report["parity"] = _parity_report(position, seq_preds, stacked_preds)
    return report


def run_batch_entry(position: str) -> None:
    """AWS Batch entry, reached via ``FF_TUNE_ENSEMBLE_AB=1`` in the job env.

    The training image's ENTRYPOINT is fixed to ``src.batch.train``, whose
    ``--mode=tune`` forwards a fixed argv into ``src.tuning.tune_nn.main``;
    that main dispatches here on the env flag (``containerOverrides.
    environment`` passes through the entrypoint untouched), so a one-off GPU
    run needs no ``src/batch`` edit — which would fire a 6-position retrain.

    Env knobs:
      FF_ENSEMBLE_SEEDS         ensemble width N           (default 8)
      FF_ENSEMBLE_FIXED_EPOCHS  fixed epochs per arm       (default 30)
      FF_ENSEMBLE_PARITY        "1" trains the sequential arm and FAILS the
                                job when the parity gate misses (default 1);
                                forces FF_FORCE_DROPOUT_ZERO=1

    The report JSON goes to stdout (CloudWatch) and, when ``S3_BUCKET`` is
    set, to ``s3://$S3_BUCKET/ensemble_ab/{POS}/report.json``.
    """
    # Fail before the multi-minute S3 data sync, not inside capture_seeds.
    if position not in ENSEMBLE_POSITIONS:
        raise SystemExit(f"ensemble mode supports {ENSEMBLE_POSITIONS}, got {position}")
    n_seeds = int(os.environ.get("FF_ENSEMBLE_SEEDS", "8"))
    fixed_epochs = int(os.environ.get("FF_ENSEMBLE_FIXED_EPOCHS", "30"))
    parity = os.environ.get("FF_ENSEMBLE_PARITY", "1").strip() == "1"
    if parity:
        os.environ["FF_FORCE_DROPOUT_ZERO"] = "1"

    from src.tuning.tune_nn import _ensure_data_from_s3

    _ensure_data_from_s3()
    report = run_ensemble_ab(position, n_seeds, fixed_epochs, parity_check=parity)
    print(json.dumps(report, indent=2))

    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        import boto3

        key = f"ensemble_ab/{position.upper()}/report.json"
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(report, indent=2).encode(),
            ContentType="application/json",
        )
        print(f"[ensemble] uploaded report to s3://{bucket}/{key}")

    # Fail the Batch job loudly on parity drift — the report is already in
    # S3/CloudWatch, and RETRY_STRATEGY exits (no retry) on task failures.
    if parity and not report["parity"]["ok"]:
        raise SystemExit(f"[ensemble] parity gate FAILED: {report['parity']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--position", required=True, choices=list(ENSEMBLE_POSITIONS))
    parser.add_argument("--seeds", type=int, default=8, help="ensemble width N")
    parser.add_argument("--fixed-epochs", type=int, default=30)
    parser.add_argument(
        "--parity-check",
        action="store_true",
        help="also train the sequential reference arm and compare predictions "
        "(requires FF_FORCE_DROPOUT_ZERO=1)",
    )
    args = parser.parse_args()
    report = run_ensemble_ab(args.position, args.seeds, args.fixed_epochs, args.parity_check)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
