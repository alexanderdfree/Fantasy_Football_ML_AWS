"""WR-only batch-size sweep — sanity check for the GPU underutilization hypothesis.

For each ``attn_batch_size`` in the configured grid, this script overrides
the WR pipeline config and runs the full training path (ridge + regular NN +
attention NN; lightgbm disabled). The interesting signal is per-epoch
wall-clock of the *attention NN*: if doubling batch_size leaves wall-clock
roughly flat, the GPU was idle waiting on the host for small batches. Originally
written to profile T4 (g4dn) underutilization; now runs on L4 (g6) post-migration.

The sweep parses per-epoch lines emitted by :class:`MultiHeadTrainer`
(``epoch_sec=X.XX peak_mem_gb=Y.YY``). The marker line ``"Attention static
features:"`` printed at the start of ``_train_attention_nn`` partitions the
regular NN's epoch lines (before) from the attention NN's (after).

Run modes:

* **Batch job** (preferred — runs on the g6.xlarge L4 we want to profile)::

    python -m src.batch.train --position WR --sweep

  ``src.batch.train`` downloads splits from S3, then delegates to
  :func:`run_sweep` below.

* **Local** (CPU dev box — useful for shape-checking the harness; per-epoch
  timing is meaningless without CUDA)::

    python -m src.scripts.batch_size_sweep --data-dir data/splits
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import os
import re
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

DEFAULT_BATCH_SIZES = [128, 256, 512, 1024, 2048, 4096]
DEFAULT_N_EPOCHS = 30
_EPOCH_LINE_RE = re.compile(r"Epoch\s+\d+\s+\|.*?epoch_sec=([\d.]+)\s+peak_mem_gb=([\d.]+)")
_ATTN_MARKER = "Attention static features:"


def _parse_attn_nn_epochs(stdout: str) -> tuple[list[float], list[float]]:
    """Pick out attention NN per-epoch (epoch_sec, peak_mem_gb) from captured stdout.

    The trainer prints both regular NN and attention NN epoch lines in the
    same format. The pipeline prints ``Attention static features: …`` right
    before attention NN training starts, so we slice the captured stdout from
    that marker forward.
    """
    marker_idx = stdout.rfind(_ATTN_MARKER)
    if marker_idx < 0:
        return [], []
    tail = stdout[marker_idx:]
    epoch_secs: list[float] = []
    peak_mems: list[float] = []
    for m in _EPOCH_LINE_RE.finditer(tail):
        epoch_secs.append(float(m.group(1)))
        peak_mems.append(float(m.group(2)))
    return epoch_secs, peak_mems


def _override_wr_config(batch_size: int, n_epochs: int) -> dict:
    """Build a WR config that runs attention NN at ``batch_size`` for ``n_epochs``.

    ``nn_epochs`` controls both the regular NN and the attention NN — we eat
    the (small) cost of the regular NN training to keep the override simple.
    Patience is set well above ``n_epochs`` so early stopping never trips and
    every batch-size run captures the same number of epochs.
    """
    from src.wr.run_pipeline import CONFIG as WR_CONFIG

    cfg = copy.deepcopy(WR_CONFIG)
    cfg["nn_epochs"] = n_epochs
    cfg["nn_patience"] = n_epochs + 100
    cfg["attn_patience"] = n_epochs + 100
    cfg["attn_batch_size"] = batch_size
    cfg["nn_log_every"] = 1
    cfg["train_lightgbm"] = False
    return cfg


def run_sweep(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    batch_sizes: list[int] = DEFAULT_BATCH_SIZES,
    n_epochs: int = DEFAULT_N_EPOCHS,
    seed: int = 42,
) -> list[dict]:
    """Run the WR pipeline once per batch_size, return a list of summary dicts."""
    from src.wr.run_pipeline import run as wr_run

    rows: list[dict] = []
    for bs in batch_sizes:
        print(f"\n=== sweep batch_size={bs} (n_epochs={n_epochs}) ===", flush=True)
        cfg = _override_wr_config(bs, n_epochs)

        buf = io.StringIO()
        # contextlib.redirect_stdout captures Python prints from the trainer.
        # The pipeline doesn't shell out so we don't miss subprocess output.
        with contextlib.redirect_stdout(buf):
            result = wr_run(train_df, val_df, test_df, seed=seed, config=cfg)
        stdout = buf.getvalue()
        # Re-emit so the live log still shows pipeline progress (useful when
        # this runs inside a Batch job whose CloudWatch logs are the user's
        # only view into what happened).
        sys.stdout.write(stdout)
        sys.stdout.flush()

        epoch_secs, peak_mems = _parse_attn_nn_epochs(stdout)
        attn_nn_train_sec = (result.get("phase_seconds", {}) or {}).get("attn_nn_train", 0.0)
        rows.append(
            {
                "batch_size": bs,
                "n_epochs_captured": len(epoch_secs),
                "median_epoch_sec": (statistics.median(epoch_secs) if epoch_secs else float("nan")),
                "p95_epoch_sec": (
                    statistics.quantiles(epoch_secs, n=20)[-1]
                    if len(epoch_secs) >= 20
                    else (max(epoch_secs) if epoch_secs else float("nan"))
                ),
                "max_peak_mem_gb": max(peak_mems) if peak_mems else float("nan"),
                "attn_nn_train_sec": attn_nn_train_sec,
            }
        )
    _print_summary(rows, n_epochs)
    return rows


def _print_summary(rows: list[dict], n_epochs: int) -> None:
    print()
    print(f"WR attention NN batch-size sweep (n_epochs={n_epochs})")
    print()
    print(
        "| batch_size | epochs | median_epoch_sec | p95_epoch_sec | "
        "max_peak_mem_gb | attn_nn_train_sec |"
    )
    print(
        "|-----------:|-------:|-----------------:|--------------:|"
        "----------------:|------------------:|"
    )
    for r in rows:
        print(
            f"| {r['batch_size']:>10d} | {r['n_epochs_captured']:>6d} | "
            f"{r['median_epoch_sec']:>16.3f} | {r['p95_epoch_sec']:>13.3f} | "
            f"{r['max_peak_mem_gb']:>15.3f} | {r['attn_nn_train_sec']:>17.2f} |"
        )
    print()
    # Decision-rule hint from the investigation plan.
    if len(rows) >= 2 and not any(_isnan(r["median_epoch_sec"]) for r in rows):
        first = rows[0]["median_epoch_sec"]
        bigger = max(r["median_epoch_sec"] for r in rows)
        ratio = bigger / first if first > 0 else float("nan")
        if ratio <= 1.2:
            print(
                f"verdict: median epoch time grew {ratio:.2f}× across the grid (≤ 1.2×) "
                "— wall-clock is flat with batch_size → GPU was idle for small batches "
                "→ T4 underutilized."
            )
        elif ratio >= 0.9 * (rows[-1]["batch_size"] / rows[0]["batch_size"]):
            print(
                f"verdict: median epoch time grew {ratio:.2f}× (near-linear with "
                "batch_size) — compute-bound → T4 well utilized."
            )
        else:
            print(
                f"verdict: median epoch time grew {ratio:.2f}× — ambiguous; consider "
                "a torch.profiler trace."
            )


def _isnan(x: float) -> bool:
    return x != x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/splits",
        help="Local directory containing train/val/test parquet splits.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Attention NN batch sizes to sweep.",
    )
    parser.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(args.data_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(args.data_dir, "test.parquet"))
    print(f"Loaded WR data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    run_sweep(
        train_df,
        val_df,
        test_df,
        batch_sizes=args.batch_sizes,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
