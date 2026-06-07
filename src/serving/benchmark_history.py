"""History tab backend: parse + project the committed ``benchmark_history/*.json``
run summaries into the row payload the UI consumes, plus the env-overridable repo
slug used for per-run GitHub links.

Pure parsing/projection over the filesystem + ``POSITION_INFO`` target labels; the
mtime-keyed row cache (``_BENCHMARK_HISTORY_CACHE``) is module-local. No serving
model/data state — extracted from ``app.py`` during the serving decomposition; the
``/api/benchmark_history`` route calls these via ``benchmark_history.<name>``.
"""

import json
import logging
import os
import re
import threading

from src.serving.metadata import POSITION_INFO
from src.serving.serialization import _safe_num

# Env-overridable so a rename or fork doesn't silently break every History
# row's GitHub link. Defense-in-depth: validate the format before trusting
# it, since the slug ends up interpolated into an href in the frontend.
# Hostile env vars are above the user-input threat model, but cheap to guard.
_BENCHMARK_REPO_SLUG_DEFAULT = "alexanderdfree/Fantasy_Football_ML_AWS"
# Conservative subset of GitHub's actual allowed characters — enough for
# every real owner/repo combo while rejecting anything that could break out
# of the href context (quotes, angle brackets, whitespace).
_REPO_SLUG_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _resolve_repo_slug(env_value: str | None) -> str:
    """Return the env value if it parses as ``owner/repo``, else the default.

    Empty / whitespace / unset → silent fallback (the common no-override case).
    Non-empty but malformed → fallback + a warning log so an operator notices
    they typo'd the env var instead of seeing broken links in prod.
    """
    candidate = (env_value or "").strip()
    if not candidate:
        return _BENCHMARK_REPO_SLUG_DEFAULT
    if _REPO_SLUG_RE.match(candidate):
        return candidate
    logging.getLogger(__name__).warning(
        "BENCHMARK_REPO_SLUG=%r does not match owner/repo — falling back to %r",
        candidate,
        _BENCHMARK_REPO_SLUG_DEFAULT,
    )
    return _BENCHMARK_REPO_SLUG_DEFAULT


_BENCHMARK_REPO_SLUG = _resolve_repo_slug(os.environ.get("BENCHMARK_REPO_SLUG"))
_BENCHMARK_MODELS = ("ridge", "nn", "attn_nn", "lgbm")
# Canonical position order for History-tab MAE cells. Every cell renders six
# pills in this sequence; positions absent from a run's ``results`` array
# produce ``mae=None`` pills that the frontend renders as ``--``. The order
# mirrors the rest of the UI (POSITION_INFO iteration in /api/position_details).
_BENCHMARK_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")

# Flat {target_key: label} merged across positions, served alongside the History
# rows so the detailed-mode per-target breakdown can render "Passing Yards" (not
# the raw "passing_yards" key) without a second /api/position_details round-trip.
# Labels are consistent per key across positions (e.g. fumbles_lost, receiving_yards
# share one label wherever they appear), so a flat merge is unambiguous. Paired
# with TARGET_UNITS (also served top-level) for the unit suffix.
_TARGET_LABELS = {t["key"]: t["label"] for info in POSITION_INFO.values() for t in info["targets"]}

# (mtime, rows) — invalidated whenever sync_benchmark_history_from_s3 (or a
# manual write) touches benchmark_history/. Lock keeps two concurrent
# /api/benchmark_history requests from re-parsing in parallel after an
# invalidation. RLock is overkill (no nesting), Lock is sufficient.
_BENCHMARK_HISTORY_CACHE: tuple[float, list[dict]] | None = None
_BENCHMARK_HISTORY_LOCK = threading.Lock()


def _benchmark_history_dir() -> str:
    # Mirrors _render_wiki_doc's repo-root resolution: src/serving/app.py is
    # two parents deep, so the Dockerfile-copied benchmark_history/ sits at
    # <repo_root>/benchmark_history/ in both local dev and the container.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, "benchmark_history")


def _extract_per_target(raw, metric: str = "mae") -> dict:
    """Flatten a stored ``{model}_per_target`` block to ``{target: <metric>}``.

    The benchmark files store per-target metrics as
    ``{target: {"mae": float, "rmse": float, "r2": float}}`` (see
    ``src/shared/benchmark_utils.py::_per_target``). The History tab's detailed
    mode renders one metric at a time, so ``metric`` ("mae" or "rmse") selects
    which value to pull; r2 is never surfaced here. Targets whose value is
    NaN/None (scrubbed by ``_safe_num``) are dropped so the frontend never
    renders ``undefined``. Returns ``{}`` when ``raw`` is absent/not a dict, or
    when no target carries the requested metric (runs predating rmse) — callers
    treat empty as "no detail" and omit the key entirely, which keeps the
    existing exact-equality pill assertions green.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for target, v in raw.items():
        if not isinstance(v, dict):
            continue
        val = _safe_num(v.get(metric))
        if val is not None:
            out[target] = round(val, 3)
    return out


def _benchmark_row(entry: dict) -> dict:
    """Project one benchmark_history JSON into the row payload the UI consumes.

    Each model's pill list always has six entries in ``_BENCHMARK_POSITIONS``
    order. A pair (position, model) whose MAE is missing from ``results``
    (untrained position on a partial run, sentinel file for a [docs-only]
    commit, or NaN scrubbed by ``_safe_num``) carries ``mae=None``; the
    frontend renders those as ``--``. ``training_skipped`` is true when the
    file is a no-train sentinel (``results=[]`` or an explicit flag from the
    CI sentinel workflow) so the UI can style the row distinctively if it
    wants.
    """
    results = entry.get("results") or []
    by_pos = {r.get("position"): r for r in results if r.get("position")}
    pills = {m: [] for m in _BENCHMARK_MODELS}
    total_elapsed = 0.0
    for pos in _BENCHMARK_POSITIONS:
        r = by_pos.get(pos, {})
        for m in _BENCHMARK_MODELS:
            mae = _safe_num(r.get(f"{m}_mae"))
            pill = {"position": pos, "mae": round(mae, 3) if mae is not None else None}
            # rmse powers the History tab's MAE/RMSE toggle. Additive and omitted
            # when absent (runs predating rmse) so the at-a-glance pill stays
            # exactly {position, mae} and the frontend renders the RMSE view as
            # "--" for those rows.
            rmse = _safe_num(r.get(f"{m}_rmse"))
            if rmse is not None:
                pill["rmse"] = round(rmse, 3)
            # Detailed mode reads these per-target breakdowns — one map per
            # metric. Omit when there's no detail so the pill stays minimal
            # (untrained positions, old-format runs).
            per_target = _extract_per_target(r.get(f"{m}_per_target"))
            if per_target:
                pill["per_target"] = per_target
            per_target_rmse = _extract_per_target(r.get(f"{m}_per_target"), "rmse")
            if per_target_rmse:
                pill["per_target_rmse"] = per_target_rmse
            pills[m].append(pill)
        elapsed = _safe_num(r.get("elapsed_sec"))
        if elapsed is not None:
            total_elapsed += elapsed
    pr_number = entry.get("pr_number")
    return {
        "timestamp": entry.get("timestamp"),
        "git_hash": entry.get("git_hash"),
        "pr_number": int(pr_number) if isinstance(pr_number, int) else None,
        "training_skipped": bool(entry.get("training_skipped")) or len(by_pos) == 0,
        "positions": [r.get("position") for r in results if r.get("position")],
        "ridge": pills["ridge"],
        "nn": pills["nn"],
        "attn_nn": pills["attn_nn"],
        "lgbm": pills["lgbm"],
        "total_elapsed_sec": round(total_elapsed, 1),
    }


def _load_benchmark_history_rows() -> list[dict]:
    """Return the cached, projected, newest-first list of benchmark rows.

    Cache key is the directory mtime: any file landing (S3 sync at boot,
    manual write) bumps it and forces a reparse. Per-request, this is O(1)
    in the steady state instead of O(N_files * parse). Returns ``[]`` if the
    dir doesn't exist (fresh container before any sync has happened).

    INVARIANT: this cache assumes writes are atomic-rename (the .tmp +
    os.replace pattern in src/shared/benchmark_utils.py::append_to_history),
    NOT in-place edits. A rename ticks the parent-directory mtime; rewriting
    an existing file in place would not, and the cache would serve stale
    rows. If a future backfill or migration script ever rewrites entries in
    place, swap the key to ``(dir_mtime, file_count, max(file_mtimes))``.
    """
    global _BENCHMARK_HISTORY_CACHE
    history_dir = _benchmark_history_dir()
    try:
        mtime = os.path.getmtime(history_dir)
    except OSError:
        return []
    with _BENCHMARK_HISTORY_LOCK:
        cached = _BENCHMARK_HISTORY_CACHE
        if cached is not None and cached[0] == mtime:
            return cached[1]
        rows: list[dict] = []
        for fn in os.listdir(history_dir):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(history_dir, fn)
            if not os.path.isfile(path):
                continue
            try:
                with open(path) as f:
                    entry = json.load(f)
            except (OSError, ValueError):
                # A malformed file shouldn't poison the whole tab.
                continue
            rows.append(_benchmark_row(entry))
        rows.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
        _BENCHMARK_HISTORY_CACHE = (mtime, rows)
        return rows
