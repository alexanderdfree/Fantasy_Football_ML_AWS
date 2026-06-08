"""Flask web application for the Fantasy Football Points Predictor.

All predictions come from position-specific models (QB, RB, WR, TE, K, DST).
No general cross-position model is used.
"""

import os
import sys
import threading
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib

matplotlib.use("Agg")

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from src.serving.metadata import _ALL_POSITIONS as _ALL_POSITIONS
from src.serving.metadata import _ALL_TARGETS as _ALL_TARGETS

# app.py is the composition root: it owns the Flask ``app`` + shared mutable
# state and re-exports the public symbol surface that tests / external callers
# import as ``src.serving.app.<name>``. The route handlers moved to routes.py
# (imported at the bottom); they pull these from the owning modules directly, so
# app's own code no longer references them — hence the explicit ``X as X``
# re-exports, which ruff preserves.
from src.serving.metadata import POSITION_INFO as POSITION_INFO
from src.serving.serialization import _EXPERT_PRED_PREFIXES as _EXPERT_PRED_PREFIXES
from src.serving.serialization import _MODEL_PRED_PREFIXES as _MODEL_PRED_PREFIXES
from src.serving.serialization import _actual_col as _actual_col
from src.serving.serialization import _pred_col as _pred_col
from src.serving.serialization import _records_to_player_rows as _records_to_player_rows
from src.serving.serialization import _round_or_none as _round_or_none
from src.serving.serialization import _safe_num as _safe_num
from src.serving.serialization import _safe_str as _safe_str
from src.serving.serialization import _validate_scoring as _validate_scoring
from src.serving.wiki import _WIKI_GITHUB_BLOB_BASE as _WIKI_GITHUB_BLOB_BASE
from src.serving.wiki import WIKI_DOCS as WIKI_DOCS
from src.serving.wiki import _render_wiki_doc as _render_wiki_doc
from src.serving.wiki import _wiki_rewrite_href as _wiki_rewrite_href

# Boot-time S3 sync lives in gunicorn.conf.py::on_starting (master-level,
# before --preload import) so this module has no import-time side effects.
# See that hook for the rationale; cross-link kept here so future readers
# don't reach for the simpler-looking module-level call.

app = Flask(__name__)

_cache = {}
# Serializes lazy model/data loads — Flask dispatches requests on multiple
# threads, so two concurrent first-hit requests would otherwise both see
# _cache as empty and race on duplicate I/O plus .loc-writes into the shared
# results DataFrame. Reentrant because _ensure_metrics nests into
# _ensure_position_loaded.
_cache_lock = threading.RLock()
# ``_apply_position_models`` writes per-position prediction columns into the
# shared ``_cache["results"]`` DataFrame. Even when row indices are disjoint
# across positions (QB rows vs RB rows, etc.), pandas' BlockManager is not
# thread-safe for concurrent ``.loc[]`` writes — the internal column-block
# representation is shared across columns, and two writers can corrupt block
# state mid-update. The parallel pre-warm path in ``_ensure_all_positions_loaded``
# spawns one worker per position; without this lock those workers race on the
# DataFrame's internals. Plain ``threading.Lock`` is correct here (no
# reentrancy needed — the write block doesn't call back into itself).
_results_write_lock = threading.Lock()
# Wiki page caching uses its own lock so a slow first-hit ``_ensure_metrics``
# (model loads, feature build) doesn't serialize wiki-tab GETs behind it.
# Wiki entries live in the SAME ``_cache`` dict (keyed by ("wiki", slug))
# because the existing module-global cache structure is shared; only the
# locking discipline diverges. Plain ``threading.Lock`` is sufficient — the
# wiki cache path doesn't nest into other cache helpers, so the RLock
# reentrancy that ``_cache_lock`` requires is overkill here.
#
# Originally split out under code-review finding L-SS4 (one RLock serializing
# two unrelated cache disciplines).
_wiki_cache_lock = threading.Lock()
# Benchmark-history cache (see ``_load_benchmark_history_rows``) is a third
# discipline because its invalidation is mtime-driven rather than write-driven
# and the cache structure is a tuple, not a dict slot. Documented at
# ``_BENCHMARK_HISTORY_LOCK`` near the rendering helpers.


@app.errorhandler(Exception)
def handle_api_error(e):
    """Return JSON errors for /api/ routes, default HTML for others."""
    if request.path.startswith("/api/"):
        # HTTPExceptions (404 NotFound, 405 MethodNotAllowed, 400 BadRequest,
        # ...) carry a real client-facing status — preserve it as JSON instead
        # of masking every 4xx as a 500 (which distorts ALB/monitoring error
        # counters). Their ``description`` is a safe, library-authored string,
        # unlike ``str(e)`` on an arbitrary exception.
        if isinstance(e, HTTPException):
            return jsonify({"error": e.description}), e.code
        # Unexpected server-side bug: log the full traceback server-side but
        # never echo exception text to the client. str(e) on a Python exception
        # can leak filesystem paths, config values, or library internals
        # (CodeQL py/stack-trace-exposure).
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500
    raise e


# Route handlers live in routes.py; importing the module registers them on
# ``app`` as a side effect. Kept here at the bottom (after ``app`` + shared
# state are defined) — the canonical Flask circular-import pattern.
from src.serving import routes  # noqa: E402, F401

if __name__ == "__main__":
    # Production runs under gunicorn (see Dockerfile CMD); this branch is the
    # local dev entrypoint. Debug defaults off — set FLASK_DEBUG=1 for the
    # Werkzeug debugger locally. Bound to 127.0.0.1 so the debugger console is
    # never reachable off-box even when enabled.
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(debug=debug, host="127.0.0.1", port=5050, use_reloader=False)
