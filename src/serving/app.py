"""Flask web application for the Fantasy Football Points Predictor.

All predictions come from position-specific models (QB, RB, WR, TE, K, DST).
No general cross-position model is used.
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

# Boot-time S3 sync lives in gunicorn.conf.py::on_starting (master-level,
# before --preload import) so this module has no import-time side effects.
# See that hook for the rationale; cross-link kept here so future readers
# don't reach for the simpler-looking module-level call.
from src.contracts.api import install_api_contract
from src.serving import state
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

_default_state = state.DEFAULT_STATE


def __getattr__(name):
    # Read-through compatibility for callers inspecting the default app.
    if name in {"_cache", "_cache_lock", "_results_write_lock", "_wiki_cache_lock"}:
        return getattr(state, name)
    raise AttributeError(name)


def handle_api_error(error):
    if request.path.startswith("/api/"):
        if isinstance(error, HTTPException):
            return jsonify({"error": error.description}), error.code
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500
    raise error


def create_app(*, serving_state=None, snapshots=None, config=None):
    """Construct an independent HTTP application over an explicit state owner."""
    from src.serving.routes import app as routes

    application = Flask(__name__)
    application.config.update(
        ALLOW_RUNTIME_INFERENCE=os.environ.get("FF_ALLOW_RUNTIME_INFERENCE", "1").lower()
        not in {"0", "false", "off"}
    )
    if config is not None:
        application.config.update(config)
    owner = serving_state if serving_state is not None else state.ServingState()
    owner.allow_runtime_inference = application.config["ALLOW_RUNTIME_INFERENCE"]
    if snapshots is not None:
        owner.snapshots = snapshots
    application.extensions["ffp_state"] = owner
    application.register_blueprint(routes)
    application.register_error_handler(Exception, handle_api_error)
    state.install_snapshot_context(application)
    install_api_contract(application)
    return application


app = create_app(serving_state=_default_state)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(debug=debug, host="127.0.0.1", port=5050, use_reloader=False)
