"""Flask context adapter for HTTP-independent snapshot state."""

import sys

from flask import current_app, has_app_context, has_request_context

from src.artifacts import snapshot_state as _state


def _application_state():
    return current_app.extensions.get("ffp_state") if has_app_context() else None


def _application_inference():
    return current_app.config.get("ALLOW_RUNTIME_INFERENCE", True) if has_app_context() else None


def install_snapshot_context(app):
    from flask import g, request

    @app.before_request
    def capture_snapshot():
        from src.serving import core

        core._discard_invalidated_generation()
        snapshot = None if request.path == "/warm" else _state.current_state().snapshots.current()
        # Local inference refreshes must reach the invalidation/reload path;
        # artifact-only workers never inspect model files or raw input state.
        if (
            snapshot is not None
            and snapshot.cache.get("prediction_inputs_fingerprint") is not None
            and not core._artifact_only()
            and (core._any_position_sentinel_advanced() or core._positions_pending())
        ):
            snapshot = None
        g.ffp_snapshot_token = _state._request_snapshot.set(snapshot)

    @app.after_request
    def snapshot_header(response):
        snapshot = _state.current_snapshot()
        if snapshot is not None:
            from src.artifacts import serving_snapshot
            from src.serving import core

            generation = snapshot.cache.get("snapshot_generation")
            if generation is not None and serving_snapshot.is_invalidated(
                core._PREDICTIONS_CACHE_DIR, generation
            ):
                # Keep contract/security headers already applied by other
                # after-request handlers, but discard the revoked entity tag.
                response.set_data('{"error":"Serving snapshot was revoked"}\n')
                response.status_code = 503
                response.mimetype = "application/json"
                response.headers.pop("ETag", None)
                response.headers.pop("X-FFP-Snapshot-Generation", None)
                response.headers["Cache-Control"] = "no-store"
                return response
            response.headers["X-FFP-Snapshot-Generation"] = snapshot.generation
        return response

    @app.teardown_request
    def release_snapshot(_error):
        token = g.pop("ffp_snapshot_token", None)
        if token is not None:
            _state._request_snapshot.reset(token)


_state._context_state = _application_state
_state._context_inference = _application_inference
_state._has_request_context = has_request_context
_state.install_snapshot_context = install_snapshot_context
sys.modules[__name__] = _state
