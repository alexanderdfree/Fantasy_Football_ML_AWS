"""Public API envelope v1 and isolated producer/consumer validation.

These definitions describe stable wire fields, not model configuration. Unknown
fields remain compatible within a major version; absent forecasts remain null.
The Flask installer only adds discovery and a version header, preserving payloads.
"""

from __future__ import annotations

import math
from urllib.parse import urlsplit

CONTRACT_VERSION = "1.0"
CONTRACT_HEADER = "X-FFP-Contract-Version"
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
SCORING_FORMATS = ("ppr", "half_ppr", "standard")
MODEL_SOURCES = ("ridge", "nn", "attn_nn", "lgbm")
EXPERT_SOURCES = ("nflcom", "rotowire", "espn")
PREDICTION_FIELDS = tuple(f"{source}_pred" for source in MODEL_SOURCES)

# An intentionally small envelope schema: structural requirements shared by the
# generated browser boundary, fixture tests, and independent Python consumers.
ENVELOPES = {
    "snapshot": {"weeks": "array", "scoring": "object", "degraded_positions": "array"},
    "predictions": {"players": "array", "scoring": "string", "total": "integer"},
    "comparison": {"scoring": "string", "model_source": "string", "subsets": "object"},
    "upcoming_available": {"available": "boolean", "scoring": "object"},
    "upcoming_unavailable": {"available": "boolean", "reason": "string"},
    "warming": {"status": "string"},
    "error": {"error": "string"},
}
API_CONTRACT = {
    "version": CONTRACT_VERSION,
    "version_header": CONTRACT_HEADER,
    "positions": list(POSITIONS),
    "scoring_formats": list(SCORING_FORMATS),
    "model_sources": list(MODEL_SOURCES),
    "expert_sources": list(EXPERT_SOURCES),
    "nullable_prediction_fields": list(PREDICTION_FIELDS),
    "envelopes": ENVELOPES,
    "endpoints": {
        "/api/snapshot": "snapshot",
        "/api/predictions": "predictions",
        "/api/comparison": "comparison",
        "/api/upcoming_week": ["upcoming_available", "upcoming_unavailable", "warming"],
    },
    "comparison": {
        "sample_basis": "shared_player_weeks",
        "actual_basis": "shared_projected_components_v2",
        "optional_metadata": [
            "coverage",
            "scoring_components",
            "excluded_sources",
            "excluded_components",
            "cohort_definitions",
        ],
    },
    "compatibility": "Optional fields and new source IDs may be added within v1; consumers reject unsupported major versions.",
}


class ContractError(ValueError):
    """A response cannot be safely consumed under the documented envelope."""


def validate_response(path: str, payload: object, status_code: int = 200) -> None:
    """Check wire envelopes and missing-versus-zero forecast semantics.

    This is deliberately not a general JSON Schema implementation. It validates
    the documented boundary for the listed endpoints and tolerates added fields.
    Domain-specific scoring/cohort consistency remains with the producer tests.
    """
    path = urlsplit(path).path
    if not isinstance(payload, dict):
        raise ContractError("Response must be an object")
    if path == "/api/upcoming_week" and payload.get("status") == "warming":
        envelope = "warming"
    elif status_code >= 400:
        envelope = "error"
    elif path == "/api/upcoming_week":
        envelope = (
            "upcoming_available" if payload.get("available") is True else "upcoming_unavailable"
        )
    else:
        envelope = API_CONTRACT["endpoints"].get(path)
    if envelope is None:
        return
    kinds = {"array": list, "object": dict, "string": str, "integer": int, "boolean": bool}
    for field, kind in ENVELOPES[envelope].items():
        if type(payload.get(field)) is not kinds[kind]:
            raise ContractError(f"{envelope}.{field} must be {kind}")
    rows = []
    if envelope in ("snapshot", "upcoming_available"):
        for scoring in SCORING_FORMATS:
            values = payload["scoring"].get(scoring)
            if not isinstance(values, list):
                raise ContractError(f"scoring.{scoring} must be an array")
            rows.extend(values)
    elif envelope == "predictions":
        rows = payload["players"]
    if envelope in ("predictions", "comparison") and payload["scoring"] not in SCORING_FORMATS:
        raise ContractError("Unknown scoring format")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("position") not in POSITIONS:
            raise ContractError(f"players[{index}] must name a known position")
        if not isinstance(row.get("player_id"), str):
            raise ContractError(f"players[{index}].player_id must be a string")
        for field in ("actual", *PREDICTION_FIELDS):
            value = row.get(field)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ContractError(f"players[{index}].{field} must be finite or null")


def install_api_contract(app) -> None:
    """Install contract discovery and headers on an independently created app."""
    from flask import jsonify, request

    if app.extensions.get("ffp_api_contract"):
        return
    app.extensions["ffp_api_contract"] = CONTRACT_VERSION
    app.add_url_rule("/api/contract", "api_contract", lambda: jsonify(API_CONTRACT))

    @app.after_request
    def contract_header(response):
        if request.path.startswith("/api/") or request.path == "/health":
            response.headers[CONTRACT_HEADER] = CONTRACT_VERSION
        return response
