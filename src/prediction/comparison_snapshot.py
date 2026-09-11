"""JSON comparison presentation assembled by the offline snapshot builder.

The numerical evaluation remains in comparison_tables. Keeping the response
assembly here lets published generations and legacy fallback share one contract
without making HTTP handlers import a builder CLI.
"""

import json
from datetime import UTC, datetime

from src.prediction import comparison
from src.shared.comparison_scoring import (
    ACTUAL_BASIS,
    EXCLUDED_COMPONENTS,
    EXCLUDED_SOURCES,
    scoring_components,
)


def build_comparison_snapshot(results, *, reference=None) -> dict:
    """Capture scores, eligibility, reference identity, and display metadata together."""
    scoring = "ppr"
    metadata = comparison._load_comparison_experts() or {}
    available = results is not None and not results.empty
    subsets, coverage, quartile_bias, rankings = comparison.comparison_tables(
        results, scoring, reference=reference
    )
    seasons = (
        sorted(int(season) for season in results["season"].dropna().unique()) if available else []
    )
    payload = {
        "scoring": scoring,
        "model_source": "live" if available else "unavailable",
        "generated_at": datetime.now(UTC).isoformat(),
        "experts_meta": metadata.get("experts_meta", {}),
        "top_n": 30,
        "top12_n": 12,
        "weekly_top_n": 24,
        "subsets": subsets,
        "coverage": coverage,
        "weekly_ranking": rankings,
        "actual_basis": ACTUAL_BASIS,
        "scoring_components": {
            pos: list(scoring_components(pos)) for pos in comparison.COMPARISON_POSITIONS
        },
        "excluded_sources": EXCLUDED_SOURCES,
        "excluded_components": EXCLUDED_COMPONENTS,
        "sample_basis": "shared_player_weeks",
        "cohort_definitions": {
            "weekly_reference_top24": "Top 24 per week by shared-component NFL.com/RotoWire mean; ESPN for K, RotoWire for DST",
            "top30": "Top 30 per season by regular-season actual shared-component points",
            "top12": "Top 12 per season by regular-season actual shared-component points",
        },
        "quartile_bias": quartile_bias,
        "quartile_bias_meta": {
            "n_quantiles": 4,
            "quartiles": list(comparison._QUARTILE_LABELS),
            "binned_by": "actual_shared_component_points",
            "bias_convention": "pred_minus_actual",
            "seasons": seasons,
        },
    }
    # Reject non-JSON/non-finite output before publication, and detach shared maps.
    return json.loads(json.dumps(payload, allow_nan=False))
