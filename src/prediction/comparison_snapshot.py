"""JSON comparison presentation assembled by the offline snapshot builder.

The numerical evaluation remains in comparison_tables. Keeping the response
assembly here lets published generations and legacy fallback share one contract
without making HTTP handlers import a builder CLI.
"""

import json
from datetime import UTC, datetime

from src.config import TEST_SEASONS
from src.prediction import comparison
from src.shared.comparison_scoring import (
    ACTUAL_BASIS,
    EXCLUDED_COMPONENTS,
    EXCLUDED_SOURCES,
    scoring_components,
)
from src.shared.comparison_uncertainty import METHOD as UNCERTAINTY_METHOD


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
            "weekly_depth_starters": (
                "Pregame depth-chart starters each week (rank 1 in the last depth chart "
                "published before game day; up to three receivers per team; every kicker "
                "and defense). Selected by neither outcomes nor any graded forecast"
            ),
            "elite_top24": (
                "The 24 players per position with the highest prior-season mean "
                "shared-component points. Selected by neither outcomes nor any graded "
                "forecast; unavailable for K and D/ST"
            ),
            "weekly_reference_top24": (
                "Top 24 per week by the archived shared-component RotoWire forecast (ESPN "
                "for K). Selected by a graded expert's own forecasts, which penalizes that "
                "expert (winner's curse); a secondary view, not the headline"
            ),
            "top30": (
                "Top 30 per season by regular-season actual shared-component points. "
                "Selected on outcomes, which favors sources that forecast stars higher"
            ),
            "top12": (
                "Top 12 per season by regular-season actual shared-component points. "
                "Selected on outcomes, which favors sources that forecast stars higher"
            ),
        },
        # The configured test season is the season A/B decisions are judged on.
        "evaluation_season_note": (
            f"Model changes were compared on {', '.join(str(season) for season in seasons)} "
            "during development, so these tables are a development-season backtest, "
            "not an untouched holdout."
            if seasons and set(seasons) <= set(TEST_SEASONS)
            else None
        ),
        "uncertainty_meta": UNCERTAINTY_METHOD,
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
