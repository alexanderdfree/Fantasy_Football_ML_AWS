"""Shared, current-game practice reasons; no inferred severity or daily history.

These columns are deliberately absent from production feature allowlists. The
practice-context A/B opts them into both flat models and attention's static path.
Historical weekly reports support a retrospective screen, not an as-of replay.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from src.data.identity import schedule_team_code_normalization

REASON_FEATURES = (
    "practice_rest_only",
    "practice_illness",
    "practice_reason_unknown",
)
LOCATION_FEATURES = (
    "practice_lower_body",
    "practice_upper_body",
    "practice_head",
    "practice_other",
)
PRACTICE_CONTEXT_FEATURES = (*REASON_FEATURES, *LOCATION_FEATURES)
PRACTICE_STATUSES = {
    "Full Participation in Practice": 2.0,
    "Limited Participation in Practice": 1.0,
    "Did Not Participate In Practice": 0.0,
}
_KEYS = ["player_id", "season", "week"]
_EMPTY = {"", "--", "-", "none", "nan", "unknown", "unspecified", "note"}
_PATTERNS = {
    "practice_lower_body": r"\b(hip|groin|glute\w*|hamstring|quad\w*|thigh|knee|calf|shin|ankle|achilles|foot|feet|toe|heel|leg)\b",
    "practice_upper_body": r"\b(back|neck|shoulder|chest|sternum|rib\w*|arm|elbow|bicep\w*|tricep\w*|hand|finger|thumb|wrist|oblique|abdom\w*|pectoral|lat|torso|spine)\b",
    "practice_head": r"\b(head\w*|concussion|nose|face|eye|ear|jaw|dental)\b",
    "practice_illness": r"\b(illness|flu|sick\w*|covid\w*|infection|dehydration)\b",
}
_REST_ONLY = re.compile(
    r"^(?:(?:not|non)[ -]injury[ -]related\s*[-:/,]?\s*)?"
    r"(?:rest(?:ing)?(?:\s+(?:player|day))?|maintenance)(?:\s+day)?$"
)


def normalize_descriptions(values) -> list[str]:
    """Normalize source text without inventing a diagnosis for missing labels."""
    if isinstance(values, str) or values is None:
        values = [values]
    return sorted(
        {
            text
            for value in values
            if isinstance(value, str) and (text := " ".join(value.lower().split())) not in _EMPTY
        }
    )


def reason_features(descriptions, *, coverage: str = "reported") -> dict[str, float]:
    """Multi-hot reasons, with rest-only requiring every description to be rest.

    An explicitly published absence is healthy. An unavailable report or a listed
    player without usable descriptions is unknown; neither becomes a rest day.
    """
    result = dict.fromkeys(PRACTICE_CONTEXT_FEATURES, 0.0)
    if coverage == "unknown":
        result["practice_reason_unknown"] = 1.0
        return result
    labels = normalize_descriptions(descriptions)
    if not labels:
        result["practice_reason_unknown"] = float(coverage != "published_absence")
        return result
    result["practice_rest_only"] = float(all(_REST_ONLY.fullmatch(s) for s in labels))
    for label in labels:
        matched = False
        for feature, pattern in _PATTERNS.items():
            if re.search(pattern, label):
                result[feature] = 1.0
                matched = True
        if not matched and not _REST_ONLY.fullmatch(label):
            result["practice_other"] = 1.0
    # A mixed label such as "knee / rest" must never receive the rest-only flag.
    return result


def observation_features(observations: list[dict]) -> pd.DataFrame:
    """One row per player-week; callers must resolve identity before this step."""
    rows = [
        {
            **{key: row[key] for key in _KEYS},
            **reason_features(row.get("injury_descriptions", []), coverage=row["coverage"]),
        }
        for row in observations
    ]
    result = pd.DataFrame(rows, columns=[*_KEYS, *PRACTICE_CONTEXT_FEATURES])
    if result.duplicated(_KEYS).any():
        raise ValueError("Practice observations have duplicate player-week identities")
    return result


def attach_observation_features(frame: pd.DataFrame, observations: list[dict]) -> pd.DataFrame:
    """Preserve frame identity/order, including non-unique dataframe indices."""
    result = frame.copy()
    features = observation_features(observations).set_index(_KEYS)
    aligned = features.reindex(pd.MultiIndex.from_frame(frame[_KEYS]))
    for column in PRACTICE_CONTEXT_FEATURES:
        default = 1.0 if column == "practice_reason_unknown" else 0.0
        result[column] = aligned[column].fillna(default).to_numpy(dtype=float)
    return result


def historical_observations(frame: pd.DataFrame, injuries: pd.DataFrame) -> list[dict]:
    """Weekly final-report context only; no fabricated report dates or trends.

    Match the existing fallback's team-week coverage rule: only a valid reported
    status establishes a covered team. Missing seasons/teams remain unknown. Multiple
    injuries for one player are combined before testing whether the report is rest-only.
    """
    labels = defaultdict(list)
    covered = set()
    teams = schedule_team_code_normalization()
    required = {"gsis_id", "season", "week"}
    if required <= set(injuries):
        for row in injuries.to_dict("records"):
            if any(pd.isna(row[key]) for key in ("gsis_id", "season", "week")):
                continue
            if row.get("game_type", row.get("season_type", "REG")) != "REG":
                continue
            key = (str(row["gsis_id"]), int(row["season"]), int(row["week"]))
            labels[key].extend(
                normalize_descriptions(
                    [row.get("practice_primary_injury"), row.get("practice_secondary_injury")]
                )
            )
            if row.get("practice_status") in PRACTICE_STATUSES:
                team = teams.get(row.get("team"), row.get("team"))
                covered.add((team, key[1], key[2]))
    observations = []
    context = frame.reindex(columns=[*_KEYS, "recent_team"]).drop_duplicates(_KEYS)
    for row in context.to_dict("records"):
        key = (str(row["player_id"]), int(row["season"]), int(row["week"]))
        team = teams.get(row.get("recent_team"), row.get("recent_team"))
        coverage = (
            "reported"
            if key in labels
            else "published_absence"
            if (team, key[1], key[2]) in covered
            else "unknown"
        )
        observations.append(
            dict(zip(_KEYS, key, strict=True))
            | {"injury_descriptions": labels.get(key, []), "coverage": coverage}
        )
    return observations


def attach_historical_context(frame: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    return attach_observation_features(frame, historical_observations(frame, injuries))
