"""Search space for the attention **game-history branch** tuner (``tune_nn --scope history``).

The attention NN's history branch is the per-game sequence over
``attn_history_stats`` padded to ``attn_max_seq_len`` (see
``src/shared/neural_net.py`` ``AttentionPool`` + ``build_game_history_arrays``).
``tune_nn`` already searches the attention *sizing* knobs (``attn_d_model`` /
``attn_n_heads`` / ``attn_encoder_hidden_dim`` / ``attn_dropout`` / ``attn_lr`` /
scheduler); the two axes that defined this branch but were never tunable are the
**sequence length** (``attn_max_seq_len``, fixed at 17) and the **per-game token
set** (``attn_history_stats``, only changeable as fixed config). This module
provides both as a small, stop-rule-safe search space.

Design
------
* **Token set = bundles, not free per-column toggles.** Each position's
  production ``attn_history_stats`` is partitioned into named, contiguous
  **bundles**: one always-on ``core`` (the irreducible primary box-score
  production + ``snap_pct_raw``) plus a handful of optional bundles. The tuner
  samples one boolean per optional bundle, so the token set is a subset search
  over ``2 ** len(optional_bundles(pos))`` combinations (≤64), tractable for TPE
  with the cheap N=24 stacked-seed evaluation. Enabling **all** optional bundles
  reproduces production exactly (set-equal; ``resolve_history_stats`` preserves
  production order), so the search is anchored on the live config.
* **Subset-only in v1 (no net-new tokens).** Every bundle column is already in
  the production set, hence already in ``data/splits/*.parquet`` — so no combo
  can trip ``build_game_history_arrays``' fail-loud ``KeyError`` on a missing
  column. Adding genuinely new per-game tokens needs a ``refresh-splits`` first
  (and the windowed-column stop-rule below); that's a deliberate future
  extension, marked CANDIDATE-ADDITIONS, not v1.
* **Stop-rule guard.** ``assert_raw_per_game`` rejects windowed / expanding /
  rolling / trend columns. The history branch is for *raw per-game* signals
  genuinely absent from the sequence; a windowed/expanding-mean token re-creates
  the double-counting the static-vs-history split exists to prevent (AGENTS.md
  "Attention static-feature whitelist" + the rejected role-inheritance token).
  Subset search can't introduce such a column today, but the guard makes the
  invariant explicit and protects the CANDIDATE-ADDITIONS path.

Dependency-free on purpose (only ``re``): unit-testable on CPU with no torch /
optuna import weight, and importable by the Batch aggregation utilities.
"""

import re

# Candidate sequence lengths (number of prior in-season games the attention
# branch pools over). 17 is production (a full regular season). The branch's
# learned positional embedding is sized to this, and the history tensor is built
# at exactly this width — so each distinct value triggers one history-array
# rebuild per trial (the trial-data-memo fingerprint already keys on it).
SEQ_LEN_CHOICES: list[int] = [8, 10, 12, 15, 17, 20]

CORE_BUNDLE = "core"

# Per-position partition of the production ``attn_history_stats`` into bundles.
# INVARIANT (contract-tested in tests/tuning/test_attn_history_space.py):
#   set(core + every optional bundle) == set(production attn_history_stats)
# Keep this in sync with src/{pos}/config.py when a position's token set changes
# (the test fails loudly on drift). Bundles are listed in production order so the
# all-optional-on resolution reproduces the production list verbatim.
ATTN_HISTORY_BUNDLES: dict[str, dict[str, list[str]]] = {
    "QB": {
        "core": [
            "passing_yards",
            "rushing_yards",
            "passing_tds",
            "rushing_tds",
            "attempts",
            "completions",
            "carries",
            "interceptions",
            "fumbles_lost",
            "snap_pct_raw",
        ],
        "sacks": ["sacks", "sack_yards"],
        "expected_pts": [
            "pass_yards_gained_exp",
            "pass_touchdown_exp",
            "pass_interception_exp",
            "rush_yards_gained_exp",
            "rush_touchdown_exp",
        ],
        "advanced": ["qbr_total", "pts_added"],
        "context": ["implied_team_total", "implied_opp_total", "is_home", "days_rest"],
        "team_script": [
            "team_points_scored",
            "opp_team_points_scored",
            "team_rush_attempts",
            "team_rushing_yards",
        ],
    },
    "RB": {
        "core": [
            "rushing_yards",
            "receiving_yards",
            "rushing_tds",
            "receiving_tds",
            "carries",
            "targets",
            "receptions",
            "fumbles_lost",
            "snap_pct_raw",
        ],
        "efficiency": ["rushing_first_downs", "receiving_first_downs"],
        "share": ["game_carry_share", "game_target_share", "game_carry_hhi", "game_target_hhi"],
        "context": ["implied_team_total", "implied_opp_total", "is_home", "days_rest"],
        "team_script": [
            "team_pass_attempts",
            "team_completions",
            "team_passing_yards",
            "team_rush_attempts",
            "team_rushing_yards",
            "team_points_scored",
            "team_turnovers",
            "opp_team_points_scored",
        ],
        "redzone": [
            "redzone_carries",
            "redzone_targets",
            "inside10_carries",
            "inside5_carries",
            "redzone_target_share",
        ],
        "expected_pts": [
            "rush_yards_gained_exp",
            "rush_touchdown_exp",
            "rec_yards_gained_exp",
            "rec_touchdown_exp",
            "receptions_exp",
        ],
    },
    "WR": {
        "core": [
            "receiving_yards",
            "rushing_yards",
            "receiving_tds",
            "rushing_tds",
            "targets",
            "receptions",
            "fumbles_lost",
            "carries",
            "snap_pct_raw",
        ],
        "expected_pts": [
            "rec_yards_gained_exp",
            "rec_touchdown_exp",
            "receptions_exp",
            "rec_first_down_exp",
        ],
        "context": ["implied_team_total", "implied_opp_total", "is_home", "days_rest"],
        "team_script": [
            "team_points_scored",
            "opp_team_points_scored",
            "team_pass_attempts",
            "team_passing_yards",
            "team_rush_attempts",
        ],
        "boom": [
            "redzone_targets",
            "redzone_target_share",
            "game_target_share",
            "game_target_hhi",
            "game_opportunity_index",
        ],
    },
    "TE": {
        "core": [
            "receiving_yards",
            "rushing_yards",
            "receiving_tds",
            "rushing_tds",
            "targets",
            "receptions",
            "fumbles_lost",
            "carries",
            "snap_pct_raw",
        ],
        "expected_pts": [
            "rec_yards_gained_exp",
            "rec_touchdown_exp",
            "receptions_exp",
            "rec_first_down_exp",
        ],
        "context": ["implied_team_total", "implied_opp_total", "is_home", "days_rest"],
        "team_script": [
            "team_points_scored",
            "opp_team_points_scored",
            "team_pass_attempts",
            "team_passing_yards",
            "team_rush_attempts",
        ],
        "boom": [
            "redzone_targets",
            "redzone_target_share",
            "game_target_share",
            "game_target_hhi",
            "game_opportunity_index",
        ],
    },
}

# CANDIDATE-ADDITIONS (future extension, NOT enabled in v1): a per-position dict
# of optional bundles holding raw per-game columns NOT in the production set.
# Each must (a) be a raw per-game signal genuinely absent from the sequence, (b)
# pass assert_raw_per_game, and (c) already exist in data/splits (else
# build_game_history_arrays KeyErrors — refresh-splits first). Left empty so the
# v1 search is strictly a subset of production and can never trip the build.
ATTN_HISTORY_CANDIDATE_BUNDLES: dict[str, dict[str, list[str]]] = {}

# Windowed / expanding / rolling markers. The history branch is for raw per-game
# signals only; a windowed token belongs on the static branch (or nowhere).
_WINDOWED_SUBSTRINGS = (
    "roll",
    "ewma",
    "trend",
    "expanding",
    "career",
    "season_to_date",
)
# ``_l3`` / ``_l5`` / ``_l8`` (any ``_l<digits>``) and ``_mean`` / ``_avg`` /
# ``_std`` / ``_median`` / ``_ytd`` suffix-segments. Anchored to underscore
# boundaries so legit names ("inside10_carries", "team_turnovers", "snap_pct_raw",
# "game_opportunity_index") never match.
_WINDOWED_RE = re.compile(r"_l\d+(?:_|$)|_(?:mean|avg|std|median|ytd)(?:_|$)")


def supported_positions() -> list[str]:
    """Positions with a defined history-branch bundle map (the flat-history,
    stacked-seed-eligible set; K/DST use nested history and are excluded)."""
    return list(ATTN_HISTORY_BUNDLES)


def is_supported(position: str) -> bool:
    return position.upper() in ATTN_HISTORY_BUNDLES


def _bundles(position: str) -> dict[str, list[str]]:
    pos = position.upper()
    try:
        return ATTN_HISTORY_BUNDLES[pos]
    except KeyError:
        raise KeyError(
            f"no attn-history bundle map for {pos!r}; supported: {supported_positions()}"
        ) from None


def core_stats(position: str) -> list[str]:
    """The always-on token bundle for ``position``."""
    return list(_bundles(position)[CORE_BUNDLE])


def optional_bundles(position: str) -> list[str]:
    """Names of the searchable (toggleable) bundles, in production order."""
    return [name for name in _bundles(position) if name != CORE_BUNDLE]


def _is_windowed(col: str) -> bool:
    c = col.lower()
    if any(sub in c for sub in _WINDOWED_SUBSTRINGS):
        return True
    return _WINDOWED_RE.search(c) is not None


def assert_raw_per_game(cols: list[str]) -> None:
    """Raise ``ValueError`` if any column looks windowed / expanding / rolling.

    Enforces the history-branch stop-rule: tokens must be raw per-game signals,
    never windowed/expanding-mean derivations (those re-create the double-count
    the static-vs-history split prevents — AGENTS.md, the rejected role-
    inheritance token, [[feedback_no_rolling_in_attn_static]]).
    """
    bad = [c for c in cols if _is_windowed(c)]
    if bad:
        raise ValueError(
            "attn_history_stats must be raw per-game signals, not windowed/"
            f"expanding/rolling features; offending: {bad}"
        )


def resolve_history_stats(position: str, enabled_optional) -> list[str]:
    """Resolve ``core + enabled optional bundles`` to an ordered token list.

    ``enabled_optional`` is any iterable of optional-bundle names. Order follows
    the bundle declaration order (production order), so enabling every optional
    bundle reproduces the production ``attn_history_stats`` verbatim. Duplicates
    are removed preserving first occurrence. Result is guarded by
    ``assert_raw_per_game``.
    """
    bundles = _bundles(position)
    enabled = set(enabled_optional)
    unknown = enabled - set(optional_bundles(position))
    if unknown:
        raise KeyError(
            f"unknown {position.upper()} history bundles: {sorted(unknown)}; "
            f"valid: {optional_bundles(position)}"
        )
    cols: list[str] = list(bundles[CORE_BUNDLE])
    for name in optional_bundles(position):
        if name in enabled:
            cols.extend(bundles[name])
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    assert_raw_per_game(out)
    return out


def production_history_stats(position: str) -> list[str]:
    """The token set with every optional bundle enabled — set-equal to (and, by
    construction, order-equal to) the position's production ``attn_history_stats``.
    Used as the search anchor and as the drift contract in the unit test.
    """
    return resolve_history_stats(position, optional_bundles(position))
