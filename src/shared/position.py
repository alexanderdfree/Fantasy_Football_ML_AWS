"""Canonical ``Position`` enum.

Six positions, ordered as they appear in every dispatch site across the
project. String-valued so existing serialisation (URL query params, S3
keys, parquet column values, JSON payloads from the serving layer) keeps
working unchanged — ``Position.QB.value == "QB"`` and
``Position.QB == "QB"`` are both true.

Callers that take a position argument can declare ``Position`` instead of
``str`` to get static-typing benefits (IDE autocomplete, typo-at-import-time
detection); the runtime contract is unchanged.

The drift-guard test at
:mod:`tests.scripts.test_scope_positions` asserts
``set(Position) == set(scope_positions.ALL_POSITIONS)`` so a refactor
that adds, removes, or renames a position has to touch the enum first.

``src/scripts/scope_positions.py`` has a zero-dependency contract (it runs
inside the CI `detect` job before the project's own imports are
available), so its ``ALL_POSITIONS`` tuple keeps the plain string form
rather than depending on this enum. The drift-guard test pins the two
sources of truth together.
"""

from __future__ import annotations

from enum import StrEnum


class Position(StrEnum):
    """Six fantasy-football positions handled by the project.

    Inherits from :class:`enum.StrEnum` (Python 3.11+) so each member is a
    real ``str`` — ``Position.QB == "QB"`` and ``f"{Position.QB}" == "QB"``
    both hold, which is what serialisation (URL query params, S3 keys,
    parquet column values, JSON payloads) relies on.
    """

    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DST = "DST"

    @classmethod
    def values(cls) -> list[str]:
        """Return the string values in canonical order — useful for places
        that consume a list of position codes (registry dispatch, app.py
        loops, parametrized tests)."""
        return [member.value for member in cls]
