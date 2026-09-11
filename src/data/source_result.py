"""Typed retrieval outcomes: observed emptiness is distinct from source failure."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Generic, TypeVar

import pandas as pd

T = TypeVar("T")


class SourceStatus(StrEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"


def _json_value(value):
    if isinstance(value, pd.DataFrame):
        return {
            "schema": [(str(name), str(dtype)) for name, dtype in value.dtypes.items()],
            "frame": _json_value(value.to_dict("split")),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(item) for item in value)
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def content_identity(data) -> str:
    """Order/schema-sensitive content hash, independent of retrieval time."""
    encoded = json.dumps(_json_value(data), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class SourceResult(Generic[T]):
    data: T
    provider: str
    status: SourceStatus
    retrieved_at: str
    content_id: str | None
    effective_at: str | None = None
    effective_period: Mapping[str, int] = field(default_factory=dict)
    coverage: Mapping[str, int] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    value_kind: str = "observed"

    def __post_init__(self):
        object.__setattr__(self, "effective_period", MappingProxyType(dict(self.effective_period)))
        object.__setattr__(self, "coverage", MappingProxyType(dict(self.coverage)))
        object.__setattr__(self, "errors", tuple(self.errors))

    @classmethod
    def capture(
        cls,
        data,
        *,
        provider: str,
        status=None,
        effective_at=None,
        effective_period=None,
        coverage=None,
        errors=(),
        value_kind="observed",
        retrieved_at=None,
    ):
        if status is None:
            status = SourceStatus.EMPTY if len(data) == 0 else SourceStatus.AVAILABLE
        status = SourceStatus(status)
        return cls(
            data=data,
            provider=provider,
            status=status,
            retrieved_at=retrieved_at or datetime.now(UTC).isoformat(),
            content_id=None if status is SourceStatus.UNAVAILABLE else content_identity(data),
            effective_at=effective_at,
            effective_period=effective_period or {},
            coverage=coverage or {"rows": len(data)},
            errors=tuple(errors),
            value_kind=value_kind,
        )

    def metadata(self) -> dict:
        return {
            "schema_version": 1,
            "provider": self.provider,
            "status": self.status.value,
            "retrieved_at": self.retrieved_at,
            "effective_at": self.effective_at,
            "effective_period": dict(self.effective_period),
            "content_id": self.content_id,
            "coverage": dict(self.coverage),
            "errors": list(self.errors),
            "value_kind": self.value_kind,
        }
