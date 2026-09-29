"""Explicit adapters for canonical point records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

CANONICAL_POINT_FIELDS = frozenset(
    {"point_id", "session_id", "page_id", "timestamp_ms", "sequence"}
)


class CanonicalPointAdapterError(ValueError):
    """Raised when an input record is not explicitly canonicalized."""


class PointRecordAdapter(Protocol):
    def adapt(self, records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]: ...


class CanonicalPointAdapter:
    """Validate canonical ingest names without guessing device aliases."""

    def adapt(self, records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        return adapt_canonical_records(records)


def adapt_canonical_records(records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    adapted: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise CanonicalPointAdapterError(f"record {index} must be a mapping")
        missing = sorted(CANONICAL_POINT_FIELDS.difference(record.keys()))
        if not ({"x", "y"} <= record.keys() or {"x_raw", "y_raw"} <= record.keys()):
            missing.extend(("x/y or x_raw/y_raw",))
        if missing:
            raise CanonicalPointAdapterError(
                f"record {index} is missing canonical fields: {', '.join(missing)}"
            )
        adapted.append(dict(record))
    return tuple(adapted)
