"""Explicit adapter for the frozen canonical point record contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

CANONICAL_POINT_FIELDS = frozenset(
    {
        "point_id",
        "session_id",
        "page_id",
        "x",
        "y",
        "timestamp_ms",
        "sequence",
    }
)


class CanonicalPointAdapterError(ValueError):
    """Raised when an input record is not explicitly canonicalized."""


def adapt_canonical_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate and copy canonical records without guessing device aliases.

    Additional fields are retained as source metadata, but they never replace a
    missing canonical field. Device-specific names such as ``id`` or ``time``
    must be mapped by a separate, explicitly named adapter.
    """

    required = CANONICAL_POINT_FIELDS
    adapted: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise CanonicalPointAdapterError(
                f"record {index} must be a mapping"
            )
        missing = sorted(required.difference(record.keys()))
        if missing:
            raise CanonicalPointAdapterError(
                f"record {index} is missing canonical fields: {', '.join(missing)}"
            )
        adapted.append(dict(record))
    return tuple(adapted)
