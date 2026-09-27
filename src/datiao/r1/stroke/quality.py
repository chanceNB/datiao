"""Quality flags and helpers shared by parser and stroke builder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..models.point import Point

MISSING_TIMESTAMP = "MISSING_TIMESTAMP"
INVALID_COORDINATE = "INVALID_COORDINATE"
DUPLICATE_POINT = "DUPLICATE_POINT"
OUT_OF_ORDER = "OUT_OF_ORDER"

INCOMPLETE = "INCOMPLETE"
PARTIAL_DATA = "PARTIAL_DATA"

POINT_QUALITY_FLAGS = frozenset(
    {MISSING_TIMESTAMP, INVALID_COORDINATE, DUPLICATE_POINT, OUT_OF_ORDER}
)
STROKE_QUALITY_FLAGS = frozenset({INCOMPLETE, PARTIAL_DATA})


def point_timestamp(point: Point) -> int | None:
    return point.normalized.timestamp_ms


def point_sequence(point: Point) -> int | float | None:
    value: Any = point.source_payload.get("sequence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def point_quality(point: Point) -> frozenset[str]:
    return frozenset(point.quality_flags)


def stroke_quality_flags(points: tuple[Point, ...]) -> tuple[str, ...]:
    qualities = set().union(*(point_quality(point) for point in points))
    flags: list[str] = []
    if qualities.intersection({MISSING_TIMESTAMP, INVALID_COORDINATE}):
        flags.append(INCOMPLETE)
    if qualities.intersection({DUPLICATE_POINT, OUT_OF_ORDER}):
        flags.append(PARTIAL_DATA)
    return tuple(flags)
