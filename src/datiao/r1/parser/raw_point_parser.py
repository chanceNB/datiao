"""Parse external records into immutable Point values without dropping data."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

from ..models import Point
from ..models.immutable import freeze_json
from ..stroke.quality import (
    DUPLICATE_POINT,
    INVALID_COORDINATE,
    MISSING_PAGE,
    MISSING_SEQUENCE,
    MISSING_TIMESTAMP,
    OUT_OF_ORDER,
)


def _nested_value(record: Mapping[str, Any], key: str) -> Any:
    normalized = record.get("normalized")
    if isinstance(normalized, Mapping) and key in normalized:
        return normalized[key]
    return record.get(key)


def _as_coordinate(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _as_timestamp(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _as_optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _sequence(record: Mapping[str, Any]) -> int | float | None:
    value = record.get("sequence")
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def parse_raw_points(records: Iterable[Mapping[str, Any]]) -> tuple[Point, ...]:
    """Parse records into Points while retaining every input record.

    The parser normalizes fields but stores the original record as an immutable
    source payload. Invalid coordinates and timestamps become quality flags;
    they do not cause the corresponding record to be discarded.
    """

    materialized = tuple(records)
    point_ids = [record.get("point_id") for record in materialized if isinstance(record, Mapping)]
    duplicate_ids = {
        point_id for point_id, count in Counter(point_ids).items() if point_id and count > 1
    }
    previous_sequences: dict[tuple[Any, Any], int | float] = {}
    previous_timestamps: dict[tuple[Any, Any], int] = {}
    points: list[Point] = []

    for raw_index, record in enumerate(materialized):
        if not isinstance(record, Mapping):
            raise TypeError("each raw point record must be a mapping")

        if record.get("point_id") in (None, ""):
            raise ValueError(f"record {raw_index} is missing canonical field: point_id")
        if record.get("session_id") in (None, ""):
            raise ValueError(f"record {raw_index} is missing canonical field: session_id")
        point_id = record["point_id"]
        session_id = record["session_id"]
        page_id = record.get("page_id")
        x = _as_coordinate(_nested_value(record, "x"))
        y = _as_coordinate(_nested_value(record, "y"))
        timestamp_value = _nested_value(record, "timestamp_ms")
        timestamp_ms = _as_timestamp(timestamp_value)
        flags: list[str] = []

        if x is None or y is None:
            flags.append(INVALID_COORDINATE)
        if timestamp_ms is None:
            flags.append(MISSING_TIMESTAMP)
        if page_id in (None, ""):
            flags.append(MISSING_PAGE)
        if point_id in duplicate_ids:
            flags.append(DUPLICATE_POINT)

        sequence = _sequence(record)
        if sequence is None:
            flags.append(MISSING_SEQUENCE)
        group_key = (session_id, page_id)
        previous_sequence = previous_sequences.get(group_key)
        if sequence is not None and previous_sequence is not None and sequence < previous_sequence:
            flags.append(OUT_OF_ORDER)
        if sequence is not None:
            previous_sequences[group_key] = sequence
        previous_timestamp = previous_timestamps.get(group_key)
        if (
            timestamp_ms is not None
            and previous_timestamp is not None
            and timestamp_ms < previous_timestamp
            and OUT_OF_ORDER not in flags
        ):
            flags.append(OUT_OF_ORDER)
        if timestamp_ms is not None:
            previous_timestamps[group_key] = timestamp_ms

        raw_payload = freeze_json(record)
        points.append(
            Point(
                point_id=str(point_id),
                session_id=str(session_id),
                page_id=str(page_id) if page_id is not None else None,
                raw_index=raw_index,
                normalized={
                    "x": x,
                    "y": y,
                    "timestamp_ms": timestamp_ms,
                    "pressure": _as_optional_float(_nested_value(record, "pressure")),
                    "tilt_x": _as_optional_float(_nested_value(record, "tilt_x")),
                    "tilt_y": _as_optional_float(_nested_value(record, "tilt_y")),
                },
                source_payload=raw_payload,
                quality_flags=tuple(flags),
            )
        )
    return tuple(points)
